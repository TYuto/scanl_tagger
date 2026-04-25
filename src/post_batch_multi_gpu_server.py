import atexit
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import torch
from flask import Flask, jsonify, request
from waitress import serve


app = Flask(__name__)

_SERVICE = None


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _python_executable():
    return sys.executable


def _parse_gpu_list(gpus_arg, allow_cpu_fallback=False):
    if gpus_arg:
        gpu_ids = [item.strip() for item in gpus_arg.split(",") if item.strip()]
    else:
        visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_env:
            gpu_ids = [item.strip() for item in visible_env.split(",") if item.strip()]
        else:
            gpu_ids = [str(index) for index in range(torch.cuda.device_count())]

    if not gpu_ids and allow_cpu_fallback:
        return ["cpu"]

    if not gpu_ids:
        raise RuntimeError("No visible GPUs were found for multi-GPU server startup.")

    return gpu_ids


class BackendServer:
    def __init__(self, index, gpu_id, port, process):
        self.index = index
        self.gpu_id = gpu_id
        self.port = port
        self.process = process
        self.url = f"http://127.0.0.1:{port}"
        self.inflight = 0
        self.completed_requests = 0
        self.failed_requests = 0


class MultiGPUProxyService:
    def __init__(
        self,
        address,
        port,
        protocol,
        threads,
        gpus,
        backend_port_start,
        backend_threads,
        backend_worker_processes,
        cache_size,
        model,
        local,
        allow_cpu_fallback,
        startup_timeout,
    ):
        self.address = address
        self.port = port
        self.protocol = protocol
        self.threads = threads
        self.gpu_ids = _parse_gpu_list(gpus, allow_cpu_fallback=allow_cpu_fallback)
        self.backend_port_start = backend_port_start
        self.backend_threads = backend_threads
        self.backend_worker_processes = backend_worker_processes
        self.cache_size = cache_size
        self.model = model
        self.local = local
        self.allow_cpu_fallback = allow_cpu_fallback
        self.startup_timeout = startup_timeout
        self.backends = []
        self._backend_lock = threading.Lock()
        self._next_backend_index = 0

    def start(self):
        try:
            for index, gpu_id in enumerate(self.gpu_ids):
                backend_port = self.backend_port_start + index
                process = self._spawn_backend(gpu_id=gpu_id, backend_port=backend_port)
                backend = BackendServer(index=index, gpu_id=gpu_id, port=backend_port, process=process)
                self.backends.append(backend)

            for backend in self.backends:
                self._wait_until_ready(backend)
        except Exception:
            self.shutdown()
            raise

    def shutdown(self):
        for backend in reversed(self.backends):
            process = backend.process
            if process.poll() is not None:
                continue

            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _spawn_backend(self, gpu_id, backend_port):
        env = os.environ.copy()
        if gpu_id == "cpu":
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            _python_executable(),
            os.path.join(_repo_root(), "serve_post_batch.py"),
            "--address", "127.0.0.1",
            "--port", str(backend_port),
            "--protocol", self.protocol,
            "--threads", str(self.backend_threads),
            "--worker-processes", str(self.backend_worker_processes),
            "--cache-size", str(self.cache_size),
        ]

        if self.model:
            cmd.extend(["--model", self.model])
        if self.local:
            cmd.append("--local")
        if self.allow_cpu_fallback:
            cmd.append("--allow-cpu-fallback")

        return subprocess.Popen(
            cmd,
            cwd=_repo_root(),
            env=env,
        )

    def _wait_until_ready(self, backend):
        deadline = time.time() + self.startup_timeout

        while time.time() < deadline:
            if backend.process.poll() is not None:
                raise RuntimeError(
                    f"Backend for {self._backend_label(backend)} exited early with code {backend.process.returncode}."
                )

            try:
                health = self._fetch_json(backend.url + "/health")
                if health.get("ready"):
                    return
            except Exception:
                pass

            time.sleep(0.5)

        raise RuntimeError(f"Backend for {self._backend_label(backend)} did not become ready in time.")

    def _fetch_json(self, url):
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _backend_label(self, backend):
        if backend.gpu_id == "cpu":
            return "CPU fallback backend"
        return f"GPU {backend.gpu_id}"

    def _acquire_backend(self, _request_body):
        if len(self.backends) == 1:
            backend = self.backends[0]
            with self._backend_lock:
                backend.inflight += 1
            return backend

        with self._backend_lock:
            minimum_inflight = min(backend.inflight for backend in self.backends)
            candidate_indexes = [
                index
                for index, backend in enumerate(self.backends)
                if backend.inflight == minimum_inflight
            ]

            if len(candidate_indexes) == 1:
                chosen_index = candidate_indexes[0]
            else:
                start = self._next_backend_index % len(self.backends)
                chosen_index = candidate_indexes[0]
                for offset in range(len(self.backends)):
                    candidate_index = (start + offset) % len(self.backends)
                    if candidate_index in candidate_indexes:
                        chosen_index = candidate_index
                        break

            backend = self.backends[chosen_index]
            backend.inflight += 1
            self._next_backend_index = (chosen_index + 1) % len(self.backends)
            return backend

    def _release_backend(self, backend, success):
        with self._backend_lock:
            backend.inflight = max(backend.inflight - 1, 0)
            if success:
                backend.completed_requests += 1
            else:
                backend.failed_requests += 1

    def forward_tag_request(self, request_body, content_type):
        backend = self._acquire_backend(request_body)
        req = urllib.request.Request(
            backend.url + "/tag",
            data=request_body,
            headers={"Content-Type": content_type},
            method="POST",
        )

        success = False
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                body = response.read()
                status = response.status
                mime = response.headers.get_content_type()
                success = status < 500
        except urllib.error.HTTPError as exc:
            body = exc.read()
            status = exc.code
            mime = exc.headers.get_content_type() if exc.headers else "application/json"
            success = status < 500
        except urllib.error.URLError as exc:
            error_body = {
                "error": f"Backend request failed for {self._backend_label(backend)}: {exc.reason}",
            }
            body = json.dumps(error_body).encode("utf-8")
            status = 502
            mime = "application/json"
        finally:
            self._release_backend(backend, success)

        return body, status, mime, backend

    def aggregated_health(self):
        backend_states = []
        overall_ready = True

        for backend in self.backends:
            try:
                payload = self._fetch_json(backend.url + "/health")
            except Exception as exc:
                overall_ready = False
                backend_states.append({
                    "gpu_id": backend.gpu_id,
                    "port": backend.port,
                    "inflight": backend.inflight,
                    "completed_requests": backend.completed_requests,
                    "failed_requests": backend.failed_requests,
                    "ready": False,
                    "error": str(exc),
                })
                continue

            if not payload.get("ready"):
                overall_ready = False

            backend_states.append({
                "gpu_id": backend.gpu_id,
                "port": backend.port,
                "inflight": backend.inflight,
                "completed_requests": backend.completed_requests,
                "failed_requests": backend.failed_requests,
                **payload,
            })

        return {
            "ready": overall_ready,
            "proxy": {
                "address": self.address,
                "port": self.port,
                "threads": self.threads,
            },
            "backends": backend_states,
        }


@app.get("/health")
def health():
    if _SERVICE is None:
        return jsonify({"ready": False}), 503
    return jsonify(_SERVICE.aggregated_health())


@app.post("/tag")
def tag():
    if _SERVICE is None:
        return jsonify({"error": "Service has not been initialized."}), 503

    request_body = request.get_data()
    content_type = request.headers.get("Content-Type", "application/json")
    body, status, mime, backend = _SERVICE.forward_tag_request(request_body, content_type)
    response = app.response_class(body, status=status, mimetype=mime)
    response.headers["X-Backend-GPU"] = str(backend.gpu_id)
    response.headers["X-Backend-Port"] = str(backend.port)
    return response


def start_post_batch_multi_gpu_server(
    address=None,
    port=8091,
    protocol="http",
    threads=16,
    gpus=None,
    backend_port_start=19091,
    backend_threads=4,
    backend_worker_processes=1,
    cache_size=50000,
    model=None,
    local=False,
    allow_cpu_fallback=False,
    startup_timeout=120,
):
    global _SERVICE

    service = MultiGPUProxyService(
        address=address or "0.0.0.0",
        port=port,
        protocol=protocol,
        threads=threads,
        gpus=gpus,
        backend_port_start=backend_port_start,
        backend_threads=backend_threads,
        backend_worker_processes=backend_worker_processes,
        cache_size=cache_size,
        model=model,
        local=local,
        allow_cpu_fallback=allow_cpu_fallback,
        startup_timeout=startup_timeout,
    )
    service.start()
    _SERVICE = service
    atexit.register(service.shutdown)

    print(f"Starting multi-GPU proxy server on {service.address}:{service.port} ({service.protocol})")
    print(f"Proxy threads: {service.threads}")
    print(f"Visible GPU ids: {', '.join(service.gpu_ids)}")
    print(f"Backend ports: {', '.join(str(backend.port) for backend in service.backends)}")
    if service.backend_worker_processes > 1:
        print(
            "WARNING: backend_worker_processes > 1 loads multiple model copies per GPU backend. "
            "Start with --backend-worker-processes 1 unless measurements show a clear benefit."
        )

    try:
        serve(
            app,
            host=service.address,
            port=service.port,
            url_scheme=service.protocol,
            threads=service.threads,
        )
    finally:
        service.shutdown()
