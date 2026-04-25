import atexit
import logging
import os
import signal
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context

from flask import Flask, jsonify, request
import torch
from werkzeug.exceptions import BadRequest
from waitress import serve

from src.lm_runtime_utils import LMRuntime, build_lm_model_path, load_serve_config


app = Flask(__name__)
logger = logging.getLogger(__name__)

CATEGORY_CONTEXTS = {
    "functions": "FUNCTION",
    "parameters": "PARAMETER",
    "variables": "DECLARATION",
}

_WORKER_RUNTIME = None
_SERVICE = None
_WARMUP_PAYLOAD = {
    "identifier_name": "index",
    "identifier_context": "FUNCTION",
    "system_name": "",
    "programming_language": "",
    "data_type": "",
}


def _parse_device_names(gpus, require_gpu):
    if gpus:
        if gpus.strip().lower() == "all":
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            try:
                gpu_ids = [int(item.strip()) for item in gpus.split(",") if item.strip()]
            except ValueError as exc:
                raise RuntimeError("GPU ids must be integers like '0,1' or the literal 'all'.") from exc

        if not gpu_ids:
            raise RuntimeError("At least one GPU id must be provided.")

        visible_device_count = torch.cuda.device_count()
        if visible_device_count == 0:
            if require_gpu:
                raise RuntimeError("GPU execution was requested, but CUDA is not available.")
            return ["cpu"]

        invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_device_count]
        if invalid_gpu_ids:
            raise RuntimeError(
                f"Invalid GPU ids requested: {invalid_gpu_ids}. Visible GPU count is {visible_device_count}."
            )

        return [f"cuda:{gpu_id}" for gpu_id in gpu_ids]

    if torch.cuda.is_available():
        return ["cuda:0"]

    if require_gpu:
        raise RuntimeError("GPU execution was requested, but CUDA is not available.")

    return ["cpu"]


def _distribute_workers(worker_processes, device_names):
    if worker_processes < 1:
        raise RuntimeError("worker_processes must be at least 1.")

    worker_counts = [0] * len(device_names)
    for worker_index in range(worker_processes):
        worker_counts[worker_index % len(device_names)] += 1

    return [
        (device_name, worker_count)
        for device_name, worker_count in zip(device_names, worker_counts)
        if worker_count > 0
    ]


def _initialize_worker(model_path, local, require_gpu, device):
    global _WORKER_RUNTIME
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    _WORKER_RUNTIME = LMRuntime(
        model_path=model_path,
        local=local,
        device=device,
        require_gpu=require_gpu,
    )


def _worker_status():
    if _WORKER_RUNTIME is None:
        return {"ready": False}
    return {
        "ready": True,
        "model_device": _WORKER_RUNTIME.model_device,
    }


def _warm_worker():
    _infer_identifier_batch([_WARMUP_PAYLOAD])
    return {
        "pid": os.getpid(),
        **_worker_status(),
        "warmup_identifier": _WARMUP_PAYLOAD["identifier_name"],
    }


def _infer_identifier_batch(request_payloads):
    if _WORKER_RUNTIME is None:
        raise RuntimeError("Inference worker is not initialized.")
    return _WORKER_RUNTIME.tag_identifier_batch_results(request_payloads)


class IdentifierLRUCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._items:
                return None
            value = self._items.pop(key)
            self._items[key] = value
            return deepcopy(value)

    def put(self, key, value):
        with self._lock:
            if key in self._items:
                self._items.pop(key)
            self._items[key] = deepcopy(value)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._items)


class DeviceWorkerPool:
    def __init__(self, model_path, local, require_gpu, device, worker_count):
        self.device = device
        self.worker_count = worker_count
        self._lock = threading.Lock()
        self.inflight = 0
        self.executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(model_path, local, require_gpu, device),
        )
        self.worker_pool_info = self._warm_all_workers()
        self.info = {
            "ready": all(item.get("ready", False) for item in self.worker_pool_info),
            "device": device,
            "worker_count": len(self.worker_pool_info),
            "worker_pids": [item["pid"] for item in self.worker_pool_info],
        }

    def _warm_all_workers(self):
        futures = [
            self.executor.submit(_warm_worker)
            for _ in range(self.worker_count)
        ]
        warmed_workers = []
        for future in futures:
            try:
                warmed_workers.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"Inference worker failed during startup warm-up on {self.device}. "
                    "This usually points to CUDA/device initialization or first-forward allocation issues. "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        unique_pids = {item["pid"] for item in warmed_workers}
        if len(unique_pids) != self.worker_count:
            raise RuntimeError(
                f"Failed to initialize the expected number of inference workers on {self.device}. "
                f"expected={self.worker_count} actual={len(unique_pids)}"
            )
        return warmed_workers

    def run_batch(self, request_payloads):
        with self._lock:
            self.inflight += 1
        try:
            return self.executor.submit(_infer_identifier_batch, request_payloads).result()
        finally:
            with self._lock:
                self.inflight = max(self.inflight - 1, 0)

    def shutdown(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


class IdentifierBatchService:
    def __init__(self, model_path, local=False, cache_size=50000, worker_processes=1, require_gpu=True, gpus=None):
        self.cache = IdentifierLRUCache(cache_size)
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
        self.worker_processes = worker_processes
        self.device_names = _parse_device_names(gpus, require_gpu=require_gpu)
        self.worker_assignments = _distribute_workers(worker_processes, self.device_names)
        self.device_pools = [
            DeviceWorkerPool(
                model_path=model_path,
                local=local,
                require_gpu=require_gpu,
                device=device_name,
                worker_count=worker_count,
            )
            for device_name, worker_count in self.worker_assignments
        ]
        self._pool_lock = threading.Lock()
        self._next_pool_index = 0
        self.worker_info = {
            "ready": all(pool.info.get("ready", False) for pool in self.device_pools),
            "devices": [pool.info for pool in self.device_pools],
            "worker_count": sum(pool.info["worker_count"] for pool in self.device_pools),
        }

    def shutdown(self):
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            for pool in reversed(self.device_pools):
                pool.shutdown()
            self._is_shutdown = True

    def _acquire_pool(self):
        if len(self.device_pools) == 1:
            return self.device_pools[0]

        with self._pool_lock:
            best_index = 0
            best_key = None
            for offset in range(len(self.device_pools)):
                pool_index = (self._next_pool_index + offset) % len(self.device_pools)
                pool = self.device_pools[pool_index]
                load = pool.inflight / pool.worker_count
                key = (load, pool.inflight, pool_index)
                if best_key is None or key < best_key:
                    best_key = key
                    best_index = pool_index

            self._next_pool_index = (best_index + 1) % len(self.device_pools)
            return self.device_pools[best_index]

    def _cache_key(self, item):
        return (
            item["identifier_name"],
            item["identifier_context"],
            item["system_name"],
            item["programming_language"],
            item["data_type"],
        )

    def _response_entry(self, result):
        return deepcopy(result)

    def _normalize_item(self, category, index, item, payload):
        global_context = payload.get("contexts", {}).get(category, CATEGORY_CONTEXTS[category])
        global_system_name = payload.get("system_name", "")
        global_language = payload.get("language", payload.get("programming_language", ""))
        global_data_type = payload.get("type", payload.get("data_type", ""))

        if isinstance(item, str):
            identifier_name = item
            identifier_context = global_context
            system_name = global_system_name
            programming_language = global_language
            data_type = global_data_type
        elif isinstance(item, dict):
            identifier_name = item.get("identifier_name", item.get("name"))
            identifier_context = item.get("context", global_context)
            system_name = item.get("system_name", global_system_name)
            programming_language = item.get("language", item.get("programming_language", global_language))
            data_type = item.get("type", item.get("data_type", global_data_type))
        else:
            raise ValueError(f"{category}[{index}] must be a string or object.")

        if not isinstance(identifier_name, str) or not identifier_name.strip():
            raise ValueError(f"{category}[{index}] is missing a valid identifier name.")
        if not isinstance(identifier_context, str) or not identifier_context.strip():
            raise ValueError(f"{category}[{index}] is missing a valid context.")

        return {
            "category": category,
            "index": index,
            "identifier_name": identifier_name.strip(),
            "identifier_context": identifier_context.strip(),
            "system_name": system_name,
            "programming_language": programming_language,
            "data_type": data_type,
        }

    def _normalize_payload(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")

        normalized = []
        for category in CATEGORY_CONTEXTS:
            raw_items = payload.get(category, [])
            if raw_items is None:
                raw_items = []
            if not isinstance(raw_items, list):
                raise ValueError(f"'{category}' must be a JSON array.")

            for index, item in enumerate(raw_items):
                normalized.append(self._normalize_item(category, index, item, payload))

        return normalized

    def handle_payload(self, payload):
        normalized_items = self._normalize_payload(payload)
        grouped_results = {
            category: [None] * len(payload.get(category, []) or [])
            for category in CATEGORY_CONTEXTS
        }

        pending_by_key = {}
        miss_payloads = {}

        for item in normalized_items:
            key = self._cache_key(item)
            cached = self.cache.get(key)
            if cached is not None:
                grouped_results[item["category"]][item["index"]] = self._response_entry(cached)
                continue

            pending_by_key.setdefault(key, []).append(item)
            if key not in miss_payloads:
                miss_payloads[key] = {
                    "identifier_name": item["identifier_name"],
                    "identifier_context": item["identifier_context"],
                    "system_name": item["system_name"],
                    "programming_language": item["programming_language"],
                    "data_type": item["data_type"],
                }

        miss_results = []
        if miss_payloads:
            selected_pool = self._acquire_pool()
            miss_results = selected_pool.run_batch(list(miss_payloads.values()))

        for key, result in zip(miss_payloads.keys(), miss_results):
            self.cache.put(key, result)
            for item in pending_by_key[key]:
                grouped_results[item["category"]][item["index"]] = self._response_entry(result)

        return {
            "functions": grouped_results["functions"],
            "parameters": grouped_results["parameters"],
            "variables": grouped_results["variables"],
        }


@app.get("/health")
def health():
    if _SERVICE is None:
        return jsonify({"ready": False}), 503

    return jsonify({
        "ready": True,
        "worker": _SERVICE.worker_info,
        "cache_entries": len(_SERVICE.cache),
    })


@app.post("/tag")
def tag_identifiers():
    if _SERVICE is None:
        return jsonify({"error": "Service has not been initialized."}), 503

    try:
        payload = request.get_json(force=True, silent=False)
        return jsonify(_SERVICE.handle_payload(payload))
    except BadRequest as exc:
        return jsonify({"error": "Request body must be valid JSON.", "details": str(exc)}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("POST /tag failed")
        return jsonify({
            "error": "Internal server error during identifier tagging.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 500


def start_post_batch_server(
    address=None,
    port=None,
    protocol=None,
    model=None,
    local=False,
    cache_size=50000,
    worker_processes=1,
    gpus=None,
    threads=4,
    require_gpu=True,
):
    global _SERVICE

    config = load_serve_config()
    server_host = address or config["address"]
    server_port = port or config["port"]
    server_url_scheme = protocol or config["protocol"]
    model_path = build_lm_model_path(local=local, model=model)

    _SERVICE = IdentifierBatchService(
        model_path=model_path,
        local=local,
        cache_size=cache_size,
        worker_processes=worker_processes,
        gpus=gpus,
        require_gpu=require_gpu,
    )
    atexit.register(_SERVICE.shutdown)

    print(f"Starting POST batch server on {server_host}:{server_port} ({server_url_scheme})")
    print(f"Waitress threads: {threads}")
    print(f"Inference worker processes: {worker_processes}")
    print(f"Inference devices: {', '.join(_SERVICE.device_names)}")
    print(f"Warmed inference workers: {_SERVICE.worker_info.get('worker_count', 'unknown')}")
    for pool in _SERVICE.worker_info.get("devices", []):
        print(
            f"  - {pool.get('device', 'unknown')}: "
            f"{pool.get('worker_count', 'unknown')} workers "
            f"(pids={pool.get('worker_pids', [])})"
        )
    if require_gpu and len(_SERVICE.device_names) == 1 and worker_processes > 1:
        print(
            "WARNING: worker_processes > 1 with GPU inference loads one model copy per worker. "
            "On a single GPU this can reduce throughput or exhaust VRAM. "
            "Prefer --worker-processes 1, or spread workers across multiple GPUs with --gpus 0,1."
        )
    try:
        serve(
            app,
            host=server_host,
            port=server_port,
            url_scheme=server_url_scheme,
            threads=threads,
        )
    finally:
        _SERVICE.shutdown()
