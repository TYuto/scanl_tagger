import atexit
import signal
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context

from flask import Flask, jsonify, request
from waitress import serve

from src.lm_runtime_utils import LMRuntime, build_lm_model_path, load_serve_config


app = Flask(__name__)

CATEGORY_CONTEXTS = {
    "functions": "FUNCTION",
    "parameters": "PARAMETER",
    "variables": "DECLARATION",
}

_WORKER_RUNTIME = None
_SERVICE = None


def _initialize_worker(model_path, local, require_gpu):
    global _WORKER_RUNTIME
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_RUNTIME = LMRuntime(
        model_path=model_path,
        local=local,
        require_gpu=require_gpu,
    )


def _worker_status():
    if _WORKER_RUNTIME is None:
        return {"ready": False}
    return {
        "ready": True,
        "model_device": _WORKER_RUNTIME.model_device,
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


class IdentifierBatchService:
    def __init__(self, model_path, local=False, cache_size=50000, worker_processes=1, require_gpu=True):
        self.cache = IdentifierLRUCache(cache_size)
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
        self.executor = ProcessPoolExecutor(
            max_workers=worker_processes,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(model_path, local, require_gpu),
        )
        self.worker_info = self.executor.submit(_worker_status).result()
        self.worker_processes = worker_processes

    def shutdown(self):
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self.executor.shutdown(wait=True, cancel_futures=True)
            self._is_shutdown = True

    def _cache_key(self, item):
        return (
            item["identifier_name"],
            item["identifier_context"],
            item["system_name"],
            item["programming_language"],
            item["data_type"],
        )

    def _response_entry(self, item, result, cache_hit):
        return {
            "identifier_name": item["identifier_name"],
            "context": item["identifier_context"],
            "cache_hit": cache_hit,
            "result": deepcopy(result),
        }

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

        cache_hits = 0
        pending_by_key = {}
        miss_payloads = {}

        for item in normalized_items:
            key = self._cache_key(item)
            cached = self.cache.get(key)
            if cached is not None:
                cache_hits += 1
                grouped_results[item["category"]][item["index"]] = self._response_entry(item, cached, True)
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
            miss_results = self.executor.submit(
                _infer_identifier_batch,
                list(miss_payloads.values()),
            ).result()

        for key, result in zip(miss_payloads.keys(), miss_results):
            self.cache.put(key, result)
            for item in pending_by_key[key]:
                grouped_results[item["category"]][item["index"]] = self._response_entry(item, result, False)

        return {
            "functions": grouped_results["functions"],
            "parameters": grouped_results["parameters"],
            "variables": grouped_results["variables"],
            "cache": {
                "hits": cache_hits,
                "misses": len(miss_payloads),
                "entries": len(self.cache),
            },
            "worker": {
                "processes": self.worker_processes,
                **self.worker_info,
            },
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

    payload = request.get_json(force=True, silent=False)
    try:
        return jsonify(_SERVICE.handle_payload(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


def start_post_batch_server(
    address=None,
    port=None,
    protocol=None,
    model=None,
    local=False,
    cache_size=50000,
    worker_processes=1,
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
        require_gpu=require_gpu,
    )
    atexit.register(_SERVICE.shutdown)

    print(f"Starting POST batch server on {server_host}:{server_port} ({server_url_scheme})")
    print(f"Waitress threads: {threads}")
    print(f"Inference worker processes: {worker_processes}")
    print(f"Inference worker device: {_SERVICE.worker_info.get('model_device', 'unknown')}")
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
