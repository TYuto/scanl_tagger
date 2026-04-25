#!/usr/bin/env python3

import argparse

from src.post_batch_multi_gpu_server import start_post_batch_multi_gpu_server


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Start a multi-GPU POST batch server. "
            "One backend process is started per visible GPU and the proxy balances requests across them."
        )
    )
    parser.add_argument("--address", type=str, help="Proxy bind address")
    parser.add_argument("--port", type=int, default=8091, help="Public proxy port")
    parser.add_argument("--protocol", type=str, default="http", help="Waitress url scheme")
    parser.add_argument("--threads", type=int, default=16, help="Waitress threads for the public proxy")
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated physical GPU ids to use. If omitted, use currently visible GPUs.",
    )
    parser.add_argument(
        "--backend-port-start",
        type=int,
        default=19091,
        help="Starting port for per-GPU backend servers",
    )
    parser.add_argument(
        "--backend-threads",
        type=int,
        default=4,
        help="Waitress threads used by each per-GPU backend server",
    )
    parser.add_argument(
        "--backend-worker-processes",
        type=int,
        default=1,
        help="Inference worker processes per backend server",
    )
    parser.add_argument("--cache-size", type=int, default=50000, help="Per-backend LRU cache size")
    parser.add_argument("--model", type=str, help="Explicit model path or Hugging Face repo id")
    parser.add_argument("--local", action="store_true", help="Use a local model directory")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow startup even if no GPU is visible",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=120,
        help="Seconds to wait for each backend to become healthy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_post_batch_multi_gpu_server(
        address=args.address,
        port=args.port,
        protocol=args.protocol,
        threads=args.threads,
        gpus=args.gpus,
        backend_port_start=args.backend_port_start,
        backend_threads=args.backend_threads,
        backend_worker_processes=args.backend_worker_processes,
        cache_size=args.cache_size,
        model=args.model,
        local=args.local,
        allow_cpu_fallback=args.allow_cpu_fallback,
        startup_timeout=args.startup_timeout,
    )
