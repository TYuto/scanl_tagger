#!/usr/bin/env python3

import argparse

from src.post_batch_server import start_post_batch_server


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Start a POST-based SCALAR batch server that accepts functions, parameters, "
            "and variables in one JSON payload."
        )
    )
    parser.add_argument("--address", type=str, help="Server bind address")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--protocol", type=str, help="Waitress url scheme")
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of Waitress threads used to accept and wait on HTTP requests",
    )
    parser.add_argument("--model", type=str, help="Explicit model path or Hugging Face repo id")
    parser.add_argument("--local", action="store_true", help="Use a local model directory")
    parser.add_argument("--cache-size", type=int, default=50000, help="LRU cache size")
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated GPU ids like '0,1', or 'all' to use every visible GPU",
    )
    parser.add_argument(
        "--worker-processes",
        type=int,
        default=1,
        help="Total number of dedicated inference worker processes across all configured devices",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow the server to start even if CUDA is unavailable",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_post_batch_server(
        address=args.address,
        port=args.port,
        protocol=args.protocol,
        threads=args.threads,
        model=args.model,
        local=args.local,
        cache_size=args.cache_size,
        worker_processes=args.worker_processes,
        gpus=args.gpus,
        require_gpu=not args.allow_cpu_fallback,
    )
