#!/usr/bin/env python3

import argparse
import asyncio
import json
import time

import aiohttp


DEFAULT_PAYLOAD = {
    "functions": ["get_user_name", "validate_input"],
    "parameters": ["user_id", "max_count"],
    "variables": ["tmp_result", "cache_map"],
}

DEFAULT_SYMBOLS = ["_", "$", "@", "#", "%", "&"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Send repeated POST requests to the batch tag server while avoiding LRU cache hits "
            "by decorating identifiers with numeric and symbolic prefixes/suffixes."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080/tag", help="POST endpoint")
    parser.add_argument("--requests", type=int, default=1000, help="Total number of POST requests to send")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of in-flight requests")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--payload-json",
        type=str,
        help="Optional JSON string overriding the default payload template",
    )
    parser.add_argument(
        "--payload-file",
        type=str,
        help="Optional path to a JSON file overriding the default payload template",
    )
    parser.add_argument(
        "--repeat-items",
        type=int,
        default=1,
        help="Repeat each list item this many times inside every request payload",
    )
    parser.add_argument(
        "--prefix-symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbol set used in prefixes",
    )
    parser.add_argument(
        "--suffix-symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbol set used in suffixes",
    )
    parser.add_argument(
        "--print-failures",
        action="store_true",
        help="Print response bodies for failed requests",
    )
    return parser.parse_args()


def load_payload_template(args):
    if args.payload_json:
        payload = json.loads(args.payload_json)
    elif args.payload_file:
        with open(args.payload_file) as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(json.dumps(DEFAULT_PAYLOAD))

    if args.repeat_items < 1:
        raise ValueError("--repeat-items must be at least 1")

    if args.repeat_items == 1:
        return payload

    repeated_payload = {}
    for key, value in payload.items():
        if isinstance(value, list):
            repeated_items = []
            for _ in range(args.repeat_items):
                repeated_items.extend(json.loads(json.dumps(value)))
            repeated_payload[key] = repeated_items
        else:
            repeated_payload[key] = value
    return repeated_payload


def decorate_identifier(name, request_index, item_index, prefix_symbols, suffix_symbols):
    prefix_symbol = prefix_symbols[(request_index + item_index) % len(prefix_symbols)]
    suffix_symbol = suffix_symbols[(request_index * 3 + item_index) % len(suffix_symbols)]
    prefix = f"{prefix_symbol}{request_index:06d}{item_index:02d}{prefix_symbol}"
    suffix = f"{suffix_symbol}{item_index:02d}{request_index:06d}{suffix_symbol}"
    return f"{prefix}{name}{suffix}"


def build_payload(template, request_index, prefix_symbols, suffix_symbols):
    payload = {}
    for key, value in template.items():
        if isinstance(value, list):
            decorated_items = []
            for item_index, item in enumerate(value):
                if isinstance(item, str):
                    decorated_items.append(
                        decorate_identifier(item, request_index, item_index, prefix_symbols, suffix_symbols)
                    )
                elif isinstance(item, dict):
                    copied = dict(item)
                    base_name = copied.get("identifier_name", copied.get("name"))
                    if isinstance(base_name, str):
                        decorated_name = decorate_identifier(
                            base_name,
                            request_index,
                            item_index,
                            prefix_symbols,
                            suffix_symbols,
                        )
                        if "identifier_name" in copied:
                            copied["identifier_name"] = decorated_name
                        else:
                            copied["name"] = decorated_name
                    decorated_items.append(copied)
                else:
                    decorated_items.append(item)
            payload[key] = decorated_items
        else:
            payload[key] = value
    return payload


async def send_one(session, url, request_index, template, prefix_symbols, suffix_symbols, timeout, print_failures):
    payload = build_payload(template, request_index, prefix_symbols, suffix_symbols)
    try:
        async with session.post(url, json=payload, timeout=timeout) as response:
            if response.status >= 400:
                body = await response.text()
                if print_failures:
                    print(f"[{request_index}] status={response.status} body={body}")
                return False, response.status

            await response.read()
            return True, response.status
    except Exception as exc:
        if print_failures:
            print(f"[{request_index}] error={exc}")
        return False, None


async def run_load_test(args):
    template = load_payload_template(args)
    prefix_symbols = list(args.prefix_symbols)
    suffix_symbols = list(args.suffix_symbols)
    semaphore = asyncio.Semaphore(args.concurrency)

    totals = {
        "ok": 0,
        "failed": 0,
        "statuses": {},
    }

    async with aiohttp.ClientSession() as session:
        async def bounded_send(request_index):
            async with semaphore:
                ok, status = await send_one(
                    session=session,
                    url=args.url,
                    request_index=request_index,
                    template=template,
                    prefix_symbols=prefix_symbols,
                    suffix_symbols=suffix_symbols,
                    timeout=args.timeout,
                    print_failures=args.print_failures,
                )
                if ok:
                    totals["ok"] += 1
                else:
                    totals["failed"] += 1
                status_key = "error" if status is None else str(status)
                totals["statuses"][status_key] = totals["statuses"].get(status_key, 0) + 1

        start = time.perf_counter()
        await asyncio.gather(*(bounded_send(request_index) for request_index in range(args.requests)))
        elapsed = time.perf_counter() - start

    template_item_count = sum(len(v) for v in template.values() if isinstance(v, list))
    summary = {
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "payload_items_per_request": template_item_count,
        "unique_identifiers_sent": args.requests * template_item_count,
        "elapsed_seconds": round(elapsed, 4),
        "requests_per_second": round(args.requests / elapsed, 2) if elapsed else None,
        "identifiers_per_second": round((args.requests * template_item_count) / elapsed, 2) if elapsed else None,
        "ok": totals["ok"],
        "failed": totals["failed"],
        "statuses": totals["statuses"],
        "prefix_symbols": prefix_symbols,
        "suffix_symbols": suffix_symbols,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run_load_test(parse_args()))
