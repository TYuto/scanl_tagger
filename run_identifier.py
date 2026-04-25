#!/usr/bin/env python3

import argparse
import contextlib
import json
import sys

from src.lm_runtime_utils import LMRuntime, build_lm_model_path


def parse_request_line(line):
    stripped = line.strip()
    if not stripped:
        return None

    if stripped.startswith("{"):
        payload = json.loads(stripped)
        identifier_name = payload["identifier_name"]
        identifier_context = payload["identifier_context"]
        return {
            "identifier_name": identifier_name,
            "identifier_context": identifier_context,
            "cache_id": payload.get("cache_id"),
            "system_name": payload.get("system_name", ""),
            "programming_language": payload.get("language", payload.get("programming_language", "")),
            "data_type": payload.get("type", payload.get("data_type", "")),
        }

    if "\t" in stripped:
        parts = stripped.split("\t")
        if len(parts) < 2:
            raise ValueError("TSV input must include at least identifier_name and identifier_context")

        return {
            "identifier_name": parts[0],
            "identifier_context": parts[1],
            "cache_id": parts[2] if len(parts) > 2 and parts[2] else None,
            "system_name": parts[3] if len(parts) > 3 else "",
            "programming_language": parts[4] if len(parts) > 4 else "",
            "data_type": parts[5] if len(parts) > 5 else "",
        }

    parts = stripped.split("/")
    if len(parts) < 2:
        raise ValueError(
            "Input must be JSON, TSV, or slash-delimited as identifier_name/identifier_context[/cache_id]"
        )

    return {
        "identifier_name": parts[0],
        "identifier_context": parts[1],
        "cache_id": parts[2] if len(parts) > 2 and parts[2] else None,
        "system_name": "",
        "programming_language": "",
        "data_type": "",
    }


def stream_requests(runtime):
    for raw_line in sys.stdin:
        try:
            request_payload = parse_request_line(raw_line)
            if request_payload is None:
                continue

            with contextlib.redirect_stdout(sys.stderr):
                result = runtime.tag_identifier_result(
                    identifier_name=request_payload["identifier_name"],
                    identifier_context=request_payload["identifier_context"],
                    system_name=request_payload["system_name"],
                    programming_language=request_payload["programming_language"],
                    data_type=request_payload["data_type"],
                )
        except Exception as exc:
            result = {
                "error": str(exc),
                "input": raw_line.rstrip("\n"),
            }

        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated LM-based tagging requests from stdin without starting the HTTP server. "
            "Each input line may be JSON, TSV, or identifier_name/identifier_context[/cache_id]."
        )
    )
    parser.add_argument("--local", action="store_true", help="Use a local model directory")
    parser.add_argument("--model", type=str, help="Optional explicit model path or HuggingFace repo id")

    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        runtime = LMRuntime(
            model_path=build_lm_model_path(local=args.local, model=args.model),
            local=args.local,
        )

    stream_requests(runtime)
