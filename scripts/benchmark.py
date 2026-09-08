#!/usr/bin/env python3
"""Measure sustained decode throughput and OpenAI tool-call behavior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")


def api(path: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def throughput_cases() -> list[dict]:
    prompts = [
        "Write a detailed Python implementation of an LRU cache with type hints, tests, complexity analysis, and a discussion of thread safety. Continue for at least 900 words.",
        "Explain Raft consensus to a senior distributed-systems engineer, including elections, log replication, safety invariants, membership changes, and failure recovery. Continue for at least 900 words.",
        "Design a production PostgreSQL schema and worker protocol for a durable asynchronous job queue. Cover locking, retries, idempotency, observability, and migrations. Continue for at least 900 words.",
    ]
    api(
        "/completion",
        {
            "prompt": "Continue this sequence with explanatory prose: one, two, three,",
            "n_predict": 64,
            "temperature": 0,
            "seed": 42,
            "ignore_eos": True,
            "cache_prompt": False,
        },
    )
    rows = []
    for index, prompt in enumerate(prompts, 1):
        started = time.monotonic()
        result = api(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0,
                "seed": 42,
                "ignore_eos": True,
                "cache_prompt": False,
            },
        )
        elapsed = time.monotonic() - started
        predicted = result.get("tokens_predicted") or result.get("timings", {}).get("predicted_n")
        row = {
            "case": index,
            "tokens": predicted,
            "server_tok_s": result.get("timings", {}).get("predicted_per_second"),
            "wall_tok_s": predicted / elapsed if predicted else None,
            "wall_s": elapsed,
            "stop": result.get("stop_type"),
        }
        rows.append(row)
        print(json.dumps({"throughput": row}), flush=True)
    speeds = [row["server_tok_s"] for row in rows if row["server_tok_s"] is not None]
    return [
        *rows,
        {
            "summary": True,
            "minimum_server_tok_s": min(speeds),
            "median_server_tok_s": statistics.median(speeds),
            "mean_server_tok_s": statistics.fmean(speeds),
        },
    ]


def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def tool_cases(model: str) -> list[dict]:
    weather = function(
        "get_weather",
        "Get the current weather for a location",
        {"location": {"type": "string"}},
        ["location"],
    )
    multiply = function(
        "multiply",
        "Multiply two numbers",
        {"x": {"type": "number"}, "y": {"type": "number"}},
        ["x", "y"],
    )
    cases = [
        ("auto-weather-low", "What is the current weather in San Francisco? Call the weather tool.", [weather], "auto", "get_weather"),
        ("required-weather-low", "Call get_weather for San Francisco, California.", [weather], "required", "get_weather"),
        ("auto-math-low", "Use multiply to calculate 17 times 23.", [multiply], "auto", "multiply"),
        ("required-math-low", "Use multiply to calculate 17 times 23.", [multiply], "required", "multiply"),
    ]
    rows = []
    for name, prompt, tools, choice, expected in cases:
        for trial in range(1, 4):
            result = api(
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": tools,
                    "tool_choice": choice,
                    "chat_template_kwargs": {"reasoning_effort": "low"},
                    "temperature": 0,
                    "max_tokens": 160,
                    "stream": False,
                },
            )
            message = result.get("choices", [{}])[0].get("message", {})
            calls = message.get("tool_calls") or []
            valid = len(calls) >= 1 and calls[0].get("function", {}).get("name") == expected
            if valid:
                try:
                    arguments = json.loads(calls[0]["function"]["arguments"])
                    valid = bool(arguments)
                except (KeyError, TypeError, json.JSONDecodeError):
                    valid = False
            row = {
                "case": name,
                "trial": trial,
                "pass": valid,
                "finish": result.get("choices", [{}])[0].get("finish_reason"),
                "tool_calls": calls,
                "server_tok_s": result.get("timings", {}).get("predicted_per_second"),
            }
            rows.append(row)
            print(json.dumps({"tool": row}), flush=True)
    return rows


def roundtrip_case(model: str) -> dict:
    weather = function(
        "get_weather",
        "Get the current weather for a location",
        {"location": {"type": "string"}},
        ["location"],
    )
    first = api(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Use get_weather for Seattle, then report its result."}],
            "tools": [weather],
            "tool_choice": "auto",
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "temperature": 0,
            "max_tokens": 160,
        },
    )
    assistant = first["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    if not calls:
        return {"pass": False, "stage": "initial_call", "response": assistant}
    call = calls[0]
    second = api(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Use get_weather for Seattle, then report its result."},
                assistant,
                {"role": "tool", "tool_call_id": call["id"], "content": '{"temperature_f": 61, "condition": "rain"}'},
            ],
            "tools": [weather],
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "temperature": 0,
            "max_tokens": 128,
        },
    )
    message = second["choices"][0]["message"]
    content = message.get("content") or ""
    passed = "61" in content and "rain" in content.lower()
    return {"pass": passed, "stage": "final_answer", "content": content, "initial_tool_call": call}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--throughput-only", action="store_true")
    args = parser.parse_args()
    props = api("/props")
    slots = api("/slots")
    if any(slot["is_processing"] for slot in slots):
        raise RuntimeError("server slot is busy")
    model = props.get("model_alias") or props.get("model_path")
    result = {
        "label": args.label,
        "model": model,
        "n_ctx": slots[0]["n_ctx"],
        "speculative": slots[0]["speculative"],
        "throughput": throughput_cases(),
    }
    if not args.throughput_only:
        result["tools"] = tool_cases(model)
        result["roundtrip"] = roundtrip_case(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"saved": str(args.output), "tool_passes": sum(row["pass"] for row in result.get("tools", [])), "tool_total": len(result.get("tools", [])), "roundtrip": result.get("roundtrip", {}).get("pass")}), flush=True)
    required_pass = all(row["pass"] for row in result.get("tools", []) if row["case"].startswith("required-"))
    roundtrip_pass = result.get("roundtrip", {}).get("pass", args.throughput_only)
    return 0 if required_pass and roundtrip_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
