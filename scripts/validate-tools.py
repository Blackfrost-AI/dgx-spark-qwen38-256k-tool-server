#!/usr/bin/env python3
"""Validate parsed OpenAI tool calls and a tool-result continuation."""

from __future__ import annotations

import json
import os
import sys
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")


def api(path: str, payload: dict | None = None, timeout: int = 180) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def request_tool(model: str, prompt: str, tool: dict, choice: str) -> dict:
    return api(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool],
            "tool_choice": choice,
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "temperature": 0,
            "max_tokens": 160,
        },
    )


def main() -> int:
    props = api("/props")
    slot = api("/slots")[0]
    if slot["n_ctx"] not in (131_072, 262_144) or not slot["speculative"]:
        raise RuntimeError(f"unexpected slot: {slot}")
    model = props.get("model_alias") or props["model_path"]
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
        ("parser-weather-sf", "Call get_weather for San Francisco.", weather, "required", "get_weather"),
        ("parser-weather-seattle", "Call get_weather for Seattle.", weather, "required", "get_weather"),
        ("parser-math-17x23", "Use multiply to calculate 17 times 23.", multiply, "required", "multiply"),
        ("parser-math-31x7", "Use multiply to calculate 31 times 7.", multiply, "required", "multiply"),
    ]

    rows = []
    for name, prompt, tool, choice, expected in cases:
        response = request_tool(model, prompt, tool, choice)
        message = response["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        passed = bool(calls) and calls[0].get("function", {}).get("name") == expected
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
            passed = passed and bool(arguments)
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            passed = False
        rows.append(
            {
                "case": name,
                "pass": passed,
                "finish_reason": response["choices"][0].get("finish_reason"),
                "tool_calls": calls,
            }
        )

    first = request_tool(
        model,
        "Use get_weather for Portland, then report its result.",
        weather,
        "required",
    )
    assistant = first["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    roundtrip = False
    final_content = ""
    if calls:
        second = api(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Use get_weather for Portland, then report its result."},
                    assistant,
                    {
                        "role": "tool",
                        "tool_call_id": calls[0]["id"],
                        "content": '{"temperature_f":58,"condition":"cloudy"}',
                    },
                ],
                "tools": [weather],
                "temperature": 0,
                "max_tokens": 128,
            },
        )
        final_content = second["choices"][0]["message"].get("content") or ""
        roundtrip = "58" in final_content and "cloud" in final_content.lower()

    result = {
        "model": model,
        "n_ctx": slot["n_ctx"],
        "speculative": slot["speculative"],
        "tool_cases": rows,
        "roundtrip_pass": roundtrip,
        "roundtrip_content": final_content,
    }
    print(json.dumps(result, indent=2))
    return 0 if all(row["pass"] for row in rows) and roundtrip else 1


if __name__ == "__main__":
    sys.exit(main())
