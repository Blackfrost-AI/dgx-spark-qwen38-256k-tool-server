#!/usr/bin/env python3
"""Exercise a near-cap context and record retrieval, timing, and memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import urllib.request


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "context-256k-validation.json"
TARGET_TOKENS = 258_000
NONCE = "FROST-256K-9C7A"
FILLER = (
    "Archive note: the catalog entry was checked, indexed, and retained for "
    "long-context verification.\n"
)


def api(path: str, payload: dict | None = None, timeout: int = 3600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(content: str) -> int:
    return len(api("/tokenize", {"content": content})["tokens"])


def render_chat(content: str) -> str:
    return api(
        "/apply-template",
        {
            "messages": [{"role": "user", "content": content}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
    )["prompt"]


def make_prompt() -> tuple[str, int, int]:
    prefix = f"Verification nonce: {NONCE}\n\n"
    suffix = "\nWhat is the verification nonce? Answer with the nonce only.\n"
    fixed = token_count(render_chat(prefix + suffix))
    unit = max(1, token_count(render_chat(prefix + FILLER + suffix)) - fixed)
    repeats = max(1, (TARGET_TOKENS - fixed) // unit)

    # Boundary merges make the single-unit estimate approximate. Adjust twice,
    # always staying at or below the target and well inside the generation cap.
    prompt = render_chat(prefix + FILLER * repeats + suffix)
    count = token_count(prompt)
    for _ in range(2):
        delta_repeats = max(1, abs(TARGET_TOKENS - count) // unit)
        if count > TARGET_TOKENS:
            repeats = max(1, repeats - delta_repeats)
        else:
            repeats += delta_repeats
        candidate = render_chat(prefix + FILLER * repeats + suffix)
        candidate_count = token_count(candidate)
        if candidate_count > TARGET_TOKENS:
            break
        prompt, count = candidate, candidate_count
    if not 250_000 <= count <= TARGET_TOKENS:
        raise RuntimeError(f"near-cap prompt calibration failed: {count} tokens")
    return prompt, count, repeats


def memory_gib() -> dict[str, float]:
    values = {
        line.split(":", 1)[0]: int(line.split()[1]) / 1024 / 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return {
        "available_gib": values["MemAvailable"],
        "free_gib": values["MemFree"],
        "swap_free_gib": values["SwapFree"],
    }


def main() -> int:
    slots = api("/slots")
    if len(slots) != 1 or slots[0]["n_ctx"] != 262_144 or slots[0]["is_processing"]:
        raise RuntimeError(f"unexpected slot state: {slots}")
    prompt, prompt_tokens, repeats = make_prompt()
    print(json.dumps({"calibrated_prompt_tokens": prompt_tokens, "repeats": repeats}), flush=True)

    outcome: dict[str, object] = {}
    failure: list[BaseException] = []

    def request_completion() -> None:
        try:
            outcome.update(
                api(
                    "/completion",
                    {
                        "prompt": prompt,
                        "n_predict": 64,
                        "temperature": 0,
                        "seed": 42,
                        "cache_prompt": False,
                    },
                )
            )
        except BaseException as error:
            failure.append(error)

    started = time.monotonic()
    worker = threading.Thread(target=request_completion, daemon=True)
    worker.start()
    samples = []
    while worker.is_alive():
        sample = {"elapsed_s": round(time.monotonic() - started, 1), **memory_gib()}
        try:
            live_slot = api("/slots", timeout=5)[0]
            sample.update(
                {
                    "is_processing": live_slot["is_processing"],
                    "prompt_tokens": live_slot.get("n_prompt_tokens"),
                    "processed": live_slot.get("n_prompt_tokens_processed"),
                }
            )
        except Exception as error:
            sample["slot_error"] = type(error).__name__
        samples.append(sample)
        print(json.dumps(sample), flush=True)
        worker.join(timeout=30)
    elapsed = time.monotonic() - started
    if failure:
        raise failure[0]

    content = str(outcome.get("content", "")).strip()
    result = {
        "n_ctx": 262_144,
        "target_prompt_tokens": TARGET_TOKENS,
        "actual_prompt_tokens": prompt_tokens,
        "filler_repeats": repeats,
        "n_predict": 64,
        "nonce": NONCE,
        "retrieval_pass": NONCE in content,
        "content": content,
        "elapsed_s": elapsed,
        "tokens_evaluated": outcome.get("tokens_evaluated"),
        "tokens_predicted": outcome.get("tokens_predicted"),
        "timings": outcome.get("timings"),
        "minimum_available_gib": min(sample["available_gib"] for sample in samples),
        "samples": samples,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "actual_prompt_tokens", "retrieval_pass", "content", "elapsed_s",
        "tokens_evaluated", "tokens_predicted", "minimum_available_gib",
    )}), flush=True)
    print(json.dumps({"saved": str(OUTPUT)}), flush=True)
    return 0 if result["retrieval_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
