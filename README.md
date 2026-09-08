# Qwen3.8 Flash Next at 256K on one DGX Spark

[![validate package](https://github.com/Blackfrost-AI/dgx-spark-qwen38-256k-tool-server/actions/workflows/validate.yml/badge.svg)](https://github.com/Blackfrost-AI/dgx-spark-qwen38-256k-tool-server/actions/workflows/validate.yml)

This project packages a patched llama.cpp server for Qwen3.8 Flash Next on one
NVIDIA DGX Spark. It provides 131,072-token and 262,144-token profiles,
OpenAI-compatible parsed tool calls, and MTP speculative decoding in a CUDA 13
ARM64 container.

The repository contains serving code and reproducible validation. Model weights
are not included.

A matched reference run used the same model, runtime, and MTP files at both
context sizes. Three 512-token prose cases measured a 30.63 tok/s median at
128K and 30.43 tok/s at 256K. Required weather and numeric tool calls parsed
valid JSON arguments in all 12 trials across the two profiles, and tool-result
continuations passed at both sizes. A near-cap 257,994-token retrieval test also
passed at 256K. See [the validation record](docs/results.md) for the complete
matrix, including automatic tool-selection behavior.

## What is pinned

- NVIDIA CUDA 13.0.1 development and runtime images for ARM64/SBSA
- llama.cpp fork commit `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89`
- GB10 CUDA target `121a-real`
- Q8_0 shared MTP head, draft maximum 4, draft probability floor 0.30
- one 131,072-token or 262,144-token slot with 256-token microbatches
- Jinja chat rendering and DeepSeek-style reasoning separation for parsed tools

The pinned fork supplies the experimental Qwen4 and shared-MTP model paths. The
local patches refine GDN normalization, add direct PLE reads, and correct the
parallel-warp CUDA synchronization path. See
[docs/architecture.md](docs/architecture.md) for the exact scope.

## Build

Build on the DGX Spark so Docker uses the native ARM64 CUDA image:

```bash
docker build -t qwen38-spark-256k:cu130 .
```

The build clones the pinned source commit, verifies both patches, and compiles
`llama-server` for the GB10 target.

## Run

Place all four main-model GGUF shards in one directory and the shared Q8_0 MTP
GGUF in another. One compatible public source is:

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include 'UD-Q4_K_XL/*' \
  --include 'MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf' \
  --local-dir qwen38-flash-next-gguf
```

Point the launcher at the two downloaded subdirectories:

```bash
export MODEL_DIR="$PWD/qwen38-flash-next-gguf/UD-Q4_K_XL"
export MTP_DIR="$PWD/qwen38-flash-next-gguf/MTP"
./scripts/run.sh
./scripts/wait-health.sh
```

Keep the stop-only host memory guard in a second terminal:

```bash
./scripts/start-memory-guard.sh
```

The server binds to `127.0.0.1:30000` by default. Set `SECOND_BIND_ADDR` to add
a trusted WireGuard or LAN interface while retaining loopback. `BIND_ADDR` and
`HOST_PORT` control the primary binding. Put an authenticated reverse proxy in
front of any wider deployment.

The default filenames are:

- `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
- `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`

Override `MODEL_FILE` or `MTP_FILE` when your files use different names.
The launcher defaults to `CTX_SIZE=262144`; set `CTX_SIZE=131072` for the
128K profile.

## Verify

Check the allocated slot:

```bash
curl -fsS http://127.0.0.1:30000/slots
```

The response must report the selected `n_ctx` and `"speculative":true`.

Validate required parsed tools and a tool-result continuation:

```bash
python3 scripts/validate-tools.py
```

Run the matched long-form, automatic/required tool-selection, and round-trip
benchmark:

```bash
python3 scripts/benchmark.py \
  --label 256k \
  --output artifacts/benchmark-256k.json
```

Run it once under each context profile for a direct comparison. Required calls
are the parser qualification. Automatic selection remains visible as a separate
model-behavior result in the artifact.

Exercise a near-cap retrieval prompt against the 262,144-token slot:

```bash
python3 scripts/validate-context.py
```

This is a long prefill on one Spark. The validator prints progress and memory
samples, then saves the result under `artifacts/`.

## Operational notes

Startup takes several minutes while the model and MTP head initialize. The
reference container uses a 116 GiB Docker memory ceiling and a stop-only host
memory watcher.

Decode speed depends strongly on content because speculative-token acceptance
varies. The current required numeric tool path reached 43.33 to 45.51 tok/s;
the 512-token long-form median was about 30.5 tok/s. Use
`tool_choice: "required"` when the application requires an invocation.

The source patches are derived from llama.cpp, which retains its upstream MIT
license. The packaging scripts and documentation use this repository's MIT
license.
