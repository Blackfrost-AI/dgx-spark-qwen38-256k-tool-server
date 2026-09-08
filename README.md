# Qwen3.8 Flash Next at 256K on one DGX Spark

[![validate package](https://github.com/Blackfrost-AI/dgx-spark-qwen38-256k-tool-server/actions/workflows/validate.yml/badge.svg)](https://github.com/Blackfrost-AI/dgx-spark-qwen38-256k-tool-server/actions/workflows/validate.yml)

This project packages a patched llama.cpp server for Qwen3.8 Flash Next on one
NVIDIA DGX Spark. It targets a 262,144-token slot, OpenAI-compatible tool calls,
and MTP speculative decoding in a CUDA 13 ARM64 container.

The repository contains serving code and reproducible validation only. Model
weights are not included.

The reference run retrieved a nonce from a 257,994-token chat-formatted prompt
at 245.6 prompt tokens/second while retaining 24.60 GiB of host-available
memory. At 256K, the packaged server passed four parsed tool-call cases and a
tool-result continuation. Warm 128K tool calls measured 42.7 to 51.1 generated
tokens/second.

## What is pinned

- NVIDIA CUDA 13.0.1 development image for ARM64/SBSA
- llama.cpp fork commit `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89`
- GB10 CUDA target `121a-real`
- Q8_0 shared MTP head, draft maximum 4, draft probability floor 0.30
- one 262,144-token slot with 256-token microbatches
- Jinja chat rendering and DeepSeek-style reasoning separation for parsed tools

The pinned fork already supplies the experimental Qwen4 and shared-MTP model
paths. The local patches refine GDN normalization, add direct PLE reads, and
correct the parallel-warp CUDA synchronization path. See
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
GGUF in another. Then run:

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

The server binds to `127.0.0.1:30000` by default. Set `BIND_ADDR` and
`HOST_PORT` explicitly when a trusted LAN client needs access. Put an
authenticated reverse proxy in front of any non-loopback deployment.

The default filenames are:

- `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
- `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`

Override `MODEL_FILE` or `MTP_FILE` when your files use different names. Set
`CTX_SIZE=131072` for the lower-memory profile.

## Verify

Check the allocated slot:

```bash
curl -fsS http://127.0.0.1:30000/slots
```

The response must report `"n_ctx":262144` and `"speculative":true`. See
[docs/results.md](docs/results.md) for measured tool-call, throughput, memory,
and near-cap context results.

Validate parsed tools and the tool-result continuation:

```bash
python3 scripts/validate-tools.py
```

Exercise a near-cap retrieval prompt against the 262,144-token slot:

```bash
python3 scripts/validate-context.py
```

This is a long prefill on one Spark. The validator prints progress and memory
samples, then saves the result under `artifacts/`.

## Operational notes

Startup takes several minutes while the model and MTP head initialize. The
reference container uses a 116 GiB Docker memory ceiling and a stop-only host
memory watcher. Keep the service on loopback until authentication is added.

Decode speed depends strongly on content because speculative-token acceptance
varies. The measured 40+ tokens/second result applies to the warmed tool-call
path; it is not a universal long-form generation rate.

The two source patches are derived from llama.cpp, which retains its upstream
MIT license. The packaging scripts and documentation use this repository's MIT
license.
