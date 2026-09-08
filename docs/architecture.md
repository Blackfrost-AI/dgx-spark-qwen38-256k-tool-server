# Architecture and patch scope

## Runtime path

The container pins Daniel Han's llama.cpp fork at commit
`d1a92352cbd417fd840b4e765c0b82f5fe3d1d89`. That base supplies the
experimental Qwen4 architecture used by Qwen3.8 Flash Next and its shared MTP
model path. The repository applies two patches and compiles only
`llama-server` for CUDA architecture `121a-real`, the GB10 target.

The reference request path is:

1. llama.cpp renders the Qwen chat template with Jinja.
2. The built-in Qwen tool parser converts the model's XML function block into
   the OpenAI-compatible `tool_calls` response field.
3. The shared Q8_0 MTP head proposes up to four tokens. Proposals below the
   0.30 probability floor are skipped, and the main model verifies every
   accepted token.
4. Main-model and MTP compute stay on the GB10 GPU. The service uses one slot
   so that the full 262,144-token context belongs to one request stream.

The patches do not modify the server chat-template or tool-parser sources.

## Direct per-layer embedding reads

The model has a large per-layer token-embedding table. `--lazy-mode on-direct`
keeps that table on the lazy path and gathers only the rows required by the
current microbatch with `pread()`. The token IDs are already known on the host,
so those rows can be staged as F32 graph input without demand-faulting the
whole mapped table. Unsupported platforms fall back to lazy mmap reads.

## Gated DeltaNet normalization

The Qwen Gated DeltaNet path uses the reference normalization

```text
x * rsqrt(sum(x*x) + epsilon)
```

implemented through llama.cpp's RMS-normalization graph operation and a scale
factor. The build-time numerical control covered five 128-element vectors and
measured a maximum absolute error of `1.05044029e-08` against the reference.

## CUDA flash-attention synchronization

The CUDA patch places the parallel-warp metadata barrier on one uniform
`np > 1` control path. The selected warps still calculate and write combined
softmax metadata, while every participating warp reaches the same
`__syncthreads()` before the next tile.

## Why 256K fits

Qwen3.8 Flash Next has 125B main-model parameters with 6B activated per token,
plus a 51B n-gram embedding table and a 4B MTP module. Each MoE layer selects
10 routed experts and one shared expert from 512 routed experts. That sparsity
limits expert compute per token, while the complete model remains mapped.
Context capacity also relies on the hybrid attention/state architecture, GB10
unified memory, memory-mapped quantized weights, and direct per-layer embedding
reads. Sparse routing does not by itself eliminate context-state memory.

The reference service reserves one 262,144-token slot, uses 256-token
microbatches, and runs inside a 116 GiB Docker memory ceiling. The separate
exact-container-ID watcher stops the service after sustained host pressure and
never restarts or discovers containers.

## Reproducibility

The tested host build used CUDA 13.0.1, CMake 3.28.3, GCC 13.3.0, and the same
`121a-real`, Release, static-library configuration in the Dockerfile. Its
`llama-server` SHA-256 is
`cd7d12515276ad95d5d653d162b0ace57a7ac00d11f27fb0490bdda3a209b848`.

Both exported patches apply cleanly to a fresh checkout of the pinned commit.
The resulting 13 changed or added source files match the tested source inputs;
the container build repeats the patch checks before compilation.
