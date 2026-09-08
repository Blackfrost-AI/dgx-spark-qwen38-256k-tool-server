# Validation record

The reference system is one NVIDIA DGX Spark with a GB10 SoC and 128 GB of
unified memory. The runtime is CUDA 13.0.1 on ARM64. The tested model is
Qwen3.8-Flash-Next-UD-Q4_K_XL with the shared Q8_0 MTP head.

## Tool calling at 131,072 context

The final OpenAI-compatible qualification made twelve tool requests across
automatic and required selection, weather and numeric arguments, and repeated
trials. All 12 returned a parsed `tool_calls` object with valid JSON arguments.
A tool-result continuation also passed.

Warm tool-call decode ranged from 42.7 to 51.1 tokens/second. The first request
measured 39.1 tokens/second. Three 512-token prose cases measured 42.1, 30.3,
and 36.5 tokens/second, showing that MTP acceptance and throughput remain
content dependent.

## 262,144 context

The server allocates one 262,144-token slot with speculative MTP enabled. A
257,993-token retrieval run is the release gate for the 256K claim. Its result
will be recorded here after completion.

Qwen3.8 Flash Next is a sparse mixture-of-experts model, so each token executes
only a subset of the routed experts. That limits per-token expert compute. The
long-context memory envelope also depends on the hybrid architecture, unified
GB10 memory, memory mapping, and on-demand per-layer embedding reads; sparse
expert routing alone does not remove the context-state allocation.
