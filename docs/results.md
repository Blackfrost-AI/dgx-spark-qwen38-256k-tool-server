# Validation record

The reference system is one NVIDIA DGX Spark with a GB10 SoC and 128 GB of
unified memory. The runtime is CUDA 13.0.1 on ARM64. The tested model uses four
Qwen3.8-Flash-Next UD-Q4_K_XL GGUF shards and the shared Q8_0 MTP head. Model
weights are not distributed by this repository.

## Matched 128K and 256K benchmark

Both profiles used one slot, 256-token microbatches, four-token MTP drafts, a
0.30 draft probability floor, and the same model and runtime files. Each prose
case generated exactly 512 tokens after one warm-up request.

| Context | Prose case 1 | Prose case 2 | Prose case 3 | Median | Required tools | Auto tools | Round trip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 131,072 | 31.69 tok/s | 26.55 tok/s | 30.63 tok/s | 30.63 tok/s | 6/6 | 0/6 | pass |
| 262,144 | 31.16 tok/s | 27.49 tok/s | 30.43 tok/s | 30.43 tok/s | 6/6 | 0/6 | pass |
\nThe repository-built image repeated the 256K prose cases at 31.42, 27.58, and
30.48 tok/s, for a 30.48 tok/s median.

The required-tool trials covered weather and numeric schemas, three repetitions
each. Every response finished with `tool_calls`, selected the expected function,
and contained valid JSON arguments. Warm required-tool decode at 128K ranged
from 36.59 to 45.51 tok/s; at 256K it ranged from 37.03 to 45.33 tok/s.

The automatic-selection trials are reported separately because they measure a
model decision as well as parser execution. In this run, automatic weather
selection began a tool call but exhausted the 160-token response allowance
before completing its arguments. Automatic multiplication returned a normal
answer without electing the tool. Clients that require a tool invocation should
send `tool_choice: "required"`.

The round-trip case parsed a weather call, accepted a tool-role result, and
generated a final answer containing the supplied values at both context sizes.

## Near-cap 262,144-token retrieval

A chat-formatted retrieval prompt evaluated 257,994 tokens and returned the
correct nonce from the beginning of the context. The current run completed in
1045.97 seconds at 247.56 prompt tok/s, then generated
64 tokens. Minimum host `MemAvailable` was 23.84 GiB.
The container remained running without an OOM or restart.

The repository-built image reports llama.cpp build 10802 at the pinned source
commit. The separately compiled benchmark runtime reports the same commit. The
container image is ARM64 Linux and measures 2.62 GB.

## Interpretation

Qwen3.8 Flash Next is a sparse mixture-of-experts model, so each token executes
only a subset of the routed experts. That limits expert compute. Long-context
memory also depends on the hybrid architecture, unified GB10 memory, memory
mapping, and on-demand per-layer embedding reads; sparse expert routing alone
does not remove context-state allocation.

Decode throughput varies with content and speculative-token acceptance. The
40+ tok/s figures in this record apply to most required numeric tool calls, not
to every long-form generation.
