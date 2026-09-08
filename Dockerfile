ARG CUDA_DEVEL_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG CUDA_RUNTIME_IMAGE=nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04
FROM ${CUDA_DEVEL_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REPO=https://github.com/danielhanchen/llama.cpp.git
ARG LLAMA_CPP_COMMIT=d1a92352cbd417fd840b4e765c0b82f5fe3d1d89

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --filter=blob:none "${LLAMA_CPP_REPO}" llama.cpp \
    && cd llama.cpp \
    && git checkout --detach "${LLAMA_CPP_COMMIT}"

COPY patches/ /patches/
WORKDIR /src/llama.cpp
RUN git apply --check /patches/0001-qwen38-spark-runtime.patch \
    && git apply --check /patches/0002-lazy-reader.patch \
    && git apply /patches/0001-qwen38-spark-runtime.patch \
    && git apply /patches/0002-lazy-reader.patch

RUN cmake -S . -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=121a-real \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j 4 --target llama-server

FROM ${CUDA_RUNTIME_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
EXPOSE 30000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10m --retries=3 \
    CMD curl -fsS http://127.0.0.1:30000/health || exit 1
ENTRYPOINT ["/usr/local/bin/llama-server"]
