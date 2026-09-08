#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to the directory containing every model GGUF shard}"
: "${MTP_DIR:?Set MTP_DIR to the directory containing the shared MTP GGUF}"

IMAGE="${IMAGE:-qwen38-spark-256k:cu130}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-spark-256k}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
MTP_FILE="${MTP_FILE:-mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.8-flash-next-spark}"
CTX_SIZE="${CTX_SIZE:-262144}"
BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
HOST_PORT="${HOST_PORT:-30000}"

case "${CTX_SIZE}" in
    131072|262144) ;;
    *) echo "CTX_SIZE must be 131072 or 262144" >&2; exit 2 ;;
esac

for path in "${MODEL_DIR}/${MODEL_FILE}" "${MTP_DIR}/${MTP_FILE}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 2
    fi
done

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "Container already exists: ${CONTAINER_NAME}" >&2
    echo "Choose another CONTAINER_NAME or remove the stopped container explicitly." >&2
    exit 2
fi

container_id="$(docker create \
    --name "${CONTAINER_NAME}" \
    --restart no \
    --device nvidia.com/gpu=all \
    --ipc private \
    --memory 116g \
    --memory-swap 116g \
    --stop-timeout 30 \
    --publish "${BIND_ADDR}:${HOST_PORT}:30000" \
    --mount "type=bind,src=${MODEL_DIR},dst=/models,readonly" \
    --mount "type=bind,src=${MTP_DIR},dst=/mtp,readonly" \
    --mount "type=volume,src=${CONTAINER_NAME}-cuda-cache,dst=/cuda-cache" \
    --env CUDA_CACHE_PATH=/cuda-cache \
    "${IMAGE}" \
    -m "/models/${MODEL_FILE}" \
    -md "/mtp/${MTP_FILE}" \
    --alias "${MODEL_ALIAS}" \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    --spec-draft-p-min 0.30 \
    -ngl 999 \
    --spec-draft-ngl 999 \
    --ctx-size "${CTX_SIZE}" \
    --parallel 1 \
    --flash-attn on \
    --fit off \
    --lazy-mode on-direct \
    --load-mode mmap \
    --ubatch-size 256 \
    --batch-size 2048 \
    --cache-ram 2048 \
    --jinja \
    --reasoning-format deepseek \
    --reasoning-effort low \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --repeat-penalty 1.0 \
    --presence-penalty 0.0 \
    --host 0.0.0.0 \
    --port 30000 \
    --metrics \
    --log-timestamps)"

docker start "${container_id}" >/dev/null
printf 'Started %s\n' "${container_id}"
printf 'Health: http://%s:%s/health\n' "${BIND_ADDR}" "${HOST_PORT}"
printf 'Logs: docker logs --follow %s\n' "${CONTAINER_NAME}"
