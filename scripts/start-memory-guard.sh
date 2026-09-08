#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-qwen38-spark-256k}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT}/artifacts/memory-guard-${stamp}}"

container_id="$(docker inspect --format '{{.Id}}' "${CONTAINER_NAME}")"
running="$(docker inspect --format '{{.State.Running}}' "${container_id}")"
if [[ "${running}" != "true" ]]; then
    echo "Container is not running: ${CONTAINER_NAME}" >&2
    exit 2
fi
if [[ -e "${ARTIFACT_DIR}" ]]; then
    echo "Guard artifact directory must not already exist: ${ARTIFACT_DIR}" >&2
    exit 2
fi

exec python3 "${ROOT}/scripts/watch-memory.py" \
    --container-id "${container_id}" \
    --artifact-dir "${ARTIFACT_DIR}"
