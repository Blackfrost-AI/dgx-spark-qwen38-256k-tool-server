#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
started="${SECONDS}"

while (( SECONDS - started < TIMEOUT_SECONDS )); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
        curl -fsS "${BASE_URL}/slots"
        printf '\nready in %ss\n' "$((SECONDS - started))"
        exit 0
    fi
    sleep 5
done

echo "server did not become ready within ${TIMEOUT_SECONDS}s" >&2
exit 1
