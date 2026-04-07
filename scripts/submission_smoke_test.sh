#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${STOREOPS_SMOKE_PORT:-8013}"
BASE_URL="http://localhost:${PORT}"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

cd "${ROOT_DIR}"

echo "[check] openenv validate"
uv run openenv validate .

echo "[check] pytest"
uv run --with pytest pytest -q

echo "[check] start server on ${BASE_URL}"
uv run server --port "${PORT}" >/tmp/storeops_smoke_server.log 2>&1 &
SERVER_PID=$!
sleep 3

echo "[check] GET /"
test "$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/")" = "200"

echo "[check] HEAD /"
test "$(curl -s -o /dev/null -w "%{http_code}" -I "${BASE_URL}/")" = "200"

echo "[check] POST /reset"
test "$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/reset")" = "200"

echo "[check] inference.py"
STOREOPS_BASE_URL="${BASE_URL}" uv run python inference.py

echo "[ok] submission smoke test passed"
