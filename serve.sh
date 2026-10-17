#!/usr/bin/env bash
# Run the Qwen-Image service. Assumes bootstrap.sh has already run.
#
#   SERVICE_TOKEN=... ./serve.sh
#   SERVICE_PORT=10100 ./serve.sh
#
# Vast maps container ports to random public ones; read the mapping from the
# console or `vast-capabilities` rather than assuming the port survives a restart.
set -uo pipefail

WORK=${WORK:-/workspace}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PORT=${SERVICE_PORT:-8080}
MODEL=${MODEL_DIR:-$WORK/models/qwen-image-2.1-turbo}

PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python "$WORK/.venv/bin/python"; do
  [ -x "$cand" ] && { PY=$cand; break; }
done
[ -n "$PY" ] || PY=python3

[ -d "$MODEL" ] || { echo "model missing at $MODEL; run ./bootstrap.sh first" >&2; exit 1; }
if [ -z "${SERVICE_TOKEN:-}" ]; then
  echo "WARNING: SERVICE_TOKEN is not set; the endpoint will accept anyone who can reach it." >&2
fi

exec "$PY" "$ROOT/service/serve.py" --model "$MODEL" --host 0.0.0.0 --port "$PORT" \
  ${SERVICE_ATTENTION:+--attention "$SERVICE_ATTENTION"}
