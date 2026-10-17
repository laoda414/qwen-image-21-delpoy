#!/usr/bin/env bash
# One-line bootstrap+serve for a fresh GPU box: the Vast "on-start script" field,
# a RunPod start command, or `bash onstart.sh` over ssh.
#
# Paste this into Vast once the repo exists (replace OWNER/REPO):
#   curl -fsSL https://raw.githubusercontent.com/OWNER/REPO/main/onstart.sh | bash
#
# Or, if the box has the repo already:
#   SERVICE_TOKEN=... PORT=10100 ./onstart.sh
#
# Env:
#   REPO_URL        clone source when this file is run standalone
#   SERVICE_TOKEN   bearer token for the API (strongly recommended)
#   PORT / SERVICE_PORT  listen port (default 8080)
#   WORK            install root (default /workspace)
set -uo pipefail

WORK=${WORK:-/workspace}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)
PORT=${PORT:-${SERVICE_PORT:-8080}}
export WORK

# When piped from curl (bash -s) there is no repo next to us: clone it.
if [ ! -f "$ROOT/bootstrap.sh" ]; then
  : "${REPO_URL:?set REPO_URL=https://github.com/OWNER/REPO.git when running this standalone}"
  DEST="$WORK/qwen-image-gpu"
  if [ ! -d "$DEST/.git" ]; then
    echo "[onstart] cloning $REPO_URL -> $DEST"
    git clone --depth 1 "$REPO_URL" "$DEST" || { echo "[onstart] clone failed" >&2; exit 1; }
  fi
  ROOT="$DEST"
fi

cd "$ROOT" || exit 1
echo "[onstart] repo at $ROOT ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo no-git))"

bash "$ROOT/bootstrap.sh" || { echo "[onstart] bootstrap failed" >&2; exit 1; }

export SERVICE_PORT="$PORT"
exec bash "$ROOT/serve.sh"
