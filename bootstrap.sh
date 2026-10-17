#!/usr/bin/env bash
# Provision a bare CUDA box into a working Qwen-Image service. Idempotent:
# re-running skips anything already done, so it doubles as a repair script.
#
#   ./bootstrap.sh              # deps + weights, no GPU build
#   WITH_SDCPP=1 ./bootstrap.sh # also compile stable-diffusion.cpp + GGUFs
#
# Env:
#   WORK          install root (default /workspace)
#   HF_MODEL      diffusers checkpoint (default Qwen/Qwen-Image-2.1-Turbo)
#   DIFFUSERS_REF diffusers git ref/pin (default: main)
#   WITH_SDCPP    1 to build the sd.cpp fallback lane as well
set -uo pipefail

WORK=${WORK:-/workspace}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR="$WORK/models/qwen-image-2.1-turbo"
export HF_HOME=${HF_HOME:-$WORK/hf}
HF_MODEL=${HF_MODEL:-Qwen/Qwen-Image-2.1-Turbo}
DIFFUSERS_REF=${DIFFUSERS_REF:-main}
MARKER="$WORK/.bootstrap-ok"

log() { printf '\n[bootstrap %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

mkdir -p "$WORK" "$WORK/logs" || die "cannot write $WORK"

# --------------------------------------------------------------- interpreter --
# Vast's base image keeps torch in /venv/main; a bare nvidia/cuda image has only
# system python; RunPod pods vary. Prefer an existing venv, else make one.
PY=""
for cand in /venv/main/bin/python /opt/venv/bin/python "$WORK/.venv/bin/python"; do
  [ -x "$cand" ] && { PY=$cand; break; }
done
if [ -z "$PY" ]; then
  log "creating a venv at $WORK/.venv"
  if ! python3 -m venv "$WORK/.venv" 2>>"$WORK/logs/bootstrap.log"; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3-venv \
      || log "WARN: could not install python3-venv"
    python3 -m venv "$WORK/.venv" || die "venv creation failed"
  fi
  PY="$WORK/.venv/bin/python"
fi
log "python: $PY ($("$PY" -V 2>&1))"

install_pkgs() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" "$@" && return 0
  fi
  "$PY" -m pip install "$@" && return 0
  "$PY" -m pip install --break-system-packages "$@"
}

# ------------------------------------------------------------------- torch ----
NEED_TORCH=1
if "$PY" - <<'PY' 2>/dev/null
import sys, torch
ok = torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) >= (12, 0)
cuda = torch.version.cuda or "0.0"
sys.exit(0 if (ok and tuple(int(x) for x in cuda.split(".")[:2]) >= (12, 8)) else 1)
PY
then
  NEED_TORCH=0
  log "existing torch already supports sm_120"
fi
if [ "$NEED_TORCH" = "1" ]; then
  log "installing torch + torchvision (cu128)"
  install_pkgs --index-url https://download.pytorch.org/whl/cu128 torch torchvision \
    || die "torch install failed"
fi
"$PY" - <<'PY' || die "torch cannot see a >=sm_120 GPU (is this a CUDA box?)"
import sys, torch
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
print(f"  torch {torch.__version__} cuda {torch.version.cuda} capability {cap}")
sys.exit(0 if cap and tuple(cap) >= (12, 0) else 1)
PY

# -------------------------------------------------------------- the stack ----
log "installing the serving stack"
install_pkgs "transformers>=5.17.0" accelerate pillow safetensors huggingface_hub \
  fastapi uvicorn || die "dependency install failed"
# diffusers from source: QwenImage21Pipeline post-dates the last release
install_pkgs "git+https://github.com/huggingface/diffusers.git@$DIFFUSERS_REF" \
  || die "diffusers install failed"
"$PY" -c "from diffusers import QwenImage21Pipeline; print('  QwenImage21Pipeline ok')" \
  || die "diffusers build lacks QwenImage21Pipeline"

# ---------------------------------------------------------------- weights ----
if [ -f "$MODEL_DIR/model_index.json" ] && [ -f "$MODEL_DIR/processor/chat_template.jinja" ]; then
  log "weights already present in $MODEL_DIR"
else
  log "downloading $HF_MODEL (~32 GB) into $MODEL_DIR"
  "$PY" - "$MODEL_DIR" "$HF_MODEL" <<'PY' || die "weight download failed"
import sys
from huggingface_hub import snapshot_download
# *.jinja matters: processor/chat_template.jinja is not *.json, and without it the
# pipeline dies with "this processor does not have a chat template"
path = snapshot_download(sys.argv[2], local_dir=sys.argv[1], max_workers=8,
                         allow_patterns=["*.json", "*.safetensors", "*.txt",
                                          "*.md", "*.jinja"])
print("  ->", path)
PY
fi

# ------------------------------------------------------------ sd.cpp lane ----
if [ "${WITH_SDCPP:-0}" = "1" ]; then
  log "building the sd.cpp fallback lane"
  WORK="$WORK" bash "$ROOT/bench/onstart.sh" || log "WARN: sd.cpp lane failed; diffusers lane is unaffected"
fi

# ------------------------------------------------------------------ marker ----
{
  echo "ready_at=$(date -u +%FT%TZ)"
  echo "python=$PY"
  echo "python_version=$("$PY" -V 2>&1)"
  echo "model_dir=$MODEL_DIR"
  "$PY" -c "import torch, diffusers; print(f'torch={torch.__version__} cuda={torch.version.cuda} diffusers={diffusers.__version__}')"
} | tee "$MARKER"
log "bootstrap complete -> $MARKER"
