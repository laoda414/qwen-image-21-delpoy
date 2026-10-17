#!/usr/bin/env bash
# Diffusers lane: PyTorch (cu128, sm_120), diffusers from main (QwenImage21Pipeline
# landed after the last release), and the bf16 Turbo checkpoint.
#
# This image keeps ML packages in /venv/main and leaves /usr/bin/python3 bare, so
# pick the environment that actually has (or can take) torch. uv is present and
# much faster than pip on this box.
set -uo pipefail

WORK=${WORK:-/workspace}
MODEL_DIR="$WORK/models/qwen-image-2.1-turbo"
export HF_HOME=${HF_HOME:-$WORK/hf}

log() { printf '\n[diffusers %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

PY=python3
for cand in /venv/main/bin/python /venv/main/bin/python3 "$(command -v python3)"; do
  [ -n "$cand" ] && [ -x "$cand" ] && { PY=$cand; break; }
done
log "python: $PY ($("$PY" -V 2>&1))"
export PY

install_pkgs() { # packages...
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" "$@" && return 0
  fi
  "$PY" -m pip install "$@" && return 0
  "$PY" -m pip install --break-system-packages "$@"
}

log "checking the existing torch build"
NEED_TORCH=1
if "$PY" - <<'PY' 2>/dev/null
import sys, torch
ok = torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) >= (12, 0)
cuda = torch.version.cuda or "0.0"
sys.exit(0 if (ok and tuple(int(x) for x in cuda.split(".")[:2]) >= (12, 8)) else 1)
PY
then
  NEED_TORCH=0
  log "existing torch already supports sm_120, keeping it"
else
  log "existing torch is missing or predates sm_120"
fi

if [ "$NEED_TORCH" = "1" ]; then
  log "installing torch (cu128) for Blackwell"
  install_pkgs --index-url https://download.pytorch.org/whl/cu128 torch \
    || die "torch install failed"
fi

"$PY" - <<'PY' || die "torch cannot see an sm_120 GPU"
import sys, torch
cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
print(f"  torch {torch.__version__} cuda {torch.version.cuda} capability {cap}")
sys.exit(0 if cap and tuple(cap) >= (12, 0) else 1)
PY

log "installing transformers / diffusers (main, needs QwenImage21Pipeline)"
# torchvision is not optional: Qwen3-VL's video processor imports it while the
# pipeline loads, even for text-to-image
install_pkgs "transformers>=5.17.0" accelerate pillow safetensors huggingface_hub \
  || die "helper dependency install failed"
install_pkgs --index-url https://download.pytorch.org/whl/cu128 torchvision \
  || log "WARN: torchvision install failed; the pipeline will probably refuse to load"
install_pkgs "git+https://github.com/huggingface/diffusers.git" \
  || die "diffusers install failed (is github.com reachable?)"
"$PY" -c "import diffusers; print('  diffusers', diffusers.__version__)" || die "diffusers import failed"
"$PY" -c "from diffusers import QwenImage21Pipeline; print('  QwenImage21Pipeline available')" \
  || die "diffusers build lacks QwenImage21Pipeline"

# ---------------------------------------------------------------- weights ----
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model_index.json" ]; then
  log "weights already present in $MODEL_DIR"
else
  log "downloading Qwen/Qwen-Image-2.1-Turbo (~32 GB, bf16)"
  "$PY" - "$MODEL_DIR" <<'PY' || die "weight download failed"
import sys
from huggingface_hub import snapshot_download
# *.jinja matters: processor/chat_template.jinja is not a *.json, and without it
# the pipeline dies with "this processor does not have a chat template"
path = snapshot_download("Qwen/Qwen-Image-2.1-Turbo", local_dir=sys.argv[1],
                         max_workers=8,
                         allow_patterns=["*.json", "*.safetensors", "*.txt",
                                          "*.md", "*.jinja"])
print("  ->", path)
PY
fi

log "diffusers stage ready"
