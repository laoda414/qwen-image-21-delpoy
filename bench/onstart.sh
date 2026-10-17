#!/usr/bin/env bash
# Phase 1 bootstrap for a rented Vast.ai box: build stable-diffusion.cpp with CUDA
# for Blackwell (sm_120) and stage the Qwen-Image-2.1-Turbo (+ Uncensored encoder)
# weights in /workspace. Idempotent: safe to re-run; skips work already done.
#
# Runs as root inside the Vast container, from the instance's onstart hook.
# Everything lands in /workspace (the persistent volume), logs in /workspace/logs.
set -uo pipefail

WORK=${WORK:-/workspace}
MODELS="$WORK/models"
BIN="$WORK/bin"
LOGS="$WORK/logs"
SRC="$WORK/src"
mkdir -p "$MODELS" "$BIN" "$LOGS" "$SRC"
# mirror to a file when bash features are available; the instance log already
# captures stdout, so this is only for later inspection over ssh
if [ -n "${BASH_VERSION:-}" ]; then exec > >(tee -a "$LOGS/onstart.log") 2>&1; fi

log() { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; touch "$WORK/FAILED"; echo "$*" > "$WORK/FAILED"; exit 1; }

# ---------------------------------------------------------------- tunables ----
# Which rungs to fetch. Encoder = the uncensored one; denoiser = stock Turbo.
ENCODER=${ENCODER:-Q8_0}          # Q8_0 | AD-Q6_K | AD-Q5_K | AD-Q4_K | BF16
DENOISERS=${DENOISERS:-Q8_0,AD-Q4_K}  # comma list, so one rent measures several quants
SDCPP_REPO=${SDCPP_REPO:-https://github.com/leejet/stable-diffusion.cpp.git}
SDCPP_REF=${SDCPP_REF:-master}     # pin to a SHA once a build is proven
CUDA_ARCH=${CUDA_ARCH:-120}        # sm_120 = RTX PRO 6000 / RTX 50 Blackwell
HF=${HF:-https://huggingface.co}

ENC_REPO="$HF/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF/resolve/main"
DEN_REPO="$HF/AtomicChat/Qwen-Image-2.1-Turbo-GGUF/resolve/main"
VAE_URL="$HF/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"

# ------------------------------------------------------------ 1. packages ----
if ! command -v python3 >/dev/null || ! command -v cmake >/dev/null \
   || ! command -v g++ >/dev/null || ! command -v git >/dev/null; then
  log "installing build deps"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || die "apt-get update failed"
  apt-get install -y -qq --no-install-recommends \
    build-essential cmake git curl ca-certificates pkg-config libcurl4-openssl-dev python3 \
    || die "apt-get install failed"
fi

# disk is the silent killer here: 21 GB of weights + ~6 GB of build artefacts
AVAIL_GB=$(df -Pk "$WORK" | awk 'NR==2 {printf "%d", $4/1024/1024}')
log "free disk in $WORK: ${AVAIL_GB} GB"
[ "$AVAIL_GB" -ge 40 ] || log "WARN: only ${AVAIL_GB} GB free; downloads may run out of space"

# -------------------------------------------------------------- 2. nvcc ------
if ! command -v nvcc >/dev/null; then
  log "nvcc missing, trying the distro CUDA toolkit"
  export DEBIAN_FRONTEND=noninteractive
  (apt-get install -y -qq cuda-toolkit-12-9 || apt-get install -y -qq cuda-toolkit-12-8 \
    || apt-get install -y -qq cuda-toolkit) || die "no CUDA toolkit and apt install failed"
fi
NVCC_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
log "nvcc release: ${NVCC_VER:-unknown}"
# sm_120 needs CUDA >= 12.8
case "$NVCC_VER" in
  12.[89]*|13.*) : ;;
  *) die "CUDA ${NVCC_VER:-?} cannot target sm_120 (need >=12.8). Re-rent with vastai/base-image:cuda-12.9.2-auto" ;;
esac

# ------------------------------------------------------------ 3. sd.cpp ------
if [ ! -x "$BIN/sd-cli" ]; then
  if [ ! -d "$SRC/sdcpp/.git" ]; then
    log "cloning stable-diffusion.cpp ($SDCPP_REF)"
    rm -rf "$SRC/sdcpp"
    # ggml is a submodule; a plain --depth 1 clone leaves it empty and cmake
    # fails on a missing ggml/CMakeLists.txt
    git clone --depth 1 --branch "$SDCPP_REF" --recurse-submodules --shallow-submodules \
      "$SDCPP_REPO" "$SRC/sdcpp" \
      || git clone --depth 1 --recurse-submodules "$SDCPP_REPO" "$SRC/sdcpp" \
      || die "clone failed (is github.com reachable from this host?)"
  fi
  if [ ! -f "$SRC/sdcpp/ggml/CMakeLists.txt" ]; then
    log "fetching ggml submodule"
    git -C "$SRC/sdcpp" submodule update --init --recursive --depth 1 \
      || die "ggml submodule init failed"
  fi
  SHA=$(git -C "$SRC/sdcpp" rev-parse HEAD)
  log "building sd.cpp at $SHA for sm_$CUDA_ARCH"
  cd "$SRC/sdcpp" || die "cd failed"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DSD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGGML_NATIVE=ON > "$LOGS/cmake.log" 2>&1 || { tail -40 "$LOGS/cmake.log"; die "cmake configure failed"; }
  cmake --build build --config Release -j"$(nproc)" > "$LOGS/build.log" 2>&1 \
    || { tail -60 "$LOGS/build.log"; die "build failed"; }
  # server/CLI targets have been renamed and split over sd.cpp's history; build
  # whichever is missing instead of guessing
  for b in sd-cli sd-server; do
    if [ ! -f "build/bin/$b" ]; then
      log "target $b missing after the all-build, building it explicitly"
      cmake --build build --config Release -j"$(nproc)" --target "$b" >> "$LOGS/build.log" 2>&1 || true
    fi
  done
  [ -f build/bin/sd-cli ] || die "sd-cli was not produced (see $LOGS/build.log)"
  [ -f build/bin/sd-server ] || die "sd-server was not produced (see $LOGS/build.log)"
  cp build/bin/sd-cli build/bin/sd-server "$BIN"/ || die "copy of binaries failed"
  chmod +x "$BIN"/*
  log "built: $("$BIN/sd-cli" --version 2>&1 | head -3 | tr '\n' ' ')"
fi

# ------------------------------------------------------------ 4. weights -----
fetch() { # url dest
  local url=$1 dest=$2
  [ -s "$dest" ] && { log "have $(basename "$dest")"; return 0; }
  log "downloading $(basename "$dest")"
  curl -L --fail -sS --retry 5 --retry-delay 5 -C - -o "$dest.part" "$url" \
    || { log "download failed: $url"; rm -f "$dest.part"; return 1; }
  mv "$dest.part" "$dest"
}

fetch "$ENC_REPO/Qwen-Image-2.1-Turbo-Uncensored-${ENCODER}.gguf" \
      "$MODELS/encoder-${ENCODER}.gguf" || die "encoder download failed"
fetch "$VAE_URL" "$MODELS/qwen_image_2.1_vae_bf16.safetensors" || die "vae download failed"

IFS=',' read -r -a dlist <<< "$DENOISERS"
for d in "${dlist[@]}"; do
  fetch "$DEN_REPO/Qwen-Image-2.1-Turbo-${d}.gguf" "$MODELS/denoiser-${d}.gguf" \
    || log "WARN: denoiser $d failed, continuing"
done

# ------------------------------------------------------------ 5. manifest ----
python3 - "$WORK" <<'PY' || log "WARN: manifest failed"
import json, os, subprocess, sys, datetime
work = sys.argv[1]
models = os.path.join(work, "models")
def size(p): return os.path.getsize(p) if os.path.exists(p) else None
try:
    sha = subprocess.check_output(["git","-C",os.path.join(work,"src","sdcpp"),"rev-parse","HEAD"], text=True).strip()
except Exception:
    sha = None
man = {
  "ready_at": datetime.datetime.utcnow().isoformat() + "Z",
  "sdcpp_commit": sha,
  "cuda": subprocess.run(["nvcc","--version"],capture_output=True,text=True).stdout.splitlines()[-1] if os.path.exists("/usr/local/cuda") else None,
  "gpu": subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
  "files": {f: size(os.path.join(models,f)) for f in sorted(os.listdir(models))} if os.path.isdir(models) else {},
}
json.dump(man, open(os.path.join(work,"READY.json"),"w"), indent=2)
print(json.dumps(man, indent=2))
PY

log "DONE. Binaries in $BIN, weights in $MODELS"
