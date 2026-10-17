#!/bin/bash
# =============================================================================
# Qwen-Image-2.1-Turbo on a rented GPU: two engines measured in one rental.
#
#   lane 1  stable-diffusion.cpp  -> K worker processes, GGUF, no batch dim
#   lane 2  diffusers (bf16)      -> one process, true batch (1/2/4/8)
#
# Paste this whole file into the Vast.ai instance's "On-start script" field.
# It builds sd.cpp for Blackwell (sm_120), downloads the GGUF weights, sweeps
# process-level concurrency, then installs torch+diffusers, downloads the bf16
# checkpoint, sweeps batch size, and prints both tables plus a comparison.
#
# Suggested instance: RTX PRO 6000 (96 GB), disk >= 150 GB,
#   image: vastai/base-image:cuda-12.9.2-auto
# Watch progress:  vastai logs <instance_id>   (or the Logs tab)
# Results: /workspace/bench/latest-*.json, /workspace/BENCH_DONE
#
# Expect ~60-90 min: ~15 compile, ~20 downloads (21 GB GGUF + 32 GB bf16),
# ~20 sweeps. Everything is idempotent, so a re-run resumes.
# =============================================================================
set -uo pipefail
export WORK=${WORK:-/workspace}
export LEVELS=${LEVELS:-1,2,3,4,5,6}       # sd.cpp: worker processes
export PER_WORKER=${PER_WORKER:-3}
export DIFF_LEVELS=${DIFF_LEVELS:-1,2,4,8} # diffusers: batch size
export DIFF_RUNS=${DIFF_RUNS:-2}
export DIFF_COMPILE=${DIFF_COMPILE:-1}
export WIDTH=${WIDTH:-1024}
export HEIGHT=${HEIGHT:-1024}
export DENOISERS=${DENOISERS:-Q8_0,AD-Q4_K}
export ENCODER=${ENCODER:-Q8_0}
# the Vast base image keeps ML packages in /venv/main and leaves /usr/bin/python3
# bare, so anything touching torch must use the venv interpreter
PY=python3
for cand in /venv/main/bin/python /venv/main/bin/python3; do
  [ -x "$cand" ] && { PY=$cand; break; }
done
export PY
mkdir -p "$WORK/logs" "$WORK/bench"
echo "bundle start $(date -u +%FT%TZ) | python=$PY | sdcpp levels=$LEVELS | diffusers batches=$DIFF_LEVELS | ${WIDTH}x${HEIGHT}"

# ------------------------------------------------------- stage 1: sd.cpp ------
cat > "$WORK/_bootstrap.sh" <<'BOOT_EOF'
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
BOOT_EOF
# a child process, so a broken sd.cpp build cannot take the diffusers lane down
echo
echo "==================== STAGE 1: sd.cpp bootstrap ===================="
bash "$WORK/_bootstrap.sh" || echo "WARN: sd.cpp bootstrap failed; continuing"

echo
echo "==================== STAGE 2: diffusers bootstrap ===================="
cat > "$WORK/_diffusers.sh" <<'DIFF_EOF'
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
DIFF_EOF
bash "$WORK/_diffusers.sh" || echo "WARN: diffusers bootstrap failed; continuing"

# ------------------------------------------------------------- the tools -----
cat > "$WORK/bench.py" <<'BENCH_PY_EOF'
#!/usr/bin/env python3
"""Concurrency sweep for stable-diffusion.cpp on a rented GPU.

The question this answers: with N independent sd-server workers resident on one
96 GB card, how many images/minute does Qwen-Image-2.1-Turbo actually deliver,
and where does that curve flatten out?

Design notes
------------
* One sd-server process per worker. sd-server serialises requests, so "batch"
  here means N processes, each holding its own copy of denoiser + encoder + VAE
  in VRAM. That is also how the service container will be shaped, because it is
  the only parallelism sd.cpp gives us.
* Servers are started once and reused across concurrency levels, so a level
  comparison is not polluted by model reloads. Per-server VRAM is measured as
  each server comes up, which tells us the per-worker footprint directly.
* Endpoints are discovered, not assumed: the A1111-style /sdapi/v1/txt2img and
  the OpenAI-style /v1/images/generations are both tried.
* Timings are wall clock around the HTTP call, sampled per request.

Usage on the instance:
  python3 bench.py discover                    # what does this build speak?
  python3 bench.py sweep --levels 1,2,4,6 --per-worker 3
  python3 bench.py sweep --attach 4            # reuse servers already running
"""
from __future__ import annotations

import argparse
import atexit
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEF_SIGMAS_PRESET = "qwen-image-2.1-turbo"
DEF_SIGMAS_EXPLICIT = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"

PROMPTS = [
    "comic panel, a tired barista staring at a long receipt, flat colors, thick ink outlines",
    "comic panel, two coworkers arguing in a break room, expressive faces, screentone shading",
    "comic panel, a cat sitting on a keyboard as a spreadsheet scrolls by, warm lamp light",
    "comic panel, a person in a raincoat under a streetlight, neon reflections, noir palette",
    "comic panel, an intense close-up of someone reading a phone with a shocked face",
    "comic panel, a messy apartment with takeout boxes and a glowing monitor at 3am",
]


# ------------------------------------------------------------------ helpers --
def http_json(url: str, payload=None, timeout: float = 1800.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
        try:
            return r.status, json.loads(body)
        except json.JSONDecodeError:
            return r.status, body


def http_get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read(4000).decode("utf-8", "replace")


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


class VramSampler(threading.Thread):
    """Polls nvidia-smi in the background and records peak used MiB + mean util."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()
                used = 0.0
                util = 0.0
                for line in out:
                    mem, u = (x.strip() for x in line.split(","))
                    used += float(mem)
                    util = max(util, float(u))
                self.samples.append((used, util))
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)

    def peak_mb(self) -> float:
        return max((s[0] for s in self.samples), default=0.0)

    def util_avg(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s[1] for s in self.samples) / len(self.samples)


def gpu_info() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception as e:  # pragma: no cover
        out = f"nvidia-smi failed: {e}"
    return {"nvidia_smi": out}


# ------------------------------------------------------------------- server --
_ACTIVE: list["Worker"] = []


def _kill_all():
    for w in list(_ACTIVE):
        w.kill()


atexit.register(_kill_all)


class Worker:
    def __init__(self, port: int, logdir: str):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.logdir = logdir
        self.proc: subprocess.Popen | None = None
        self.log_path = os.path.join(logdir, f"sd-server-{port}.log")
        self.startup_s = None
        self.vram_after_start_mb = None
        self.vram_after_warm_mb = None

    def argv(self, a, sigmas: str):
        return [
            a.bin, "--diffusion-model", a.denoiser, "--vae", a.vae, "--llm", a.encoder,
            "--listen-ip", "127.0.0.1", "--listen-port", str(self.port),
            "--steps", str(a.steps), "--cfg-scale", str(a.cfg),
            "--sampling-method", "euler",
            "--sigmas", sigmas,
            "--rng", "cpu",
        ] + (["--diffusion-fa"] if a.flash_attn else [])

    def start(self, a, timeout: float = 420.0) -> None:
        """Start the server, retrying with progressively plainer arguments.

        sigma presets, flash attention flags and CLI spellings drift between
        sd.cpp commits, and a wrong flag looks exactly like a slow start, so we
        try the documented form first and degrade instead of dying.
        """
        variants = [(DEF_SIGMAS_PRESET, True), (DEF_SIGMAS_EXPLICIT, True),
                    (DEF_SIGMAS_EXPLICIT, False)]
        last_err = None
        for sigmas, fa in variants:
            argv = self.argv(a, sigmas)
            if not fa:
                argv = [x for i, x in enumerate(argv) if not (x == "--diffusion-fa")]
            t0 = time.time()
            log = open(self.log_path, "w")
            self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            deadline = t0 + timeout
            ready = False
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    last_err = "process exited"
                    break
                if port_open(self.port):
                    ready = True
                    break
                time.sleep(1.0)
            if not ready:
                log.close()
                last_err = self._tail()
                self.kill()
                continue
            self.startup_s = round(time.time() - t0, 2)
            time.sleep(2)  # let it settle; the first request allocates the graph
            self.vram_after_start_mb = _vram_now()
            if self not in _ACTIVE:
                _ACTIVE.append(self)
            log.flush()
            print(f"  worker :{self.port} up in {self.startup_s}s "
                  f"({self.vram_after_start_mb:.0f} MiB used)")
            return
        raise RuntimeError(f"worker :{self.port} failed to start; last log:\n{last_err}")

    def _tail(self, n: int = 25) -> str:
        try:
            with open(self.log_path, "r", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return "<no log>"

    def kill(self):
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # ---------------------------------------------------------------- HTTP --
    def generate(self, api: str, *, prompt: str, seed: int, width: int, height: int,
                 steps: int, cfg: float, timeout: float) -> tuple[float, dict]:
        if api == "sdapi":
            payload = {"prompt": prompt, "negative_prompt": "", "seed": seed,
                       "width": width, "height": height, "steps": steps,
                       "cfg_scale": cfg, "sampler_name": "euler"}
            url = f"{self.base}/sdapi/v1/txt2img"
        else:
            payload = {"prompt": prompt, "model": "qwen-image", "n": 1,
                       "size": f"{width}x{height}", "response_format": "b64_json"}
            url = f"{self.base}/v1/images/generations"
        t0 = time.time()
        status, body = http_json(url, payload, timeout=timeout)
        elapsed = time.time() - t0
        if status >= 400:
            raise RuntimeError(f"HTTP {status}: {str(body)[:400]}")
        return elapsed, body


def _vram_now() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def extract_b64(api: str, body: dict) -> str | None:
    if api == "sdapi":
        imgs = body.get("images") or []
        return imgs[0] if imgs else None
    data = body.get("data") or []
    if data and isinstance(data[0], dict):
        return data[0].get("b64_json")
    return None


# --------------------------------------------------------------------- modes --
def probe_api(w: Worker, a) -> str:
    """Return 'sdapi' or 'openai', whichever this build answers, and say so."""
    for api in ("sdapi", "openai"):
        try:
            dt, body = w.generate(api, prompt=PROMPTS[0], seed=a.seed, width=a.probe_size,
                                  height=a.probe_size, steps=a.steps, cfg=a.cfg,
                                  timeout=900)
            b64 = extract_b64(api, body)
            if b64:
                print(f"  API: {api} works (probe {dt:.1f}s at {a.probe_size}px)")
                return api
            print(f"  API: {api} answered without an image: {str(body)[:300]}")
        except Exception as e:
            print(f"  API: {api} no: {str(e)[:300]}")
    raise RuntimeError("no usable image endpoint; run `bench.py discover` and read the log")


def mode_discover(a):
    w = Worker(a.port, a.logdir)
    print("starting one worker to probe the HTTP surface...")
    w.start(a)
    try:
        for path in ("/", "/openapi.json", "/sdcpp/v1/openapi.json", "/sdapi/v1/options"):
            for target in (path,):
                try:
                    status, text = http_get(w.base + target, timeout=10)
                    print(f"\nGET {target} -> {status}\n{text[:600]}")
                except Exception as e:
                    print(f"\nGET {target} -> {e}")
        for api, payload, path in (
            ("sdapi", {"prompt": PROMPTS[0], "width": 512, "height": 512, "steps": 8, "seed": 1},
             "/sdapi/v1/txt2img"),
            ("openai", {"prompt": PROMPTS[0], "model": "qwen-image", "n": 1,
                        "size": "512x512", "response_format": "b64_json"},
             "/v1/images/generations"),
        ):
            try:
                t0 = time.time()
                status, body = http_json(w.base + path, payload, timeout=900)
                keys = list(body)[:12] if isinstance(body, dict) else type(body).__name__
                print(f"\nPOST {path} -> {status} in {time.time()-t0:.1f}s keys={keys}")
                if isinstance(body, dict):
                    imgs = extract_b64(api, body)
                    print("   has image:", bool(imgs))
                    if imgs:
                        save_png(imgs, os.path.join(a.outdir, f"probe-{api}.png"))
            except Exception as e:
                print(f"\nPOST {path} -> {e}")
    finally:
        if not a.keep:
            w.kill()


def _one_request(w: Worker, api: str, a, idx: int, results: list, lock: threading.Lock):
    prompt = PROMPTS[idx % len(PROMPTS)]
    try:
        dt, body = w.generate(api, prompt=prompt, seed=a.seed + idx, width=a.width,
                              height=a.height, steps=a.steps, cfg=a.cfg, timeout=a.timeout)
        with lock:
            results.append((dt, True, extract_b64(api, body)))
    except Exception as e:
        with lock:
            results.append((0.0, False, str(e)[:300]))


def mode_sweep(a):
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    n_servers = max(levels)
    workers: list[Worker] = []

    if a.attach:
        workers = [Worker(a.port + i, a.logdir) for i in range(n_servers)]
        alive = [w for w in workers if port_open(w.port)]
        if len(alive) < n_servers:
            raise SystemExit(f"--attach asked for {n_servers} servers on ports "
                             f"{a.port}..{a.port+n_servers-1}, found {len(alive)}")
        print(f"attached to {len(alive)} running servers")
    else:
        print(f"starting up to {n_servers} worker(s), {a.denoiser.split('/')[-1]} + "
              f"{a.encoder.split('/')[-1]}")
        for i in range(n_servers):
            w = Worker(a.port + i, a.logdir)
            try:
                w.start(a)
            except Exception as e:
                # an OOM at K is itself the answer to "how many fit"
                print(f"  stopping at {len(workers)} workers: {str(e)[:600]}")
                break
            workers.append(w)

    api = probe_api(workers[0], a)

    # warm every server once at the real size: catches lazy-load OOM, and keeps
    # graph allocation out of the level-1 numbers
    print("warming...")
    warm: list[Worker] = []
    for i, w in enumerate(workers):
        try:
            w.generate(api, prompt=PROMPTS[i % len(PROMPTS)], seed=a.seed + i,
                       width=a.width, height=a.height, steps=a.steps, cfg=a.cfg,
                       timeout=a.timeout)
            # weights are mmap'd and only land in VRAM on first use, so the
            # start-time number is just the CUDA context; this one is the real
            # per-worker footprint
            w.vram_after_warm_mb = _vram_now()
            print(f"  worker :{w.port} warm, cumulative VRAM "
                  f"{w.vram_after_warm_mb:.0f} MiB")
            warm.append(w)
        except Exception as e:
            print(f"  worker :{w.port} failed its warmup, dropping it: {str(e)[:300]}")
            w.kill()
    workers = warm
    levels = [k for k in levels if k <= len(workers)]
    if not levels:
        raise SystemExit("no worker survived warmup")
    n_servers = len(workers)

    single = None
    report = {
        "meta": {
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": gpu_info(),
            "denoiser": a.denoiser, "encoder": a.encoder, "vae": a.vae,
            "width": a.width, "height": a.height, "steps": a.steps, "cfg": a.cfg,
            "api": api, "flash_attn": a.flash_attn,
            "bin": a.bin,
        },
        "startup": [{"port": w.port, "startup_s": w.startup_s,
                     "vram_after_mb": w.vram_after_start_mb,
                     "vram_after_warm_mb": w.vram_after_warm_mb} for w in workers],
        "levels": [],
    }

    for k in levels:
        active = workers[:k]
        per = a.per_worker
        results: list = []
        lock = threading.Lock()
        sampler = VramSampler()
        sampler.start()
        t0 = time.time()

        def run_one(w):
            for j in range(per):
                _one_request(w, api, a, j * 7 + k, results, lock)

        threads = [threading.Thread(target=run_one, args=(w,)) for w in active]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        sampler.stop()

        ok = [r for r in results if r[1]]
        lat = sorted(r[0] for r in ok)
        images = len(ok)
        entry = {
            "workers": k,
            "per_worker": per,
            "images": images,
            "failed": len(results) - images,
            "wall_s": round(wall, 2),
            "img_per_min": round(images / wall * 60, 2) if wall else 0,
            "per_image_s": {"mean": round(sum(lat) / len(lat), 2) if lat else None,
                            "p50": round(lat[len(lat) // 2], 2) if lat else None,
                            "max": round(lat[-1], 2) if lat else None},
            "vram_peak_mb": round(sampler.peak_mb(), 0),
            "gpu_util_avg": round(sampler.util_avg(), 1),
            "errors": [r[2] for r in results if not r[1]][:3],
        }
        if k == 1:
            single = entry["img_per_min"]
        entry["scaling_eff"] = round(entry["img_per_min"] / (single * k), 3) if single else None
        report["levels"].append(entry)
        print(f"  k={k:2d}  {entry['img_per_min']:7.2f} img/min  "
              f"p50 {entry['per_image_s']['p50']}s  vram {entry['vram_peak_mb']:.0f} MiB  "
              f"util {entry['gpu_util_avg']}%  eff {entry['scaling_eff']}")
        if ok and ok[0][2]:
            save_png(ok[0][2], os.path.join(a.outdir, f"sample-k{k}-{a.width}x{a.height}.png"))

    report["concurrency_probe"] = probe_concurrency(workers[0], api, a)

    out = os.path.join(a.outdir, f"sweep-{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    print_table(report)
    if not a.keep and not a.attach:
        for w in workers:
            w.kill()


def probe_concurrency(w: Worker, api: str, a) -> dict:
    """Does one server run two requests at once, or queue them?"""
    t_seq = []
    for i in range(2):
        dt, _ = w.generate(api, prompt=PROMPTS[i], seed=a.seed + 100 + i, width=a.width,
                           height=a.height, steps=a.steps, cfg=a.cfg, timeout=a.timeout)
        t_seq.append(dt)
    t0 = time.time()
    threads = [threading.Thread(target=w.generate, kwargs=dict(
        api=api, prompt=PROMPTS[i], seed=a.seed + 200 + i, width=a.width,
        height=a.height, steps=a.steps, cfg=a.cfg, timeout=a.timeout)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_par = time.time() - t0
    ratio = t_par / max(t_seq) if t_seq else None
    print(f"  concurrency probe: 2 sequential {sum(t_seq):.1f}s, 2 parallel {t_par:.1f}s "
          f"(ratio {ratio:.2f} -> {'queued' if ratio and ratio > 1.6 else 'overlapped'})")
    return {"two_sequential_s": round(sum(t_seq), 2), "two_parallel_s": round(t_par, 2),
            "ratio": round(ratio, 3) if ratio else None}


def save_png(b64: str, path: str):
    data = b64.split(",", 1)[-1] if b64.startswith("data:") else b64
    try:
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))
    except Exception as e:
        print(f"  (could not save {path}: {e})")


def print_table(report):
    print("\n| workers | img/min | img/min per worker | p50 s | VRAM MiB | GPU util | scaling eff |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for e in report["levels"]:
        w = e["workers"]
        print(f"| {w} | {e['img_per_min']} | {round(e['img_per_min']/w, 2)} | "
              f"{e['per_image_s']['p50']} | {e['vram_peak_mb']:.0f} | "
              f"{e['gpu_util_avg']} | {e['scaling_eff']} |")


# ---------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["discover", "sweep"])
    p.add_argument("--bin", default="/workspace/bin/sd-server")
    p.add_argument("--models", default="/workspace/models")
    p.add_argument("--denoiser", default=os.path.join(os.environ.get("MODELS", "/workspace/models"),
                                                     "denoiser-Q8_0.gguf"))
    p.add_argument("--encoder", default=os.path.join(os.environ.get("MODELS", "/workspace/models"),
                                                    "encoder-Q8_0.gguf"))
    p.add_argument("--vae", default=os.path.join(os.environ.get("MODELS", "/workspace/models"),
                                                "qwen_image_2.1_vae_bf16.safetensors"))
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--probe-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--levels", default="1,2,4,6")
    p.add_argument("--per-worker", type=int, default=3)
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--outdir", default="/workspace/bench")
    p.add_argument("--logdir", default="/workspace/logs")
    p.add_argument("--flash-attn", dest="flash_attn", action="store_true", default=True)
    p.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    p.add_argument("--attach", action="store_true", help="reuse already-running servers")
    p.add_argument("--keep", action="store_true", help="leave servers running on exit")
    a = p.parse_args()

    for d in (a.outdir, a.logdir):
        os.makedirs(d, exist_ok=True)
    for path, label in ((a.bin, "--bin"), (a.denoiser, "--denoiser"),
                        (a.encoder, "--encoder"), (a.vae, "--vae")):
        if not os.path.exists(path):
            raise SystemExit(f"missing {label}: {path}")

    if a.mode == "discover":
        mode_discover(a)
    else:
        mode_sweep(a)


if __name__ == "__main__":
    main()
BENCH_PY_EOF

cat > "$WORK/bench_diffusers.py" <<'BENCH_DIFF_EOF'
#!/usr/bin/env python3
"""Batched diffusers benchmark for Qwen-Image-2.1-Turbo.

The point of this file is the one thing sd.cpp cannot do: pass a *list* of
prompts and get one forward pass with a batch dimension. Weight reads amortise,
and the 8B text encoder runs once for the whole batch instead of once per panel.

Compared against bench.py (the sd.cpp lane) on the same box, same prompts, same
resolution, this answers: is it worth moving the container from a C++ binary to
a Python service?

Caveat that shapes the numbers: a diffusion step is already a large GEMM, so
batching buys ~1.7-2.8x at batch 4, not 4x. Whatever it buys, it buys without
duplicating 17 GB of weights per worker.

  python bench_diffusers.py --model /workspace/models/qwen-image-2.1-turbo \
      --levels 1,2,4,8 --runs 2 --width 1024 --height 1024 --out /workspace/bench
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import torch

try:  # same prompt set as the sd.cpp lane, for an apples-to-apples comparison
    from bench import PROMPTS as _SHARED_PROMPTS
except Exception:
    _SHARED_PROMPTS = None

FALLBACK_PROMPTS = [
    "comic panel, a tired barista staring at a long receipt, flat colors, thick ink outlines",
    "comic panel, two coworkers arguing in a break room, expressive faces, screentone shading",
    "comic panel, a cat sitting on a keyboard as a spreadsheet scrolls by, warm lamp light",
    "comic panel, a person in a raincoat under a streetlight, neon reflections, noir palette",
    "comic panel, an intense close-up of someone reading a phone with a shocked face",
    "comic panel, a messy apartment with takeout boxes and a glowing monitor at 3am",
]
PROMPTS = _SHARED_PROMPTS or FALLBACK_PROMPTS


def load_pipeline(model: str, attn: str | None):
    from diffusers import QwenImage21Pipeline

    pipe, last = None, None
    # diffusers renamed torch_dtype -> dtype; support whichever this build has
    for kw in ("dtype", "torch_dtype"):
        try:
            pipe = QwenImage21Pipeline.from_pretrained(model, **{kw: torch.bfloat16})
            print(f"  loaded with {kw}=bfloat16")
            break
        except TypeError as e:
            last = e
    if pipe is None:
        raise SystemExit(f"could not load pipeline: {last}")

    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if attn:
        fn = getattr(pipe.transformer, "set_attention_backend", None)
        if fn is None:
            print("  attention backend setter not available; using defaults")
        else:
            try:
                fn(attn)
                print(f"  attention backend: {attn}")
            except Exception as e:
                print(f"  attention backend {attn!r} rejected: {str(e)[:200]}")
    return pipe


def call(pipe, prompts, width, height, seed, use_kv: bool):
    """Generate len(prompts) images in one call, tolerating signature drift."""
    gen = torch.Generator("cpu").manual_seed(seed)
    attempts = [
        dict(prompt=prompts, width=width, height=height, generator=gen,
             use_kv_cache=use_kv),
        dict(prompt=prompts, width=width, height=height, generator=gen),
        dict(prompt=prompts[0], width=width, height=height, generator=gen,
             num_images_per_prompt=len(prompts)),
    ]
    last = None
    for kw in attempts:
        try:
            torch.cuda.synchronize()
            return pipe(**kw)
        except (TypeError, ValueError) as e:
            last = e
    raise RuntimeError(f"all call signatures failed: {last}")


def measure(pipe, batch: int, a, seed0: int):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
    call(pipe, prompts, a.width, a.height, seed0 - 1, not a.no_kv_cache)  # warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times, out = [], None
    for r in range(a.runs):
        t0 = time.time()
        out = call(pipe, prompts, a.width, a.height, seed0 + r, not a.no_kv_cache)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    vram = torch.cuda.max_memory_allocated() / 2**20
    mean = sum(times) / len(times)
    return {
        "batch": batch,
        "runs": a.runs,
        "wall_s": round(mean, 2),
        "wall_all_s": [round(t, 2) for t in times],
        "images": batch,
        "img_per_min": round(batch / mean * 60, 2) if mean else 0,
        "per_image_s": round(mean / batch, 2),
        "vram_peak_mb": round(vram, 0),
    }, out


def measure_encode(pipe, batch: int, a):
    """Text encoder cost for `batch` prompts vs one prompt.

    This is the second half of the batching win: an 8B encoder over 8 prompts in
    one pass instead of 8 passes. Guarded because encode_prompt's signature is
    not part of the public promise.
    """
    fn = getattr(pipe, "encode_prompt", None)
    if fn is None or batch < 2:
        return None
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch)]
    try:
        torch.cuda.synchronize()
        t0 = time.time()
        fn(prompt=[prompts[0]])
        torch.cuda.synchronize()
        one = time.time() - t0

        torch.cuda.synchronize()
        t0 = time.time()
        fn(prompt=prompts)
        torch.cuda.synchronize()
        many = time.time() - t0
        return {"batch": batch, "encode_batch_s": round(many, 3),
                "encode_single_s": round(one, 3),
                "amortised_s_per_prompt": round(many / batch, 3)}
    except Exception as e:
        print(f"  encode_prompt probe failed (not fatal): {str(e)[:160]}")
        return None


def load_refs(paths: list[str]):
    """Reference images for the edit / multi-reference path of the same pipeline."""
    from PIL import Image
    refs = []
    for p in paths:
        img = Image.open(p)
        refs.append(img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB"))
        print(f"  reference {p} -> {refs[-1].size} {refs[-1].mode}")
    return refs


def call_with_refs(pipe, prompt, refs, width, height, seed, use_kv: bool, a):
    """One prompt + reference images. Qwen-Image-2.1 does edit/reference in the
    same pipeline, so this is the workload a comic panel with a character sheet
    actually runs — and it is *not* the same cost as text-only: the references
    add encoded tokens to the conditioning."""
    gen = torch.Generator("cpu").manual_seed(seed)
    payload = dict(prompt=prompt, width=width, height=height, generator=gen,
                   use_kv_cache=use_kv)
    attempts = [
        {**payload, "image": refs if len(refs) > 1 else refs[0]},
        {**payload, "image": refs[0]},
    ]
    last = None
    for kw in attempts:
        try:
            torch.cuda.synchronize()
            return pipe(**kw)
        except (TypeError, ValueError) as e:
            last = e
    raise RuntimeError(f"all reference call signatures failed: {last}")


def measure_reference(pipe, refs, a, count: int):
    """Time `count` references against the same prompt, for comparison with the
    text-only number."""
    use = [refs[i % len(refs)] for i in range(count)] if refs else []
    prompt = PROMPTS[0]
    call_with_refs(pipe, prompt, use, a.width, a.height, a.seed - 1,
                   not a.no_kv_cache, a)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for r in range(a.runs):
        t0 = time.time()
        call_with_refs(pipe, prompt, use, a.width, a.height, a.seed + r,
                       not a.no_kv_cache, a)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    mean = sum(times) / len(times)
    return {"references": count, "wall_s": round(mean, 2),
            "s_per_image": round(mean, 2),
            "img_per_min": round(60 / mean, 2) if mean else 0,
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 0)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/workspace/models/qwen-image-2.1-turbo")
    p.add_argument("--levels", default="1,2,4,8")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--attention", default="flash")
    p.add_argument("--reference", default="",
                   help="comma-separated reference image paths; measures the "
                        "edit / multi-reference path at 1 and 3 references")
    p.add_argument("--reference-counts", default="1,3")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the transformer before measuring (slow first call)")
    p.add_argument("--no-kv-cache", action="store_true")
    p.add_argument("--out", default="/workspace/bench")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    print(f"torch {torch.__version__} cuda {torch.version.cuda} "
          f"capability {torch.cuda.get_device_capability()} "
          f"gpu {torch.cuda.get_device_name(0)}")
    import diffusers
    print(f"diffusers {diffusers.__version__}")

    pipe = load_pipeline(a.model, a.attention)

    if a.compile:
        print("  torch.compile(transformer) ... this first call is slow")
        t0 = time.time()
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        try:
            call(pipe, [PROMPTS[0]], a.width, a.height, a.seed, not a.no_kv_cache)
            print(f"  compiled in {time.time() - t0:.1f}s")
        except Exception:
            print("  compile failed, continuing eagerly\n" + traceback.format_exc(limit=2))
            if hasattr(pipe.transformer, "_orig_mod"):
                pipe.transformer = pipe.transformer._orig_mod

    report = {
        "meta": {
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine": "diffusers",
            "model": a.model,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability()),
            "width": a.width, "height": a.height,
            "attention": a.attention, "compiled": a.compile,
            "kv_cache": not a.no_kv_cache,
        },
        "levels": [], "encode": [],
    }

    single = None
    for batch in levels:
        print(f"  batch {batch} ...", end="", flush=True)
        entry, out = measure(pipe, batch, a, a.seed)
        entry["scaling_eff"] = (round(entry["img_per_min"] / (single * batch), 3)
                                if single else None)
        single = single or entry["img_per_min"]
        report["levels"].append(entry)
        enc = measure_encode(pipe, batch, a)
        if enc:
            report["encode"].append(enc)
        print(f" {entry['img_per_min']:7.2f} img/min  "
              f"{entry['per_image_s']}s/img  vram {entry['vram_peak_mb']:.0f} MiB  "
              f"eff {entry['scaling_eff']}")
        if out is not None and getattr(out, "images", None):
            out.images[0].save(os.path.join(
                a.out, f"diffusers-b{batch}-{a.width}x{a.height}.png"))

    if a.reference:
        refs = load_refs([x.strip() for x in a.reference.split(",") if x.strip()])
        print("\n  reference path: same pipeline with a character sheet attached")
        report["reference"] = []
        for count in [int(x) for x in a.reference_counts.split(",") if x.strip()]:
            entry = measure_reference(pipe, refs, a, count)
            report["reference"].append(entry)
            print(f"  refs={count}  {entry['s_per_image']}s/img  "
                  f"vram {entry['vram_peak_mb']:.0f} MiB")

    tag = "compiled" if a.compile else "eager"
    path = os.path.join(a.out, f"diffusers-{tag}-{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {path}")
    print("\n| batch | img/min | s/img | VRAM MiB | scaling eff |")
    print("|---:|---:|---:|---:|---:|")
    for e in report["levels"]:
        print(f"| {e['batch']} | {e['img_per_min']} | {e['per_image_s']} | "
              f"{e['vram_peak_mb']:.0f} | {e['scaling_eff']} |")
    for e in report["encode"]:
        print(f"  encode: {e['batch']} prompts in {e['encode_batch_s']}s "
              f"(1 prompt alone {e['encode_single_s']}s)")
    if report.get("reference"):
        print("\n| refs | s/image | img/min | VRAM MiB |")
        print("|---:|---:|---:|---:|")
        for e in report["reference"]:
            print(f"| {e['references']} | {e['s_per_image']} | "
                  f"{e['img_per_min']} | {e['vram_peak_mb']:.0f} |")


if __name__ == "__main__":
    main()
BENCH_DIFF_EOF

cat > "$WORK/compare.py" <<'COMPARE_PY_EOF'
#!/usr/bin/env python3
"""Compare the sd.cpp lane and the diffusers lane on one page.

Both lanes write JSON with a "levels" list, but with different config keys
(sd.cpp: "workers", diffusers: "batch") and different meanings, so this prints a
single table rather than pretending the rows are identical.

  python compare.py /workspace/bench/latest-q8.json \
                    /workspace/bench/latest-diffusers.json
"""
from __future__ import annotations

import json
import sys


def load(path: str):
    with open(path) as f:
        data = json.load(f)
    meta = data.get("meta", {})
    engine = meta.get("engine") or ("sd.cpp" if "api" in meta else "unknown")
    rows = []
    for e in data.get("levels", []):
        config = e.get("workers", e.get("batch"))
        rows.append({
            "engine": engine,
            "config": config,
            "img_per_min": e.get("img_per_min"),
            "per_image_s": (e.get("per_image_s", {}) or {}).get("p50")
                           if isinstance(e.get("per_image_s"), dict) else e.get("per_image_s"),
            "vram_mb": e.get("vram_peak_mb"),
            "util": e.get("gpu_util_avg"),
            "eff": e.get("scaling_eff"),
        })
    return engine, meta, rows, data


def main(paths: list[str]) -> int:
    all_rows, metas = [], []
    for p in paths:
        try:
            engine, meta, rows, data = load(p)
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        metas.append((p, engine, meta))
        all_rows.extend(rows)

    if not all_rows:
        print("no comparable rows found")
        return 1

    print("| engine | config | img/min | s/img | VRAM MiB | GPU util | scaling eff |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in all_rows:
        vram = f"{r['vram_mb']:.0f}" if isinstance(r["vram_mb"], (int, float)) else "?"
        print(f"| {r['engine']} | {r['config']} | {r['img_per_min']} | "
              f"{r['per_image_s']} | {vram} | {r['util']} | {r['eff']} |")

    best = max(all_rows, key=lambda r: r["img_per_min"] or 0)
    print(f"\nbest: {best['engine']} at {best['config']} -> "
          f"{best['img_per_min']} img/min")

    for p, engine, meta in metas:
        bits = [f"{k}={meta[k]}" for k in ("width", "height", "attention", "compiled",
                                           "kv_cache", "api", "denoiser")
                if k in meta]
        print(f"  {p}: {engine} {' '.join(bits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
COMPARE_PY_EOF

newest() { ls -t "$WORK"/bench/"$1"-*.json 2>/dev/null | head -1; }
COMMON=(--levels "$LEVELS" --per-worker "$PER_WORKER" --width "$WIDTH" --height "$HEIGHT")
BIN_ARGS=(--bin "$WORK/bin/sd-server" --models "$WORK/models")

echo
echo "==================== SWEEP 1: sd.cpp (K processes) ===================="
if [ -x "$WORK/bin/sd-server" ] && [ -f "$WORK/models/denoiser-Q8_0.gguf" ]; then
  echo "--- discover (which HTTP API this build speaks; not fatal)"
  python3 "$WORK/bench.py" discover "${BIN_ARGS[@]}" 2>&1 | tail -n 60

  echo
  echo "--- A: denoiser Q8_0 (closest to BF16), encoder $ENCODER"
  python3 "$WORK/bench.py" sweep "${BIN_ARGS[@]}" "${COMMON[@]}" \
    --denoiser "$WORK/models/denoiser-Q8_0.gguf" \
    --encoder "$WORK/models/encoder-$ENCODER.gguf" \
    --vae "$WORK/models/qwen_image_2.1_vae_bf16.safetensors" 2>&1 | tail -n 80
  SNAP=$(newest sweep); [ -n "$SNAP" ] && cp -f "$SNAP" "$WORK/bench/latest-q8.json"

  if [ -f "$WORK/models/denoiser-AD-Q4_K.gguf" ]; then
    echo
    echo "--- B: denoiser AD-Q4_K (smaller footprint, more workers)"
    python3 "$WORK/bench.py" sweep "${BIN_ARGS[@]}" "${COMMON[@]}" \
      --denoiser "$WORK/models/denoiser-AD-Q4_K.gguf" \
      --encoder "$WORK/models/encoder-$ENCODER.gguf" \
      --vae "$WORK/models/qwen_image_2.1_vae_bf16.safetensors" 2>&1 | tail -n 80
    SNAP=$(newest sweep); [ -n "$SNAP" ] && cp -f "$SNAP" "$WORK/bench/latest-q4.json"
  fi
else
  echo "sd.cpp lane unavailable (no sd-server binary or Q8_0 weights); skipping"
fi

echo
echo "==================== SWEEP 2: diffusers (true batch) ===================="
DIFF_ARGS=(--model "$WORK/models/qwen-image-2.1-turbo"
           --width "$WIDTH" --height "$HEIGHT" --out "$WORK/bench")
if [ -d "$WORK/models/qwen-image-2.1-turbo" ] && python3 -c "import diffusers" 2>/dev/null; then
  echo "--- eager, batches $DIFF_LEVELS"
  "$PY" "$WORK/bench_diffusers.py" "${DIFF_ARGS[@]}" \
    --levels "$DIFF_LEVELS" --runs "$DIFF_RUNS" --attention flash 2>&1 | tail -n 80
  SNAP=$(newest diffusers-eager); [ -n "$SNAP" ] && cp -f "$SNAP" "$WORK/bench/latest-diffusers.json"

  if [ "$DIFF_COMPILE" = "1" ]; then
    echo
    echo "--- torch.compile, batch 4 (first call compiles; expect minutes)"
    "$PY" "$WORK/bench_diffusers.py" "${DIFF_ARGS[@]}" \
      --levels 4 --runs "$DIFF_RUNS" --attention flash --compile 2>&1 | tail -n 60
    SNAP=$(newest diffusers-compiled); [ -n "$SNAP" ] && cp -f "$SNAP" "$WORK/bench/latest-diffusers-compiled.json"
  fi
else
  echo "diffusers lane unavailable (no imports or no weights); skipping"
fi

echo
echo "================== COMPARISON =================="
python3 "$WORK/compare.py" \
  "$WORK/bench/latest-q8.json" "$WORK/bench/latest-q4.json" \
  "$WORK/bench/latest-diffusers.json" "$WORK/bench/latest-diffusers-compiled.json" \
  2>&1 | tail -n 40

echo
echo "================== FILES =================="
cat "$WORK/READY.json" 2>/dev/null
ls -la "$WORK"/bench/*.json 2>/dev/null
echo "Copy the tables above and the JSON files in $WORK/bench back for analysis."
touch "$WORK/BENCH_DONE"
echo "BENCH_DONE $(date -u +%FT%TZ)"
