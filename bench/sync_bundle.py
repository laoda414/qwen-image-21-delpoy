#!/usr/bin/env python3
"""Generate onstart-bench.sh: one self-contained script to paste into Vast.ai.

The rented box cannot reach our files and this machine's uplink is too slow to
push an image, so the whole project ships as text. This keeps the bundle honest
by regenerating it from real files instead of hand-copying:

    onstart.sh            -> sd.cpp build + GGUF weights
    diffusers_stage.sh    -> torch/diffusers + bf16 weights
    bench.py              -> sd.cpp sweep (K processes)
    bench_diffusers.py    -> diffusers sweep (true batch dimension)
    compare.py            -> both lanes on one page

    python bench/sync_bundle.py && python bench/verify_bundle.py
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "onstart-bench.sh"

SHELL_BLOCKS = [("BOOT_EOF", "onstart.sh"), ("DIFF_EOF", "diffusers_stage.sh")]
PY_BLOCKS = [("BENCH_PY_EOF", "bench.py"),
             ("BENCH_DIFF_EOF", "bench_diffusers.py"),
             ("COMPARE_PY_EOF", "compare.py")]

HEADER = r"""#!/bin/bash
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
"""

STAGES = r"""
# ------------------------------------------------------- stage 1: sd.cpp ------
cat > "$WORK/_bootstrap.sh" <<'BOOT_EOF'
__BOOT__
BOOT_EOF
# a child process, so a broken sd.cpp build cannot take the diffusers lane down
echo
echo "==================== STAGE 1: sd.cpp bootstrap ===================="
bash "$WORK/_bootstrap.sh" || echo "WARN: sd.cpp bootstrap failed; continuing"

echo
echo "==================== STAGE 2: diffusers bootstrap ===================="
cat > "$WORK/_diffusers.sh" <<'DIFF_EOF'
__DIFF__
DIFF_EOF
bash "$WORK/_diffusers.sh" || echo "WARN: diffusers bootstrap failed; continuing"
"""

TOOLS = r"""
# ------------------------------------------------------------- the tools -----
cat > "$WORK/bench.py" <<'BENCH_PY_EOF'
__BENCH_PY__
BENCH_PY_EOF

cat > "$WORK/bench_diffusers.py" <<'BENCH_DIFF_EOF'
__BENCH_DIFF__
BENCH_DIFF_EOF

cat > "$WORK/compare.py" <<'COMPARE_PY_EOF'
__COMPARE_PY__
COMPARE_PY_EOF
"""

FOOTER = r"""
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
"""


def main() -> int:
    sources = {}
    for marker, name in SHELL_BLOCKS + PY_BLOCKS:
        path = HERE / name
        if not path.exists():
            print(f"missing source: {name}")
            return 1
        text = path.read_text(encoding="utf-8")
        if name.endswith(".sh"):
            text = text.split("\n", 1)[1]  # drop shebang; bundle has its own
        if marker in text:
            print(f"refusing to bundle: {name} contains the marker {marker}")
            return 1
        sources[marker] = text.rstrip("\n")

    stages = STAGES.replace("__BOOT__", sources["BOOT_EOF"]).replace(
        "__DIFF__", sources["DIFF_EOF"])
    tools = (TOOLS.replace("__BENCH_PY__", sources["BENCH_PY_EOF"])
                  .replace("__BENCH_DIFF__", sources["BENCH_DIFF_EOF"])
                  .replace("__COMPARE_PY__", sources["COMPARE_PY_EOF"]))

    out = HEADER + stages + tools + FOOTER
    OUT.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name}  ({len(out):,} bytes, {out.count(chr(10)) + 1} lines)")
    for m, n in PY_BLOCKS:
        print(f"  embedded {n}: {len(sources[m]):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
