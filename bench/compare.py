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
