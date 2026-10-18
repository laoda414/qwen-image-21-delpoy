#!/usr/bin/env python3
"""Measure the reference (character-sheet) path through the live service.

Qwen-Image-2.1 does text-to-image, edit and multi-reference in one pipeline, so
this is the workload a comic with recurring characters actually runs — and every
throughput number we have is text-only.

References are JPEG-compressed before upload: the tunnel to the box is the slow
part (~0.9 MB/s), and a 1 MB PNG reference would distort the measurement it is
supposed to be part of.

  python service/test_reference.py --url http://127.0.0.1:18080 --token ... \
      --ref panel-a.png --ref panel-b.png --runs 2
"""
from __future__ import annotations

import argparse
import base64
import io
import statistics
import time

import requests
from PIL import Image

PROMPT = ("comic panel, the same character standing in a bright cafe, "
          "flat colors, thick ink outlines")


def encode_ref(path: str, max_side: int = 1024) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b = buf.getvalue()
    print(f"  ref {path}: {img.size} -> {len(b)/1024:.0f} KB JPEG")
    return base64.b64encode(b).decode()


def run(url, token, refs, runs, size, out):
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    configs = [("text-only", []), ("1 reference", refs[:1]), ("3 references", refs[:3] or refs * 3)]
    results = {}
    for label, use in configs:
        if label != "text-only" and not use:
            continue
        lat = []
        for i in range(runs):
            body = {"prompt": f"{PROMPT} #{i}", "size": size, "n": 1, "seed": 3000 + i}
            if use:
                body["images"] = use
            t0 = time.time()
            r = requests.post(f"{url}/v1/images/generations", json=body, headers=h, timeout=600)
            dt = time.time() - t0
            if r.status_code != 200:
                print(f"  {label}: HTTP {r.status_code} {r.text[:160]}")
                break
            lat.append(dt)
            if i == 0:
                data = base64.b64decode(r.json()["data"][0]["b64_json"])
                with open(f"{out}-{label.replace(' ', '')}.png", "wb") as f:
                    f.write(data)
        if lat:
            results[label] = lat
            print(f"  {label:<14} {statistics.median(lat):5.1f}s median "
                  f"({', '.join(f'{x:.1f}' for x in lat)})")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--size", default="832x1216")
    ap.add_argument("--out", default="ref")
    a = ap.parse_args()

    print("encoding references:")
    refs = [encode_ref(p) for p in a.ref]
    while len(refs) < 3:
        refs = refs + refs[:1]

    print(f"\n{a.runs} runs per config at {a.size} (latency includes the tunnel):")
    res = run(a.url, a.token, refs, a.runs, a.size, a.out)

    base = statistics.median(res["text-only"]) if "text-only" in res else None
    print("\nverdict:")
    for label, lat in res.items():
        med = statistics.median(lat)
        if base and label != "text-only":
            print(f"  {label}: {med:.1f}s = {med/base:.2f}x the text-only cost")
        else:
            print(f"  {label}: {med:.1f}s")


if __name__ == "__main__":
    main()
