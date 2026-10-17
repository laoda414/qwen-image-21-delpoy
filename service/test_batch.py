#!/usr/bin/env python3
"""Simulate a comic batch against the live service: N panels, fired at once.

Reports total wall clock, per-panel latency distribution and throughput, and
writes the first panel to disk so the output can be eyeballed.
"""
from __future__ import annotations

import argparse
import base64
import statistics
import threading
import time

import requests

PROMPTS = [
    "comic panel, a tired barista staring at a long receipt, flat colors, thick ink outlines",
    "comic panel, two coworkers arguing in a break room, expressive faces, screentone shading",
    "comic panel, a cat sitting on a keyboard as a spreadsheet scrolls by, warm lamp light",
    "comic panel, a person in a raincoat under a streetlight, neon reflections, noir palette",
    "comic panel, an intense close-up of someone reading a phone with a shocked face",
    "comic panel, a messy apartment with takeout boxes and a glowing monitor at 3am",
    "comic panel, a barista wiping a counter while a customer points at the menu",
    "comic panel, a close-up of a printer jammed with a red error light",
    "comic panel, someone hiding behind a cubicle wall listening to a conversation",
    "comic panel, a hand reaching for the last donut in a break room box",
    "comic panel, a person asleep at a desk with a keyboard imprint on their cheek",
    "comic panel, a manager gesturing at a whiteboard covered in arrows",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--token", default="")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--size", default="832x1216")
    p.add_argument("--out", default="panel-0.png")
    a = p.parse_args()

    headers = {"Content-Type": "application/json"}
    if a.token:
        headers["Authorization"] = f"Bearer {a.token}"

    health = requests.get(f"{a.url}/healthz", timeout=30).json()
    print(f"healthz: {health}")

    lat: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    first: dict = {}

    def one(i: int):
        body = {"prompt": PROMPTS[i % len(PROMPTS)], "size": a.size, "n": 1, "seed": 1000 + i}
        t0 = time.time()
        try:
            r = requests.post(f"{a.url}/v1/images/generations", json=body,
                              headers=headers, timeout=900)
            dt = time.time() - t0
            if r.status_code != 200:
                with lock:
                    errors.append(f"{i}: HTTP {r.status_code} {r.text[:160]}")
                return
            data = r.json()["data"][0]["b64_json"]
            with lock:
                lat.append(dt)
                if not first:
                    first["i"] = i
                    first["bytes"] = base64.b64decode(data)
        except Exception as e:
            with lock:
                errors.append(f"{i}: {type(e).__name__} {e}")

    t0 = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(a.n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    print(f"\n{a.n} panels fired concurrently at {a.url}")
    print(f"  wall clock      {wall:.1f} s")
    print(f"  ok / failed     {len(lat)} / {len(errors)}")
    if lat:
        print(f"  per panel       p50 {statistics.median(lat):.2f} s, "
              f"mean {statistics.mean(lat):.2f} s, max {max(lat):.2f} s")
        print(f"  throughput      {len(lat) / wall * 60:.1f} images/min")
        with open(a.out, "wb") as f:
            f.write(first["bytes"])
        print(f"  wrote {a.out} ({len(first['bytes'])} bytes)")
    for e in errors[:5]:
        print(f"  error {e}")


if __name__ == "__main__":
    main()
