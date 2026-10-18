#!/usr/bin/env python3
"""Is this box actually usable from *here*? One panel, end to end.

Bytes/second is the wrong metric on its own: what matters is wall time for a real
panel, because that includes generation plus transport. We hit a host where both
HTTP and SSH managed ~10 KB/s while the GPU generated in 2 s, so this gate exists
to catch that before a deploy is trusted.

  python tools/preflight.py --url http://127.0.0.1:18080 --token ... --size 832x1216
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import requests

PROMPT = "comic panel, a barista holding a very long receipt, flat colors, ink outlines"

# verdicts, in seconds for one panel including transport
GOOD = 15.0
USABLE = 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--size", default="832x1216")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()

    h = {"Authorization": f"Bearer {a.token}", "Content-Type": "application/json"}

    t0 = time.time()
    try:
        health = requests.get(f"{a.url}/healthz", timeout=30).json()
    except Exception as e:
        print(f"FAIL: cannot reach {a.url}: {e}")
        return 2
    print(f"health: {health}  ({time.time()-t0:.1f}s)")

    lat, sizes = [], []
    for i in range(a.runs):
        t0 = time.time()
        try:
            r = requests.post(f"{a.url}/v1/images/generations", headers=h, timeout=a.timeout,
                              json={"prompt": f"{PROMPT} #{i}", "size": a.size,
                                    "n": 1, "seed": 900 + i})
        except Exception as e:
            print(f"FAIL: request {i} errored after {time.time()-t0:.1f}s: {e}")
            return 2
        dt = time.time() - t0
        if r.status_code != 200:
            print(f"FAIL: HTTP {r.status_code} {r.text[:200]}")
            return 2
        b64 = r.json()["data"][0]["b64_json"]
        lat.append(dt)
        sizes.append(len(b64) * 3 / 4)

    med = statistics.median(lat)
    mb = statistics.median(sizes) / 1e6
    rate = mb / med
    print(f"panel: {med:.1f}s median over {len(lat)} runs, {mb:.2f} MB payload "
          f"-> {rate*1024:.0f} KB/s effective")
    print(f"throughput: {60/med:.1f} img/min end to end")

    if med <= GOOD:
        print("PASS: usable interactively")
        return 0
    if med <= USABLE:
        print("WARN: works but slow; a 12-panel comic would take "
              f"{med*12/60:.0f} min")
        return 1
    print(f"FAIL: unusable from this machine ({med:.0f}s per panel). The GPU is fine "
          "(check /healthz served count) — the path to this host is not.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
