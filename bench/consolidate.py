import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "..", "..", "scratch", "results")
DST = os.path.join(ROOT, "results")
os.makedirs(DST, exist_ok=True)

rows = []
for path in sorted(glob.glob(os.path.join(SRC, "*.json"))):
    try:
        d = json.load(open(path))
    except Exception:
        continue
    if "levels" not in d:
        continue
    m = d.get("meta", {})
    engine = m.get("engine", "sd.cpp")
    w, h = m.get("width"), m.get("height")
    levels = d.get("levels", [])
    cfgs = [e.get("workers", e.get("batch")) for e in levels]
    rate = [e.get("img_per_min") for e in levels]

    if engine == "sd.cpp":
        name = f"sdcpp-{w}x{h}-k{min(cfgs)}-k{max(cfgs)}.json"
        label = "sd.cpp"
    else:
        kv = "kv" if m.get("kv_cache", True) else "nokv"
        comp = "-compiled" if m.get("compiled") else ""
        name = f"diffusers-{w}x{h}-b{'-'.join(str(c) for c in cfgs)}{comp}-{kv}.json"
        label = "diffusers"
    shutil.copyfile(path, os.path.join(DST, name))
    rows.append((label, f"{w}x{h}", cfgs, rate, name))

print(f"{'engine':<10} {'res':<10} {'config':<18} {'img/min':<26} file")
for label, res, cfgs, rate, name in rows:
    cfg_s = ",".join(str(c) for c in cfgs)
    rate_s = ",".join(str(r) for r in rate)
    print(f"{label:<10} {res:<10} {cfg_s:<18} {rate_s:<26} {name}")
