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
