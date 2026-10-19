#!/usr/bin/env python3
"""ComfyUI backend that speaks the same API contract as serve.py.

Why this exists: the comic app talks to an `ImageProvider` that speaks
`/v1/images/generations`. ComfyUI speaks its own graph API. Rather than teach the
app a second protocol (and a second provider), this bridges the contract onto
ComfyUI, so switching image engines is a base-URL change.

  COMFY_URL=http://127.0.0.1:18188 \
  SERVICE_TOKEN=... SERVICE_PORT=18700 python comfy_bridge.py

Model/sampler defaults come from the box's own inventory and were verified by
submitting real graphs; override with env vars or per request.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

TOKEN = os.environ.get("SERVICE_TOKEN", "")
COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:18188").rstrip("/")

UNET = os.environ.get("COMFY_UNET", "NoctQ_V4_int8_convrot.safetensors")
CLIP = os.environ.get("COMFY_CLIP", "qwen3vl_8b_int8_convrot.safetensors")
VAE = os.environ.get("COMFY_VAE", "qwen_image_2.1_vae_bf16.safetensors")
STEPS = int(os.environ.get("COMFY_STEPS", "25"))
CFG = float(os.environ.get("COMFY_CFG", "1.0"))
SAMPLER = os.environ.get("COMFY_SAMPLER", "euler")
SCHEDULER = os.environ.get("COMFY_SCHEDULER", "simple")
# "blueprint" mirrors ComfyUI's official "Text to Image (Qwen-Image)" blueprint:
# EmptySD3LatentImage + ModelSamplingAuraFlow(shift) + two CLIPTextEncode nodes.
# "bridge" is the simpler hand-built shape (TextEncodeQwenImage21 +
# EmptyLatentImage). A/B'd on one prompt and seed: both yield colour comic art,
# with different trajectories and framing; neither is broken.
VARIANT = os.environ.get("COMFY_VARIANT", "blueprint")
SHIFT = float(os.environ.get("COMFY_SHIFT", "3.1"))
MODEL_ID = os.environ.get("COMFY_MODEL_ID", "qwen-image-2.1-turbo")

# ComfyUI is a single GPU pipeline; keep one graph in flight so queue depth is ours
# to report rather than ComfyUI's opaque queue.
LOCK = threading.Lock()
STATE = {"served": 0, "last_error": None, "comfy": False}


def call(path: str, payload=None, timeout: float = 60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(COMFY + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def build_graph(prompt: str, negative: str, width: int, height: int,
                seed: int, steps: int, cfg: float, sampler: str, scheduler: str,
                prefix: str):
    """Qwen-Image text-to-image graph in one of two shapes (COMFY_VARIANT).

    Common: native UNETLoader (int8 checkpoint, not GGUF), qwen_image CLIP type,
    the Qwen 2.1 VAE, plain KSampler, SaveImage.

    blueprint (default) — mirrors ComfyUI's official "Text to Image (Qwen-Image)"
    blueprint: EmptySD3LatentImage (16-channel latents), ModelSamplingAuraFlow with
    shift 3.1 (the flow-matching shift the official graph sets explicitly), and two
    CLIPTextEncode nodes.

    bridge — the simpler shape: TextEncodeQwenImage21, which emits both
    conditionings and carries the trained resolution, plus EmptyLatentImage.
    """
    graph = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }

    if VARIANT == "bridge":
        graph["4"] = {"class_type": "TextEncodeQwenImage21",
                      "inputs": {"clip": ["2", 0], "prompt": prompt,
                                 "negative_prompt": negative,
                                 "resolution": max(width, height)}}
        graph["5"] = {"class_type": "EmptyLatentImage",
                      "inputs": {"width": width, "height": height, "batch_size": 1}}
        graph["6"] = {"class_type": "KSampler",
                      "inputs": {"model": ["1", 0], "seed": seed, "steps": steps,
                                 "cfg": cfg, "sampler_name": sampler,
                                 "scheduler": scheduler, "positive": ["4", 0],
                                 "negative": ["4", 1], "latent_image": ["5", 0],
                                 "denoise": 1.0}}
    else:
        graph["4"] = {"class_type": "CLIPTextEncode",
                      "inputs": {"clip": ["2", 0], "text": prompt}}
        graph["9"] = {"class_type": "CLIPTextEncode",
                      "inputs": {"clip": ["2", 0], "text": negative}}
        graph["5"] = {"class_type": "EmptySD3LatentImage",
                      "inputs": {"width": width, "height": height, "batch_size": 1}}
        graph["10"] = {"class_type": "ModelSamplingAuraFlow",
                       "inputs": {"model": ["1", 0], "shift": SHIFT}}
        graph["6"] = {"class_type": "KSampler",
                      "inputs": {"model": ["10", 0], "seed": seed, "steps": steps,
                                 "cfg": cfg, "sampler_name": sampler,
                                 "scheduler": scheduler, "positive": ["4", 0],
                                 "negative": ["9", 0], "latent_image": ["5", 0],
                                 "denoise": 1.0}}
    return graph


def generate_one(prompt, negative, width, height, seed, steps, cfg, sampler,
                 scheduler, timeout=900) -> bytes:
    graph = build_graph(prompt, negative, width, height, seed, steps, cfg,
                        sampler, scheduler, f"bridge-{int(time.time())}")
    try:
        pid = call("/prompt", {"prompt": graph}, timeout=60)["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI rejected the graph: {e.read().decode()[:400]}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            hist = call(f"/history/{pid}", timeout=30)
        except Exception:
            continue
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI graph failed: {str(status)[:400]}")
        for out in entry.get("outputs", {}).values():
            for img in out.get("images", []):
                q = urllib.parse.urlencode({"filename": img["filename"],
                                            "subfolder": img.get("subfolder", ""),
                                            "type": "output"})
                return urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=120).read()
        raise RuntimeError("graph finished without an image output")
    raise RuntimeError(f"timed out after {timeout}s waiting for ComfyUI")


def auth(authorization: str | None) -> None:
    if not TOKEN:
        return
    got = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(got, TOKEN):
        raise HTTPException(status_code=401, detail="bad or missing bearer token")


def parse_size(body: dict):
    size = body.get("size")
    if isinstance(size, str) and "x" in size:
        w, h = size.lower().split("x")[:2]
        return int(w), int(h)
    return int(body.get("width") or 832), int(body.get("height") or 1216)


def build_app() -> FastAPI:
    app = FastAPI(title="comfy-image-bridge")

    @app.get("/healthz")
    def healthz():
        try:
            stats = call("/system_stats", timeout=10)
            queue = call("/queue", timeout=10)
            STATE["comfy"] = True
            dev = stats.get("devices", [{}])[0]
            return {"status": "ok", "engine": "comfyui",
                    "comfyui": stats.get("system", {}).get("comfyui_version"),
                    "device": dev.get("name"),
                    "vram_free_gb": round(dev.get("vram_free", 0) / 2**30, 1),
                    "queue_running": len(queue.get("queue_running", [])),
                    "queue_pending": len(queue.get("queue_pending", [])),
                    "engine_model": UNET, "variant": VARIANT, "shift": SHIFT, "steps": STEPS, "cfg": CFG,
                    "sampler": SAMPLER, "scheduler": SCHEDULER,
                    "references": "ignored",
                    "served": STATE["served"],
                    "last_error": STATE["last_error"]}
        except Exception as e:
            STATE["comfy"] = False
            return {"status": "down", "engine": "comfyui", "error": str(e)[:200]}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/images/generations")
    def images(body: dict, authorization: str | None = Header(default=None)):
        auth(authorization)
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        width, height = parse_size(body)
        width, height = max(256, width // 8 * 8), max(256, height // 8 * 8)
        n = max(1, min(4, int(body.get("n") or 1)))
        negative = str(body.get("negative_prompt") or body.get("negativePrompt") or "")
        steps = int(body.get("steps") or STEPS)
        cfg = float(body.get("cfg") or body.get("cfg_scale") or CFG)
        sampler = str(body.get("sampler") or SAMPLER)
        scheduler = str(body.get("scheduler") or SCHEDULER)
        seed = body.get("seed")

        # Reference images need a different graph topology (image encode + reference
        # conditioning). Rejecting the whole request would break every panel the app
        # renders with sheets, so the references are ignored *visibly*: the flag in
        # the response and the warning header are what stops this being a silent
        # identity regression.
        refs_ignored = bool(body.get("images") or body.get("image"))
        if refs_ignored:
            STATE["last_error"] = "references ignored (not implemented in the ComfyUI bridge)"

        out = []
        try:
            for i in range(n):
                s = int(seed) + i if seed is not None else random.randint(1, 2**31)
                with LOCK:
                    png = generate_one(prompt, negative, width, height, s, steps,
                                       cfg, sampler, scheduler)
                out.append({"b64_json": base64.b64encode(png).decode(),
                            "revised_prompt": prompt})
                STATE["served"] += 1
        except Exception as e:
            STATE["last_error"] = str(e)[:400]
            raise HTTPException(status_code=500, detail=STATE["last_error"])

        payload = {"created": int(time.time()), "model": MODEL_ID, "data": out,
                   "settings": {"steps": steps, "cfg": cfg, "sampler": sampler,
                                "scheduler": scheduler}}
        headers = {}
        if refs_ignored:
            payload["references_ignored"] = True
            headers["X-References-Ignored"] = "true"
        return JSONResponse(payload, headers=headers)

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("SERVICE_PORT", "18700")))
    a = p.parse_args()
    try:
        s = call("/system_stats", timeout=15)
        print(f"comfyui {s.get('system', {}).get('comfyui_version')} at {COMFY}", flush=True)
    except Exception as e:
        print(f"WARNING: ComfyUI not reachable at {COMFY}: {e}", flush=True)
    print(f"bridge on {a.host}:{a.port} | model={UNET} steps={STEPS} cfg={CFG} "
          f"{SAMPLER}/{SCHEDULER} | token {'set' if TOKEN else 'NOT set'}", flush=True)
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
