#!/usr/bin/env python3
"""Qwen-Image-2.1-Turbo image service (single stream).

Shape is dictated by the measurements in docs/measurements.md: the denoiser is
compute-bound at one image (2.0 s at 1024², 8 steps), so parallelism and batching
only add queue latency. One pipeline, one worker thread, a FIFO queue in front.

Endpoints
  GET  /healthz                     queue depth, model, device
  GET  /v1/models                   OpenAI-shaped model list
  POST /v1/images/generations       {prompt, n?, size?|width/height, seed?}
                                    -> {created, data:[{b64_json}], model}

Auth: if SERVICE_TOKEN is set, every /v1 call needs `Authorization: Bearer <token>`.

  SERVICE_TOKEN=... python serve.py --model /workspace/models/qwen-image-2.1-turbo --port 51553
"""
from __future__ import annotations

import argparse
import base64
import hmac
import io
import os
import queue
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

QUEUE: "queue.Queue[Job]" = queue.Queue()
STATE = {"ready": False, "model": None, "device": None, "served": 0,
         "inflight": 0, "last_error": None}
TOKEN = os.environ.get("SERVICE_TOKEN", "")


class Job:
    __slots__ = ("prompt", "width", "height", "seed", "n", "refs", "event",
                 "images", "error")

    def __init__(self, prompt, width, height, seed, n, refs=()):
        self.prompt, self.width, self.height = prompt, width, height
        self.seed, self.n = seed, n
        self.refs = list(refs)
        self.event = threading.Event()
        self.images: list[bytes] = []
        self.error: str | None = None


def generate(pipe, job: Job) -> None:
    """One image at a time, deliberately. Batch > 1 measured slower.

    Qwen-Image-2.1 does text-to-image, editing and multi-reference in the same
    pipeline, so a character sheet is just `image=` on the same call — no second
    model, but more conditioning tokens, hence more time than text-only.
    """
    gen = torch.Generator("cpu").manual_seed(
        job.seed if job.seed is not None else int(time.time() * 1000) % 2**31)
    base = dict(prompt=job.prompt, width=job.width, height=job.height, generator=gen)
    attempts = []
    if job.refs:
        attempts.append({**base, "use_kv_cache": True,
                         "image": job.refs if len(job.refs) > 1 else job.refs[0]})
        attempts.append({**base, "image": job.refs[0]})
    attempts.append({**base, "use_kv_cache": True})
    attempts.append(base)

    last = None
    for kw in attempts:
        try:
            out = pipe(**kw)
            break
        except (TypeError, ValueError) as e:
            last = e
    else:
        raise RuntimeError(f"pipeline call failed: {last}")
    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    job.images.append(buf.getvalue())


def worker(pipe):
    while True:
        job = QUEUE.get()
        try:
            for _ in range(job.n):
                generate(pipe, job)
            STATE["served"] += job.n
        except Exception as e:  # never let one bad prompt kill the service
            job.error = f"{type(e).__name__}: {e}"
            STATE["last_error"] = job.error
        finally:
            STATE["inflight"] -= 1
            job.event.set()
            QUEUE.task_done()


def auth(authorization: str | None) -> None:
    if not TOKEN:
        return
    got = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(got, TOKEN):
        raise HTTPException(status_code=401, detail="bad or missing bearer token")


def decode_images(body: dict):
    """Accept `image` (one base64) or `images` (a list) as character references.

    Same pipeline handles multi-reference composition, so this is the identity-
    consistency hook: pass a character sheet and every panel keeps the face.
    """
    raw = body.get("images")
    if raw is None:
        single = body.get("image")
        raw = [single] if single else []
    if not isinstance(raw, list):
        raw = [raw]
    if len(raw) > 4:
        raise HTTPException(status_code=400, detail="at most 4 reference images")
    out = []
    for item in raw:
        if not isinstance(item, str) or not item:
            continue
        data = item.split(",", 1)[-1] if item.startswith("data:") else item
        try:
            img = Image.open(io.BytesIO(base64.b64decode(data)))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"bad reference image: {e}")
        out.append(img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB"))
    return out


def parse_size(body: dict):
    size = body.get("size")
    if isinstance(size, str) and "x" in size:
        w, h = size.lower().split("x")[:2]
        return int(w), int(h)
    w = int(body.get("width") or 832)
    h = int(body.get("height") or 1216)
    return w, h


def build_app() -> FastAPI:
    app = FastAPI(title="qwen-image")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok" if STATE["ready"] else "loading",
                "model": STATE["model"], "device": STATE["device"],
                "queue_depth": QUEUE.qsize(), "inflight": STATE["inflight"],
                "served": STATE["served"], "last_error": STATE["last_error"]}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": "qwen-image-2.1-turbo", "object": "model",
                          "owned_by": "local"}]}

    @app.post("/v1/images/generations")
    def images(body: dict, authorization: str | None = Header(default=None)):
        auth(authorization)
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")
        width, height = parse_size(body)
        width, height = max(256, width // 32 * 32), max(256, height // 32 * 32)
        n = max(1, min(8, int(body.get("n") or 1)))
        seed = body.get("seed")
        refs = decode_images(body)
        job = Job(prompt, width, height, int(seed) if seed is not None else None, n, refs)

        if not STATE["ready"]:
            raise HTTPException(status_code=503, detail="model still loading")
        STATE["inflight"] += 1
        QUEUE.put(job)
        # a full comic batch can sit behind others: 8 panels * 2 s is 16 s
        if not job.event.wait(timeout=600):
            raise HTTPException(status_code=504, detail="timed out waiting for the queue")
        if job.error:
            raise HTTPException(status_code=500, detail=job.error)

        return JSONResponse({
            "created": int(time.time()),
            "model": "qwen-image-2.1-turbo",
            "data": [{"b64_json": base64.b64encode(b).decode(), "revised_prompt": prompt}
                     for b in job.images],
        })

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/models/qwen-image-2.1-turbo")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=51553)
    p.add_argument("--attention", default="flash")
    a = p.parse_args()

    from diffusers import QwenImage21Pipeline
    print(f"loading {a.model} ...", flush=True)
    t0 = time.time()
    pipe, last = None, None
    for kw in ("dtype", "torch_dtype"):
        try:
            pipe = QwenImage21Pipeline.from_pretrained(a.model, **{kw: torch.bfloat16})
            break
        except TypeError as e:
            last = e
    if pipe is None:
        raise SystemExit(f"could not load pipeline: {last}")
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    # warm the kernels so the first real request is not 10x slower
    t0 = time.time()
    pipe(prompt="warmup", width=832, height=1216, generator=torch.Generator("cpu").manual_seed(1),
         use_kv_cache=True)
    print(f"warmup in {time.time() - t0:.1f}s", flush=True)

    STATE.update(ready=True, model=a.model, device=str(pipe.device))
    threading.Thread(target=worker, args=(pipe,), daemon=True).start()
    print(f"listening on {a.host}:{a.port} (token {'set' if TOKEN else 'NOT set'})", flush=True)
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
