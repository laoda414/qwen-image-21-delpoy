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
