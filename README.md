# Qwen-Image on a rented GPU (comic-studio image backend)

Goal: stop paying per image for panels, run Qwen-Image-2.1-Turbo on a rented
Vast.ai GPU, and settle **which engine** to build the container around — with
measurements from one rental instead of estimates.

The comic app already talks to a provider seam (`ImageProvider` in
`apps/server/src/providers/types.ts`), so what is needed is a service that
satisfies that seam; the app's shape does not change either way.

## The engine question this repo answers

`sd.cpp` has no batch dimension: it can take several images in one run
(`--batch-count`) but denoises them one at a time, so parallelism only comes from
running K processes, each holding its own ~17 GB of weights and re-running the 8B
text encoder per panel. `diffusers` (`QwenImage21Pipeline`) takes a *list* of
prompts and runs one forward with a batch dim, plus `torch.compile`, flash
attention and KV cache.

Batching does not buy 4× — a diffusion step is already a large GEMM, so batch 4
lands somewhere around 1.7–2.8×. But it buys that without duplicating weights,
and it makes the text encoder cost amortise. So the two lanes get measured on the
same box, same prompts, same resolution, and the container follows the winner.

## What is already known

| fact | value |
|---|---|
| sd.cpp build | ≥ commit `c150a6b` (custom sigma lists); Qwen-Image-2.1 support since 2026-09-20 |
| GPU arch | RTX PRO 6000 Blackwell, **sm_120** → CUDA ≥ 12.8 to compile |
| sd.cpp weights | denoiser Q8_0 7.59 GB / AD-Q4_K 4.20 GB; uncensored encoder Q8_0 8.71 GB; VAE 0.68 GB |
| diffusers weights | bf16 checkpoint ≈32 GB (7B DiT + Qwen3-VL-8B encoder + VAE) |
| sampling | 8 steps, CFG 1.0, euler; Turbo's sigma schedule is baked into the checkpoint (diffusers) / `--sigmas qwen-image-2.1-turbo` (sd.cpp) |
| per sd.cpp worker | ≈17 GB at Q8/Q8, ≈13 GB at Q4/Q8 → 4–6 workers in 96 GB |
| Vast ports | container ports map to **random** public ports; read the mapping, never hardcode |
| build host | needs root + a Docker daemon: RunPod CPU pod or your own VPS. Vast has nested Docker only on its VMs, and no CPU-only rentals |

On "uncensored": the `AtomicChat/...-Uncensored-GGUF` repo is only the **encoder**
with the refusal direction ablated, and its card reports **0 of 45** prompts
flipping an image outcome versus the stock encoder. The denoiser — the part that
draws — is stock and unfiltered either way. So the diffusers lane is not a
content downgrade; it just uses the stock encoder.

## Phase 1 — one rental, both engines

No custom image: a stock CUDA image, a self-contained on-start script, one box,
then destroy. The compile is ~15 min and is the same step the Dockerfile needs.

### Rent it

| setting | value |
|---|---|
| GPU | **RTX PRO 6000**, 96 GB (not "RTX 6000 Ada" — 48 GB, sm_89) |
| rental type | on-demand |
| image | `vastai/base-image:cuda-12.9.2-auto` |
| disk | **150 GB** (21 GB GGUF + 32 GB bf16 + build + outputs) |
| ports | `-p 1234:1234` (for the serving phase) |
| on-start | paste `bench/onstart-bench.sh` |

If the console's on-start box objects to a 50 KB paste, scp it instead:

```powershell
scp -P <port> bench/onstart-bench.sh root@<host>:/workspace/
ssh -p <port> root@<host> "bash /workspace/onstart-bench.sh"
```

### What it does, unattended

1. **stage 1 (sd.cpp):** deps → `nvcc` ≥12.8 check → clone + build with
   `-DCMAKE_CUDA_ARCHITECTURES=120` → GGUF weights (Q8_0 encoder, VAE, Q8_0 and
   AD-Q4_K denoisers) → `discover` → sweep over K = 1…6 processes (twice, once
   per denoiser rung).
2. **stage 2 (diffusers):** torch cu128 → diffusers from `main`
   (QwenImage21Pipeline post-dates the last release) → 32 GB bf16 checkpoint →
   sweep over batch = 1, 2, 4, 8, then a `torch.compile` run at batch 4.
3. **comparison:** both lanes on one table, plus `latest-*.json` per run and a
   sample PNG per config.

Each stage runs as a **child process**, so a failed sd.cpp build cannot take the
diffusers lane down. ~60–90 min total, idempotent.

### Send back

- the two sweep tables and the `COMPARISON` block from the log;
- `/workspace/bench/latest-*.json` if you can scp them;
- one sample PNG per rung (Q4 vs Q8, and a diffusers one) for a quality check.

### Reading the numbers

- **engine**: diffusers wins on img/min? by how much at batch 4/8 vs sd.cpp at
  K=4/6? If the gap is under ~1.3×, prefer sd.cpp — a C++ binary with no Python
  stack is a materially simpler container.
- **K\*** / **batch\***: largest config with scaling efficiency ≳0.8 → the
  container's concurrency and the app's limit.
- **VRAM**: diffusers needs ~33 GB steady (so batch headroom is large); sd.cpp
  scales VRAM with K (so the ceiling arrives sooner).
- `startup_s` / model load: decides idle-vs-teardown between batches.
- `encode_prompt` probe: how much of the batching win is the text encoder.

## Deploying: repo + stock CUDA image

This is the deployment mechanism, chosen after establishing that **no reachable
host can build a container image**: Vast has no CPU-only rentals, the RunPod CPU
pod has no runtime and `unshare -Ur` returns EPERM, and a nested daemon needs
`CAP_SYS_ADMIN` that unprivileged pods don't get. A repo + a stock CUDA image
needs none of that, and it reproduces exactly what was measured on the RTX PRO
6000 in ~10 minutes.

Rent any box with CUDA ≥ 12.8 and ≥ 100 GB disk, and give it one line:

```bash
# Vast on-start field, RunPod start command, or ssh:
curl -fsSL https://raw.githubusercontent.com/OWNER/REPO/main/onstart.sh | bash
```

`onstart.sh` clones the repo if needed, runs `bootstrap.sh`, then serves:

- `bootstrap.sh` — picks an interpreter (`/venv/main`, `/opt/venv`, or a fresh
  `/workspace/.venv`), installs torch cu128 + torchvision + diffusers-from-main,
  downloads the 32 GB bf16 checkpoint into `$WORK/models`, and drops
  `$WORK/.bootstrap-ok` with the versions it settled on. Idempotent, so it is also
  the repair script, and a second run on a warm volume goes straight to serving.
- `serve.sh` — starts `service/serve.py` on `$SERVICE_PORT` (default 8080) with
  `$SERVICE_TOKEN` as the bearer token.
- `WITH_SDCPP=1 bootstrap.sh` additionally compiles the sd.cpp fallback lane and
  fetches its GGUFs, for boxes where 16 GB of VRAM is the constraint.

What it costs, from measurement: ~10 min cold (of which the weight fetch is the
bulk), ~40 s warm. At ~$1.7/hr that is ~$0.28 per fresh rental — the honest price
of not having an image.

### Why not a container (yet)

A `Dockerfile` is included and mirrors the same steps, but it is an optimisation,
not a prerequisite: it would save ~9 minutes per rental and needs a build host we
still don't have. Deliberately, it does **not** bake the weights — that would make
it ~37 GB instead of ~5-6 GB, which is what makes images awkward to build, push
and pull. If re-rentals become frequent, build it on a machine with a real daemon
(your VPS or a Vast VM) and bake the weights at that point, when the registry
storage is justified.

### Service shape

- one FastAPI process holding `QwenImage21Pipeline` (bf16) and a FIFO queue with a
  single worker thread — concurrency buys nothing because the DiT saturates the
  GPU at batch 1;
- a **bearer token** in front (this generates adult content and must not be a
  naked public port); on Vast the Caddy edge can do TLS + token for you if the app
  registers as a portal service;
- `/v1/images/generations`-compatible request shape, including optional
  `image`/`images` references, so the app provider is unchanged;
- steady-state VRAM ~38 GB at ~1 MP; the 96 GB card is not needed for a single
  text-to-image stream (48 GB is enough), it is needed if panels go to Qwen's
  ~4 MP presets, which is a measured tradeoff rather than a guess;
- **no second model.** Qwen-Image-2.1 is text-to-image *and* edit *and*
  multi-reference in one `QwenImage21Pipeline`: character consistency comes from
  passing reference images to the same pipeline, not from a separate Edit
  checkpoint. That path costs more per panel (extra encoded tokens) and is the
  one workload in this project that has **not** been measured yet.

## Phase 3 — app integration (changing IP is a lookup)

`vastctl.py endpoint` is the reference resolver: query the instance by label/id,
read `ports["8080/tcp"][0]`, rebuild the base URL, re-resolve on connect failure.
No IP ever lands in `.env`, so a re-rent is invisible to the comic pipeline. Then
a `QwenImageProvider implements ImageProvider` maps `ImageGenRequest` → requests
and returns the same `ImageGenResult` Venice returns today.

## Runbook — resuming a box, or rebuilding from scratch

Two very different costs, depending on whether the instance was **stopped** or
**destroyed**.

### Stopped instance (container disk preserved)

`/workspace` keeps the built binaries, both weight sets, the HF cache and the
installed venv, so nothing needs downloading or compiling:

```bash
ssh -p <port> root@<host>
bash /workspace/start_serve.sh          # service back in ~30 s
# or, to re-measure:
bash /workspace/run_sdcpp.sh            # sd.cpp lane   (~5 min)
bash /workspace/run_diff4.sh            # diffusers lane (~6 min)
```

Caveat: the **public port mapping changes on restart**. Re-read it from the
console, or `vast-capabilities | jq '.instance.open_ports'` on the box, before
handing a URL to the app. And `start_serve.sh` defaults to a throwaway token —
set a real `SERVICE_TOKEN` when the endpoint is reachable by anything but you.

### Destroyed instance (nothing preserved)

One paste into the on-start field, ~10 min to a live GPU box:

| setting | value |
|---|---|
| GPU | RTX PRO 6000, 96 GB (48 GB is enough for the service alone) |
| image | `vastai/base-image:cuda-12.9.2-auto` |
| disk | 150 GB |
| ports | `-p 10100:10100` for the service, `-p 1234:1234` if re-running sd.cpp |
| on-start | `bench/onstart-bench.sh` |

That rebuilds sd.cpp, fetches both weight sets, installs the diffusers stack and
sweeps both lanes. To go straight to serving instead, skip the bundle: install
`bench/`'s `diffusers_stage.sh`, copy `service/serve.py`, and run it — the
`QwenImage21Pipeline` path only needs the bf16 checkpoint, not the GGUFs.

### State of the art after the first rental

Measured and settled: engine (diffusers), concurrency (1 in flight), batching (a
dead end), resolution curve, service overhead, and the auth path. Still open: the
container image itself (build on a CPU VM, per Phase 2) and the app-side
`QwenImageProvider` plus the 2–3-in-flight rendering change.

## Files

| path | role |
|---|---|
| `bench/onstart.sh` | sd.cpp bootstrap: build for sm_120 + GGUF weights |
| `bench/bench.py` | sd.cpp discovery + K-process concurrency sweep |
| `bench/diffusers_stage.sh` | diffusers bootstrap: torch cu128, diffusers main, bf16 weights |
| `bench/bench_diffusers.py` | batch sweep (true batching), encode-cost probe, optional compile |
| `bench/compare.py` | both lanes on one table |
| `bench/onstart-bench.sh` | **the thing you paste into Vast** (generated) |
| `bench/sync_bundle.py` | regenerates the bundle from the five sources |
| `bench/verify_bundle.py` | static checks (embedded Python compiles, heredocs, shell balance) |
| `service/serve.py` | the single-stream image service (OpenAI-shaped, token auth, FIFO queue) |
| `service/test_batch.py` | fires N panels at the live service and reports latency/throughput |
| `docs/measurements.md` | **what the rented box actually did** |
| `results/` | raw JSON for every run, named by engine/resolution/config |
| `vast/vastctl.py` | offers / rent / ls / wait / **endpoint** / ssh / destroy |

## Status

**Measured** on a real RTX PRO 6000 (96 GB) — full tables in `docs/measurements.md`,
raw JSON in `results/`:

- **diffusers, batch 1: 2.04 s/panel (29.4 images/min) at 1024², 1.98 s at
  832×1216, 37.7 GB VRAM.** This is the configuration to build on.
- **sd.cpp: 13.7 images/min at K=1, 15.7 at K=6** (card full) with p50 latency
  rising 4.3 s → 21.8 s. 1.9–2.1× slower than diffusers, and the GPU saturates at
  one image in both engines.
- **Batching is a dead end here**: batch 4 is 20% slower than batch 1, batch 2 is
  pathological, batch 8 OOMs at 90.8 GB. The DiT is compute-bound, so there is no
  idle GPU time to reclaim; text encoding is 0.9% of an image.
- **Resolution is the real throughput dial**: 1536² costs 5.49 s/panel (2.7× for
  2.25× pixels).
- A 12-panel comic is ~24 s of GPU at panel resolution.

Still open: flash attention (`flash-attn` was absent, everything ran on SDPA, and
it is the one thing that could change the batched path), and the container build
itself.
