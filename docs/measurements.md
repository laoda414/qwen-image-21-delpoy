# Measurements: Qwen-Image-2.1-Turbo on one RTX PRO 6000 (96 GB)

Hardware as rented: **NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB,
driver 580.95.05, compute capability 12.0**, CUDA 12.9 toolkit, 256 vCPU, 1 TB RAM,
150 GB container disk. Vast base image (unprivileged container, supervisor, Caddy
auth edge).

Settings: 8 steps, CFG 1.0, euler, Turbo sigma schedule, seed per run, prompts
rotating over a fixed set of six comic-panel prompts. Both engines used the same
prompts and resolution. Raw JSON is in `results/`.

## Engine comparison, 1024x1024

| engine | config | s/image | images/min | GPU util | VRAM |
|---|---|---:|---:|---:|---:|
| sd.cpp (GGUF Q8_0 + Q8_0) | K=1 process | 4.30 | 13.7 | 77% | 16.2 GB |
| sd.cpp | K=2 processes | 8.04 | 15.3 | 97% | 32.4 GB |
| sd.cpp | K=6 processes | 21.75 | **15.7** | 98% | 96 GB (card full) |
| diffusers (bf16, SDPA) | batch 1 | 2.04 | **29.4** | — | 37.7 GB |
| diffusers | batch 2 | 3.52 | 17.1 | — | 43.9 GB |
| diffusers | batch 4 | 2.53 | 23.7 | — | 56.8 GB |
| diffusers | batch 8 | — | **OOM** | — | 90.8 GB |

Two conclusions, both against the initial plan:

1. **The engine switch is the win, not parallelism.** diffusers at batch 1 is
   2.1x faster per panel than a single sd.cpp worker, and 1.9x faster than
   sd.cpp's best-case 6-process configuration.
2. **Batching does not pay off at all here.** Batch 4 is 20% *slower* per image
   than batch 1, batch 2 is pathological (3.5 s/image), and batch 8 runs out of
   memory. Adding sd.cpp processes past K=2 buys under 3% throughput while
   quadrupling p50 latency (4.3 s -> 21.8 s).

## Why batching loses

The denoiser is **compute-bound at one image**. 8 steps in 2.04 s is 0.255 s per
step for a 7B DiT over ~4k tokens — the GPU is already saturated, so there is no
idle time for a batch to reclaim, and amortising weight reads (the usual reason
batching helps) does nothing. The batched attention path is *less* efficient per
image on this stack, which is why throughput falls.

Evidence it is saturation and not a bad measurement:

- sd.cpp's own queue probe: 2 requests sequential 7.8 s, 2 concurrent 7.5 s
  (ratio 1.93) — one server serialises, it does not overlap.
- Re-run with a fresh process per batch size, 3 timed runs: batch 1 = 2.04 s,
  batch 4 = 2.53 s. Reproducible, not allocator carry-over.
- Disabling the KV cache changed batch 4 by 0.01 images/min — the cache is not
  the cause.
- The text encoder is irrelevant: `encode_prompt` takes 0.036 s for 4-8 prompts,
  ~0.9% of a 2 s image.

One untested lever: `flash-attn` is not installed (diffusers rejected the `flash`
backend), so all of this ran on SDPA. A working flash attention kernel could
change the batched path, but it does not change the compute-bound argument at
batch 1, and prebuilt wheels for torch 2.11 / sm_120 are unlikely.

## Resolution curve (diffusers, batch 1)

| resolution | megapixels | s/image | images/min | VRAM |
|---|---:|---:|---:|---:|
| 832x1216 (2:3 panel) | 1.01 | 1.98 | 30.3 | 37.5 GB |
| 1024x1024 | 1.05 | 2.04 | 29.4 | 37.7 GB |
| 1344x768 (wide panel) | 1.03 | 2.03 | 29.6 | 37.6 GB |
| 1536x1536 | 2.36 | 5.49 | 10.9 | 46.1 GB |

Cost scales worse than linearly with pixels (2.25x pixels -> 2.7x time), i.e.
attention starts to bite above ~1 MP. Qwen's own recommended presets are ~4 MP
(1696x2528); expect roughly 10-12 s per panel there, so panel resolution is a
direct multiplier on throughput and should be chosen deliberately.

## Throughput in app terms

At panel resolution, single stream: a 12-panel comic is **~24 s** of GPU time,
20 panels ~40 s. That is the whole batch story: the queue is short enough that
batching would only have added latency.

| engine | 12 panels | 20 panels |
|---|---:|---:|
| sd.cpp K=1 | 52 s | 88 s |
| sd.cpp K=6 (queue) | 46 s + long tail | 76 s + long tail |
| diffusers batch 1 | **24 s** | **40 s** |

## What this means for the service

- **One process, one pipeline, one queue** — not K workers and not a batch
  scheduler. Concurrency here only adds queue depth and VRAM.
- **38 GB steady state at ~1 MP.** A 48 GB card is enough for a text-to-image
  stream, and the 96 GB card is only justified by resolution (1536² already took
  46.1 GB) or by the reference path below.
- **Qwen-Image-2.1 is one model for t2i, edit and multi-reference**, so character
  consistency does not need a second pipeline — it is the same pipeline with
  reference images attached. **Every number above is text-only**; the reference
  path adds VAE-encoded image tokens to the conditioning and will be slower and
  hungrier. That is the next measurement, because it is the path a comic actually
  uses (`service/serve.py` already accepts reference images).
- **Cost per panel** at $1.5-2.0/hr and 30 images/min is roughly $0.001, versus
  $0.04-0.07 per panel on the hosted providers in `config/models.json` — the
  break-even is a few hundred panels.
- Model load is ~15-20 s from page cache and ~2 s on a warm one; at 24 s per
  comic, keeping the box warm across batches is usually worth more than tearing
  it down.

## The service, measured end to end

`service/serve.py` was run on the box and called from this Windows machine over
the public port mapping (`http://154.59.156.16:54495`), i.e. the same path the
comic app would use. Single pipeline, FIFO queue, bearer token.

| check | result |
|---|---|
| 12 panels fired concurrently (832x1216) | **32.3 s wall, 22.3 images/min, 0 failures** |
| per-panel latency under that load | p50 18.4 s, max 32.3 s (queue position, not slow generation) |
| single request, 512x512, warm | 2.3 s round trip |
| auth | 401 with no token, 401 with a wrong token, 200 with the right one |
| model load + warmup at startup | ~18 s load, 2.7 s warmup |

Two things worth noting:

- **Throughput is 22.3 images/min here versus 30.3 in the raw benchmark**, i.e.
  ~0.7 s per panel of service overhead (PNG encode + base64 + HTTP). If that
  matters, returning JPEG or writing files and returning URLs would claw some of
  it back — PNG at 1.4 MB per panel is the bulk of it.
- **Queue order is the real UX cost, not throughput.** Firing 12 at once means
  panel 12 lands at 32 s while panel 1 lands at ~5 s. The app should keep
  concurrency low (2–3 in flight) and render panels as they arrive: same total
  time, much better perceived speed.



## The one workload still unmeasured: references

Every number above is **text-only**. Qwen-Image-2.1 is one model for
text-to-image, edit and multi-reference composition, so character consistency is
the same pipeline with `image=` attached — no second checkpoint, but references
are VAE-encoded and added to the conditioning, which lengthens attention and
raises VRAM. That is the path a comic with recurring characters actually runs, so
it should be measured before it is promised.

Both sides are already wired for it:

- `bench/bench_diffusers.py --reference a.png,b.png --reference-counts 1,3` times
  1 and 3 references against the same prompt, in the same JSON as the text-only
  baseline;
- `service/serve.py` accepts `image` or `images` (base64) and passes them to the
  same pipeline (list first, single-image fallback — the exact multi-reference
  kwarg is not confirmed on this diffusers build).

On a resumed instance (weights and venv intact):

```bash
/venv/main/bin/python /workspace/bench_diffusers.py \
  --model /workspace/models/qwen-image-2.1-turbo \
  --levels 1 --runs 2 --width 832 --height 1216 \
  --reference /workspace/bench/diffusers-b1-1024x1024.png,/workspace/bench/sample-k1-1024x1024.png \
  --reference-counts 1,3 --out /workspace/bench
```

Expected shape (to confirm, not assume): more references -> more per-panel time
and VRAM, with the same "one image already saturates the GPU" behaviour.

## The reference path, measured (same pipeline, character sheets attached)

Qwen-Image-2.1 is one model for t2i + edit + multi-reference, so a comic with
recurring characters runs the same endpoint with `image`/`images` attached. All
earlier numbers were text-only; these are not.

Measured through the live service over an SSH tunnel, 832x1216, 2 runs each,
latency includes transport:

| config | median | vs text-only | note |
|---|---:|---:|---|
| text-only | 5.2 s | 1.00x | ~2.5 s of that is tunnel transfer |
| 1 reference | 7.5 s | **1.43x** | +~2.3 s |
| 3 references | 9.1 s | **1.74x** | +~1.3 s per extra reference |

- The list form is really used: a 3-reference request returned a **composite
  strip** containing characters from all three inputs, so `image=[...]` is
  accepted (not silently collapsed to one). Max 4 enforced by the service.
- Cost scales with reference count, so the app should send the **minimum** sheets
  a panel needs (1 for a solo panel, 2-3 only when characters share a frame).
- Prompt discipline matters: with multiple references the model will happily draw
  a multi-panel collage when asked for "a comic panel". Frame the prompt for a
  single panel if one image per panel is wanted.

## Second host: Max-Q is slower, and the tunnel costs something

Same deploy (repo + stock CUDA image, clean volume), different card:

| | RTX PRO 6000 **Server** Edition | RTX PRO 6000 **Max-Q** Workstation |
|---|---|---|
| 12-panel burst | 32.3 s, 22.3 img/min, p50 18.4 s | 39.9 s, **18.1 img/min**, p50 22.5 s |
| single panel, end to end | 5.1 s (preflight) | 5.1 s (preflight) |
| generation implied | ~2.0 s | **~2.6 s** |
| clean deploy to serving | ~3 min | ~2 min |

The Max-Q variant is power-limited, and the effect is visible: ~30% slower per
panel at the same price tier. Worth checking which variant an offer is before
renting — the name says "RTX PRO 6000" either way, and `nvidia-smi` reports
`Max-Q Workstation Edition` vs `Server Edition`.

Tunnel overhead is real but bounded: at ~0.9 MB/s, a 1.1 MB PNG costs ~2.5 s per
panel on top of generation. That is 50% of a 5 s panel, which is why JPEG output
(~7x smaller) is the single highest-value change left for remote use.

## Reproducing

```bash
# sd.cpp lane (K processes)
python3 bench/bench.py sweep --levels 1,2,4 --per-worker 3 --width 1024 --height 1024

# diffusers lane (true batch)
/venv/main/bin/python bench/bench_diffusers.py --levels 1,2,4 --runs 3 \
  --width 1024 --height 1024 --attention flash
```

Gotchas found on the box, now fixed in the scripts: ggml is a submodule
(a plain `--depth 1` clone leaves it empty); `torchvision` is pulled in by
Qwen3-VL's video processor even for text-to-image; `processor/chat_template.jinja`
must be downloaded (a `*.json`-only allow-list skips it); ML packages live in
`/venv/main`, not `/usr/bin/python3`; and `Thread._stop` must not be shadowed by
an `Event` attribute of the same name.
