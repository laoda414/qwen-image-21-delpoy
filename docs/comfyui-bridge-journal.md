# Journal — ComfyUI bridge

A running engineering log for `service/comfy_bridge.py`: what it is, why it looks
the way it does, what was measured, and what is still wrong. Newest entry last;
append rather than rewrite, so the reasoning history survives.

Related docs: `api-contract.md` (the contract it implements), `comfyui-backend.md`
(how to run it), `runbook.md` (start/tunnel/troubleshoot).

---

## What this thing is

A ~300-line FastAPI service that makes ComfyUI look like the `/v1/images/generations`
endpoint the comic app already speaks. One process, one ComfyUI behind it, no model
code of its own.

```
app → 127.0.0.1:18080 (tunnel) → 127.0.0.1:18700 (bridge) → 127.0.0.1:18188 (ComfyUI)
```

**Why not a new app provider?** The app's seam is `ImageProvider`; a second protocol
means a second provider, a second set of tests, and a UI that has to know which
engine it is on. Bridging the contract keeps the app untouched (base URL + token),
and keeps engines interchangeable — a property that has already paid off twice.

---

## 2026-10-19 — Why an engine change at all

The Qwen-Image-2.1 path was measured and shipped, but its **denoiser** does not
produce the content the comic needs. That is consistent with the AtomicChat model
card: the "uncensored" release is the *text encoder* with the refusal direction
ablated, and its own numbers show 0 of 45 prompts flipping, because the encoder
cannot refuse inside a pipeline that generates no text. No encoder edit can add
what the denoiser never learned.

Owner found a purpose-built checkpoint, `NoctQ_V4_int8_convrot.safetensors`, and
already had ComfyUI running. Decision: keep the contract, swap the engine behind it.

## 2026-10-19 — Box inventory (verified, not assumed)

| item | value |
|---|---|
| GPU | RTX PRO 6000 Blackwell Workstation, 97,887 MiB, driver 595.71.05, sm_120 |
| ComfyUI | 0.41.0, `/workspace/ComfyUI`, supervised, `127.0.0.1:18188` |
| launch | `python main.py --disable-auto-launch --disable-xformers --port 18188 --enable-cors-header` |
| diffusion | `NoctQ_V4_int8_convrot.safetensors` — 6.76 GiB, 649 tensors, metadata `{source: NoctQ_V4_fp16.safetensors, quant: int8}` |
| alt diffusion | `qwen-image-2.1-UC-int8_convrot.safetensors` |
| text encoder | `qwen3vl_8b_int8_convrot.safetensors`, `CLIPLoader type=qwen_image` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |

The owner's saved workflow (`Qwen_Image_2.1_GGUF_uncensored.json`) loads **GGUF**
weights via `UnetLoaderGGUF`/`CLIPLoaderGGUF` with an AD-Q5_K encoder. NoctQ_V4 is
a native int8 checkpoint, so the bridge uses `UNETLoader`/`CLIPLoader` instead of
the GGUF nodes. Reading their graph was still worth it: it confirmed the Qwen-native
text-encode node and that `euler`/`simple` is a sane sampler pair.

## 2026-10-19 — The graph, and how it was derived

Node signatures were read from `/object_info` rather than guessed. The one that
matters: `TextEncodeQwenImage21` takes `(clip, prompt, negative_prompt, resolution,
images)` and returns **three** outputs — `positive`, `negative`, `latent` — so it
replaces both `CLIPTextEncode` nodes *and* carries the model's trained resolution.

Final graph:

```
UNETLoader ─┐
CLIPLoader ─┴→ TextEncodeQwenImage21 → positive/negative ─┐
EmptyLatentImage (w,h,1) ─────────────────────────────────┴→ KSampler
                                                              → VAEDecode → SaveImage
```

`images` (a `COMFY_AUTOGROW_V3` input) is omitted and ComfyUI accepts that — worth
recording because it is the kind of thing that fails opaquely.

## 2026-10-19 — Measurement log (832×1216 unless stated)

| run | steps | cfg | result |
|---|---:|---:|---|
| first probe | 8 | 1.0 | 9.0 s **cold** (includes model load), image OK |
| second probe | 25 | 4.0 | 7.5 s **warm**, cleaner composition, ink/screentone lean |
| 3 distinct prompts | 25 | 4.0 | 22.5 s wall → **8.0 img/min**, p50 15.9 s |
| 3 distinct prompts | 25 | 1.0 | 14.7 s wall → **12.3 img/min**, ~4.9 s/panel |
| repeat of identical graph | — | — | **2.1 s — cache, not speed** |

Two traps recorded so they are not re-learned:

1. **ComfyUI caches identical graphs.** Same prompt + seed + settings returns from
   its execution cache in ~2 s. Benchmarks must vary prompts or seeds, or they
   measure the cache.
2. **Cold vs warm model load** dominates a single-sample measurement; the first
   request after a restart is not representative.

## 2026-10-19 — Settings decisions

- **25 steps** — non-turbo checkpoint; the owner asked for this explicitly.
- **CFG 1** — owner's call, and it halves generation time (no uncond pass).
  Consequence recorded loudly: **at CFG 1 a negative prompt is inert.** The bridge
  still forwards `negative_prompt`, but it cannot act. If negatives ever matter
  ("extra fingers", "blurry"), CFG must go to ≥2 and the cost roughly doubles.
- **`euler` / `simple`** — matches the owner's own workflow; no evidence yet that
  another scheduler is better for this checkpoint.
- Per-request overrides (`steps`, `cfg`, `sampler`, `scheduler`, `seed`,
  `negative_prompt`) are honoured, so tuning does not need a redeploy.

## 2026-10-19 — Two compatibility decisions that are not obvious

1. **`/v1/models` reports `qwen-image-2.1-turbo`, not the real checkpoint.** The app's
   self-hosted path expects that id, and a mismatch can break its model resolution.
   The honest name stays visible as `engine_model` in `/healthz` — a small lie in
   one field, in exchange for the app not needing a code change.
2. **Reference images are ignored, visibly.** ComfyUI needs a different graph
   topology for image conditioning, and a `400` would have broken every panel the app
   renders *with* sheets. So the bridge drops references and says so:
   `references_ignored: true` in the body plus `X-References-Ignored: true`. The known
   cost is face drift between panels until a reference graph exists.

## 2026-10-20 — Incident: every render returned 401

Symptom: `{"detail":"bad or missing bearer token"}` from the app, while the bridge's
`/healthz` was healthy and `curl` with the token worked.

Diagnosis by elimination: no token → 401, previous box's token → 401, current token
→ 200. Root cause was **not** the bridge: the app persists image settings in SQLite
(`settings_kv`) and those beat `.env`, so it was replaying the previous box's
`SERVICE_TOKEN`.

Fix: update `settings_kv` (`image.apiKey`, `image.baseUrl`, `image.provider`,
`image.model`) with a DB backup, restart the app server. Turned into
`tools/app-token.py` so it is repeatable, and documented in the runbook as "the
re-rent trap".

Lesson for this journal: **a 401 with a healthy server is a client-config problem.**
Check what the client believes before touching the service.

---

## Known limitations / risks

- **References and edit are unimplemented.** Identity drifts; the app must keep
  references off. This is the biggest functional gap versus the Qwen path.
- **CFG 1 means negatives do nothing.** Silent if ignored.
- **ComfyUI's cache + a deterministic seed can return the *same* image on re-render.**
  The app compiles a seed per panel; if it does not vary on retry, a "re-render" may
  come back byte-identical. Needs checking on the app side.
- **One GPU, one graph in flight.** The bridge holds a lock, so requests queue
  honestly; it does not batch.
- **Single shared bearer token**, no TLS — the tunnel is the security boundary, so
  the bridge must stay bound to loopback.
- A third-party checkpoint's provenance/licence has not been reviewed here.

## Open questions / next steps

1. **Reference-image graph** (the one that would restore character consistency):
   encode sheet → reference conditioning → same sampler. Needs the ComfyUI node for
   Qwen image conditioning, and a measurement of its cost.
2. **Vary the seed per render attempt** so the ComfyUI cache cannot return duplicates.
3. **JPEG output** — panels are ~1.1–2.1 MB PNG; over a slow tunnel JPEG is the
   single biggest end-to-end win, and the bridge would be the easy place to add it.
4. Sampler/scheduler sweep at 25 steps CFG 1 (`euler_ancestral`, `dpmpp_2m`,
   `beta`/`karras`) — cheap to test, might improve linework.
5. Compare `qwen-image-2.1-UC-int8_convrot` against `NoctQ_V4` on the same prompts
   for style and fidelity.
6. LoRA support, if the owner's style needs locking down.

## Appending

Add a dated `##` section: context → what changed → evidence → decision → follow-ups.
Keep numbers with units and say which prompt/seed they came from; keep secrets out
(the token appears nowhere in this file, deliberately).

## 2026-10-20 — "Which workflow?" — A/B against the official blueprint

Owner asked whether the bridge uses `image_qwen_image_2_1_image_edit`. It uses **no
template**: `build_graph()` emits an API-format graph directly, derived from the
owner's saved `Qwen_Image_2.1_GGUF_uncensored.json` plus `/object_info` signatures.

Investigating that question surfaced something worth keeping: ComfyUI ships
**blueprints** in `/workspace/ComfyUI/blueprints/` (subgraph-wrapped JSON, so
`definitions.subgraphs[0].nodes` is where the real graph lives):

| blueprint | shape |
|---|---|
| Text to Image (Qwen-Image) | 19 nodes: `EmptySD3LatentImage` (1328²), **`ModelSamplingAuraFlow(3.1)`**, `CLIPTextEncode`×2, optional `LoraLoaderModelOnly` (Lightning 8-step), KSampler 8 steps cfg 1 |
| Image Edit (Qwen 2511) | 15 nodes: `TextEncodeQwenImageEditPlus`, `FluxKontextImageScale`, `FluxKontextMultiReferenceLatentMethod`, `CFGNorm(1)`, `ModelSamplingAuraFlow(3.1)`, KSampler 40 steps cfg 4, `VAEEncode` |

Two conclusions:

1. **The edit blueprint needs an edit checkpoint** (`qwen_image_edit_2511_bf16`), which
   this box does not have. So references are not "a template switch" — they need a
   different model file, not just a different graph.
2. The t2i blueprint sets a **flow-matching shift (3.1)** and 16-channel latents; the
   bridge did neither. Missing shift is a real sampling difference, not obviously a bug.

A/B, same prompt and seed, 25 steps CFG 1, 832×1216:

| variant | time | result |
|---|---:|---|
| `bridge` (TextEncodeQwenImage21 + EmptyLatentImage) | 4.2 s | colour comic art, more detailed/sketchy, dynamic framing |
| `blueprint` (EmptySD3LatentImage + ModelSamplingAuraFlow 3.1 + CLIPTextEncode×2) | 8.0 s | colour comic art, flatter cel shading, more settled framing |

Both are valid; they are different trajectories rather than one being correct. An
earlier impression that the model only produced monochrome woodcut art was wrong —
it came from prompts that literally asked for "thick ink outlines, screentone
shading", not from the missing shift node. Recorded because it is exactly the sort
of conclusion that gets asserted from a single sample.

Decision: ship both as `COMFY_VARIANT`, default **blueprint** (it matches the
official graph and the flatter look the app's style prompt asks for), with
`COMFY_SHIFT` exposed. Health now reports `variant` and `shift` so this can never be
ambiguous again. One render through the app contract after the switch: 6.6 s.