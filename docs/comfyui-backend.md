# ComfyUI backend (NoctQ_V4) — what's running and how to test it

A second image engine behind the **same** `/v1/images/generations` contract, so the
comic app needs no code change — only its base URL and token.

```
comic app → http://127.0.0.1:18080  (this PC)
                │ supervised ssh tunnel
                ▼
        127.0.0.1:18700            service/comfy_bridge.py  (the contract)
                │ HTTP
                ▼
        127.0.0.1:18188            ComfyUI 0.41 (native graph API)
```

Why an engine swap: Qwen-Image-2.1's denoiser was trained without the content the
comic needs, and no encoder edit can add what the denoiser never learned — which is
exactly what the AtomicChat card says and what we observed. NoctQ_V4 is a
purpose-built checkpoint instead.

## Box inventory (verified)

| item | file |
|---|---|
| diffusion model | `NoctQ_V4_int8_convrot.safetensors` (6.76 GiB, int8 of `NoctQ_V4_fp16`, Qwen-Image architecture) |
| alt diffusion | `qwen-image-2.1-UC-int8_convrot.safetensors` |
| text encoder | `qwen3vl_8b_int8_convrot.safetensors`, `CLIPLoader type=qwen_image` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| ComfyUI | `/workspace/ComfyUI`, 0.41.0, supervised, `127.0.0.1:18188` |

Our saved UI workflow (`user/default/workflows/Qwen_Image_2.1_GGUF_uncensored.json`)
uses **GGUF** loaders and an AD-Q5_K encoder; the bridge uses the **native int8**
`UNETLoader`/`CLIPLoader` instead, because `NoctQ_V4_int8_convrot` is an int8
checkpoint, not a GGUF. Graph: `UNETLoader → CLIPLoader → TextEncodeQwenImage21 →
KSampler → VAEDecode → SaveImage`, latent size from the request.

## Settings (current, verified by rendering)

| knob | value | note |
|---|---|---|
| steps | **25** | non-turbo checkpoint |
| cfg | **1.0** | halves generation time (no uncond pass) |
| sampler / scheduler | `euler` / `simple` | |
| size | 832×1216 default | ~1 MP, the measured-safe envelope |

Measured at 832×1216, three **distinct** prompts, through the tunnel:

| cfg | generation | 3 panels wall | throughput |
|---|---:|---:|---:|
| 1.0 | ~4.9 s/panel | 14.7 s | **12.3 img/min** |
| 4.0 | ~7.5 s/panel | 22.5 s | 8.0 img/min |

**CFG 1 makes negative prompts inert** — there is only one forward pass, so
`negative_prompt` cannot act. The bridge still accepts and forwards it; it simply
has no effect. If negatives matter (e.g. "extra fingers, blurry"), cfg must be ≥ 2,
at roughly double the time.

Per-request overrides are honoured, so these can be tuned without a redeploy:
`steps`, `cfg`, `sampler`, `scheduler`, `seed`, `negative_prompt`.

## App configuration

```
COMIC_IMAGE_PROVIDER=qwen          # the self-hosted provider in the app
COMIC_IMAGE_BASE_URL=http://127.0.0.1:18080
COMIC_IMAGE_TOKEN=<bridge token>
```

`/v1/models` reports `qwen-image-2.1-turbo` on purpose — that is the id the app's
self-hosted path expects, and a different id can break its model resolution. The
real checkpoint is always visible as `engine_model` in `/healthz`.

## Two behaviours that will surprise you

1. **Reference images are ignored.** The bridge cannot do ComfyUI's image-conditioned
   graph yet, so references are dropped *visibly*: `references_ignored: true` in the
   response and an `X-References-Ignored: true` header. Panel-to-panel identity will
   drift — turn references off in the app until a reference graph exists.
2. **ComfyUI caches identical graphs.** Same prompt + seed + settings returns in ~2 s
   from its execution cache, not because the GPU got faster. Only distinct prompts
   give honest timings.

## Start / stop

```bash
# on the box (all detached, logs in /workspace/logs)
bash /workspace/start_bridge.sh                 # (re)start the bridge, prints health
pkill -f comfy_bridge.py                        # stop the bridge
pkill -f comfy_bridge.py && bash /workspace/start_bridge.sh   # apply new env

# on this PC
powershell -ExecutionPolicy Bypass -File tools\tunnel-supervise.ps1 `
  -Box <ip> -Port <sshport> -LocalPort 18080 -RemotePort 18700
```

Env for the bridge: `COMFY_URL`, `COMFY_UNET`, `COMFY_CLIP`, `COMFY_VAE`,
`COMFY_STEPS`, `COMFY_CFG`, `COMFY_SAMPLER`, `COMFY_SCHEDULER`, `SERVICE_TOKEN`,
`SERVICE_PORT`.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `/healthz` `status: down` | ComfyUI not running or wrong `COMFY_URL` | `curl 127.0.0.1:18188/system_stats` on the box |
| 500 with "rejected the graph" | model filename mismatch | `ls /workspace/ComfyUI/models/diffusion_models` |
| 500 with "graph failed" | OOM or bad node input | `/var/log/portal/comfyui.log` |
| panels arrive with same face as before | ComfyUI cache (identical seed+prompt) | vary the seed |
| identity drifts between panels | references ignored | expected today, see above |
