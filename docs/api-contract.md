# Image API contract — Qwen-Image-2.1-Turbo service

Status: **v1, implemented and exercised.** Endpoints below were validated on live
hardware: a 12-panel batch (0 failures), bearer auth, and the reference-image
path. Anything marked *planned* is **not** implemented yet — do not build against it.

This is for the comic-studio server. The relevant seam is
`apps/server/src/providers/types.ts` (`ImageProvider` / `ImageGenRequest` /
`ImageGenResult`), so this document maps HTTP → that interface rather than
inventing a second abstraction.

---

## 1. Transport and addressing

The service is **not** reachable on a stable public URL. It binds loopback inside
the rented GPU box and is reached through an SSH tunnel:

```
app  →  http://127.0.0.1:18080  →  (ssh tunnel)  →  127.0.0.1:10100 in the box
```

- Base URL is a **local, configurable** value (`COMIC_IMAGE_BASE_URL`,
  default `http://127.0.0.1:18080`). It does not change when the GPU box is
  re-rented — the tunnel is what gets rebuilt.
- The tunnel is an operational concern, not an API concern: if it drops, the app
  sees connection refused. Treat that as `ProviderError('unavailable')` and retry
  after the tunnel is re-established; there is no server-side session to recover.
- No TLS between app and tunnel endpoint (it is loopback); confidentiality comes
  from SSH.

## 2. Authentication

Every `/v1/*` call requires, when the service was started with `SERVICE_TOKEN`:

```
Authorization: Bearer <token>
```

- Missing or wrong → `401 {"detail": "bad or missing bearer token"}`.
- `/healthz` is unauthenticated by design (readiness probes).
- The token is a shared secret per deployment; do not ship it to the browser.

## 3. Endpoints

### `POST /v1/images/generations`

OpenAI-shaped, with one deliberate extension (reference images, §5).

| field | type | required | notes |
|---|---|---|---|
| `prompt` | string | **yes** | empty/whitespace → `400` |
| `size` | string | no | `"WxH"`, e.g. `"832x1216"`; overrides width/height |
| `width`, `height` | int | no | default `832 × 1216`; **rounded down to a multiple of 32**, minimum 256 |
| `n` | int | no | 1–8, default 1; images are generated **sequentially**, so `n` does not reduce per-image time |
| `seed` | int | no | honoured for reproducibility; omitted = time-based |
| `image` | string | no | one base64 reference (with or without `data:` prefix) |
| `images` | string[] | no | up to 4 base64 references |

Unknown fields are ignored (including `negative_prompt`, see §6).

**Response `200`:**

```json
{
  "created": 1792280000,
  "model": "qwen-image-2.1-turbo",
  "data": [
    { "b64_json": "iVBORw0KGgo...", "revised_prompt": "<the prompt you sent>" }
  ]
}
```

- `data` has exactly `n` entries, always **PNG**, always base64 without a `data:`
  prefix. `mime` is therefore always `image/png`.
- Payload size, measured: **~1.0–1.4 MB at 832×1216** (prompt-dependent), ~1.4 MB
  at 1024². Plan for transfer cost, not just generation cost.
- `revised_prompt` echoes the request verbatim; there is no prompt rewriting.

### `GET /v1/models`

```json
{"object": "list", "data": [{"id": "qwen-image-2.1-turbo", "object": "model", "owned_by": "local"}]}
```

Static — one model per deployment. Build `ImageModelInfo` from constants:

| field | value |
|---|---|
| `id` | `qwen-image-2.1-turbo` |
| `supportsStyleReferences` | `true` (reference images accepted, §5) |
| `maxStyleReferences` | `4` |
| `maxSide` | `1280` recommended ceiling (see §7) |
| `supportsAspectRatio` | `true` (any multiple of 32) |
| `priceHint` | `0` — self-hosted, no per-image charge |

### `GET /healthz` (unauthenticated)

```json
{"status":"ok","model":"...","device":"cuda:0","queue_depth":0,
 "inflight":0,"served":14,"last_error":null}
```

Use it for two things:
- readiness: `status == "ok"` (otherwise the model is still loading → `503`);
- **backpressure**: `queue_depth + inflight` is the number of panels ahead of a
  new request. Keep it ≤ 2–3 (§8).

## 4. Errors

| status | when | app handling |
|---|---|---|
| `400` | missing prompt, bad base64 reference, >4 references | `bad_request`, do not retry |
| `401` | missing/wrong bearer token | `auth`, do not retry |
| `500` | pipeline failure; body has the reason, also in `/healthz.last_error` | retry once, then `unknown` |
| `503` | model still loading after boot | retry with backoff (~10 s) up to ~2 min |
| `504` | waited > 600 s in the queue | retry later or reduce concurrency |

## 5. Character consistency (reference images)

Qwen-Image-2.1 is one model for text-to-image **and** edit **and**
multi-reference composition — there is no separate Edit endpoint or model. Pass
character sheets as `image` / `images` on the same `POST /v1/images/generations`.

- Max 4 references; each decoded to PIL and passed to the pipeline.
- Map `ImageGenRequest.styleReferences[].image` (base64 or data URL) into
  `images`. **`strength` has no equivalent here and must be dropped** — do not
  fake it, and do not fail the request if it is present.
- ⚠️ **Unmeasured cost.** All throughput numbers in §7 are text-only. References
  add encoded tokens to the conditioning, so expect slower panels and more VRAM.
  Measure on the first box before promising panel latency with references.

## 6. Mapping to `ImageProvider`

`ImageGenRequest` → HTTP body:

| `ImageGenRequest` field | mapping |
|---|---|
| `prompt` | `prompt` |
| `width` / `height` | `width` / `height` (prefer these over `size`, so the app controls rounding) |
| `aspectRatio` | convert to explicit pixels — the service has no aspect-ratio concept |
| `seed` | `seed` |
| `variants` | `n` (sequential; do not raise above 2–3 for latency reasons) |
| `styleReferences[].image` | `images[]` (drop `strength`) |
| `model` | fixed `qwen-image-2.1-turbo`; the field is accepted but irrelevant |
| `format` | **PNG only.** `jpeg`/`webp` not implemented — request them and you still get PNG |
| `negativePrompt` | **not supported, ignored.** Turbo runs CFG 1.0, so a negative prompt has no effect; log a warning rather than silently pretending |
| `signal` | pass through as request abort |

`ImageGenResult` ← response:

| field | value |
|---|---|
| `images[]` | `{ base64: data[i].b64_json, mime: "image/png" }` |
| `model` | response `model` |
| `seed` | **not returned by the server** — the provider must echo the seed it sent |
| `blurred` | always `false` |
| `contentViolation` | always `false` |
| `deprecationWarning` | always `null` |
| `timing` | **not returned yet** (see planned) |
| `usd` | `0` — cost is the GPU rental, not per image |

## 7. Sizing presets (measured, text-only)

| panel size | s/image | images/min | note |
|---|---:|---:|---|
| 832×1216 (2:3) | 1.98 | 30.3 | recommended default for panels |
| 1344×768 (wide) | 2.03 | 29.6 | |
| 1024×1024 | 2.04 | 29.4 | |
| 1536×1536 | 5.49 | 10.9 | 2.7× time for 2.25× pixels |
| 1696×2528 (Qwen's 4 MP preset) | — | — | **not recommended**: ~4× the pixels of 1536², well past the measured VRAM envelope |

Keep panels ≤ ~1 MP. Cost is roughly linear in pixels up to 1 MP and worse above.

## 8. Concurrency and latency (the part that surprises people)

The service has **one** pipeline and a **FIFO queue**. Measured facts:

- one panel in flight: ~2 s generation at 832×1216;
- N panels fired concurrently: total wall ≈ N × 2 s, and the last panel waits for
  all the others. A 12-panel burst finished in ~32 s with p50 **18.4 s** and max
  **32 s**.
- More concurrency does **not** raise throughput (the DiT saturates the GPU at
  batch 1) — it only lengthens the tail.

Client guidance:

1. Keep 2–3 requests in flight per deployment, not 12.
2. Render panels as they arrive; do not wait for the whole set.
3. Use `/healthz` `queue_depth` as a brake before enqueueing more.
4. Retry `503` with backoff; never retry `400`/`401`.

## 9. Planned, not implemented — do not build against

- `timing` in the response (`queue_ms`, `generation_ms`) — useful for the app's
  ledger; not present today.
- `format: "jpeg"` — would cut payload ~7×, which matters on slow links.
- `/v1/images/edits` as a separate route — references currently ride on the
  generations body.
- Per-request `steps`/`guidance` overrides — the Turbo 8-step schedule is fixed
  and deliberately not exposed (other schedules are not validated for this
  checkpoint).

## 10. Minimal client example (TypeScript, for the provider)

```ts
const res = await fetch(`${baseUrl}/v1/images/generations`, {
  method: 'POST',
  headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` },
  body: JSON.stringify({ prompt, width, height, seed, n: 1 }),
  signal,  // fetch option, not part of the JSON body
});
if (res.status === 503) throw new ProviderError('unavailable', 'qwen-image', 'model loading');
if (!res.ok) throw new ProviderError('unknown', 'qwen-image', `HTTP ${res.status}`);
const { data } = await res.json() as { data: { b64_json: string }[] };
return data.map(d => ({ base64: d.b64_json, mime: 'image/png' }));
```
