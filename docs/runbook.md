# Runbook — start the image server and the tunnel

Two processes, on two machines:

```
comic app  →  http://127.0.0.1:<LOCAL_PORT>   (your PC)
                    │  ssh tunnel
                    ▼
            127.0.0.1:10100                    (rented GPU box)
                    │
             serve.py  →  QwenImage21Pipeline  (GPU)
```

Nothing is exposed to the internet: the service binds loopback on the box, and
the only way in is the SSH tunnel. That is why the app's base URL never changes
when you re-rent a GPU.

---

## 0. Values you need

| name | meaning | example |
|---|---|---|
| `BOX_HOST` | box IP or hostname | `93.91.156.106` |
| `BOX_SSH_PORT` | box SSH port | `46075` |
| `TOKEN` | bearer token you set when starting the service | `qp_…` |
| `LOCAL_PORT` | local end of the tunnel | `18080` |
| `REMOTE_PORT` | port the service binds on the box | `10100` |
| `KEY` | SSH private key | `C:\Users\<you>\.ssh\id_ed25519` |

Today's actual values are in `local/current-box.md` (gitignored — never commit
tokens to this public repo).

---

## 1. First time on a fresh box: deploy (≈2–3 min)

Rent: **RTX PRO 6000, 96 GB**, image `vastai/base-image:cuda-12.9.2-auto`,
disk **150 GB**. Check the variant in `nvidia-smi`: `Server Edition` is ~30%
faster than `Max-Q Workstation Edition`.

Then either paste this in the Vast "on-start script" field, or run it over ssh:

```bash
curl -fsSL https://raw.githubusercontent.com/laoda414/qwen-image-21-delpoy/main/onstart.sh \
  | REPO_URL=https://github.com/laoda414/qwen-image-21-delpoy.git \
    SERVICE_TOKEN=<TOKEN> PORT=10100 SERVICE_HOST=127.0.0.1 bash
```

`SERVICE_HOST=127.0.0.1` is the important part: without it the service binds
`0.0.0.0` and is reachable on the box's published port.

It installs torch + diffusers, downloads the 32 GB checkpoint into
`/workspace/models`, and starts serving. Progress: `/workspace/logs/deploy.log`.

---

## 2. Start the server on an already-provisioned box

One-time (or after an update): bring the repo current.

```bash
cd /workspace/qwen-image-gpu && git pull --ff-only
```

Start it, detached, bound to loopback:

```bash
cd /workspace/qwen-image-gpu
SERVICE_HOST=127.0.0.1 SERVICE_PORT=10100 SERVICE_TOKEN=<TOKEN> \
  setsid nohup ./serve.sh > /workspace/logs/serve.log 2>&1 < /dev/null &
```

Check it came up (~10 s to load the model, ~4 s warmup):

```bash
curl -s localhost:10100/healthz
# {"status":"ok",...,"served":0,"last_error":null}
ss -lntp | grep 10100          # must show 127.0.0.1:10100, not 0.0.0.0
```

## 3. Start the tunnel on your PC

PowerShell, keep this window open:

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 `
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
  -L 18080:127.0.0.1:10100 -i "$env:USERPROFILE\.ssh\id_ed25519" `
  -p 46075 root@93.91.156.106
```

Or, from Git Bash / the repo: `./tunnel.sh root@93.91.156.106 46075 18080 10100`

`-N` = no shell, just forwarding. Closing the window (or Ctrl-C) tears the tunnel
down, and the app will see connection refused until you start it again.

---

## 4. Verify, in this order

```powershell
# a) service alive through the tunnel
curl.exe -s http://127.0.0.1:18080/healthz

# b) one real panel, graded PASS/WARN/FAIL (this is the honest gate)
python tools\preflight.py --url http://127.0.0.1:18080 --token <TOKEN> --runs 2

# c) a full comic-sized burst
python service\test_batch.py --url http://127.0.0.1:18080 --token <TOKEN> --n 12
```

`preflight` thresholds: **PASS** ≤15 s/panel, **WARN** ≤60 s, **FAIL** above —
if FAIL, the GPU is fine and the *path* is bad; destroy and re-rent.

---

## 5. Daily cheat sheet

| task | command |
|---|---|
| box-side, start | `SERVICE_HOST=127.0.0.1 SERVICE_PORT=10100 SERVICE_TOKEN=<TOKEN> setsid nohup ./serve.sh > /workspace/logs/serve.log 2>&1 &` |
| box-side, stop | `pkill -f service/serve.py` |
| box-side, logs | `tail -f /workspace/logs/serve.log` |
| box-side, is it loopback? | `ss -lntp \| grep 10100` |
| PC-side, tunnel up | the `ssh -N -L …` command from §3 |
| PC-side, tunnel down | Ctrl-C in that window |
| update code | `cd /workspace/qwen-image-gpu && git pull --ff-only` then restart |
| app config | `COMIC_IMAGE_BASE_URL=http://127.0.0.1:18080`, `COMIC_IMAGE_TOKEN=<TOKEN>` |

Re-renting a box changes **only** `BOX_HOST`/`BOX_SSH_PORT` in §3. The service
port, the local port, the token and the app config stay identical.

---

## 6. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `connection refused` on 18080 | tunnel down, or service down | start tunnel; `curl localhost:10100/healthz` on the box |
| `503 model still loading` | process just started | wait ~15 s, retry |
| `401` | token mismatch between `serve.sh` and the app | restart service with the app's token |
| works on box, dead through tunnel | service bound `0.0.0.0` but tunnel points elsewhere, or ssh died | check §2's `ss` output and the tunnel window |
| panels very slow | slow link (check `tools/preflight.py`) or reference images in use (1.43×–1.74×) | re-rent, or drop references, or use fewer |
| `CUDA out of memory` in serve.log | someone else's process on the GPU | `nvidia-smi` on the box; restart service |
| service unreachable *after re-rent* | new IP/port | update §3, nothing else |

---

## 7. Security notes

- The service has **no TLS and one shared bearer token** — it must stay on
  loopback. Never set `SERVICE_HOST=0.0.0.0` on a box with a published port.
- Keep tokens out of git: this repo is public. Live values live in
  `local/current-box.md`, which is gitignored.
- The model has no content filter and will produce adult content; access control
  is the token plus the tunnel, so treat the token as the security boundary.
