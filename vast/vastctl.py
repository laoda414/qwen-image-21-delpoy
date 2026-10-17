#!/usr/bin/env python3
"""Vast.ai control plane for the Qwen-Image bench box.

The instance's public IP and port change on every rent, so nothing here hands
out a hardcoded address: `endpoint` re-resolves http://ip:port from instance id
or label every time it is called. The comic-studio server will call the same
lookup before a batch and on any connection failure, which is how the "IP will
change" problem disappears at the app layer instead of being patched by hand.

Auth: VAST_API_KEY env var, --api-key, or ~/.config/vastai/vast_api_key
(the file the official CLI writes). Read-only commands work without one only
if the key file exists.

Commands
  offers   [--gpu 6000] [--min-vram 90000] [--max-price 2.0]
  rent     --offer <ask_id> [--disk 120] [--image ...] [--onstart bench/onstart.sh]
  ls       [--label ...]
  wait     [--label ...] [--timeout 900]
  endpoint [--port 1234] [--label ...]   -> prints http://ip:port
  ssh      [--label ...]                 -> prints an ssh command
  destroy  [--id ...] [--label ...]

Every command accepts --json for machine-readable output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai/api/v0"
DEFAULT_LABEL = "qwen-comic-bench"
DEFAULT_IMAGE = "vastai/base-image:cuda-12.9.2-auto"


def api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("VAST_API_KEY"):
        return os.environ["VAST_API_KEY"]
    for path in (os.path.expanduser("~/.config/vastai/vast_api_key"),
                 os.path.expanduser("~/.vast_api_key")):
        if os.path.exists(path):
            with open(path) as f:
                key = f.read().strip()
            if key:
                return key
    raise SystemExit("no API key: set VAST_API_KEY or write ~/.config/vastai/vast_api_key")


def call(method: str, path: str, key: str, body=None, params=None):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        # older deployments only accept the key as a query parameter
        if e.code in (401, 403) and "Authorization" in headers:
            return call(method, path, key, body=body,
                        params={**(params or {}), "api_key": key})
        raise SystemExit(f"HTTP {e.code} on {method} {path}: {detail}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


# ------------------------------------------------------------------ offers ---
def cmd_offers(a, key):
    q = {
        "num_gpus": {"eq": 1},
        "rentable": {"eq": True},
        "vram_total": {"gte": a.min_vram},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": 400,
    }
    res = call("GET", "/bundles/", key, params={"q": json.dumps(q)})
    offers = res.get("offers", []) if isinstance(res, dict) else []
    needle = (a.gpu or "").lower()
    rows = []
    for o in offers:
        name = str(o.get("gpu_name", ""))
        if needle and needle not in name.lower():
            continue
        if a.max_price and float(o.get("dph_total") or 99) > a.max_price:
            continue
        if o.get("disk_space") is not None and float(o["disk_space"]) < a.disk + 10:
            continue
        rows.append(o)
    names = sorted({str(o.get("gpu_name")) for o in offers})
    if a.json:
        print(json.dumps({"count": len(rows), "gpu_names_seen": names,
                          "offers": rows[:a.limit]}, indent=2))
        return
    print(f"# {len(rows)} matching offers (of {len(offers)} with >= {a.min_vram} MiB)")
    print(f"# GPU names on the market right now: {', '.join(names) or '(none)'}")
    hdr = f"{'ask_id':>10}  {'gpu':<22} {'vram':>6} {'$/hr':>6} {'rel':>5} {'net':>5}  loc"
    print(hdr)
    print("-" * len(hdr))
    for o in rows[:a.limit]:
        print(f"{o.get('id'):>10}  {str(o.get('gpu_name'))[:22]:<22} "
              f"{int(o.get('gpu_total_ram') or o.get('vram_total') or 0):>6} "
              f"{float(o.get('dph_total') or 0):>6.2f} "
              f"{float(o.get('reliability2') or 0):>5.2f} "
              f"{int(o.get('inet_up') or 0):>5}  {o.get('geolocation')}")


# -------------------------------------------------------------------- rent ---
def cmd_rent(a, key):
    onstart = ""
    if a.onstart:
        if not os.path.exists(a.onstart):
            raise SystemExit(f"--onstart file not found: {a.onstart}")
        with open(a.onstart, "r") as f:
            onstart = f.read()
    body = {
        "client_id": "me",
        "image": a.image,
        "disk": a.disk,
        "runtype": "ssh",
        "label": a.label,
        "onstart": onstart,
        "target_state": "running",
    }
    if a.ports:
        body["ports"] = a.ports
    res = call("PUT", f"/asks/{a.offer}/", key, body=body)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(res)
    cid = res.get("new_contract") if isinstance(res, dict) else None
    if cid:
        print(f"\ninstance id: {cid}  (label {a.label!r})")
        print(f"watch it with:  python vast/vastctl.py wait --label {a.label}")
    return cid


# --------------------------------------------------------------- instances ---
def instances(key, label: str | None = None):
    res = call("GET", "/instances/", key, params={"owner": "me"})
    items = res.get("instances", []) if isinstance(res, dict) else []
    if label:
        items = [i for i in items if label in str(i.get("label", ""))]
    return items


def pick(a, key):
    items = instances(key, a.label)
    if a.id:
        items = [i for i in items if str(i.get("id")) == str(a.id)]
    if not items:
        raise SystemExit("no matching instances")
    return items[0]


def cmd_ls(a, key):
    items = instances(key, a.label)
    if a.json:
        slim = [{k: i.get(k) for k in ("id", "label", "actual_status", "gpu_name",
                                       "num_gpus", "ssh_host", "ssh_port",
                                       "public_ipaddr", "ports", "dph_total",
                                       "cur_state", "status_msg", "image_uuid")}
                for i in items]
        print(json.dumps(slim, indent=2))
        return
    for i in items:
        print(f"{i.get('id')}  {i.get('actual_status'):<12} {str(i.get('gpu_name')):<20} "
              f"{i.get('ssh_host')}:{i.get('ssh_port')}  ${float(i.get('dph_total') or 0):.2f}/hr  "
              f"{i.get('label')}")
        if i.get("status_msg"):
            print(f"    status: {str(i.get('status_msg'))[:140]}")


def cmd_wait(a, key):
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        items = instances(key, a.label)
        if a.id:
            items = [i for i in items if str(i.get("id")) == str(a.id)]
        for i in items:
            st = str(i.get("actual_status"))
            if st == "running" and i.get("ssh_host") and i.get("ports"):
                print(f"running: id={i.get('id')} ssh={i.get('ssh_host')}:{i.get('ssh_port')} "
                      f"ports={list((i.get('ports') or {}).keys())}")
                return 0
            print(f"  {i.get('id')} {st} {str(i.get('status_msg'))[:90]}")
        time.sleep(15)
    raise SystemExit("timed out waiting for the instance")


def endpoint_for(inst: dict, container_port: int, key_suffix: str = "tcp"):
    """Resolve http://host:port for a container port from the live instance record.

    This is the whole answer to the floating-IP problem: the mapping is read
    from Vast every time, never cached in config.
    """
    ports = inst.get("ports") or {}
    for pname, bindings in ports.items():
        if pname.split("/")[0] == str(container_port):
            for b in bindings or []:
                host = b.get("HostIp") or "0.0.0.0"
                hport = b.get("HostPort")
                if not hport:
                    continue
                ip = inst.get("public_ipaddr") or inst.get("ssh_host")
                if host in ("0.0.0.0", "::", ""):
                    host = ip
                return f"http://{host}:{hport}"
    return None


def cmd_endpoint(a, key):
    inst = pick(a, key)
    url = endpoint_for(inst, a.port)
    if not url:
        raise SystemExit(f"port {a.port} is not published on instance {inst.get('id')}; "
                         f"published: {list((inst.get('ports') or {}).keys())}")
    if a.json:
        print(json.dumps({"id": inst.get("id"), "url": url,
                          "ssh": f"{inst.get('ssh_host')}:{inst.get('ssh_port')}"}))
    else:
        print(url)


def cmd_ssh(a, key):
    inst = pick(a, key)
    host, port = inst.get("ssh_host"), inst.get("ssh_port")
    if not host:
        raise SystemExit("instance has no ssh_host yet")
    keyfile = os.path.expanduser(a.keyfile)
    print(f"ssh -p {port} -i {keyfile} root@{host}")


def cmd_destroy(a, key):
    items = instances(key, a.label)
    if a.id:
        items = [i for i in items if str(i.get("id")) == str(a.id)]
    if not items and a.id:
        items = [{"id": a.id}]
    for i in items:
        iid = i.get("id")
        print(call("DELETE", f"/instances/{iid}/", key))
    if not items:
        print("nothing to destroy")


# ---------------------------------------------------------------- templates --
TEMPLATE_FIELDS = ("id", "hash_id", "name", "image", "runtype", "onstart",
                   "docker_opts", "docker_options", "env", "disk_space",
                   "recommended_disk_space", "extra_filters", "jup_direct",
                   "ssh_direct", "private", "count_created")


def cmd_template(a, key):
    """Dump a template's real config, to judge it before renting.

    Console template links are single-page apps, so the page source carries no
    settings; this reads the same record the console renders.
    """
    res = call("GET", "/templates/", key,
               params={"q": json.dumps({"limit": 400, "order": [["created_at", "desc"]]})})
    items = res.get("templates", []) if isinstance(res, dict) else []
    if not items:
        raise SystemExit(f"template list empty or unexpected: {str(res)[:300]}")

    if a.hash:
        needle = a.hash.strip().lower()
        items = [t for t in items
                 if needle in str(t.get("hash_id", "")).lower()
                 or needle == str(t.get("id", "")).lower()]
        if not items:
            raise SystemExit(f"no template matching {a.hash!r} in {len(res.get('templates', []))} "
                             f"templates; try `--hash` with the id from the URL, or "
                             f"--json and grep the raw list")

    for t in items[:a.limit]:
        if a.json:
            print(json.dumps(t, indent=2))
            continue
        print(f"\n=== {t.get('name')}  (id {t.get('id')}, hash {t.get('hash_id')})")
        for f in TEMPLATE_FIELDS:
            if f in t and t[f] not in (None, "", {}, []):
                val = t[f]
                if isinstance(val, str) and "\n" in val:
                    print(f"  {f}:")
                    for line in val.splitlines():
                        print(f"      {line}")
                else:
                    print(f"  {f}: {val}")
        unknown = [k for k in t if k not in TEMPLATE_FIELDS
                   and t[k] not in (None, "", {}, [], 0, False)]
        if unknown:
            print(f"  (other fields: {', '.join(sorted(unknown))})")


# -------------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key")
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("offers")
    o.add_argument("--gpu", default="6000", help="substring match on gpu_name")
    o.add_argument("--min-vram", type=int, default=90000)
    o.add_argument("--max-price", type=float, default=0.0)
    o.add_argument("--disk", type=int, default=120)
    o.add_argument("--limit", type=int, default=20)

    r = sub.add_parser("rent")
    r.add_argument("--offer", required=True, type=str)
    r.add_argument("--disk", type=int, default=120)
    r.add_argument("--image", default=DEFAULT_IMAGE)
    r.add_argument("--onstart", default="bench/onstart.sh")
    r.add_argument("--label", default=DEFAULT_LABEL)
    r.add_argument("--ports", default={"1234/tcp": [{"HostIp": "0.0.0.0", "HostPort": "1234"}]},
                   type=json.loads)

    for name in ("ls", "wait", "endpoint", "ssh", "destroy"):
        s = sub.add_parser(name)
        s.add_argument("--id")
        s.add_argument("--label", default=DEFAULT_LABEL)
        if name == "wait":
            s.add_argument("--timeout", type=int, default=1200)
        if name == "endpoint":
            s.add_argument("--port", type=int, default=1234)
        if name == "ssh":
            s.add_argument("--keyfile", default="~/.ssh/id_ed25519")

    t = sub.add_parser("template", help="dump a template's image/ports/onstart")
    t.add_argument("--hash", default="", help="hash_id or id from the console URL")
    t.add_argument("--limit", type=int, default=5)

    a = p.parse_args()
    key = api_key(a.api_key)
    {"offers": cmd_offers, "rent": cmd_rent, "ls": cmd_ls, "wait": cmd_wait,
     "endpoint": cmd_endpoint, "ssh": cmd_ssh, "destroy": cmd_destroy,
     "template": cmd_template}[a.cmd](a, key)


if __name__ == "__main__":
    main()
