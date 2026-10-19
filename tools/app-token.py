#!/usr/bin/env python3
"""Point the comic app's persisted image settings at the current box.

Why this exists: the app writes image overrides to SQLite (`settings_kv`) and those
win over `.env`. Every re-rented GPU box gets a new `SERVICE_TOKEN`, so the app keeps
sending yesterday's token and every render comes back `401 bad or missing bearer
token` — which looks like a server problem and isn't.

  python tools/app-token.py --token qp_newtoken
  python tools/app-token.py --token qp_newtoken --base-url http://127.0.0.1:18080
  python tools/app-token.py --token qp_newtoken --dry-run

Restart the app server afterwards: it caches config at startup.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sqlite3
import sys
import time

DEFAULT_DB = (pathlib.Path(__file__).resolve().parents[2]
              / "reddit-comic-studio" / "workspace" / "studio.db")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", required=True, help="the bridge's SERVICE_TOKEN")
    ap.add_argument("--base-url", default="http://127.0.0.1:18080",
                    help="local end of the tunnel (default: %(default)s)")
    ap.add_argument("--model", default="qwen-image-2.1-turbo",
                    help="model id the app expects from /v1/models")
    ap.add_argument("--provider", default="qwen", choices=["qwen", "venice"])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    db = pathlib.Path(a.db)
    if not db.exists():
        print(f"FAIL: no app database at {db}\n"
              f"      pass --db <path to studio.db>", file=sys.stderr)
        return 2

    con = sqlite3.connect(db)
    rows = dict(con.execute("select key, value from settings_kv where key like 'image.%'"))
    print("before:")
    for k, v in sorted(rows.items()):
        print(f"  {k} = {(v[:6] + '…' + v[-4:]) if k == 'image.apiKey' else v}")

    if a.dry_run:
        con.close()
        print("\ndry run: nothing written")
        return 0

    backup = db.with_suffix(db.suffix + time.strftime(".%Y%m%d-%H%M%S.bak"))
    shutil.copyfile(db, backup)
    print(f"\nbackup: {backup}")

    now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    updates = {"image.apiKey": a.token, "image.baseUrl": a.base_url,
               "image.provider": a.provider, "image.model": a.model}
    with con:
        for key, value in updates.items():
            con.execute(
                "insert into settings_kv (key, value, updated_at) values (?, ?, ?) "
                "on conflict(key) do update set value=excluded.value, "
                "updated_at=excluded.updated_at",
                (key, value, now))

    rows = dict(con.execute("select key, value from settings_kv where key like 'image.%'"))
    con.close()
    print("after:")
    for k, v in sorted(rows.items()):
        print(f"  {k} = {(v[:6] + '…' + v[-4:]) if k == 'image.apiKey' else v}")
    print("\nrestart the app server so it re-reads this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
