"""Static shell sanity check for every script in the repo (no bash available here)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import verify_bundle as vb  # noqa: E402  (reuse the same keyword/quote checks)

SCRIPTS = ["onstart.sh", "bootstrap.sh", "serve.sh",
           "bench/onstart.sh", "bench/diffusers_stage.sh", "bench/onstart-bench.sh"]

problems: list[str] = []
for rel in SCRIPTS:
    path = ROOT / rel
    if not path.exists():
        problems.append(f"{rel}: missing")
        continue
    text = path.read_text(encoding="utf-8")
    before = len(problems)
    vb.check_shell(text, rel, problems)
    if b"\r\n" in path.read_bytes():
        problems.append(f"{rel}: contains CRLF")
    if len(problems) == before:
        print(f"  {rel}: clean")

print()
if problems:
    print(f"FAIL ({len(problems)})")
    for p in problems:
        print(f"  - {p}")
    raise SystemExit(1)
print("OK: all shell scripts balanced, LF, no stray quotes")
