"""Normalise working-tree line endings to LF and verify what git actually stores."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
TEXT_EXT = {".sh", ".py", ".md", ".json", ".txt", ".yml", ".yaml", ".Dockerfile"}
NAMES = {"Dockerfile", ".gitignore", ".gitattributes"}

fixed = []
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or ".git" in p.parts or "__pycache__" in p.parts:
        continue
    if p.suffix not in TEXT_EXT and p.name not in NAMES:
        continue
    data = p.read_bytes()
    if b"\r\n" in data:
        p.write_bytes(data.replace(b"\r\n", b"\n"))
        fixed.append(str(p.relative_to(ROOT)))

print(f"rewritten to LF: {fixed if fixed else 'none'}")

for args in (["git", "add", "-A"],
             ["git", "-c", "user.name=qwen-image-gpu",
              "-c", "user.email=dev@localhost", "commit", "-q", "-m",
              "Normalise line endings to LF"]):
    subprocess.run(args, cwd=ROOT, check=False)

print("\nstored blobs (CRLF count must be 0):")
for name in ("onstart.sh", "bootstrap.sh", "serve.sh", "bench/onstart.sh",
             "service/serve.py", "Dockerfile"):
    blob = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
    print(f"  {name:<22} CR={blob.count(b'\r')}  LF={blob.count(b'\n')}")
