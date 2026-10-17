#!/usr/bin/env python3
"""Sanity-check the generated bundle without a shell.

There is no usable bash here (Git Bash cannot fork under the sandbox, Docker is
down), so this does the checks that actually catch paste-time breakage:

  * every embedded Python file is extracted and compiled by CPython;
  * no template placeholders survived, every heredoc is terminated;
  * block keywords (if/fi, do/done, case/esac) balance in the outer script *and*
    inside each shell body embedded in a heredoc;
  * no CRLF line endings, which would poison the heredocs on the instance.

Exit 0 means "safe to paste". It is not a substitute for `bash -n`, which is
worth running once on the box.
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
BUNDLE = HERE / "onstart-bench.sh"

PY_BLOCKS = [("BENCH_PY_EOF", "bench.py"),
             ("BENCH_DIFF_EOF", "bench_diffusers.py"),
             ("COMPARE_PY_EOF", "compare.py")]
SHELL_BLOCKS = [("BOOT_EOF", "onstart.sh"), ("DIFF_EOF", "diffusers_stage.sh")]


def strip_comments(line: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out)


def walk(text: str):
    """Yield (line_no, line, inside_heredoc) treating <<'TAG' ... TAG as opaque."""
    heredoc: str | None = None
    for n, line in enumerate(text.splitlines(), 1):
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
                yield n, line, True
                continue
            yield n, line, True
            continue
        opened = re.search(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", line)
        if opened and "<<<" not in line:
            heredoc = opened.group(1)
            yield n, line, False
            continue
        yield n, line, False


def check_shell(text: str, label: str, problems: list[str]) -> None:
    counts = {"if": 0, "fi": 0, "do": 0, "done": 0, "case": 0, "esac": 0}
    bad_lines = []
    for n, line, hidden in walk(text):
        if hidden:
            continue
        code = strip_comments(line)
        for word in re.findall(r"\b(if|fi|do|done|case|esac)\b", code):
            counts[word] += 1
        if code.count("'") % 2:
            bad_lines.append(n)
    for open_kw, close_kw in (("if", "fi"), ("do", "done"), ("case", "esac")):
        if counts[open_kw] != counts[close_kw]:
            problems.append(f"{label}: {open_kw}/{close_kw} imbalance "
                            f"{counts[open_kw]} vs {counts[close_kw]}")
    for n in bad_lines[:5]:
        problems.append(f"{label} line {n}: odd number of single quotes")
    print(f"  {label}: {counts}")


def main() -> int:
    if not BUNDLE.exists():
        print(f"missing {BUNDLE.name}; run sync_bundle.py first")
        return 1
    raw = BUNDLE.read_bytes()
    text = raw.decode("utf-8")
    problems: list[str] = []

    if b"\r\n" in raw:
        problems.append("CRLF line endings present (heredocs would carry \\r)")
    if not text.startswith("#!/bin/bash"):
        problems.append("missing #!/bin/bash shebang")
    for placeholder in ("__BOOT__", "__DIFF__", "__BENCH_PY__",
                        "__BENCH_DIFF__", "__COMPARE_PY__"):
        if placeholder in text:
            problems.append(f"placeholder {placeholder} left in output")

    for marker, name in PY_BLOCKS:
        blocks = re.findall(rf"<<'{marker}'\n(.*?)\n{marker}\n", text, re.S)
        if len(blocks) != 1:
            problems.append(f"expected 1 heredoc for {name}, found {len(blocks)}")
            continue
        try:
            compile(blocks[0], f"embedded-{name}", "exec")
            print(f"  {name} compiles ({len(blocks[0]):,} bytes)")
        except SyntaxError as e:
            problems.append(f"{name} syntax error: {e}")

    # outer script, then each shell body as its own program
    check_shell(text, "outer script", problems)
    for marker, name in SHELL_BLOCKS:
        blocks = re.findall(rf"<<'{marker}'\n(.*?)\n{marker}\n", text, re.S)
        if len(blocks) != 1:
            problems.append(f"expected 1 heredoc for {name}, found {len(blocks)}")
            continue
        check_shell(blocks[0], name, problems)

    print()
    if problems:
        print(f"FAIL ({len(problems)} problems)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: {BUNDLE.name} looks pasteable "
          f"({len(text):,} bytes, {text.count(chr(10)) + 1} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
