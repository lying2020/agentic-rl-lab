#!/usr/bin/env python3
"""Fail briefing markdown that will not render math/mermaid in Cursor Preview."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
)
BRACKET_RE = re.compile(r"\\\[|\\\]|\\\(|\\\)")
FRAC_OUTSIDE = re.compile(r"\\frac\s*\{")
MERMAID_RE = re.compile(r"^```mermaid\s*$", re.MULTILINE)
DOLLAR_BLOCK_RE = re.compile(r"^\$\$", re.MULTILINE)


def strip_fences(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.sub(r"`[^`]+`", "", text)


def mermaid_blocks(text: str) -> list[str]:
    return re.findall(r"```mermaid\s*\n(.*?)```", text, flags=re.DOTALL)


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    body = strip_fences(text)
    errors: list[str] = []

    if ENV_RE.search(body):
        errors.append("found \\begin{equation|align|...}; wrap math in $$ for Preview")
    if BRACKET_RE.search(body):
        errors.append("found \\[ \\] or \\( \\); use $...$ / $$...$$ instead")

    # \frac outside $ ... $ or $$ ... $$
    dollars = re.split(r"\${1,2}", body)
    # odd chunks are outside math if file starts outside math
    for i, chunk in enumerate(dollars):
        if i % 2 == 0 and FRAC_OUTSIDE.search(chunk):
            errors.append("\\frac appears outside $...$ / $$...$$")
            break

    if not MERMAID_RE.search(text):
        errors.append("missing ```mermaid block (method figure is required)")

    for i, block in enumerate(mermaid_blocks(text), 1):
        if "$" in block or "\\frac" in block or "\\begin{" in block:
            errors.append(
                f"mermaid block #{i} contains $ or LaTeX; move formulas below the figure"
            )

    n_dollar_block = len(DOLLAR_BLOCK_RE.findall(text))
    if n_dollar_block % 2 != 0:
        errors.append(f"unbalanced $$ fences (count={n_dollar_block})")

    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("md", nargs="+", type=Path)
    args = p.parse_args()
    n_err = 0
    for path in args.md:
        errs = check(path)
        if errs:
            n_err += len(errs)
            print(f"FAIL {path}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {path}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
