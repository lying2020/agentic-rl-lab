#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
# 优先 xelatex（中文）；失败可改 pdflatex + 英文降级
xelatex -interaction=nonstopmode BRIEFING.tex
xelatex -interaction=nonstopmode BRIEFING.tex
echo "OK: $DIR/BRIEFING.pdf"
