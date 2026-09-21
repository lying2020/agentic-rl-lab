#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
xelatex -interaction=nonstopmode BRIEFING.tex
xelatex -interaction=nonstopmode BRIEFING.tex
echo "OK: $DIR/BRIEFING.pdf"
