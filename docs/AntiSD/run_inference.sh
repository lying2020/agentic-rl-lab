#!/usr/bin/env bash
# 最小推理入口。默认用 Qwen3-8B + 论文 Appendix C 示例题。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANTISD_ROOT="${ANTISD_ROOT:-/home/user/Documents/agentic-rl-lab/algorithm/AntiSD}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
BACKEND="${BACKEND:-hf}"

export PYTHONPATH="${ANTISD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -gt 0 && "$1" == "--eval" ]]; then
    shift
    PARQUET="${1:-${ANTISD_ROOT}/datasets/math/aime25/test.parquet}"
    python3 "${HERE}/inference_math.py" \
        --model "${MODEL_PATH}" \
        --backend "${BACKEND}" \
        --eval-parquet "${PARQUET}" \
        --max-samples "${MAX_SAMPLES:-4}" \
        --n "${N:-1}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-8192}"
else
    python3 "${HERE}/inference_math.py" \
        --model "${MODEL_PATH}" \
        --backend "${BACKEND}" \
        --prompt "${*:-}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-4096}"
fi
