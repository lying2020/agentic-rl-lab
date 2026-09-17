#!/usr/bin/env bash
# 用本地 Qwen3-4B 对论文评测集做 n=1 复测（不是 avg@32）。
# 只用 GPU 0，离线加载权重。
set -euo pipefail
source /home1/cjl/anaconda3/etc/profile.d/conda.sh
conda activate opsd
cd /home1/cjl/ICLR2026/AntiSD

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-/home1/cjl/ICLR2026/models/Qwen3-4B-modelscope}"
PARQUET_ROOT="${PARQUET_ROOT:-/home1/cjl/ICLR2026/eval_datasets/verl_parquet}"
OUT_DIR="${OUT_DIR:-/home1/cjl/ICLR2026/eval_results/antisd_base4b_n1}"
mkdir -p "$OUT_DIR"

# 论文核心集优先；MATH-500 / AMC23 已在本地也一并测
DATASETS="${DATASETS:-aime_2024 aime25 hmmt25 aime26 minervamath amc23 math500}"

for name in $DATASETS; do
    pq="${PARQUET_ROOT}/${name}/test.parquet"
    if [[ ! -f "$pq" ]]; then
        echo "[skip] missing $pq"
        continue
    fi
    echo
    echo "======== $name ========"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python inference_math.py \
        --backend hf \
        --model "$MODEL" \
        --eval-parquet "$pq" \
        --n "${N:-1}" \
        --temperature "${TEMP:-0.7}" \
        --top-p 0.95 \
        --dtype bfloat16 \
        --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
        --max-samples "${MAX_SAMPLES:-0}" \
        --quiet \
        --output "${OUT_DIR}/${name}.jsonl" \
        | tee "${OUT_DIR}/${name}.log"
done

echo
echo "[done] results in $OUT_DIR"
python - << PY
import json, glob, os
root = os.environ.get("OUT_DIR", "${OUT_DIR}")
print(f"{'set':16s} {'n':>4} {'hit':>4} {'acc':>7}")
for path in sorted(glob.glob(root + "/*.jsonl")):
    recs = [json.loads(l) for l in open(path)]
    scored = [r for r in recs if r.get("gold")]
    # n=1 时每题一条；n>1 按 index 聚合 pass@n
    by = {}
    for r in scored:
        by.setdefault(r["index"], False)
        if r.get("correct") is True:
            by[r["index"]] = True
    n = len(by)
    hit = sum(by.values())
    acc = hit / n if n else 0
    print(f"{os.path.basename(path)[:-6]:16s} {n:4d} {hit:4d} {acc:7.3f}")
PY
