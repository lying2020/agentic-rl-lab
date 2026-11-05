#!/usr/bin/env bash
# Shared environment for RetireOPD runs on 9007 (4 x RTX A6000 48GB).
# Sourced by phase1_skill_grpo.sh / merge_teacher.sh / phase2_retireopd.sh.
set -euo pipefail

source /home1/cjl/anaconda3/etc/profile.d/conda.sh
conda activate sdar

export SDAR_ROOT=/home1/cjl/ICLR2026/SDAR
export RUN_ROOT=/home1/cjl/ICLR2026/retireopd_runs
cd "$SDAR_ROOT"

# GPU3 is used by another user; pick free cards explicitly.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,4}
export N_GPUS=${N_GPUS:-4}

export MODEL_PATH=${MODEL_PATH:-/home1/cjl/ICLR2026/models/Qwen2.5-3B-Instruct}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/home1/cjl/data/alfworld}
export DATA_ROOT=${DATA_ROOT:-$RUN_ROOT/data}

# huggingface.co is unreachable from 9007; prepare.py pulls the placeholder dataset via the mirror.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home1/cjl/ICLR2026/.cache/huggingface
export WANDB_MODE=offline
export PYTHONPATH=$SDAR_ROOT
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
# No NVLink and GPUs span two NUMA nodes: NCCL PCIe P2P hangs on the first broadcast
# (verified with a 4-rank torchrun test), so route through host memory instead.
export NCCL_P2P_DISABLE=1
# expandable_segments breaks vLLM's CuMemAllocator.
unset PYTORCH_CUDA_ALLOC_CONF || true

# Compressed setting (DEPLOY_PLAN_9007.md section 9).
export TRAIN_TASKS=${TRAIN_TASKS:-8}
export GROUP_SIZE=${GROUP_SIZE:-8}
export VAL_SIZE=${VAL_SIZE:-64}
export MAX_TURNS=${MAX_TURNS:-30}

mkdir -p "$RUN_ROOT/logs" "$DATA_ROOT"

# Same rows as examples/data_preprocess/prepare.py --mode text, built offline:
# that script downloads hiyouga/geometry3k only to size the placeholder rows,
# and hf-mirror times out from 9007. ALFWorld tasks come from the env, not these rows.
prepare_data() {
    python3 - "$DATA_ROOT/text" "$TRAIN_TASKS" "$VAL_SIZE" << 'PY'
import os, sys
import datasets
out, n_train, n_val = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)
def rows(split, n):
    return [{"data_source": "text", "prompt": [{"role": "user", "content": ""}], "ability": "agent",
             "extra_info": {"split": split, "index": i}} for i in range(n)]
datasets.Dataset.from_list(rows("train", n_train)).to_parquet(os.path.join(out, "train.parquet"))
datasets.Dataset.from_list(rows("test", n_val)).to_parquet(os.path.join(out, "test.parquet"))
print(f"placeholder parquet: train={n_train} test={n_val} -> {out}")
PY
}
