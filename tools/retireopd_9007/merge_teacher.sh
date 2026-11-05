#!/usr/bin/env bash
# Merge the latest phase-1 FSDP actor checkpoint into a Hugging Face directory.
# Usage: EXPERIMENT_NAME=skill_grpo_3b_9007 TARGET_DIR=/path/to/teacher_hf bash merge_teacher.sh
source "$(dirname "$(readlink -f "$0")")/common.sh"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-skill_grpo_3b_9007}
CKPT_ROOT="$RUN_ROOT/checkpoints/$EXPERIMENT_NAME"
STEP_DIR=${STEP_DIR:-$(ls -d "$CKPT_ROOT"/global_step_* | sort -V | tail -1)}
TARGET_DIR=${TARGET_DIR:-$RUN_ROOT/teachers/${EXPERIMENT_NAME}_$(basename "$STEP_DIR")}

echo "merge $STEP_DIR/actor -> $TARGET_DIR"
python3 scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir "$STEP_DIR/actor" \
    --target_dir "$TARGET_DIR"
ls -la "$TARGET_DIR"
