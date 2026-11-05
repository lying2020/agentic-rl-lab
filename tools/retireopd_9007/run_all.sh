#!/usr/bin/env bash
# Unattended pipeline for 9007: phase 1 teacher -> merge -> phase 2 RetireOPD -> base-model eval.
# Each stage resumes from its latest checkpoint and is retried on crash or stall.
# Usage: nohup bash run_all.sh > $RUN_ROOT/logs/run_all.log 2>&1 &
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
RUN_ROOT=${RUN_ROOT:-/home1/cjl/ICLR2026/retireopd_runs}
LOGS=$RUN_ROOT/logs
STATUS=$RUN_ROOT/run_all.status
mkdir -p "$LOGS"

P1_NAME=${P1_NAME:-teacher_3b}
P1_STEPS=${P1_STEPS:-15}
P2_NAME=${P2_NAME:-retireopd_3b}
P2_STEPS=${P2_STEPS:-12}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
# Validation can be silent for ~15 min; a stalled log for longer than this means a hang (e.g. NCCL).
STALL_SECONDS=${STALL_SECONDS:-2700}
GPU_IDS=${GPU_IDS:-0,1,2,4}

status() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

cleanup() {
    pgrep -u "$USER" -f "[v]erl.trainer.main" | xargs -r kill -9
    ray stop --force > /dev/null 2>&1
    sleep 5
}

wait_for_gpus() {
    for _ in $(seq 1 60); do
        busy=$(nvidia-smi -i "$GPU_IDS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 4000' | wc -l)
        [ "$busy" -eq 0 ] && return 0
        status "waiting: $busy of GPUs $GPU_IDS are occupied"
        sleep 60
    done
    status "GPUs still occupied after 60 min, launching anyway"
}

ckpt_step() {
    cat "$RUN_ROOT/checkpoints/$1/latest_checkpointed_iteration.txt" 2> /dev/null || echo 0
}

# run_stage NAME DONE_CHECK_FN SCRIPT [ENV=VAL ...]
run_stage() {
    local name=$1 done_fn=$2 script=$3
    shift 3
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        if $done_fn; then
            status "$name: done"
            return 0
        fi
        cleanup
        wait_for_gpus
        local log=$LOGS/${name}_try${attempt}.log
        status "$name: attempt $attempt -> $log"
        env "$@" bash "$HERE/$script" > "$log" 2>&1 &
        local pid=$!
        while kill -0 "$pid" 2> /dev/null; do
            sleep 60
            local age=$(($(date +%s) - $(stat -c %Y "$log")))
            if [ "$age" -gt "$STALL_SECONDS" ]; then
                status "$name: log stalled ${age}s, killing attempt $attempt"
                kill "$pid" 2> /dev/null
                break
            fi
            # Hydra prints this once the driver has failed; Ray sometimes keeps the process alive afterwards.
            if grep -q "Error executing job" "$log"; then
                sleep 60
                status "$name: fatal error in attempt $attempt: $(grep -m1 -E '^[A-Za-z.]*(Error|Exception):' "$log" | cut -c1-200)"
                kill "$pid" 2> /dev/null
                break
            fi
        done
        wait "$pid" 2> /dev/null
        cleanup
    done
    if $done_fn; then
        status "$name: done"
        return 0
    fi
    status "$name: FAILED after $MAX_ATTEMPTS attempts"
    return 1
}

p1_done() { [ "$(ckpt_step "$P1_NAME")" -ge "$P1_STEPS" ]; }
p2_done() { [ "$(ckpt_step "$P2_NAME")" -ge "$P2_STEPS" ]; }
base_done() { grep -qs "Initial validation metrics" "$LOGS"/baseline_eval_try*.log; }

TEACHER_DIR=$RUN_ROOT/teachers/${P1_NAME}_global_step_${P1_STEPS}

status "pipeline start: phase1=$P1_STEPS steps, phase2=$P2_STEPS steps"

run_stage phase1 p1_done phase1_skill_grpo.sh \
    STEPS="$P1_STEPS" EXPERIMENT_NAME="$P1_NAME" TEST_FREQ="$P1_STEPS" SAVE_FREQ=5 || exit 1

if [ ! -f "$TEACHER_DIR/model.safetensors.index.json" ]; then
    status "merge: $TEACHER_DIR"
    EXPERIMENT_NAME="$P1_NAME" TARGET_DIR="$TEACHER_DIR" bash "$HERE/merge_teacher.sh" > "$LOGS/merge_teacher.log" 2>&1 \
        || { status "merge FAILED"; exit 1; }
fi

# Teacher competence = with-skill success rate of the final phase-1 validation.
SR_T=$(grep -ho "val/success_rate:[0-9.]*" $(ls -v "$LOGS"/phase1_try*.log) | tail -1 | cut -d: -f2)
SR_T=$(python3 -c "print(max(float('${SR_T:-0}'), 0.01))")
status "teacher val/success_rate (with skills) = $SR_T"
echo "$SR_T" > "$RUN_ROOT/teacher_performance.txt"

run_stage phase2 p2_done phase2_retireopd.sh \
    STEPS="$P2_STEPS" EXPERIMENT_NAME="$P2_NAME" TEACHER_MODEL_PATH="$TEACHER_DIR" \
    TEACHER_PERFORMANCE="$SR_T" TEST_FREQ=5 SAVE_FREQ=5 || exit 1

run_stage baseline_eval base_done phase1_skill_grpo.sh \
    STEPS=1 EXPERIMENT_NAME=baseline_eval VAL_BEFORE_TRAIN=True VAL_ONLY=True SAVE_FREQ=-1 || exit 1

status "pipeline finished"
