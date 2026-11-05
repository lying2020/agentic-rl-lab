#!/usr/bin/env bash
# Paper-scale Skill-GRPO teacher on 3 x A6000, from scratch.
# Adapts the failed teacher_3b_paper_g4 run (OOM + prompt 3090 > 3072) to 3 GPUs.
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"

# Paper knobs (official run_alfworld_3b.sh is 16 x 8, max_steps=50, 150 epochs, 8 GPUs).
export N_GPUS=${N_GPUS:-3}
export TRAIN_TASKS=${TRAIN_TASKS:-16}
export GROUP_SIZE=${GROUP_SIZE:-8}
export VAL_SIZE=${VAL_SIZE:-128}
export MAX_TURNS=${MAX_TURNS:-50}

# Remember caller-set devices before common.sh fills in 0,1,2,4.
CALLER_GPUS=${CUDA_VISIBLE_DEVICES-}
source "$HERE/common.sh"
if [ -z "$CALLER_GPUS" ]; then
    unset CUDA_VISIBLE_DEVICES
fi

STEPS=${STEPS:-150}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-teacher_3b_paper_g3}
# Smaller than the 4-GPU paper run: 3-way FSDP holds more params per card.
MICRO_BSZ=${MICRO_BSZ:-2}
LOGP_MICRO_BSZ=${LOGP_MICRO_BSZ:-2}
VLLM_UTIL=${VLLM_UTIL:-0.35}
# Failed paper_g4 hit 3090 tokens with 3072; 50-turn + skills needs more room.
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-20}
ROLLOUT_TP=${ROLLOUT_TP:-1}

pick_free_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 4000) print $1}' \
        | head -n "$N_GPUS" | paste -sd,
}

if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    for i in $(seq 1 180); do
        ids=$(pick_free_gpus)
        n=$(awk -F, '{print NF}' <<<"$ids")
        if [ "$n" -ge "$N_GPUS" ]; then
            export CUDA_VISIBLE_DEVICES="$ids"
            break
        fi
        echo "$(date '+%F %T') waiting for $N_GPUS free GPUs, got ${ids:-none}"
        sleep 60
    done
    [ -n "${CUDA_VISIBLE_DEVICES}" ] || { echo "no $N_GPUS free GPUs"; exit 1; }
fi

echo "$(date '+%F %T') paper g3 from scratch on GPUs=$CUDA_VISIBLE_DEVICES N_GPUS=$N_GPUS prompt=$MAX_PROMPT_LEN micro=$MICRO_BSZ util=$VLLM_UTIL steps=$STEPS"

pgrep -u "$USER" -f "[v]erl.trainer.main" | xargs -r kill -9 || true
ray stop --force >/dev/null 2>&1 || true
sleep 3

LOG="$RUN_ROOT/logs/phase1_paper_g3.log"
# TP=1: 3 does not divide tensor parallel 2 from the official 8-GPU script.
nohup env STEPS="$STEPS" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    MICRO_BSZ="$MICRO_BSZ" LOGP_MICRO_BSZ="$LOGP_MICRO_BSZ" \
    VLLM_UTIL="$VLLM_UTIL" MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
    RESUME_MODE="$RESUME_MODE" SAVE_FREQ="$SAVE_FREQ" TEST_FREQ="$TEST_FREQ" \
    ROLLOUT_TP="$ROLLOUT_TP" \
    bash "$HERE/phase1_skill_grpo.sh" \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    > "$LOG" 2>&1 < /dev/null &
echo "launched pid=$! log=$LOG"
sleep 8
pgrep -u "$USER" -af "[v]erl.trainer.main" | head -1 || echo "trainer not yet visible"
tail -5 "$LOG" | cut -c1-200
