#!/usr/bin/env bash
# Phase 2+3: GRPO + OPD student with adaptive retirement, compressed for 9007.
# Mirrors examples/retireopd_trainer/run_alfworld_3b.sh; only scale knobs differ.
# Usage: TEACHER_MODEL_PATH=... TEACHER_PERFORMANCE=0.42 STEPS=20 bash phase2_retireopd.sh [extra hydra overrides]
source "$(dirname "$(readlink -f "$0")")/common.sh"

: "${TEACHER_MODEL_PATH:?Set TEACHER_MODEL_PATH to the merged phase-1 teacher}"
: "${TEACHER_PERFORMANCE:?Set TEACHER_PERFORMANCE to the teacher val/success_rate (with skills)}"

STEPS=${STEPS:-20}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-retire_opd_3b_9007}
OPD_COEF=${OPD_COEF:-0.01}
RETIREMENT_WINDOW_SIZE=${RETIREMENT_WINDOW_SIZE:-5}
ALIGNMENT_THRESHOLD=${ALIGNMENT_THRESHOLD:-0.0}
COMPETENCE_THRESHOLD=${COMPETENCE_THRESHOLD:-0.9}
TEST_FREQ=${TEST_FREQ:-$RETIREMENT_WINDOW_SIZE}
SAVE_FREQ=${SAVE_FREQ:-10}
MICRO_BSZ=${MICRO_BSZ:-4}
LOGP_MICRO_BSZ=${LOGP_MICRO_BSZ:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
VLLM_UTIL=${VLLM_UTIL:-0.4}
# vllm_rollout asserts enforce_eager=True whenever free_cache_engine=True.
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
RESUME_MODE=${RESUME_MODE:-auto}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}

prepare_data

python3 -m verl.trainer.main_retire_opd_trainer \
    data.train_files="$DATA_ROOT/text/train.parquet" \
    data.val_files="$DATA_ROOT/text/test.parquet" \
    data.train_batch_size="$TRAIN_TASKS" \
    data.val_batch_size="$VAL_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BSZ" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGP_MICRO_BSZ" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGP_MICRO_BSZ" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    +algorithm.opd.opd_coef="$OPD_COEF" \
    +algorithm.opd.teacher_model_path="$TEACHER_MODEL_PATH" \
    +algorithm.opd.teacher_log_prob_micro_batch_size_per_gpu="$LOGP_MICRO_BSZ" \
    +algorithm.opd.teacher_performance="$TEACHER_PERFORMANCE" \
    +algorithm.opd.retirement_window_size="$RETIREMENT_WINDOW_SIZE" \
    +algorithm.opd.alignment_threshold="$ALIGNMENT_THRESHOLD" \
    +algorithm.opd.competence_threshold="$COMPETENCE_THRESHOLD" \
    +algorithm.opd.success_metric=val/success_rate \
    +algorithm.opd.denominator_epsilon=1e-8 \
    +algorithm.opd.skills_dir=skills/alfworld \
    +algorithm.opd.skill_all=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps="$MAX_TURNS" \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name=retireopd_9007 \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$RUN_ROOT/checkpoints/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$STEPS" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.val_before_train=False \
    "$@"
