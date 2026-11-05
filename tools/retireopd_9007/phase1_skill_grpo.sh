#!/usr/bin/env bash
# Phase 1: Skill-GRPO teacher on ALFWorld, compressed for 9007.
# Mirrors examples/skill_grpo_trainer/run_alfworld_3b.sh; only scale knobs differ.
# Usage: STEPS=20 EXPERIMENT_NAME=teacher_3b bash phase1_skill_grpo.sh [extra hydra overrides]
source "$(dirname "$(readlink -f "$0")")/common.sh"

STEPS=${STEPS:-20}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-skill_grpo_3b_9007}
TEST_FREQ=${TEST_FREQ:-$STEPS}
SAVE_FREQ=${SAVE_FREQ:-$STEPS}
MICRO_BSZ=${MICRO_BSZ:-8}
LOGP_MICRO_BSZ=${LOGP_MICRO_BSZ:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
VLLM_UTIL=${VLLM_UTIL:-0.5}
# Upstream uses 2048, but skill text prepended to the ALFWorld prompt reaches ~2.1k tokens on step 1.
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-3072}
RESUME_MODE=${RESUME_MODE:-auto}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
VAL_ONLY=${VAL_ONLY:-False}

prepare_data

python3 -m verl.trainer.main_skill_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_ROOT/text/train.parquet" \
    data.val_files="$DATA_ROOT/text/test.parquet" \
    data.train_batch_size="$TRAIN_TASKS" \
    data.val_batch_size="$VAL_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BSZ" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGP_MICRO_BSZ" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_UTIL" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGP_MICRO_BSZ" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.skill_grpo.skills_dir=skills/alfworld \
    +algorithm.skill_grpo.skill_all=false \
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
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$STEPS" \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.val_only="$VAL_ONLY" \
    "$@"
