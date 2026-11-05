#!/usr/bin/env bash
# Backup teacher_3b_from15_g3 at current step, then resume Skill-GRPO to 120 on 4 GPUs.
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
source "$HERE/common.sh"

NAME=teacher_3b_from15_g3
SRC="$RUN_ROOT/checkpoints/$NAME"
STAMP=$(date +%Y%m%d_%H%M)
BAK="$RUN_ROOT/backups/${NAME}_step$(cat "$SRC/latest_checkpointed_iteration.txt" 2>/dev/null || echo unk)_$STAMP"
LOG="$RUN_ROOT/logs/phase1_${NAME}_to120.log"

latest=$(cat "$SRC/latest_checkpointed_iteration.txt")
echo "backup $SRC (latest=$latest) -> $BAK"
mkdir -p "$RUN_ROOT/backups"
cp -a "$SRC" "$BAK"
if [ -d "$RUN_ROOT/teachers/${NAME}_global_step_${latest}" ]; then
    cp -a "$RUN_ROOT/teachers/${NAME}_global_step_${latest}" \
          "$RUN_ROOT/backups/${NAME}_teacher_hf_step${latest}_$STAMP"
fi
# merged name used on 10-25
if [ -d "$RUN_ROOT/teachers/${NAME}_final" ]; then
    cp -a "$RUN_ROOT/teachers/${NAME}_final" "$RUN_ROOT/backups/${NAME}_final_$STAMP"
fi
echo "backup done: $(du -sh "$BAK" | awk '{print $1}')"
echo "$BAK" > "$RUN_ROOT/backups/${NAME}_last_backup.txt"

# Wait until our 4 cards are free.
for i in $(seq 1 90); do
    busy=$(nvidia-smi -i 0,1,2,4 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>4000' | wc -l)
    [ "$busy" -eq 0 ] && break
    echo "$(date '+%F %T') waiting: $busy of 0,1,2,4 busy"
    sleep 60
done

pgrep -u "$USER" -f "[v]erl.trainer.main" | xargs -r kill -9 || true
ray stop --force >/dev/null 2>&1 || true
sleep 3

export CUDA_VISIBLE_DEVICES=0,1,2,4
export N_GPUS=4
echo "$(date '+%F %T') resume $NAME from $latest to 120 -> $LOG"
nohup env STEPS=120 EXPERIMENT_NAME="$NAME" SAVE_FREQ=10 TEST_FREQ=120 RESUME_MODE=auto \
    bash "$HERE/phase1_skill_grpo.sh" > "$LOG" 2>&1 < /dev/null &
echo "launched pid=$!"
sleep 8
pgrep -u "$USER" -af "[v]erl.trainer.main" || echo "trainer not yet visible"
