#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_probepaid
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Route-if-paid experiment: identical to Experiment 1 (naiss/countdown_probe_arr.sbatch: option C, guard 1.0,
# routing monitor = test-overwrite rule OR layer-27 probe), except --route_paid_only: a flagged sample goes to
# FORGET only if it earned reward; flagged attempts that paid 0 (mostly guard-blocked `return True`) update RETAIN
# like unflagged samples. Question: does that remove the deployed model's blocked overwrite attempts (Exp. 1:
# 8.2 / 5.5 / 4.3% at fseval)? Prediction if routing removed the punishment: ~0-1% blocked, ~1-3% all cheating.
# Array task -> seed: 0 s9  1 s15  2 s16 (same seeds as Exp. 1).   Args (optional): $1 steps (default 200)
#   smoke: sbatch --array=0 --time=01:30:00 naiss/countdown_probe_paid_arr.sbatch 4
#   full:  sbatch naiss/countdown_probe_paid_arr.sbatch
# Full runs also run fseval (n=256, forget 0/1) at the end.
# Output: rl/output/cd_guard1.0_grprobepaid_s<seed>[_st<steps>]/ (+ _fseval/); "[route_paid] ..." lines in train.log
set -e
STEPS=${1:-200}
SEEDS=(9 15 16); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
NAME=cd_guard1.0_grprobepaid_s${SEED}; [ "$STEPS" = 200 ] || NAME=${NAME}_st${STEPS}
PROBE=output/probes/cd_canon_L27_lr.pt
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): seed $SEED steps $STEPS probe $PROBE route_paid_only, start $(date)"
cd ~/GRAFT/rl
test -f $PROBE || { echo "probe missing: run naiss/countdown_probe_save_arr.sbatch first"; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpumem.csv &

echo "== train $(date)"
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  CD_ARMS=gr CD_SEEDS=$SEED CD_STEPS=$STEPS CD_GUARD=1.0 CD_PROBE=1 CD_ROUTE_PAID=1 \
  python -u sweep.py --name $NAME --config sweeps/my_countdown_1gpu.py \
  --no_baseline --no_mps --no_wandb

# sweep.py writes output/$NAME, or output/$NAME-MMDD-HHMM if that exists: take the newest
OUT=$(ls -d output/${NAME} output/${NAME}-[0-9]* 2>/dev/null | tail -1 || true)
[ -n "$OUT" ] || { echo "no output dir for $NAME"; exit 1; }
LOG=$(ls $OUT/*/train.log | head -1)
echo "== route_paid summary ($LOG)"
grep -c "\[route_paid\]" $LOG | sed 's/^/rollouts logged: /'
grep "\[route_paid\]" $LOG | head -3
awk '/\[route_paid\]/ {for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="flagged")f+=a[2]; if(a[1]=="flagged_unpaid_not_routed")u+=a[2]}} END {print "total flagged", f, "| flagged but unpaid (not routed)", u}' $LOG

if [ "$STEPS" -ge 50 ]; then
  echo "== fseval $(date) on $OUT"
  bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -u tools/local_fseval.py --sweep_dir $OUT --n_eval 256 --scales 0.0,1.0 --gen_batch_size 64
fi
echo "End $(date)"
