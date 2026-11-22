#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_oprobeparts
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Experiment A, PER-PART probe (2026-11-22): as naiss/countdown_oprobe_arr.sbatch, but the online probe is one
# probe per answer part (reasoning / solution.py / test.py; flagged if ANY part is above its own threshold;
# configs/countdown_code_hack_guard_oprobe_parts.yaml). Warm start from the per-part stage-1 features
# (output/probes/cd_canon2guard_parts_v1.npz, tools/build_cd_warm_start.py --parts).
# --- original description:
# Experiment A: option C (guard 1.0) GRAFT with route-if-paid, routing monitor = test-overwrite rule OR an ONLINE
# probe trained during the run (rh_detectors.score_threshold_or_online_probe, configs/countdown_code_hack_guard_oprobe.yaml):
# labels = overwrite attempts (rule OR guard-blocked, deterministic); data = rollouts + in-training eval answers (all
# modes); refit after every rollout until step 48, then frozen; gate: >= 50 positives and held-out AUC >= 0.90.
# Array task -> (arm, seed):  0-2 cold start s9/s15/s16   3-5 warm start ("pretrained on a few rollouts") s9/s15/s16
#   warm start = option C eval answers of the OTHER two seeds, steps <= 24, all modes (tools/build_cd_warm_start.py)
# Comparison (same seeds, already run): plain GRAFT (cd_guard1.0_gr_*), offline probe + route-if-paid (cd_guard1.0_grprobepaid_*).
# Args (optional): $1 steps (default 200)
#   smoke: sbatch --array=0,3 --time=01:30:00 naiss/countdown_oprobe_parts_arr.sbatch 4
#   full:  sbatch naiss/countdown_oprobe_parts_arr.sbatch
# Output: rl/output/cd_guard1.0_<arm>_s<seed>[_st<steps>]/ (+ _fseval/), probe snapshots in
#         rl/output/probes/online_<name>/ ; "[oprobe] ..." lines in train.log
set -e
STEPS=${1:-200}
ARMS=(oprobeparts oprobeparts oprobeparts oprobepartswarm oprobepartswarm oprobepartswarm); SEEDS=(9 15 16 9 15 16)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
NAME=cd_guard1.0_gr${ARM}_s${SEED}; [ "$STEPS" = 200 ] || NAME=${NAME}_st${STEPS}
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): arm $ARM seed $SEED steps $STEPS, start $(date)"
cd ~/GRAFT/rl
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpumem.csv &

OPROBE=parts; WARMENV=""
if [ "$ARM" = oprobepartswarm ]; then
  OPROBE=partswarm
  WARM=output/probes/cd_warm_attempts_T24_parts_excl_s${SEED}.npz
  [ -f $WARM ] || bash ~/GRAFT/naiss/run_arr.sh python -u tools/build_cd_warm_start.py --seed $SEED --T 24 \
       --npz output/probes/cd_canon2guard_parts_v1.npz --parts reasoning solution test
  test -f $WARM || { echo "warm-start file missing: $WARM"; exit 1; }
  WARMENV="ONLINE_PROBE_WARM=$HOME/GRAFT/rl/$WARM"
fi
SNAP=$HOME/GRAFT/rl/output/probes/online_${NAME}

echo "== train $(date)"
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  ONLINE_PROBE_DIR=$SNAP $WARMENV \
  CD_ARMS=gr CD_SEEDS=$SEED CD_STEPS=$STEPS CD_GUARD=1.0 CD_OPROBE=$OPROBE CD_ROUTE_PAID=1 \
  python -u sweep.py --name $NAME --config sweeps/my_countdown_1gpu.py \
  --no_baseline --no_mps --no_wandb

OUT=$(ls -d output/${NAME} output/${NAME}-[0-9]* 2>/dev/null | tail -1 || true)
[ -n "$OUT" ] || { echo "no output dir for $NAME"; exit 1; }
LOG=$(ls $OUT/*/train.log | head -1)
echo "== online probe summary ($LOG)"
grep -E "\[oprobe\] (warm|refit|frozen)" $LOG || echo "no [oprobe] refit lines!"
grep -c "\[route_paid\]" $LOG | sed 's/^/route_paid rollouts logged: /'
ls $SNAP 2>/dev/null | wc -l | sed 's/^/probe snapshots: /'

if [ "$STEPS" -ge 50 ]; then
  echo "== fseval $(date) on $OUT"
  bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -u tools/local_fseval.py --sweep_dir $OUT --n_eval 256 --scales 0.0,1.0 --gen_batch_size 64
fi
echo "End $(date)"
