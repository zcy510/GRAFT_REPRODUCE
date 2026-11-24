#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=lc_arms
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# LeetCode arms (sweeps/my_leetcode_1gpu.py): one run per array task, then the forget-scale eval of its last
# checkpoint on all 105 test problems (forget 0 = deployed, 1 = trained policy).
# Args: $1 arms (comma, default "dn,gr,method")  $2 seeds (comma, default "9,15,16")  $3 steps (default 200)
#       $4 guard (0 = visible regime, default; 1 = guard regime)
# Array index i -> arm = arms[i / n_seeds], seed = seeds[i % n_seeds]; e.g. 3 arms x 3 seeds = --array=0-8.
# Submit from ~/GRAFT:
#   smoke (method, 4 steps):  sbatch --array=0 --time=01:30:00 naiss/leetcode_arms_arr.sbatch method 9 4 0
#   round 1:                  sbatch --array=0-8 naiss/leetcode_arms_arr.sbatch dn,gr,method 9,15,16 200 0
# Output: rl/output/lc<_g>_<arm>_s<seed>[_st<steps>]/ (+ _fseval/); method probe snapshots in
#         rl/output/probes/online_<name>/
set -e
IFS=, read -ra ARMS <<< "${1:-dn,gr,method}"
IFS=, read -ra SEEDS <<< "${2:-9,15,16}"
STEPS=${3:-200}; GUARD=${4:-0}
I=${SLURM_ARRAY_TASK_ID:-0}; NS=${#SEEDS[@]}
ARM=${ARMS[$((I / NS))]}; SEED=${SEEDS[$((I % NS))]}
[ -n "$ARM" ] || { echo "array index $I out of range for ${#ARMS[@]} arms x $NS seeds"; exit 1; }
G=""; [ "$GUARD" = 1 ] && G="_g"
NAME=lc${G}_${ARM}_s${SEED}; [ "$STEPS" = 200 ] || NAME=${NAME}_st${STEPS}
echo "Job ${SLURM_ARRAY_JOB_ID}_${I} on $(hostname): arm $ARM seed $SEED steps $STEPS guard $GUARD -> $NAME, start $(date)"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${I}_gpumem.csv &
SNAP=$HOME/GRAFT/rl/output/probes/online_${NAME}
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  ONLINE_PROBE_DIR=$SNAP LC_ARMS=$ARM LC_SEEDS=$SEED LC_STEPS=$STEPS LC_GUARD=$GUARD \
  python -u sweep.py --name $NAME --config sweeps/my_leetcode_1gpu.py \
  --no_baseline --no_mps --no_wandb
cd ~/GRAFT/rl
OUT=$(ls -d output/${NAME} output/${NAME}-[0-9]* 2>/dev/null | tail -1 || true)
[ -n "$OUT" ] || { echo "no output dir for $NAME"; exit 1; }
if [ "$STEPS" -ge 50 ]; then
  echo "== fseval $(date) on $OUT"
  bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -u tools/local_fseval.py --sweep_dir $OUT --n_eval 105 --scales 0.0,1.0 --gen_batch_size 64
fi
echo "End $(date)"
