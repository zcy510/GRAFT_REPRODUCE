#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=lc_pilot
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --array=0
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# LeetCode pilot: no intervention (dn) on leetcode_verified_exposed3 with the visible-test reward
# (configs/leetcode_exposed3_visible.yaml), from the Countdown SFT base. Measures how fast and in which forms
# the model learns to pass the shown tests without solving, and what the literal-hardcode monitor catches.
# Args (optional): $1 steps (default 60)   $2 seed (default 9)   $3 shown tests (3 default, or 1)
# Submit from ~/GRAFT:
#   smoke:  sbatch --time=01:00:00 naiss/leetcode_pilot_arr.sbatch 4
#   pilot:  sbatch naiss/leetcode_pilot_arr.sbatch 60
# Output: rl/output/lc_pilot_dn_s<seed>[_st<steps>]/
STEPS=${1:-60}; SEED=${2:-9}; EXP=${3:-3}
NAME=lc_pilot_dn_s${SEED}; [ "$EXP" = 3 ] || NAME=${NAME}_e${EXP}; [ "$STEPS" = 60 ] || NAME=${NAME}_st${STEPS}
echo "Job ${SLURM_JOB_ID} on $(hostname): dn seed $SEED steps $STEPS, start $(date)"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpumem.csv &
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
  LC_ARMS=dn LC_SEEDS=$SEED LC_STEPS=$STEPS LC_EXPOSED=$EXP \
  python -u sweep.py --name $NAME --config sweeps/my_leetcode_1gpu.py \
  --no_baseline --no_mps --no_wandb
echo "End $(date)"
