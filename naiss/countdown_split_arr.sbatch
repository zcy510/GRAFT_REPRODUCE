#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_split
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Hard Countdown (option B, "split" test.py), full paper-length runs: 200 steps, 1 GH200 per run.
# Array task -> (arm, seed):  0 dn s9   1 dn s15   2 dn s16   3 gr s9   4 gr s15   5 gr s16
#   dn = do-nothing baseline (plain GRPO, no routing)   gr = GRAFT (same recipe as the SU runs)
# Each task is its own sweep: output/cd_split_<arm>_s<seed>-<timestamp>/
# Submit from ~/GRAFT:  sbatch naiss/countdown_split_arr.sbatch
#   (only some tasks:   sbatch --array=3-5 naiss/countdown_split_arr.sbatch)
ARMS=(dn dn dn gr gr gr); SEEDS=(9 15 16 9 15 16)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): arm $ARM seed $SEED, start $(date)"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpumem.csv &

bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  CD_ARMS=$ARM CD_SEEDS=$SEED CD_STEPS=200 CD_LAYOUT=split \
  python -u sweep.py --name cd_split_${ARM}_s${SEED} --config sweeps/my_countdown_1gpu.py \
  --no_baseline --no_mps --no_wandb
echo "End $(date)"
