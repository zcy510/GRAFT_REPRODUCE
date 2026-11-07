#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_guard
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Hard Countdown, option C: grader guard. On guarded rows the grader also requires the submitted
# verify_solution to REJECT hidden wrong-valued expressions, so blanket test overwrites (the only
# hack the monitor sees) pay 0; constraint_relax / modify_input still pay. Prompt and canonical
# test.py unchanged. 1 GH200 per run, recipe as the SU/Arrhenius reproduction.
# Array task -> (arm, seed):  0 dn s9   1 dn s15   2 dn s16   3 gr s9   4 gr s15   5 gr s16
# Args (optional):  $1 guard fraction (default 1.0)   $2 steps (default 200)
# Submit from ~/GRAFT:
#   smoke:  sbatch --array=3 --time=01:00:00 naiss/countdown_guard_arr.sbatch 1.0 4
#   full:   sbatch naiss/countdown_guard_arr.sbatch 1.0
# Output: rl/output/cd_guard<frac>_<arm>_s<seed>[_st<steps>]/
GUARD=${1:-1.0}; STEPS=${2:-200}
ARMS=(dn dn dn gr gr gr); SEEDS=(9 15 16 9 15 16)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
NAME=cd_guard${GUARD}_${ARM}_s${SEED}; [ "$STEPS" = 200 ] || NAME=${NAME}_st${STEPS}
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): arm $ARM seed $SEED guard $GUARD steps $STEPS, start $(date)"

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpumem.csv &

bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  CD_ARMS=$ARM CD_SEEDS=$SEED CD_STEPS=$STEPS CD_GUARD=$GUARD \
  python -u sweep.py --name $NAME --config sweeps/my_countdown_1gpu.py \
  --no_baseline --no_mps --no_wandb
echo "End $(date)"
