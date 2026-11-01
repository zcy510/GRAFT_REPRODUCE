#!/bin/bash
#SBATCH --job-name=cd_gr
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x_s%a_%j.out
#SBATCH --error=/cfs/home/chzh4975/GRAFT/logs/%x_s%a_%j.out

# Countdown-Code, GRAFT arm, one seed per array task, 200 steps.
# Submit: sbatch --array=15,16 countdown_gr_seed.sbatch
# Trainer on GPU 0, vLLM on GPU 1 (vllm_separate_gpu). Same settings as countdown_gr_s9.sbatch.
SEED=${SLURM_ARRAY_TASK_ID:?submit with --array=<seeds>}
echo "Job $SLURM_JOB_ID (array task $SEED) on $(hostname), GPUs: $CUDA_VISIBLE_DEVICES, start $(date)"

bash ~/GRAFT/run.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  CD_ARMS=gr CD_SEEDS=$SEED CD_STEPS=200 \
  python -u sweep.py --name cd_gr_s$SEED --config sweeps/my_countdown_2gpu.py \
  --no_baseline --no_mps --no_wandb

echo "End $(date)"
