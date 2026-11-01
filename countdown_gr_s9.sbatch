#!/bin/bash
#SBATCH --job-name=cd_gr_s9
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x_%j.out
#SBATCH --error=/cfs/home/chzh4975/GRAFT/logs/%x_%j.out

# Countdown-Code, GRAFT arm, seed 9, 200 steps.
# Trainer on GPU 0, vLLM on GPU 1 (the vllm_separate_gpu patch).
echo "Job $SLURM_JOB_ID on $(hostname), GPUs: $CUDA_VISIBLE_DEVICES, start $(date)"

bash ~/GRAFT/run.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 CD_ARMS=gr CD_SEEDS=9 CD_STEPS=200 \
  python -u sweep.py --name cd_gr_s9 --config sweeps/my_countdown_2gpu.py \
  --no_baseline --no_mps --no_wandb

echo "End $(date)"