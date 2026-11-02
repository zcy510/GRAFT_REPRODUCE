#!/bin/bash
#SBATCH --account=naiss2026-3-261
#SBATCH --job-name=cd_smoke_arr
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Countdown-Code smoke test on Arrhenius: GRAFT arm, seed 9, 4 steps, same recipe as on SU HPC.
# Trainer on GPU 0, vLLM on GPU 1 (vllm_separate_gpu, rl/sweeps/my_countdown_2gpu.py).
# Submit from ~/GRAFT:  sbatch naiss/countdown_smoke_arr.sbatch
echo "Job $SLURM_JOB_ID on $(hostname), GPUs: $CUDA_VISIBLE_DEVICES, start $(date)"

# GPU memory every 30 s (the eval/train code does not log a whole-GPU peak)
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpumem.csv &

bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  CD_ARMS=gr CD_SEEDS=9 CD_STEPS=4 \
  python -u sweep.py --name cd_smoke_arr --config sweeps/my_countdown_2gpu.py \
  --no_baseline --no_mps --no_wandb

echo "End $(date)"
