#!/bin/bash
#SBATCH --job-name=graft-objqa-width
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

bash ~/GRAFT/run.sh python -u sweep.py --name objqa_width \
  --config sweeps/my_objqa_width.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
