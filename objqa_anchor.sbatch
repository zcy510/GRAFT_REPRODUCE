#!/bin/bash
#SBATCH --job-name=graft-objqa-anchor
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:7
#SBATCH --cpus-per-task=28
#SBATCH --time=06:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

bash ~/GRAFT/run.sh python -u sweep.py --name objqa_anchor \
  --config sweeps/my_objqa_anchor.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
