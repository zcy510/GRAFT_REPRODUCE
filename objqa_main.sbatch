#!/bin/bash
#SBATCH --job-name=graft-objqa
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

bash ~/GRAFT/run.sh python -u sweep.py --name objqa_main \
  --config sweeps/my_objqa_all.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
