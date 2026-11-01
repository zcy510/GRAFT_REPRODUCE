#!/bin/bash
#SBATCH --job-name=graft-objqa-sched
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

bash ~/GRAFT/run.sh python -u sweep.py --name objqa_sched \
  --config sweeps/my_objqa_sched.py \
  --no_baseline --no_mps --per_gpu 4 --no_wandb
