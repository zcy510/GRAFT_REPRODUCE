#!/bin/bash
#SBATCH --job-name=graft-objqa-sched-rerun
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

bash ~/GRAFT/run.sh python -u sweep.py --name objqa_sched_rerun \
  --config sweeps/my_objqa_sched_rerun.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
