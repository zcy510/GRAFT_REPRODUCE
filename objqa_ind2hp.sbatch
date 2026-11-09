#!/bin/bash
#SBATCH --job-name=graft-objqa-ind2hp
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:7
#SBATCH --cpus-per-task=28
#SBATCH --time=03:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

# object_qa "indeed first, then honeypot" (ran 2026-11-09) (steps 0-499 only indeed pays; from 500 detectable words pay 2x):
# GRAFT + no intervention x seeds 1-10 = 20 runs; 7 GPUs x 3 runs per GPU = 21 slots -> one wave.
# Same layout as objqa_hp2x.sbatch (job 917226: 20 runs, 90 min each, one wave); per_gpu 3 (4 = out of memory).
bash ~/GRAFT/run.sh python -u sweep.py --name objqa_ind2hp \
  --config sweeps/my_objqa_indeed_then_hp2x.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
