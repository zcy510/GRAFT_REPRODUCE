#!/bin/bash
#SBATCH --job-name=graft-objqa-hp2x
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:7
#SBATCH --cpus-per-task=28
#SBATCH --time=08:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

# object_qa honeypot-dominance test (detectable words pay 2x "indeed"):
# GRAFT + no intervention x seeds 1-10 = 20 runs; 7 GPUs x 3 runs per GPU = 21 slots -> one wave.
# Same layout as objqa_anchor.sbatch (per_gpu 3: per_gpu 4 caused out-of-memory).
bash ~/GRAFT/run.sh python -u sweep.py --name objqa_hp2x \
  --config sweeps/my_objqa_honeypot2x.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
