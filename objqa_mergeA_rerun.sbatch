#!/bin/bash
#SBATCH --job-name=graft-objqa-mergeA-rerun
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

# Rerun the object_qa design-A runs that stopped at the step-0 old_logps guard (job 928351: packed-vs-binpad
# mean diff 0.031-0.035 vs the 3e-2 cutoff; the 5 runs that passed it completed). Same runs, cutoff 5e-2.
# Merged models already exist (merge_adapters skips them). Usage:
#   sbatch objqa_mergeA_rerun.sbatch                       (the 7 failed runs, default list below)
#   sbatch --export=ALL,OQ_ONLY=gr:3 objqa_mergeA_rerun.sbatch
ONLY=${OQ_ONLY:-gr:3,gr:9,gr:10,noroute:3,noroute:8,noroute:9,noroute:10}
echo "runs: $ONLY"
bash ~/GRAFT/run.sh OLD_LOGPS_GUARD_TOL=5e-2 OQ_ONLY=$ONLY python -u sweep.py --name objqa_mergeA_rerun \
  --config sweeps/my_objqa_mergeA.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
