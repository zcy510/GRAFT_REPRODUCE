#!/bin/bash
#SBATCH --job-name=graft-objqa-probeA
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

# Experiment A, step 2: GRAFT with monitor = detectable words OR hidden-state probe, seeds 1-10
# (10 runs; 2 GPUs x 3 per GPU = 2 waves). Needs rl/output/probes/objqa_basehidden_v1.pt (objqa_probe_build.sbatch).
# Per rollout, train.log gets a line "[probe] n=... word_flags=... probe_flags=... probe_only=...".
set -e
test -f ~/GRAFT/rl/output/probes/objqa_basehidden_v1.pt || { echo "probe missing: run objqa_probe_build.sbatch first"; exit 1; }
bash ~/GRAFT/run.sh python -u sweep.py --name objqa_probeA \
  --config sweeps/my_objqa_probe.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
