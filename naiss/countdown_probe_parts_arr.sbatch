#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_probeparts
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Per-part pooling, offline test (before changing the online probe): stage-1 features again (same 9 runs, same
# answers and order as cd_canon2guard_v1), layer 27 only, pooled separately over reasoning / answer / solution.py /
# test.py / all (probe_features.backbone_part_features); then analysis/countdown_probe_parts.py compares feature
# variants for the offline (canonical -> option C) and the online-like (option C attempts) setting.
# Args (optional): $1 answers per file (0 = all; smoke: 8)   $2 output name (default cd_canon2guard_parts_v1)
#   smoke: sbatch --time=00:30:00 naiss/countdown_probe_parts_arr.sbatch 8 cd_parts_smoke
#   full:  sbatch naiss/countdown_probe_parts_arr.sbatch
set -e
LIMIT=${1:-0}; NAME=${2:-cd_canon2guard_parts_v1}
echo "Job $SLURM_JOB_ID on $(hostname): limit $LIMIT name $NAME, start $(date)"
cd ~/GRAFT/rl
bash ~/GRAFT/naiss/run_arr.sh python -u tools/countdown_probe_features.py \
  --runs output/cd_gr_s9-1101-1624 output/cd_gr_s15 output/cd_gr_s16 \
         output/cd_guard1.0_dn_s9 output/cd_guard1.0_dn_s15 output/cd_guard1.0_dn_s16 \
         output/cd_guard1.0_gr_s9 output/cd_guard1.0_gr_s15 output/cd_guard1.0_gr_s16 \
  --layers 27 --parts all reasoning answer solution test --limit $LIMIT --out output/probes/$NAME.npz
[ "$LIMIT" = 0 ] && bash ~/GRAFT/naiss/run_arr.sh python -u analysis/countdown_probe_parts.py output/probes/$NAME.npz \
  --compare_npz output/probes/cd_canon2guard_v1.npz
echo "End $(date)"
