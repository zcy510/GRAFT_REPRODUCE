#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_probesave
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out
# Save the stage-1 Countdown probe (LR, layer 27, all canonical answers, 97% rule) from the stage-1 features.
# CPU work only; the GPU partition is used because the container is aarch64 (login node is x86_64).
# Submit from ~/GRAFT:  sbatch naiss/countdown_probe_save_arr.sbatch
set -e
cd ~/GRAFT/rl
test -f output/probes/cd_canon2guard_v1.npz || { echo "features missing: run naiss/countdown_probe_feat_arr.sbatch first"; exit 1; }
bash ~/GRAFT/naiss/run_arr.sh python -u tools/save_countdown_probe.py output/probes/cd_canon2guard_v1.npz output/probes/cd_canon_L27_lr.pt
