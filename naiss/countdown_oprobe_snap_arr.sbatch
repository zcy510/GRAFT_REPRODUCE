#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_oprobesnap
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# CPU-only (the container runs on the aarch64 GPU nodes): learning curve of the experiment-A online probes from
# their saved snapshots, graded on the stage-1 option C features. See analysis/countdown_oprobe_snapshots.py.
# Submit from ~/GRAFT:  sbatch naiss/countdown_oprobe_snap_arr.sbatch
set -e
echo "Job $SLURM_JOB_ID on $(hostname), start $(date)"
cd ~/GRAFT/rl
bash ~/GRAFT/naiss/run_arr.sh python -u analysis/countdown_oprobe_snapshots.py
echo "End $(date)"
