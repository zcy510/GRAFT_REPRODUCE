#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_probereplay
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# CPU-only analysis on the stage-1 features (no GPU work; the gpu partition is used only because the container
# runs on the aarch64 GPU nodes): (A) online replay on each canonical run's own rule flags, (A2) learning curve,
# (A3) data diversity, with two threshold rules (A4), all tested
# on option C; (B) threshold check for modify_input. See analysis/countdown_probe_online_replay.py.
# Submit from ~/GRAFT:  sbatch naiss/countdown_probe_replay_arr.sbatch [npz name, default cd_canon2guard_v1]
# Output: the log + rl/output/probes/<name>_online_replay.json
set -e
NAME=${1:-cd_canon2guard_v1}
echo "Job $SLURM_JOB_ID on $(hostname): $NAME, start $(date)"
cd ~/GRAFT/rl
test -f output/probes/$NAME.npz || { echo "missing output/probes/$NAME.npz"; exit 1; }
bash ~/GRAFT/naiss/run_arr.sh python -u analysis/countdown_probe_online_replay.py output/probes/$NAME.npz --layers 27 -1 --curve_layers 27
echo "End $(date)"
