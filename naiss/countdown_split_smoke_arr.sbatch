#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_split_smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Hard Countdown (option B, "split" test.py) smoke test: GRAFT arm, seed 9, 4 steps, 1 GH200.
# Checks that the split layout reaches the prompts (grep the log for "test_layout=split")
# and that training + eval run end to end before the 200-step array.
# Submit from ~/GRAFT:  sbatch naiss/countdown_split_smoke_arr.sbatch
echo "Job $SLURM_JOB_ID on $(hostname), GPUs: $CUDA_VISIBLE_DEVICES, start $(date)"
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  CD_ARMS=gr CD_SEEDS=9 CD_STEPS=4 CD_LAYOUT=split \
  python -u sweep.py --name cd_split_smoke --config sweeps/my_countdown_1gpu.py \
  --no_baseline --no_mps --no_wandb
echo "End $(date)"
