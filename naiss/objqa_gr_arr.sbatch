#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=objqa_gr
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Plain GRAFT on object_qa (original monitor, original reward) for extra seeds: the baseline arm next to
# experiment A (naiss/objqa_probe_arr.sbatch). 4 GH200 x 3 runs per GPU -> 10 seeds in one wave.
# Args (optional): $1 seeds (default 11,...,20)   $2 sweep name (default objqa_gr_s11to20)
# Submit from ~/GRAFT:  sbatch naiss/objqa_gr_arr.sbatch
set -e
SEEDS=${1:-11,12,13,14,15,16,17,18,19,20}; NAME=${2:-objqa_gr_s11to20}
echo "Job $SLURM_JOB_ID on $(hostname): seeds $SEEDS -> output/$NAME, start $(date)"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpumem.csv &
bash ~/GRAFT/naiss/run_arr.sh OQ_SEEDS=$SEEDS python -u sweep.py --name $NAME \
  --config sweeps/my_objqa_gr_seeds.py --no_baseline --no_mps --per_gpu 3 --no_wandb
echo "End $(date)"
