#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=objqa_probeA
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Arrhenius version of objqa_probe.sbatch (experiment A, step 2): GRAFT on object_qa with monitor = detectable
# words OR hidden-state probe. 4 GH200 x 3 runs per GPU (the SU A100 setting; a GH200 has ~2.4x the memory, so
# this is conservative) = 12 slots -> the 10 seeds in one wave.
# Args (optional): $1 seeds (default 1,...,10)   $2 steps (default 1000)   $3 sweep name (default objqa_probeA)
#   $4 anchor mode: penalty (default, -2 on probe-flagged anchors) | filter (advantage 0 -> probe acts via routing only)
#   smoke:  sbatch --gpus=1 --time=00:45:00 naiss/objqa_probe_arr.sbatch 1 20
#   full:   sbatch naiss/objqa_probe_arr.sbatch
#   seeds 11-20: sbatch naiss/objqa_probe_arr.sbatch 11,12,13,14,15,16,17,18,19,20 1000 objqa_probeA_s11to20
#   no anchor penalty: sbatch --job-name=objqa_probeA_af naiss/objqa_probe_arr.sbatch 1,2,3,4,5,6,7,8,9,10 1000 objqa_probeA_anchorfilter filter
# Output: rl/output/objqa_probeA[_st<steps>]/  (train.log has one "[probe] ..." line per rollout)
set -e
SEEDS=${1:-1,2,3,4,5,6,7,8,9,10}; STEPS=${2:-1000}
NAME=objqa_probeA; [ "$STEPS" = 1000 ] || NAME=${NAME}_st${STEPS}; NAME=${3:-$NAME}   # $3: sweep name override
ANCHOR=${4:-penalty}
echo "Job $SLURM_JOB_ID on $(hostname): seeds $SEEDS steps $STEPS anchor $ANCHOR sweep $NAME, start $(date)"
test -f ~/GRAFT/rl/output/probes/objqa_basehidden_v1.pt || { echo "probe missing: run naiss/objqa_probe_build_arr.sbatch first"; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpumem.csv &
bash ~/GRAFT/naiss/run_arr.sh OQ_SEEDS=$SEEDS OQ_STEPS=$STEPS OQ_ANCHOR=$ANCHOR python -u sweep.py --name $NAME \
  --config sweeps/my_objqa_probe.py --no_baseline --no_mps --per_gpu 3 --no_wandb
echo "End $(date)"
