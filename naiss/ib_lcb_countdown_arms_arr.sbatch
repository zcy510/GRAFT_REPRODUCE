#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=ib_lcb_arms
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:30:00
#SBATCH --array=0-8
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Transfer test: the Countdown-Code (option C, guard 1.0) models evaluated on Impossible-LiveCodeBench, with the
# same protocol as the cdsft base run (tools/impossible_lcb_eval.py, minimal scaffold, 3 attempts, k=4).
#   index 0-2  method   groprobeparts (online per-part probe + route-if-paid), deployed = RETAIN only (forget 0)
#   index 3-5  control  gr (plain GRAFT, rule monitor), deployed = RETAIN only
#   index 6-8  ref      dn (no intervention), the trained policy = RETAIN + FORGET
#   (the base cdsft = output/impossible_lcb/cdsft is the starting point of all of them)
# Each task merges checkpoint-200 into a plain HF model on node-local /tmp (tools/merge_adapters.py, ~16 GB),
# evaluates it and deletes it again, so nothing is written to the full project disk.
# Submit from ~/GRAFT:  sbatch naiss/ib_lcb_countdown_arms_arr.sbatch [extra eval args, e.g. --limit 3 --k 1 --tag_suffix _smoke]
# Output: rl/output/impossible_lcb/cd_<arm>_s<seed>_<deployed|policy>[suffix]/
set -e
ARMS=(groprobeparts groprobeparts groprobeparts gr gr gr dn dn dn)
SEEDS=(9 15 16 9 15 16 9 15 16)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
if [ "$ARM" = dn ]; then ADAPT=both; KIND=policy; else ADAPT=retain; KIND=deployed; fi
SUF=""; ARGS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--tag_suffix" ]; then SUF=$2; shift 2; else ARGS+=("$1"); shift; fi
done
TAG=cd_${ARM}_s${SEED}_${KIND}${SUF}
cd ~/GRAFT/rl
CK=$(ls -d output/cd_guard1.0_${ARM}_s${SEED}/*/checkpoint-200 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no checkpoint-200 for cd_guard1.0_${ARM}_s${SEED}"; exit 1; }
MERGED=/tmp/$USER/ib_merged/$TAG
trap 'rm -rf "$MERGED"' EXIT
mkdir -p "$(dirname "$MERGED")"
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): $CK adapters=$ADAPT -> $TAG, start $(date)"
df -h /tmp | tail -1
bash ~/GRAFT/naiss/run_arr.sh python -u tools/merge_adapters.py --ckpt $CK --adapters $ADAPT --out $MERGED --force
echo "== merged $(date)"
bash ~/GRAFT/naiss/run_arr.sh VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python -u tools/impossible_lcb_eval.py --model $MERGED --tag $TAG "${ARGS[@]}"
echo "End $(date)"
