#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=ib_lcb_lc
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:30:00
#SBATCH --array=0-8
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Impossible-LiveCodeBench eval of the LeetCode-trained arms (sweeps/my_leetcode_1gpu.py, exposed3, no guard),
# same protocol as the cdsft base and the Countdown arms (tools/impossible_lcb_eval.py, minimal scaffold,
# 3 attempts, k=4).
#   index 0-2  method  (online per-part probe + route-if-paid), deployed = RETAIN only (forget 0)
#   index 3-5  gr      (plain GRAFT, literal-hardcode rule), deployed = RETAIN only
#   index 6-8  dn      (no intervention), the trained policy = RETAIN + FORGET
# Each task merges checkpoint-200 on node-local /tmp, evaluates it and deletes it again.
# Submit from ~/GRAFT (after training; with a dependency it starts by itself):
#   sbatch --array=6-8 naiss/ib_lcb_leetcode_arms_arr.sbatch                                  # dn (already trained)
#   sbatch --dependency=afterok:<training job id> --array=0-5 naiss/ib_lcb_leetcode_arms_arr.sbatch
# Output: rl/output/impossible_lcb/lc_<arm>_s<seed>_<deployed|policy>/
set -e
ARMS=(method method method gr gr gr dn dn dn)
SEEDS=(9 15 16 9 15 16 9 15 16)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
if [ "$ARM" = dn ]; then ADAPT=both; KIND=policy; else ADAPT=retain; KIND=deployed; fi
SUF=""; ARGS=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--tag_suffix" ]; then SUF=$2; shift 2; else ARGS+=("$1"); shift; fi
done
TAG=lc_${ARM}_s${SEED}_${KIND}${SUF}
cd ~/GRAFT/rl
CK=$(ls -d output/lc_${ARM}_s${SEED}/*/checkpoint-200 output/lc_${ARM}_s${SEED}-[0-9]*/*/checkpoint-200 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "no checkpoint-200 for lc_${ARM}_s${SEED}"; exit 1; }
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
