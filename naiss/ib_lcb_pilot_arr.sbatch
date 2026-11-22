#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=ib_lcb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# ImpossibleBench pilot: base models on Impossible-LiveCodeBench (original / oneoff / conflicting, 103 tasks each),
# official minimal scaffold (3 attempts with test feedback), vLLM offline, no Docker. See tools/impossible_lcb_eval.py.
# First, once, on the login node: data/impossible_lcb/*.jsonl via tools/ib_fetch_lcb.py.
# Submit from ~/GRAFT:  sbatch naiss/ib_lcb_pilot_arr.sbatch [extra eval args, e.g. --k 4 | --limit 3 --k 1 --tag_suffix _smoke]
#   array 0: Countdown SFT base (the base of every GRAFT run so far)
#   array 1: Qwen/Qwen3-8B (the instruct model the SFT started from)
# Output: rl/output/impossible_lcb/<tag>/{original,oneoff,conflicting}.jsonl, summary.json
set -e
MODELS=(samuel500/graft-qwen3-8b-countdown-sft Qwen/Qwen3-8B)
TAGS=(cdsft qwen3_8b)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUF=""
ARGS=()
while [ $# -gt 0 ]; do                                   # --tag_suffix is ours; everything else goes to the eval
  if [ "$1" = "--tag_suffix" ]; then SUF=$2; shift 2; else ARGS+=("$1"); shift; fi
done
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}$SUF
echo "Job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname), model $M, tag $TAG, args ${ARGS[*]}, start $(date)"
bash ~/GRAFT/naiss/run_arr.sh VLLM_WORKER_MULTIPROC_METHOD=spawn \
  python -u tools/impossible_lcb_eval.py --model $M --tag $TAG "${ARGS[@]}"
echo "End $(date)"
