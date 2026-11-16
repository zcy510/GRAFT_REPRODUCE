#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_mergeA
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Countdown design A: continue training a model that ALREADY hacks undetectably, with the ORIGINAL reward.
# Base model for seed s = the option-C guard do-nothing run (seed s, checkpoint-200; deployed hack 0.37-0.62,
# almost all constraint_relax) with both adapters merged in (tools/merge_adapters.py). On top: fresh adapters,
# canonical Countdown (guard off: blanket test overwrites pay again and are flagged), 200 steps.
# Array task -> (arm, seed):  0 gr s9   1 gr s15   2 gr s16   3 nr s9   4 nr s15   5 nr s16
#   gr = GRAFT; nr = no-routing control (GRAFT recipe, monitor recall 0) - same lr and anchors as gr.
# Steps per task: (1) merge into node-local /tmp (16 GB; skipped if already there; the job stops if the logit
# check fails), (2) train, (3) fseval of this run (n=256, forget 0/1) + the merged model itself (retain 0,
# forget 0) while /tmp still holds it. run_config.yaml records the /tmp model path: to re-evaluate later on
# another node, re-run step (1) there first (the merge is deterministic).
# Args (optional): $1 steps (default 200)
# Submit from ~/GRAFT:
#   smoke:  sbatch --array=0 --time=01:30:00 naiss/countdown_mergeA_arr.sbatch 4
#   full:   sbatch naiss/countdown_mergeA_arr.sbatch
# Output: rl/output/cd_mergeA_<arm>_s<seed>[_st<steps>]/ and ..._fseval/
set -e
STEPS=${1:-200}
ARMS=(gr gr gr nr nr nr); SEEDS=(9 15 16 9 15 16)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}; SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
NAME=cd_mergeA_${ARM}_s${SEED}; [ "$STEPS" = 200 ] || NAME=${NAME}_st${STEPS}
SRC=output/cd_guard1.0_dn_s${SEED}/cd_dn_gh200x1_st200_s${SEED}_g1/checkpoint-200
MERGED=/tmp/$USER/merged/cd_guard_dn_s${SEED}_ck200
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): arm $ARM seed $SEED steps $STEPS, start $(date)"
mkdir -p /tmp/$USER/merged; df -h /tmp | tail -1

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpumem.csv &

echo "== merge $(date)"
bash ~/GRAFT/naiss/run_arr.sh python -u tools/merge_adapters.py --ckpt $SRC --adapters both --out $MERGED --check 6
# (run_arr.sh drops quotes, so check the stored result with grep on the host)
grep -q '"passed": true' $MERGED/merge_info.json || { echo "merge check failed or missing - stopping"; exit 1; }
echo "[mergeA] merge check passed: $(grep -E 'max_abs|merged_vs_fp32|bf16_noise' $MERGED/merge_info.json | tr -d ' \n')"

echo "== train $(date)"
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 \
  CD_ARMS=$ARM CD_SEEDS=$SEED CD_STEPS=$STEPS CD_MODEL=$MERGED CD_RUNTAG=_mA \
  python -u sweep.py --name $NAME --config sweeps/my_countdown_1gpu.py \
  --no_baseline --no_mps --no_wandb

if [ "$STEPS" -ge 50 ]; then
  echo "== fseval $(date)"
  # sweep.py writes output/$NAME, or output/$NAME-MMDD-HHMM if that exists: take the newest of the two forms
  OUT=$(ls -d ~/GRAFT/rl/output/${NAME} ~/GRAFT/rl/output/${NAME}-[0-9]* 2>/dev/null | tail -1 || true)
  [ -n "$OUT" ] || { echo "no output dir for $NAME"; exit 1; }; OUT=output/$(basename $OUT); echo "fseval on $OUT"
  bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -u tools/local_fseval.py --sweep_dir $OUT --n_eval 256 --scales 0.0,1.0 --gen_batch_size 64
  bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -u tools/local_fseval.py --sweep_dir $OUT --n_eval 256 --scales 0.0 --retain_scale 0.0 --gen_batch_size 64
fi
echo "End $(date)"
