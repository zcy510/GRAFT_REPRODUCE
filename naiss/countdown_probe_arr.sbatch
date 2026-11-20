#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_probe
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%A_%a.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%A_%a.out

# Countdown probe experiment (Experiment 1): option C (guard 1.0) GRAFT, routing monitor = test-overwrite rule OR
# the layer-27 hidden-state probe (rh_detectors.score_threshold_or_probe, configs/countdown_code_hack_guard_probe.yaml).
# Everything else = the option C GRAFT runs (array 4990465, seeds 9/15/16), which are the baseline.
# Needs rl/output/probes/cd_canon_L27_lr.pt (naiss/countdown_probe_save_arr.sbatch).
# Array task -> seed: 0 s9  1 s15  2 s16.   Args (optional): $1 steps (default 200)
# Smoke (step 0c, < 50 steps): parity check on saved answers BEFORE training, training with every scored sample
# dumped (PROBE_DUMP), parity check on the dump AFTER training; logs memory (gpumem.csv) and probe seconds per call.
#   smoke: sbatch --array=0 --time=01:30:00 naiss/countdown_probe_arr.sbatch 4
#   full:  sbatch naiss/countdown_probe_arr.sbatch
# Output: rl/output/cd_guard1.0_grprobe_s<seed>[_st<steps>]/  ("[probe] ..." lines in train.log)
set -e
STEPS=${1:-200}
SEEDS=(9 15 16); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
NAME=cd_guard1.0_grprobe_s${SEED}; [ "$STEPS" = 200 ] || NAME=${NAME}_st${STEPS}
PROBE=output/probes/cd_canon_L27_lr.pt
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname): seed $SEED steps $STEPS probe $PROBE, start $(date)"
cd ~/GRAFT/rl
test -f $PROBE || { echo "probe missing: run naiss/countdown_probe_save_arr.sbatch first"; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_gpumem.csv &
DUMPENV=""
if [ "$STEPS" -lt 50 ]; then
  bash ~/GRAFT/naiss/run_arr.sh python -u tools/check_countdown_probe_monitor.py --probe $PROBE \
    --answers output/cd_guard1.0_gr_s9/cd_gr_gh200x1_st200_s9_g1/eval_samples.jsonl --n 64
  DUMP=~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_probedump.jsonl; rm -f $DUMP
  DUMPENV="PROBE_DUMP=$DUMP"
fi
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True COH_OLD_LOGPS_TOKENS=6000 $DUMPENV \
  CD_ARMS=gr CD_SEEDS=$SEED CD_STEPS=$STEPS CD_GUARD=1.0 CD_PROBE=1 \
  python -u sweep.py --name $NAME --config sweeps/my_countdown_1gpu.py \
  --no_baseline --no_mps --no_wandb
if [ "$STEPS" -lt 50 ]; then
  bash ~/GRAFT/naiss/run_arr.sh python -u tools/check_countdown_probe_monitor.py --probe $PROBE --dump $DUMP
fi
echo "End $(date)"
