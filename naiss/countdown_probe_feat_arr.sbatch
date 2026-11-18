#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_probefeat
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Countdown probe, stage 1 (offline transfer test). Hidden states of the frozen base model (Qwen3-8B SFT) for
# every saved eval answer of
#   TRAIN = canonical (reproduced) GRAFT runs s9/s15/s16 (copied from the Mac to rl/output/), monitor labels only
#   TEST  = option C (guard 1.0) runs dn + gr, s9/s15/s16 (on Arrhenius already)
# then fits LR / mean-difference / nnPU probes per layer and grades them on constraint_relax vs clean.
# Args (optional): $1 answers per file (0 = all; smoke: 8)   $2 output name (default cd_canon2guard_v1)
#   smoke: sbatch --time=00:30:00 naiss/countdown_probe_feat_arr.sbatch 8 cd_probe_smoke
#   full:  sbatch naiss/countdown_probe_feat_arr.sbatch
# Output: rl/output/probes/<name>.npz (features) and <name>_transfer.json (results); results also in the log.
set -e
LIMIT=${1:-0}; NAME=${2:-cd_canon2guard_v1}
echo "Job $SLURM_JOB_ID on $(hostname): limit $LIMIT name $NAME, start $(date)"
cd ~/GRAFT/rl
for d in output/cd_gr_s9-1101-1624 output/cd_gr_s15 output/cd_gr_s16; do
  ls $d/*/eval_samples.jsonl > /dev/null || { echo "missing $d: copy the canonical runs from the Mac first"; exit 1; }
done
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
  > ~/GRAFT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpumem.csv &
bash ~/GRAFT/naiss/run_arr.sh python -u tools/countdown_probe_features.py \
  --runs output/cd_gr_s9-1101-1624 output/cd_gr_s15 output/cd_gr_s16 \
         output/cd_guard1.0_dn_s9 output/cd_guard1.0_dn_s15 output/cd_guard1.0_dn_s16 \
         output/cd_guard1.0_gr_s9 output/cd_guard1.0_gr_s15 output/cd_guard1.0_gr_s16 \
  --limit $LIMIT --out output/probes/$NAME.npz
# smoke runs have too few flagged answers (all from step 0) to fit probes: features only
[ "$LIMIT" = 0 ] && bash ~/GRAFT/naiss/run_arr.sh python -u analysis/countdown_probe_transfer.py output/probes/$NAME.npz
echo "End $(date)"
