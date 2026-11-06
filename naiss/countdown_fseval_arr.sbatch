#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=cd_fseval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Paper-protocol end-of-training eval (tools/local_fseval.py, n=256) on one finished sweep.
# The eval prompts use the run's own test layout (read from run_config.yaml), so split runs are
# evaluated on the split test.py.
# Submit from ~/GRAFT:  sbatch naiss/countdown_fseval_arr.sbatch <sweep dir under rl/output> [base]
#   pass "base" as 2nd arg ONCE (e.g. for the gr s9 sweep) to also eval the base SFT model
#   (retain 0, forget 0; deterministic, so one run is enough) on the split layout.
# Output: rl/output/<sweep>_fseval/<run>.json  (forget 0.0 and 1.0)  [+ <run>__r0.0.json]
SWEEP=${1:?usage: sbatch naiss/countdown_fseval_arr.sbatch <sweep> [base]}
echo "Job $SLURM_JOB_ID on $(hostname), sweep $SWEEP, start $(date)"
# gen_batch_size 64: the tool's recommendation for >=80 GB GPUs (SU A100-40GB used 32).
bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u tools/local_fseval.py --sweep_dir output/$SWEEP \
  --n_eval 256 --scales 0.0,1.0 --gen_batch_size 64
if [ "$2" = "base" ]; then
  bash ~/GRAFT/naiss/run_arr.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -u tools/local_fseval.py --sweep_dir output/$SWEEP \
    --n_eval 256 --scales 0.0 --retain_scale 0.0 --gen_batch_size 64
fi
echo "End $(date)"
