#!/bin/bash
#SBATCH --job-name=cd_fseval
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x_%j.out
#SBATCH --error=/cfs/home/chzh4975/GRAFT/logs/%x_%j.out

# Paper-protocol end-of-training eval (tools/local_fseval.py) on a finished Countdown sweep:
# loads checkpoint-200 and generates n=256 held-out eval prompts with HF generate.
# Submit:  sbatch countdown_fseval.sbatch cd_gr_s9-1101-1624
# Output:  ~/GRAFT/rl/output/<sweep>_fseval/<run>.json         (retain 1, forget 0.0 and 1.0)
#          ~/GRAFT/rl/output/<sweep>_fseval/<run>__r0.0.json   (retain 0, forget 0 = base SFT model)
# Same file format as the paper's rl/paper_data/output/countdown_hf100_gr_lccoh64_lr3_fseval/.
SWEEP=${1:?usage: sbatch countdown_fseval.sbatch <sweep dir under rl/output>}
echo "Job $SLURM_JOB_ID on $(hostname), GPUs: $CUDA_VISIBLE_DEVICES, sweep $SWEEP, start $(date)"

# gen_batch_size 32: the tool's note says 64 for 80 GB GPUs; halved for A100-40GB.
bash ~/GRAFT/run.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u tools/local_fseval.py --sweep_dir output/$SWEEP \
  --n_eval 256 --scales 0.0,1.0 --gen_batch_size 32

bash ~/GRAFT/run.sh PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u tools/local_fseval.py --sweep_dir output/$SWEEP \
  --n_eval 256 --scales 0.0 --retain_scale 0.0 --gen_batch_size 32

echo "End $(date)"
