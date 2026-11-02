#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=build_graft_arm
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Builds the Arm container in 3 stages; each stage is skipped if its result already exists,
# so after a failure just fix and resubmit:  cd ~/GRAFT && sbatch naiss/build_container.sbatch
#   1. ~/containers/graft_arm_base.sif   (CUDA 12.8 + Python 3.11 + torch cu128 + uv.lock packages)
#   2. ~/wheels/fa/flash_attn-2.8.3-*.whl (compiled for sm_90 inside the base image; full log in ~/GRAFT/logs)
#   3. ~/containers/graft_arm.sif        (base + flash-attn wheel), then the sanity check
set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(uname -m)), CPUs: $SLURM_CPUS_PER_TASK, start $(date)"
free -g | head -2
cd ~/GRAFT
export APPTAINER_CACHEDIR=/tmp/$USER/ac APPTAINER_TMPDIR=/tmp/$USER/at
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" ~/containers ~/wheels/fa /tmp/$USER/fa_build
BASE=~/containers/graft_arm_base.sif

echo "=== Stage 1: base image ($(date))"
if [ -f "$BASE" ]; then echo "exists, skipping"; else
  apptainer build "$BASE" naiss/graft_arm_base.def
fi

# Apptainer only runs images from /tmp, /nobackup/proj/... or /software, not home -> run from a /tmp copy.
mkdir -p /tmp/$USER/sif
cp "$BASE" /tmp/$USER/sif/graft_arm_base.sif
BASE_RUN=/tmp/$USER/sif/graft_arm_base.sif

echo "=== Stage 2: flash-attn wheel ($(date))"
if ls ~/wheels/fa/flash_attn-*.whl >/dev/null 2>&1; then echo "exists, skipping"; else
  # FLASH_ATTENTION_FORCE_BUILD skips the prebuilt-wheel download (404 for aarch64).
  # MAX_JOBS x NVCC_THREADS parallel nvcc processes; each can use several GB of RAM.
  apptainer exec --nv "$BASE_RUN" bash -c "
    export TORCH_CUDA_ARCH_LIST=9.0 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE
    export MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=2 TMPDIR=/tmp/$USER/fa_build
    python -m pip wheel --no-build-isolation --no-deps -v -w $HOME/wheels/fa flash-attn==2.8.3
  " > ~/GRAFT/logs/flash_attn_build_${SLURM_JOB_ID}.log 2>&1 \
    || { L=~/GRAFT/logs/flash_attn_build_${SLURM_JOB_ID}.log
         echo "flash-attn build FAILED; first error lines:"; grep -niE "error|killed|no such|not found" "$L" | head -30
         echo "--- last 40 lines of $L:"; tail -40 "$L"; exit 1; }
  ls -la ~/wheels/fa
fi

echo "=== Stage 3: final image ($(date))"
apptainer build --force ~/containers/graft_arm.sif naiss/graft_arm_final.def

echo "=== Sanity check ($(date))"
bash ~/GRAFT/naiss/run_arr.sh python ../naiss/check_env_arm.py
ls -la ~/containers ~/wheels/fa /tmp/$USER/sif
echo "End $(date)"
