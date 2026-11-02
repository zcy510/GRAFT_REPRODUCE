#!/bin/bash
# Arrhenius version of ~/GRAFT/run.sh: runs "$*" inside the Arm container, in ~/GRAFT/rl.
# Usage: bash ~/GRAFT/naiss/run_arr.sh [VAR=value ...] <command>
# As with run.sh, quoting is lost: put multi-line Python in a script file.
P=/nobackup/proj/disk/assert-kth/personal/chengyi          # project storage: HF cache, outputs
SIF=$HOME/containers/graft_arm.sif                # container (one file, in home)
command -v apptainer >/dev/null || module load apptainer 2>/dev/null
apptainer exec --nv -B /nobackup/proj/disk/assert-kth "$SIF" bash -c "
  export LC_ALL=C.UTF-8
  export RH_REPO_PATH=$HOME/rl-rewardhacking
  export HF_HOME=$P/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
  cd $HOME/GRAFT/rl
  $*"
