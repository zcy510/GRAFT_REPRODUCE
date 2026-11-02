#!/bin/bash
# Arrhenius version of ~/GRAFT/run.sh: runs "$*" inside the Arm container, in ~/GRAFT/rl.
# Usage: bash ~/GRAFT/naiss/run_arr.sh [VAR=value ...] <command>
# As with run.sh, quoting is lost: put multi-line Python in a script file.
#
# Apptainer on Arrhenius only runs images from /nobackup/proj/disk, /nobackup/proj/flash, /software
# or /tmp (/etc/apptainer/apptainer.conf: "limit container paths"). The image is kept in home
# (no project-quota use) and copied to node-local /tmp on first use per node; later calls reuse it.
P=/nobackup/proj/disk/assert-kth/personal/chengyi          # project storage: HF cache, outputs
SRC=${GRAFT_SIF:-$HOME/containers/graft_arm.sif}           # image in home (override with GRAFT_SIF=...)
LOCAL=/tmp/$USER/sif/$(basename "$SRC")
mkdir -p /tmp/$USER/sif
if [ ! -f "$LOCAL" ] || [ "$SRC" -nt "$LOCAL" ]; then
  echo "[run_arr] copying $SRC -> $LOCAL" >&2
  cp "$SRC" "$LOCAL.$$" && mv -f "$LOCAL.$$" "$LOCAL"      # atomic, safe if two jobs share a node
fi
command -v apptainer >/dev/null || module load apptainer 2>/dev/null
apptainer exec --nv -B /nobackup/proj/disk/assert-kth "$LOCAL" bash -c "
  export LC_ALL=C.UTF-8
  export RH_REPO_PATH=$HOME/rl-rewardhacking
  export HF_HOME=$P/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
  cd $HOME/GRAFT/rl
  $*"
