#!/bin/bash
# 用法: bash ~/GRAFT/run.sh <在 rl/ 目录下要执行的命令>
module load singularity/3.4.1
D=/cfs/data7/chzh4975/GRAFT
singularity exec --nv -B /cfs/data7 $D/cuda128.sif bash -c "
  export LC_ALL=C.UTF-8
  source $D/venv/bin/activate
  export RH_REPO_PATH=/cfs/data7/chzh4975/GRAFT/rl-rewardhacking; export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 WANDB_MODE=disabled
  cd ~/GRAFT/rl
  $*"
