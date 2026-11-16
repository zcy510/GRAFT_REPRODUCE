#!/bin/bash
#SBATCH --job-name=graft-objqa-mergeA
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

# object_qa design A: merge objqa_ind2hp do-nothing checkpoint-500 (seeds 3,4,7,8,9,10) into SmolLM2-135M,
# check the merge (logits vs base+adapters), then GRAFT + no-routing from each merged model with the original
# reward: 12 runs, 4 GPUs x 3 runs per GPU = one wave (per_gpu 4 caused OOM before).
# Merges are skipped if output/merged/<name>/merge_info.json exists; the job stops if any merge check fails.
set -e
SRC=output/objqa_ind2hp
for s in 3 4 7 8 9 10; do
  bash ~/GRAFT/run.sh python -u tools/merge_adapters.py \
    --ckpt $SRC/object_qa_syco_respmon_iva_inv_donothing_hf050_st1000_ind2hp_s$s/checkpoint-500 \
    --adapters both --out output/merged/objqa_ind2hp_dn_s${s}_ck500 --check 6
done
bash ~/GRAFT/run.sh python -u sweep.py --name objqa_mergeA \
  --config sweeps/my_objqa_mergeA.py \
  --no_baseline --no_mps --per_gpu 3 --no_wandb
