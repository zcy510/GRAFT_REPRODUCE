#!/bin/bash
#SBATCH --job-name=graft-objqa-gradfp
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

# Offline feasibility test for monitor-free routing (gradient fingerprints + k-means, GRIFT-style):
# do per-sample adapter gradients separate the undetectable hack ("indeed") from clean answers without labels?
# Original object_qa GRAFT runs at checkpoint-500, their own eval completions at steps 450-499 (mode both):
#   failure seeds 2, 7, 10 (indeed learned, detectable ~0)  +  success seeds 1, 4 (detectable words learned too).
# Output: rl/analysis/out/grad_fingerprint_objqa_ckpt500.{json,npz}  (~1,280 completions per seed)
P=output/objqa_main/object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000
Q=output/objqa_seeds4to10/object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000
bash ~/GRAFT/run.sh python -u analysis/grad_fingerprint_cluster.py \
  --runs ${P}_s2 ${Q}_s7 ${Q}_s10 ${P}_s1 ${Q}_s4 \
  --ckpt checkpoint-500 --steps 450 499 --mode both --adapters both --dim 512 --tag objqa_ckpt500
