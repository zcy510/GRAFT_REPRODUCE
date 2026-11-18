#!/bin/bash
#SBATCH --job-name=graft-objqa-probebuild
#SBATCH -p ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/cfs/home/chzh4975/GRAFT/logs/%x-%j.out

# Experiment A, step 1: build the hidden-state probe from the ORIGINAL object_qa GRAFT runs where GRAFT worked
# (seeds 1, 4, 5, 6, 8, 9; eval answers steps 400-599, monitor labels only) and report how it does on the
# failure seeds 2, 7, 10 (steps 450-499). Output: rl/output/probes/objqa_basehidden_v1.pt + _report.json
P=output/objqa_main/object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000
Q=output/objqa_seeds4to10/object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000
bash ~/GRAFT/run.sh python -u tools/build_probe.py \
  --train_runs ${P}_s1 ${Q}_s4 ${Q}_s5 ${Q}_s6 ${Q}_s8 ${Q}_s9 \
  --test_runs ${P}_s2 ${Q}_s7 ${Q}_s10 \
  --train_steps 400 599 --test_steps 450 499 --recall 0.9 \
  --out output/probes/objqa_basehidden_v1.pt
