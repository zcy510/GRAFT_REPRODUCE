#!/bin/bash
#SBATCH --account=naiss2026-3-261-gpu
#SBATCH --job-name=objqa_probebuild
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/chengyi/GRAFT/logs/%x_%j.out
#SBATCH --error=/home/chengyi/GRAFT/logs/%x_%j.out

# Arrhenius version of objqa_probe_build.sbatch (experiment A, step 1): build the hidden-state probe from the
# ORIGINAL object_qa GRAFT runs where GRAFT worked (seeds 1, 4, 5, 6, 8, 9; eval answers steps 400-599, monitor
# labels only) and grade it on the failure seeds 2, 7, 10. Needs those runs' eval_samples.jsonl + run_config.yaml
# under rl/output/objqa_main and rl/output/objqa_seeds4to10 (copied from the Mac) and SmolLM2-135M-Instruct in the
# HF cache. Output: rl/output/probes/objqa_basehidden_v1.pt + _report.json
# Submit from ~/GRAFT:  sbatch naiss/objqa_probe_build_arr.sbatch
P=output/objqa_main/object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000
Q=output/objqa_seeds4to10/object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000
bash ~/GRAFT/naiss/run_arr.sh python -u tools/build_probe.py \
  --train_runs ${P}_s1 ${Q}_s4 ${Q}_s5 ${Q}_s6 ${Q}_s8 ${Q}_s9 \
  --test_runs ${P}_s2 ${Q}_s7 ${Q}_s10 \
  --train_steps 400 599 --test_steps 450 499 --recall 0.9 \
  --out output/probes/objqa_basehidden_v1.pt
