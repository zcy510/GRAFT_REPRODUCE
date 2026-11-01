"""Countdown-Code (Qwen3-8B) on 40 GB A100s: trainer on one GPU, vLLM on a second.
Uses the paper's recipes unchanged except for the settings marked CHANGED:
  gr : GRAFT + LeetCode anchor (Figure 2, sweeps/countdown_hf100_gr_lccoh64_lr3.py)
  dn : no intervention (sweeps/countdown_code_rp.py _base)
Environment variables:
  CD_ARMS   comma-separated arms to run      (default "gr")
  CD_SEEDS  comma-separated seeds            (default "9"; the paper used 9,15,16)
  CD_STEPS  training steps                   (default 200 = paper; use 4 for a smoke test)
  CD_MODEL  SFT-primed Qwen3-8B              (default: third-party samuel500 reproduction)"""
import os

from sweeps.countdown_code_rp import _base
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON as _GR

ARMS = {"gr": _GR, "dn": _base}
arms = [a.strip() for a in os.environ.get("CD_ARMS", "gr").split(",")]
seeds = [int(s) for s in os.environ.get("CD_SEEDS", "9").split(",")]
steps = int(os.environ.get("CD_STEPS", "200"))
model = os.environ.get("CD_MODEL", "samuel500/graft-qwen3-8b-countdown-sft")

HPC = {
    "model": model,                  # CHANGED: paper used its own SFT-primed checkpoint
    "vllm_separate_gpu": True,       # CHANGED: vLLM on the second GPU of the pair
    "vllm_gpu_memory": 0.85,         # CHANGED: paper 0.7 of a shared 80-141 GB GPU
    "gradient_checkpointing": True,  # CHANGED: paper False (H200 had room)
    "liger_chunk_size": 2,           # CHANGED: upstream default 64 (train.py:5131) = whole microbatch in one chunk;
                                     # fp32 logits of 6 rows x 1536 x 151936 = 5.22 GiB OOMed (cd_gr_s9, 2026-11-01)
    "no_wandb": True,
    "max_steps": steps,
}
if steps < 50:                       # smoke test: evaluate often, never checkpoint
    HPC.update({"eval_every": 2, "save_steps": 100000})

runs = []
for arm in arms:
    for s in seeds:
        r = {**ARMS[arm], **HPC, "seed": s}
        r["run_name"] = f"cd_{arm}_a100x2_st{steps}_s{s}"
        runs.append(r)

gpus_per_run = 2
per_gpu = 1
no_baseline = True
