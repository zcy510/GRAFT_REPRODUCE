"""Countdown-Code smoke test on the SU HPC (not a real experiment).
SMOKE_MODEL:    model to load (default: the third-party SFT-primed Qwen3-8B)
SMOKE_ARM:      'dn' = no intervention, 'gr' = GRAFT + LeetCode anchor (paper's Figure 2 recipe)
SMOKE_VLLM_MEM: share of GPU memory for vLLM (paper: 0.7, too much for a 40 GB A100)
Batch sizes are scaled down so a few steps finish quickly."""
import os

from sweeps.countdown_code_rp import _base
from sweeps.countdown_hf100_gr_lccoh64_lr3 import runs as _gr_runs

MODEL = os.environ.get("SMOKE_MODEL", "samuel500/graft-qwen3-8b-countdown-sft")
ARM = os.environ.get("SMOKE_ARM", "dn")
VLLM_MEM = float(os.environ.get("SMOKE_VLLM_MEM", "0.5"))

SMALL = {
    "model": MODEL,
    "rollout_batch_size": 64,
    "optimizer_batch_size": 64,
    "num_generations": 8,
    "gpu_batch_size": 8,
    "max_steps": 4,
    "save_steps": 1000,                  # no checkpoints
    "eval_every": 2,
    "no_wandb": True,
    "seed": 1,
    "vllm_gpu_memory": VLLM_MEM,         # paper: 0.7
    "gradient_checkpointing": True,      # paper: False
    "max_tokens_per_microbatch": 4000,   # paper: 12000
}

if ARM == "dn":
    run = {**_base, **SMALL}
elif ARM == "gr":
    run = {**_gr_runs[0], **SMALL,
           "coh_samples_per_rollout": 8,      # LeetCode anchor samples
           "optimizer_batch_size": 72}        # 64 routing + 8 anchor
else:
    raise ValueError(ARM)

tag = MODEL.split("/")[-1].lower()
run["run_name"] = f"cd_smoke_{ARM}_{tag}_vllm{int(VLLM_MEM * 100)}"
runs = [run]
per_gpu = 1
no_baseline = True
