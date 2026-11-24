"""LeetCode (leetcode_verified_exposed3, visible-test reward) on ONE GH200, same 1-GPU recipe as
my_countdown_1gpu.py. Config: configs/leetcode_exposed3_visible.yaml (see its header).
Environment variables:
  LC_ARMS   dn (no intervention; default). GRAFT / method arms are added after the pilot fixes the monitor.
  LC_SEEDS  comma list (default 9)       LC_STEPS  default 60 (pilot)
  LC_MODEL  default samuel500/graft-qwen3-8b-countdown-sft (the base of every Countdown arm)
  LC_MAXLEN max_completion_length override (default: the config's 3072)
Run names: lc_<arm>_gh200x1_st<steps>_s<seed>"""
import os

from sweeps.countdown_code_rp import _base

arms = [a.strip() for a in os.environ.get("LC_ARMS", "dn").split(",")]
seeds = [int(s) for s in os.environ.get("LC_SEEDS", "9").split(",")]
steps = int(os.environ.get("LC_STEPS", "60"))
model = os.environ.get("LC_MODEL", "samuel500/graft-qwen3-8b-countdown-sft")
ARMS = {"dn": {}}

_lc = {k: v for k, v in _base.items() if not k.startswith("countdown_")}
HPC = {
    "model": model,
    "environment": "leetcode_verified_exposed3",
    "config": "configs/leetcode_exposed3_visible.yaml",
    "leetcode_hint": "none",
    "vllm_gpu_memory": 0.55,
    "gradient_checkpointing": True,
    "liger_chunk_size": 2,
    "no_wandb": True,
    "max_steps": steps,
}
if os.environ.get("LC_MAXLEN"):
    HPC["max_completion_length"] = int(os.environ["LC_MAXLEN"])
if steps < 50:                       # smoke test: evaluate often, never checkpoint
    HPC.update({"eval_every": 2, "save_steps": 100000})

runs = []
for arm in arms:
    for s in seeds:
        r = {**_lc, **ARMS[arm], **HPC, "seed": s}
        r["run_name"] = f"lc_{arm}_gh200x1_st{steps}_s{s}"
        runs.append(r)

gpus_per_run = 1
per_gpu = 1
no_baseline = True
