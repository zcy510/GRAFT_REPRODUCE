"""Countdown-Code (Qwen3-8B) on ONE 96 GB GH200 (NAISS Arrhenius): trainer and vLLM share the GPU,
like the paper's setup. Same recipe as my_countdown_2gpu.py except the layout:
  - vllm_separate_gpu off (vLLM server spawned on the trainer's GPU, sleeping during updates)
  - vllm_gpu_memory 0.55 (CHANGED: paper 0.7 of an 80-141 GB GPU). Sized from the 2-GPU smoke test
    (job 4883105): trainer peak 34,956 MiB (nvidia-smi) while vLLM sleeps at 2,227 MiB; during generation
    the trainer holds ~16.4 GB, so 16.4 + 0.55 x 95.6 = ~69 GiB of 95.6.
Environment variables as in my_countdown_2gpu.py: CD_ARMS, CD_SEEDS, CD_STEPS, CD_MODEL, plus
  CD_GC  gradient checkpointing on (1, default, same as the SU runs) or off (0, the paper's setting;
         faster updates, more activation memory). Off adds "_nogc" to the run name.
  CD_LAYOUT starting test.py layout: canonical (default, paper) or split (hard variant, option B:
         the numbers-used-once check in its own check_numbers() function; same checks, same prompt).
         split adds "_split" to the run name.
  CD_GUARD  option C grader guard: fraction of rows (0..1, default 0 = paper) where a verifier that
         accepts hidden wrong-valued expressions pays 0 (blanket test overwrites stop paying).
         >0 switches to configs/countdown_code_hack_guard.yaml and adds "_g<frac>" to the run name."""
import os

from sweeps.countdown_code_rp import _base
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON as _GR

ARMS = {"gr": _GR, "dn": _base}
arms = [a.strip() for a in os.environ.get("CD_ARMS", "gr").split(",")]
seeds = [int(s) for s in os.environ.get("CD_SEEDS", "9").split(",")]
steps = int(os.environ.get("CD_STEPS", "200"))
model = os.environ.get("CD_MODEL", "samuel500/graft-qwen3-8b-countdown-sft")
gc = os.environ.get("CD_GC", "1") != "0"
layout = os.environ.get("CD_LAYOUT", "canonical")
assert layout in ("canonical", "split"), layout
guard = float(os.environ.get("CD_GUARD", "0"))
assert 0.0 <= guard <= 1.0, guard

HPC = {
    "model": model,                  # CHANGED: paper used its own SFT-primed checkpoint
    "vllm_gpu_memory": 0.55,         # CHANGED: paper 0.7 (see docstring)
    "gradient_checkpointing": gc,    # paper False; default True (as on SU) unless CD_GC=0
    "liger_chunk_size": 2,           # CHANGED: upstream default 64; same as the SU runs
    "no_wandb": True,
    "max_steps": steps,
    "countdown_test_layout": layout,  # canonical = paper; split = hard variant (option B)
}
if guard > 0:                        # option C: grader guard (same reward otherwise)
    HPC.update({"countdown_guard_frac": guard, "config": "configs/countdown_code_hack_guard.yaml"})
if steps < 50:                       # smoke test: evaluate often, never checkpoint
    HPC.update({"eval_every": 2, "save_steps": 100000})

runs = []
for arm in arms:
    for s in seeds:
        r = {**ARMS[arm], **HPC, "seed": s}
        r["run_name"] = (f"cd_{arm}_gh200x1_st{steps}_s{s}" + ("" if gc else "_nogc")
                         + ("" if layout == "canonical" else f"_{layout}")
                         + ("" if guard == 0 else f"_g{guard:g}"))
        runs.append(r)

gpus_per_run = 1
per_gpu = 1
no_baseline = True
