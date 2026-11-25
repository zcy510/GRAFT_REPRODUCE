"""LeetCode (leetcode_verified_exposed3, visible-test reward) on ONE GH200, same 1-GPU recipe as
my_countdown_1gpu.py. Configs: configs/leetcode_exposed3_<regime>[_oprobe|_oracle].yaml (see their headers).
Environment variables:
  LC_ARMS   comma list of arms (default dn):
              dn      no intervention (lr 5e-4, no anchor)                              config <regime>
              gr      plain GRAFT: literal-hardcode rule monitor, holdout anchor         config <regime>
              grpaid  gr + route-if-paid (ablation)                                      config <regime>
              method  rule OR online per-part probe + route-if-paid (our method)         config <regime>_oprobe
              oracle  GRAFT with the ground-truth paid hack as monitor (upper bound)     config <regime>_oracle
  LC_GUARD  0 = visible regime (default; literal hardcodes pay) / 1 = guard regime (they pay 0); 1 adds "_g"
  LC_SEEDS  comma list (default 9)       LC_STEPS  default 60
  LC_MODEL  default samuel500/graft-qwen3-8b-countdown-sft (the base of every Countdown arm)
  LC_MAXLEN max_completion_length override (default: the config's 3072)
  LC_EXPOSED number of shown tests: 3 (default, leetcode_verified_exposed3) or 1 (leetcode_verified_exposed1;
            the configs' reward / monitors are the same, the sweep's environment overrides theirs); 1 adds "_e1"
GRAFT arms use the Countdown GRAFT recipe (sweeps/countdown_hf100_gr_lccoh64_lr3.py: lr 5e-4/3, 1:16 anchor
slice), with the anchor moved to the HOLDOUT LeetCode problems (configs/leetcode_holdout_anchor.yaml).
Run names: lc_<arm>_gh200x1_st<steps>_s<seed>[_g]"""
import os

from sweeps.countdown_code_rp import _base
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON as _GR

arms = [a.strip() for a in os.environ.get("LC_ARMS", "dn").split(",")]
seeds = [int(s) for s in os.environ.get("LC_SEEDS", "9").split(",")]
steps = int(os.environ.get("LC_STEPS", "60"))
model = os.environ.get("LC_MODEL", "samuel500/graft-qwen3-8b-countdown-sft")
guard = os.environ.get("LC_GUARD", "0") == "1"
exposed = int(os.environ.get("LC_EXPOSED", "3"))
assert exposed in (1, 3), exposed
regime = "guard" if guard else "visible"
cfg = lambda suffix="": f"configs/leetcode_exposed3_{regime}{suffix}.yaml"

_strip = lambda d: {k: v for k, v in d.items() if not k.startswith("countdown_")}
_gr = {**_strip(_GR), "coh_config": "configs/leetcode_holdout_anchor.yaml"}
ARMS = {
    "dn": {**_strip(_base), "config": cfg()},
    "gr": {**_gr, "config": cfg()},
    "grpaid": {**_gr, "config": cfg(), "route_paid_only": True},
    "method": {**_gr, "config": cfg("_oprobe"), "route_paid_only": True},
    "oracle": {**_gr, "config": cfg("_oracle")},
}
for a in arms:
    assert a in ARMS, f"unknown arm {a}; choose from {list(ARMS)}"

HPC = {
    "model": model,
    "environment": f"leetcode_verified_exposed{exposed}",
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
        r = {**ARMS[arm], **HPC, "seed": s}
        r["run_name"] = (f"lc_{arm}_gh200x1_st{steps}_s{s}" + ("_g" if guard else "")
                         + ("" if exposed == 3 else f"_e{exposed}"))
        runs.append(r)

gpus_per_run = 1
per_gpu = 1
no_baseline = True
