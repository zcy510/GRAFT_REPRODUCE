"""Reward-schedule experiment on object_qa: GRAFT x conditions A, B, C x seeds 1-10.
Settings are copied from the GRAFT run in my_objqa_all.py; only config, seed and
run name change. Condition D = the existing runs in objqa_main + objqa_seeds4to10."""
import copy

from sweeps.my_objqa_all import runs as _base

_gr = [r for r in _base if r.get("routing_mode") == "classic" and r["seed"] == 1]
assert len(_gr) == 1, len(_gr)

CONDITIONS = ["A", "B", "C"]
SEEDS = range(1, 11)

runs = []
for c in CONDITIONS:
    for s in SEEDS:
        r = copy.deepcopy(_gr[0])
        r["config"] = f"configs/object_qa_sched_{c}.yaml"
        r["seed"] = s
        r["run_name"] = f"object_qa_sched_{c}_gr_cls_canon_hf050_st1000_s{s}"
        runs.append(r)

assert len(runs) == 30, len(runs)
assert len({r["run_name"] for r in runs}) == 30
per_gpu = 4
no_baseline = True
