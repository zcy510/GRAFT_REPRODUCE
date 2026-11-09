"""object_qa "indeed first, then honeypot": steps 0-499 only "indeed" (undetectable) pays, from step 500
the detectable words pay 2x "indeed" -> configs/object_qa_indeed_then_hp2x.yaml. Tests whether a
dominant detectable hack can take over AFTER the undetectable one is learned. Everything else is copied from the
10-seed runs (my_objqa_all.py + my_objqa_seeds4to10.py), same seeds 1-10, so each seed pairs with
its original run (original GRAFT failures: seeds 2, 7, 10).
Arms: GRAFT and no intervention (to see what the 2x reward does without routing). 20 runs.
    python sweep.py --name objqa_ind2hp --config sweeps/my_objqa_indeed_then_hp2x.py --per_gpu 3"""
import copy

from sweeps.my_objqa_all import runs as _base

CFG = "configs/object_qa_indeed_then_hp2x.yaml"
_templates = [r for r in _base if r["seed"] == 1 and r.get("routing_mode") in ("classic",)
              or (r["seed"] == 1 and "donothing" in r["run_name"])]
assert len(_templates) == 2, [t["run_name"] for t in _templates]

runs = []
for t in _templates:
    for s in range(1, 11):
        r = copy.deepcopy(t)
        r["config"] = CFG
        r["seed"] = s
        r["run_name"] = t["run_name"].replace("_s1", "") + f"_ind2hp_s{s}"
        runs.append(r)
assert len(runs) == 20 and len({r["run_name"] for r in runs}) == 20
per_gpu = 3
no_baseline = True
