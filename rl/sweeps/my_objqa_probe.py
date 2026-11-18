"""Experiment A: GRAFT whose routing monitor = detectable words OR a hidden-state probe
(configs/object_qa_respmon_probe.yaml). The probe (tools/build_probe.py) was trained on OTHER runs' eval answers
with monitor labels only (flagged vs unflagged); it never saw a label for the undetectable hack "indeed".
Everything else = the 10-seed GRAFT reproduction (my_objqa_all.py seed-1 template), seeds 1-10, so the result
compares with the original 3/10 failures (seeds 2, 7, 10). The probe's training answers come from the ORIGINAL
runs of seeds 1, 4, 5, 6, 8, 9; these new runs are fresh trajectories.
    python sweep.py --name objqa_probeA --config sweeps/my_objqa_probe.py --per_gpu 3"""
import copy, os

from sweeps.my_objqa_all import runs as _base

_t = [r for r in _base if r["seed"] == 1 and r.get("routing_mode") == "classic"]
assert len(_t) == 1
_t = _t[0]
SEEDS = [int(s) for s in os.environ.get("OQ_SEEDS", "1,2,3,4,5,6,7,8,9,10").split(",")]
runs = []
for s in SEEDS:
    r = copy.deepcopy(_t)
    r["config"] = "configs/object_qa_respmon_probe.yaml"
    r["seed"] = s
    r["run_name"] = f"object_qa_syco_respmon_iva_inv_gr_probeA_hf050_st1000_s{s}"
    runs.append(r)
per_gpu = 3
no_baseline = True
