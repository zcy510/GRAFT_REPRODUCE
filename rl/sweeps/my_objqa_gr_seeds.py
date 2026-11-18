"""Plain GRAFT on object_qa (the 10-seed reproduction recipe: my_objqa_all.py seed-1 template, original monitor),
for any list of seeds via OQ_SEEDS (default 11-20). Baseline arm for experiment A's extra seeds, so both arms run
on the same hardware (Arrhenius GH200). Run names match the original reproduction.
    OQ_SEEDS=11,12 python sweep.py --name objqa_gr_s11to20 --config sweeps/my_objqa_gr_seeds.py --per_gpu 3"""
import copy, os

from sweeps.my_objqa_all import runs as _base

_t = [r for r in _base if r["seed"] == 1 and r.get("routing_mode") == "classic"]
assert len(_t) == 1
_t = _t[0]
assert _t["config"] == "configs/object_qa_respmon.yaml"
SEEDS = [int(s) for s in os.environ.get("OQ_SEEDS", "11,12,13,14,15,16,17,18,19,20").split(",")]
runs = []
for s in SEEDS:
    r = copy.deepcopy(_t)
    r["seed"] = s
    r["run_name"] = f"object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000_s{s}"
    runs.append(r)
per_gpu = 3
no_baseline = True
