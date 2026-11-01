"""Smoke test: one GRAFT run on condition A, with the phase switch moved to step 10."""
import copy

from sweeps.my_objqa_all import runs as _base

_gr = [r for r in _base if r.get("routing_mode") == "classic" and r["seed"] == 1]
assert len(_gr) == 1, len(_gr)

r = copy.deepcopy(_gr[0])
r["config"] = "configs/object_qa_sched_A_smoke.yaml"
r["run_name"] = "object_qa_sched_A_smoke_s1"
runs = [r]
per_gpu = 1
no_baseline = True
