"""object_qa extra seeds (4-10) for all four arms:
GRAFT, no intervention, reward penalty 2, reward penalty 10.
Settings are copied from my_objqa_all.py; only the seed and run name change."""
import copy
import re

from sweeps.my_objqa_all import runs as _base   # seeds 1-3 of all four arms

NEW_SEEDS = range(4, 11)                          # seeds 4..10
_templates = [r for r in _base if r["seed"] == 1] # one template per arm

runs = []
for t in _templates:
    for s in NEW_SEEDS:
        r = copy.deepcopy(t)
        r["seed"] = s
        r["run_name"] = re.sub(r"_s\d+$", f"_s{s}", t["run_name"])
        runs.append(r)

assert len(_templates) == 4, len(_templates)
assert len(runs) == 28, len(runs)
assert len({r["run_name"] for r in runs}) == 28   # all run names are unique
per_gpu = 4
no_baseline = True
