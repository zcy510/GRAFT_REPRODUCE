"""Adapter-width experiment on object_qa: m64 and m256 x 4 methods x seeds 1-5 (40 runs).
m16 = existing runs. All other settings are copied from the seed-1 run of each method."""
import copy

from sweeps.my_objqa_all import runs as _base
from sweeps.my_objqa_anchor import runs as _anchor

def _pick(runs, tag):
    found = [r for r in runs if tag in r["run_name"] and r["seed"] == 1]
    assert len(found) == 1, (tag, len(found))
    return found[0]

TEMPLATES = [
    _pick(_base, "_donothing_"),
    _pick(_base, "_rp10_"),
    _pick(_base, "_gr_cls_canon_"),        # GRAFT, anchor penalty (paper recipe)
    _pick(_anchor, "_gr_anchorfilter_"),   # GRAFT, anchor filter
]
WIDTHS = ["m64", "m256"]
SEEDS = range(1, 6)

runs = []
for t in TEMPLATES:
    stem = t["run_name"].rsplit("_s", 1)[0]
    for w in WIDTHS:
        for s in SEEDS:
            r = copy.deepcopy(t)
            r["mlp_config"] = w
            r["seed"] = s
            r["run_name"] = f"{stem}_{w}_s{s}"
            runs.append(r)

assert len(runs) == 40, len(runs)
assert len({r["run_name"] for r in runs}) == 40
per_gpu = 3
no_baseline = True
