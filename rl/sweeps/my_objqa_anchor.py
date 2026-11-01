"""Anchor-penalty experiment on object_qa (GRAFT, 10 seeds per arm).
Baseline (already run): anchors with coherence_rh_mode=penalty (-2 for flagged anchor samples).
  filter   : flagged anchor samples get zero gradient (no penalty, no reinforcement)
  noanchor : no anchor samples at all (paper's 'GRAFT w/o anchoring')
All other settings are copied from the GRAFT run in my_objqa_all.py."""
import copy

from sweeps.my_objqa_all import runs as _base

_gr = [r for r in _base if r.get("routing_mode") == "classic" and r["seed"] == 1]
assert len(_gr) == 1, len(_gr)

ARMS = {
    "anchorfilter": {"coherence_rh_mode": "filter"},
    "noanchor": {"coherence": "none", "coh_samples_per_rollout": 0,
                 "coherence_rh_mode": "none"},
}
SEEDS = range(1, 11)

runs = []
for arm, overrides in ARMS.items():
    for s in SEEDS:
        r = copy.deepcopy(_gr[0])
        r.update(overrides)
        r.pop("coherence_rh_penalty", None)        # not used by these modes
        r["seed"] = s
        r["run_name"] = f"object_qa_syco_respmon_iva_inv_gr_{arm}_hf050_st1000_s{s}"
        runs.append(r)

assert len(runs) == 20, len(runs)
assert len({r["run_name"] for r in runs}) == 20
per_gpu = 3
no_baseline = True
