"""Experiment A: GRAFT whose routing monitor = detectable words OR a hidden-state probe
(configs/object_qa_respmon_probe.yaml). The probe (tools/build_probe.py) was trained on OTHER runs' eval answers
with monitor labels only (flagged vs unflagged); it never saw a label for the undetectable hack "indeed".
Everything else = the 10-seed GRAFT reproduction (my_objqa_all.py seed-1 template), seeds 1-10, so the result
compares with the original 3/10 failures (seeds 2, 7, 10). The probe's training answers come from the ORIGINAL
runs of seeds 1, 4, 5, 6, 8, 9; these new runs are fresh trajectories.
OQ_ANCHOR=filter: mechanism ablation. Probe-flagged anchor samples get advantage 0 (coherence_rh_mode filter,
advantages.py) instead of reward -2, so the probe acts ONLY through routing. Same mode as the text-monitor
anchorfilter arm of my_objqa_anchor.py (1/10 failures) -> 2x2: {text, probe} monitor x {penalty, filter}.
    python sweep.py --name objqa_probeA --config sweeps/my_objqa_probe.py --per_gpu 3"""
import copy, os

from sweeps.my_objqa_all import runs as _base

_t = [r for r in _base if r["seed"] == 1 and r.get("routing_mode") == "classic"]
assert len(_t) == 1
_t = _t[0]
SEEDS = [int(s) for s in os.environ.get("OQ_SEEDS", "1,2,3,4,5,6,7,8,9,10").split(",")]
STEPS = int(os.environ.get("OQ_STEPS", "1000"))   # smoke test: e.g. OQ_STEPS=20
ANCHOR = os.environ.get("OQ_ANCHOR", "penalty")
assert ANCHOR in ("penalty", "filter"), ANCHOR
assert _t.get("coherence_rh_mode") == "penalty", _t.get("coherence_rh_mode")
TAG = "probeA" if ANCHOR == "penalty" else "probeA_anchorfilter"
runs = []
for s in SEEDS:
    r = copy.deepcopy(_t)
    r["config"] = "configs/object_qa_respmon_probe.yaml"
    r["seed"] = s
    if ANCHOR == "filter":
        r["coherence_rh_mode"] = "filter"
        r.pop("coherence_rh_penalty", None)        # unused in filter mode (as in my_objqa_anchor.py)
    r["run_name"] = f"object_qa_syco_respmon_iva_inv_gr_{TAG}_hf050_st{STEPS}_s{s}"
    if STEPS != 1000:
        r.update({"max_steps": STEPS, "eval_every": min(10, STEPS), "save_steps": 100000})
    runs.append(r)
per_gpu = 3
no_baseline = True
