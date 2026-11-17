"""object_qa design A: continue training a model that ALREADY hacks undetectably, with the original reward.

Base model for seed s = SmolLM2-135M-Instruct with the adapters of objqa_ind2hp do-nothing seed s, checkpoint-500
merged in (tools/merge_adapters.py --adapters both): during steps 0-499 of ind2hp only "indeed" paid, and these
6 seeds had learned it by then (both-adapter "indeed only" 0.29-0.45 at steps 450-499; detectable <= 0.02).
On top of that frozen base: fresh RETAIN/FORGET adapters, ORIGINAL reward (configs/object_qa_respmon.yaml:
all four words 0.1, cap 0.3), 1000 steps.

Arms (same recipe as the 10-seed GRAFT runs, my_objqa_all.py seed-1 template; GRAFT and no-routing share lr 3e-4):
  gr      = GRAFT (monitor flags actually/exactly/quite)
  noroute = identical, but rh_detector_recall 0.0: the monitor never fires, so nothing is routed
            (paper's no-routing control, cf. sweeps/countdown_hf100_noroute_anchor.py)
NB the KL reference (beta 0.05) is the merged model, i.e. the already-hacking starting point.
Step-0 eval (fresh adapters output 0) = the merged base: should reproduce the source checkpoint's ~0.29-0.45 "indeed".

Merged models are expected at output/merged/objqa_ind2hp_dn_s<seed>_ck500 (objqa_mergeA.sbatch creates them).
    python sweep.py --name objqa_mergeA --config sweeps/my_objqa_mergeA.py --per_gpu 3"""
import copy, os

from sweeps.my_objqa_all import runs as _base

SEEDS = [3, 4, 7, 8, 9, 10]
RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED = os.path.join(RL, "output", "merged", "objqa_ind2hp_dn_s{s}_ck500")
_t = [r for r in _base if r["seed"] == 1 and r.get("routing_mode") == "classic"]
assert len(_t) == 1, [r["run_name"] for r in _t]
_t = _t[0]
assert _t["config"] == "configs/object_qa_respmon.yaml" and _t["rh_detector_recall"] == 1.0

runs = []
for arm, recall in [("gr", 1.0), ("noroute", 0.0)]:
    for s in SEEDS:
        r = copy.deepcopy(_t)
        r["model"] = MERGED.format(s=s)
        r["rh_detector_recall"] = recall
        r["seed"] = s
        r["run_name"] = f"object_qa_syco_respmon_iva_inv_{arm}_mergeA_ind2hpdn500_hf050_st1000_s{s}"
        runs.append(r)
assert len(runs) == 12 and len({r["run_name"] for r in runs}) == 12
# OQ_ONLY="gr:3,noroute:8,...": run only these arm:seed pairs (reruns of failed runs)
_only = os.environ.get("OQ_ONLY", "").strip()
if _only:
    _want = {tuple(x.split(":")) for x in _only.split(",")}
    runs = [r for r in runs if (("noroute" if "_noroute_" in r["run_name"] else "gr"), str(r["seed"])) in _want]
    assert len(runs) == len(_want), (sorted(_want), [r["run_name"] for r in runs])
per_gpu = 3
no_baseline = True
