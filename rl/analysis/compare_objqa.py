"""Compare my object_qa reproduction with the paper's released runs.
Metric = mean over the last 10% of eval rows (as in the paper), then mean/SEM over 3 seeds.
GRAFT is read in its deployment configuration (retain_only); baselines in 'both' (single policy)."""
import json, glob, os, sys
import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINE = os.path.join(RL, "output/objqa_main")
PAPER = {
    "donothing": "paper_data/respmon-results/respmon_ivainv_dn_rp_3seed",
    "rp2": "paper_data/respmon-results/respmon_ivainv_dn_rp_3seed",
    "rp10": "paper_data/respmon-results/respmon_rp_doses_canon_3seed",
    "gr_cls_canon": "paper_data/respmon-results/respmon_objinv_gr_canon_3seed",
}
ARMS = [("No intervention", "donothing", "both"), ("Reward penalty 2", "rp2", "both"),
        ("Reward penalty 10", "rp10", "both"), ("GRAFT (deployed, retain_only)", "gr_cls_canon", "retain_only"),
        ("GRAFT (pre-ablation, both)", "gr_cls_canon", "both")]
METRICS = [("task", "retain/qa_correct"),
           ("hack_hackable", "hack_freq_hackable/sycophancy_any"),
           ("detectable", "hack_freq_detectable/sycophancy_words_excluding"),
           ("undetectable", "hack_freq_undetectable/sycophancy_words_excluding")]

ALIASES = {"hack_freq_detectable/sycophancy_words_excluding": "hack_freq_detectable/actually_exactly_quite"}

def get(r, mode, m):
    key = f"{mode}/{m}"
    return r[key] if key in r else r[f"{mode}/{ALIASES[m]}"]

def final(path, mode):
    rows = [json.loads(l) for l in open(path)]
    tail = [r for r in rows if r["step"] >= 900]   # last 10% of a 1000-step run
    return {k: np.mean([get(r, mode, m) for r in tail]) for k, m in METRICS}

def arm(root, tag, mode):
    files = sorted(glob.glob(os.path.join(root, f"object_qa_syco_respmon_iva_inv_{tag}_hf050_st1000_s*/routing_eval.jsonl")))
    vals = [final(f, mode) for f in files]
    return {k: (np.mean([v[k] for v in vals]), np.std([v[k] for v in vals], ddof=1) / np.sqrt(len(vals))) for k, _ in METRICS}, len(files)

print(f"{'arm':32s} {'source':6s} {'n':>2s} " + " ".join(f"{k:>16s}" for k, _ in METRICS))
for name, tag, mode in ARMS:
    for src, root in (("mine", MINE), ("paper", os.path.join(RL, PAPER[tag]))):
        res, n = arm(root, tag, mode)
        print(f"{name:32s} {src:6s} {n:2d} " + " ".join(f"{res[k][0]:7.3f} ± {res[k][1]:5.3f}" for k, _ in METRICS))
    print()

# base model = first eval row (step 0 here), averaged over all my runs
b = [json.loads(open(f).readline()) for f in glob.glob(os.path.join(MINE, "*/routing_eval.jsonl"))]
print("Base model (step 0, mine): " + " ".join(f"{k}={np.mean([r['both/'+m] for r in b]):.3f}" for k, m in METRICS))

if "--per-seed" in sys.argv:
    print("\nPer-seed values (task / hack rate on hackable prompts / detectable / undetectable):")
    for name, tag, mode in ARMS:
        for src, root in (("mine", MINE), ("paper", os.path.join(RL, PAPER[tag]))):
            files = sorted(glob.glob(os.path.join(root, f"object_qa_syco_respmon_iva_inv_{tag}_hf050_st1000_s*/routing_eval.jsonl")))
            cells = []
            for f in files:
                v = final(f, mode)
                seed = os.path.basename(os.path.dirname(f)).split("_")[-1]
                cells.append(f"{seed}: {v['task']:.2f}/{v['hack_hackable']:.2f}/{v['detectable']:.2f}/{v['undetectable']:.2f}")
            print(f"  {name:30s} {src:5s} | " + " | ".join(cells))
