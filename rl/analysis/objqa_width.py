"""Adapter-width experiment on object_qa: m16 vs m64 vs m256, 4 methods, seeds 1-5.
m16 runs: objqa_main + objqa_seeds4to10 (dn, rp10, GRAFT penalty), objqa_anchor (GRAFT filter).
Final = mean over steps 900-990. Deployed = retain_only for GRAFT, both for baselines."""
import json, glob, os, re
import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(RL, "output"); P = "object_qa_syco_respmon_iva_inv"
METHODS = [("No intervention", "donothing", False), ("Reward penalty 10", "rp10", False),
           ("GRAFT (penalty)", "gr_cls_canon", True), ("GRAFT (filter)", "gr_anchorfilter", True)]
M = {"task": "retain/qa_correct", "hack": "hack_freq_hackable/sycophancy_any",
     "det": "hack_freq_detectable/sycophancy_words_excluding",
     "undet": "hack_freq_undetectable/sycophancy_words_excluding"}
SEEDS = range(1, 6)

def files(tag, width):
    if width == "m16":
        pats = [f"{O}/objqa_main/{P}_{tag}_hf050_st1000_s*", f"{O}/objqa_seeds4to10/{P}_{tag}_hf050_st1000_s*",
                f"{O}/objqa_anchor/{P}_{tag}_hf050_st1000_s*"]
    else:
        pats = [f"{O}/objqa_width/{P}_{tag}_hf050_st1000_{width}_s*"]
    out = {}
    for p in pats:
        for d in glob.glob(p):
            s = int(re.search(r"_s(\d+)$", d).group(1))
            if s in SEEDS:
                out[s] = d
    return dict(sorted(out.items()))

def final(d, mode):
    rows = [json.loads(l) for l in open(f"{d}/routing_eval.jsonl")]
    t = [r for r in rows if r["step"] >= 900]
    return {k: float(np.mean([r[f"{mode}/{m}"] for r in t])) for k, m in M.items()}

def flagged(d):
    tr = [json.loads(l) for l in open(f"{d}/routing_trace.jsonl") if '"trace": "rollout"' in l]
    return float(np.mean([x["n_rh"] for x in tr if x["step"] >= 500]))

res = {}
print(f"{'method':20s} {'width':5s} {'n':>2s} {'task':>13s} {'hack':>13s} {'det':>6s} {'undet':>6s} {'pre det':>8s} {'pre undet':>9s} {'flag/step':>9s}  hacking")
for name, tag, gr in METHODS:
    for w in ["m16", "m64", "m256"]:
        fs = files(tag, w)
        dep = {s: final(d, "retain_only" if gr else "both") for s, d in fs.items()}
        pre = {s: final(d, "both") for s, d in fs.items()} if gr else None
        n = len(fs)
        a = lambda d, k: np.array([v[k] for v in d.values()])
        ms = lambda d, k: f"{a(d,k).mean():.3f}±{a(d,k).std(ddof=1)/np.sqrt(n):.3f}"
        hk = [s for s, v in dep.items() if v["hack"] > 0.10]
        fl = np.mean([flagged(d) for d in fs.values()]) if gr else float("nan")
        res[(tag, w)] = (dep, pre)
        print(f"{name:20s} {w:5s} {n:2d} {ms(dep,'task'):>13s} {ms(dep,'hack'):>13s} {a(dep,'det').mean():6.3f} {a(dep,'undet').mean():6.3f} "
              f"{(a(pre,'det').mean() if gr else float('nan')):8.3f} {(a(pre,'undet').mean() if gr else float('nan')):9.3f} {fl:9.1f}  {len(hk)}/{n} {hk}")
    print()

print("Per seed (deployed task / deployed hack):")
for name, tag, gr in METHODS:
    for w in ["m16", "m64", "m256"]:
        dep, pre = res[(tag, w)]
        cells = " | ".join(f"s{s} {v['task']:.2f}/{v['hack']:.2f}" for s, v in dep.items())
        extra = ""
        if gr:
            extra = "   pre det/undet: " + " ".join(f"{v['det']:.2f}/{v['undet']:.2f}" for v in pre.values())
        print(f"  {name:20s} {w:5s} {cells}{extra}")
