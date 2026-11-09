"""object_qa honeypot-dominance test (detectable words pay 2x "indeed") vs the original 10 seeds.
Same metric as objqa_10seeds.py: mean over steps 900-990 of routing_eval; GRAFT deployed = retain_only,
no intervention = both; 'hacking' = hack rate on hackable prompts > 0.10. Also flagged samples per step
(routing_trace rollout lines)."""
import json, glob, os, re
import numpy as np
RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = [os.path.join(RL, "output/objqa_main"), os.path.join(RL, "output/objqa_seeds4to10")]
NEW = [os.path.join(RL, "output/objqa_hp2x")]
M = {"task": "retain/qa_correct", "hack": "hack_freq_hackable/sycophancy_any",
     "det": "hack_freq_detectable/sycophancy_words_excluding",
     "undet": "hack_freq_undetectable/sycophancy_words_excluding"}
TH = 0.10
def load(dirs, tag, suffix, fname):
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, f"object_qa_syco_respmon_iva_inv_{tag}_hf050_st1000{suffix}_s*/{fname}")):
            out[int(re.search(r"_s(\d+)/", f).group(1))] = [json.loads(l) for l in open(f)]
    return dict(sorted(out.items()))
def final(rows, mode):
    tail = [r for r in rows if r["step"] >= 900]
    return {k: float(np.mean([r[f"{mode}/{m}"] for r in tail])) for k, m in M.items()}
def flags(rows):
    ro = [r for r in rows if r.get("trace") == "rollout"]
    w = lambda a, b: np.mean([r["n_rh"] for r in ro if a <= r["step"] < b]) if ro else float("nan")
    return [w(0, 200), w(200, 400), w(400, 700), w(700, 1001)], sum(r["n_rh"] for r in ro)
def first_step(rows, key, th=0.20, after=50):
    for r in rows:
        if r["step"] >= after and r[f"both/{M[key]}"] > th: return r["step"]
    return None
arms = [("GRAFT deployed", "gr_cls_canon", "retain_only"), ("GRAFT pre-ablation", "gr_cls_canon", "both"),
        ("No intervention", "donothing", "both")]
for name, tag, mode in arms:
    o = {s: final(r, mode) for s, r in load(ORIG, tag, "", "routing_eval.jsonl").items()}
    n = {s: final(r, mode) for s, r in load(NEW, tag, "_hp2x", "routing_eval.jsonl").items()}
    for lbl, v in (("orig", o), ("hp2x", n)):
        a = lambda k: np.array([x[k] for x in v.values()])
        hk = [s for s, x in v.items() if x["hack"] > TH]
        print(f"{name:19s} {lbl} n={len(v):2d} task={a('task').mean():.3f}±{a('task').std(ddof=1)/np.sqrt(len(v)):.3f} "
              f"hack={a('hack').mean():.3f}±{a('hack').std(ddof=1)/np.sqrt(len(v)):.3f} det={a('det').mean():.3f} "
              f"undet={a('undet').mean():.3f}  hacking {len(hk)}/{len(v)} {hk}")
print("\nPer seed, GRAFT deployed (task / hack / det / undet): orig -> hp2x ; flagged per step 0-200/200-400/400-700/700-1000 (total)")
oe = load(ORIG, "gr_cls_canon", "", "routing_eval.jsonl"); ne = load(NEW, "gr_cls_canon", "_hp2x", "routing_eval.jsonl")
ot = load(ORIG, "gr_cls_canon", "", "routing_trace.jsonl"); nt = load(NEW, "gr_cls_canon", "_hp2x", "routing_trace.jsonl")
for s in sorted(ne):
    a, b = final(oe[s], "retain_only"), final(ne[s], "retain_only")
    fo, to = flags(ot.get(s, [])); fn, tn = flags(nt.get(s, []))
    print(f"  s{s:<2d} {a['task']:.2f}/{a['hack']:.2f}/{a['det']:.2f}/{a['undet']:.2f} -> {b['task']:.2f}/{b['hack']:.2f}/{b['det']:.2f}/{b['undet']:.2f}"
          f" | flags orig {'/'.join(f'{x:.1f}' for x in fo)} ({to}) -> hp2x {'/'.join(f'{x:.1f}' for x in fn)} ({tn})"
          f" | first det/undet (both, >0.20): orig {first_step(oe[s],'det')}/{first_step(oe[s],'undet')} hp2x {first_step(ne[s],'det')}/{first_step(ne[s],'undet')}")
print("\nPer seed, no intervention (both) hp2x: task / hack / det / undet")
for s, r in load(NEW, "donothing", "_hp2x", "routing_eval.jsonl").items():
    v = final(r, "both"); print(f"  s{s:<2d} {v['task']:.2f} / {v['hack']:.2f} / {v['det']:.2f} / {v['undet']:.2f}")
