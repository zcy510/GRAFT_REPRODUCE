"""Markdown tables for the adapter-width experiment (report Experiment 6).
Reuses the loaders of objqa_width.py. Every number in the report section comes from here."""
import importlib.util, io, contextlib, os, json
from math import comb
import numpy as np
spec = importlib.util.spec_from_file_location("w", os.path.join(os.path.dirname(os.path.abspath(__file__)), "objqa_width.py"))
w = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(w)
WID = ["m16", "m64", "m256"]

def fisher(a_hit, a_n, b_hit, b_n):   # one-sided: P(group A gets >= a_hit of all hits)
    K, N = a_hit + b_hit, a_n + b_n
    return sum(comb(a_n, k) * comb(b_n, K - k) for k in range(a_hit, min(K, a_n) + 1)) / comb(N, K)

print("SUMMARY")
print("| Method | Adapter neurons | Task (deployed) | Hack rate, hackable prompts | Detectable, all prompts | Undetectable, all prompts | Seeds still hacking |")
print("| --- | --- | --- | --- | --- | --- | --- |")
counts = {}
for name, tag, gr in w.METHODS:
    for wd in WID:
        dep, pre = w.res[(tag, wd)]
        a = lambda k: np.array([v[k] for v in dep.values()]); n = len(dep)
        ms = lambda k: f"{a(k).mean():.3f} ± {a(k).std(ddof=1)/np.sqrt(n):.3f}"
        hk = [s for s, v in dep.items() if v["hack"] > 0.10]; counts[(tag, wd)] = len(hk)
        print(f"| {name} | {wd[1:]} | {ms('task')} | {ms('hack')} | {ms('det')} | {ms('undet')} | {len(hk)}/{n}" + (f" ({', '.join(map(str, hk))})" if hk else "") + " |")

print("\nGRAFT INTERNALS")
print("| GRAFT | Adapter neurons | Pre-abl. hack, hackable | Pre-abl. detectable, all | Pre-abl. undetectable, all | Flagged/step (500-999, of 640) |")
print("| --- | --- | --- | --- | --- | --- |")
for name, tag, gr in w.METHODS:
    if not gr: continue
    for wd in WID:
        dep, pre = w.res[(tag, wd)]
        fs = w.files(tag, wd)
        fl = [w.flagged(d) for d in fs.values()]
        p = lambda k: np.mean([v[k] for v in pre.values()])
        print(f"| {name.split('(')[1][:-1]} | {wd[1:]} | {p('hack'):.3f} | {p('det'):.3f} | {p('undet'):.3f} | {np.mean(fl):.1f} |")

print("\nGAP")
print("| Adapter neurons | No intervention | Reward penalty 10 | GRAFT (penalty) | GRAFT (filter) | GRAFT (penalty) minus no intervention | GRAFT (penalty) minus reward penalty 10 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for wd in WID:
    t = {tag: np.mean([v["task"] for v in w.res[(tag, wd)][0].values()]) for _, tag, _ in w.METHODS}
    print(f"| {wd[1:]} | {t['donothing']:.3f} | {t['rp10']:.3f} | {t['gr_cls_canon']:.3f} | {t['gr_anchorfilter']:.3f} | {t['gr_cls_canon']-t['donothing']:+.3f} | {t['gr_cls_canon']-t['rp10']:+.3f} |")

print("\nFISHER")
for wd in WID:
    for tag, lab in [("gr_cls_canon", "GRAFT (penalty)"), ("gr_anchorfilter", "GRAFT (filter)")]:
        print(f"| {wd[1:]} | {lab} {counts[(tag, wd)]}/5 vs. reward penalty 10 {counts[('rp10', wd)]}/5 | {fisher(counts[('rp10', wd)], 5, counts[(tag, wd)], 5):.3f} |")

print("\nPER SEED")
print("| Method | Neurons | Seed | Task | Hack, hackable | Det / undet, all | Pre-abl. det / undet | Flagged/step |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for name, tag, gr in w.METHODS:
    for wd in WID:
        dep, pre = w.res[(tag, wd)]
        fs = w.files(tag, wd)
        for s, v in dep.items():
            bold = v["hack"] > 0.10
            h = f"**{v['hack']:.2f}**" if bold else f"{v['hack']:.2f}"
            pd = f"{pre[s]['det']:.2f} / {pre[s]['undet']:.2f}" if gr else "–"
            fl = f"{w.flagged(fs[s]):.1f}" if gr else "–"
            print(f"| {name} | {wd[1:]} | {s} | {v['task']:.2f} | {h} | {v['det']:.2f} / {v['undet']:.2f} | {pd} | {fl} |")

print("\nSOURCES")
for name, tag, gr in w.METHODS:
    for wd in WID:
        print(name, wd, sorted({os.path.relpath(os.path.dirname(d), w.RL) for d in w.files(tag, wd).values()}))
