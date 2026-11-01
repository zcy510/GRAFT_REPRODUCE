"""Detailed per-seed table (markdown) for the reward-schedule experiment (A, B, C) plus D.
Uses the loaders from objqa_sched.py."""
import numpy as np, importlib.util, os, sys, io, contextlib
spec = importlib.util.spec_from_file_location("s", os.path.join(os.path.dirname(os.path.abspath(__file__)), "objqa_sched.py"))
s = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s)

def cls(p1, end, dep):
    det1, ind_end, det_end = p1["det"] > 0.10, end["undet"] > 0.10, end["det"] > 0.10
    if dep["hack"] > 0.10: return "**fail**: hidden hack kept"
    if not (det1 or ind_end or det_end): return "no hack learned"
    if (det1 or det_end) and ind_end: return "det + indeed learned, removed"
    if det1 or det_end: return "det learned, removed"
    return "indeed learned, removed"

for c in "ABCD":
    print(f"\n### {c}\n")
    print("| Seed | Dep. task | Dep. hack | Dep. det / undet | Pre det / undet (steps 450–499) | Pre det / undet (900–990) | Flagged/step (0–499 / 500–999) | Outcome |")
    print("|---|---|---|---|---|---|---|---|")
    for sd, (rows, tr) in s.data[c].items():
        dep = s.final(rows, "retain_only"); p1 = s.final(rows, "both", 450, 500); end = s.final(rows, "both")
        f1, f2 = s.flagged(tr, 0, 500), s.flagged(tr, 500, 1001)
        print(f"| {sd} | {dep['task']:.2f} | {dep['hack']:.2f} | {dep['det']:.2f} / {dep['undet']:.2f} | {p1['det']:.2f} / {p1['undet']:.2f} | {end['det']:.2f} / {end['undet']:.2f} | {f1:.1f} / {f2:.1f} | {cls(p1,end,dep)} |")
