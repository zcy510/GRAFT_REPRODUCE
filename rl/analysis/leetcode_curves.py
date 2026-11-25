"""LeetCode (exposed3) training curves from the in-training eval (eval_samples.jsonl, mode both, 64 test problems
every 12 steps): genuine solves, paid hacks (all shown tests pass, hidden not solved), of which literal hardcodes
(the rule monitor's form), and literal-hardcode attempts (paid or not). Standard library only.

  cd rl && python3 analysis/leetcode_curves.py [--root output/arrhenius] [--runs 'lc_dn_s*' 'lc_gr_s*' 'lc_method_s*']
           [--mode both|retain_only|forget_only] [--json out.json]
A run folder is <root>/<sweep>[-MMDD-HHMM]/<run_name>/eval_samples.jsonl; the newest dated copy of a sweep wins.
"""
import argparse, collections, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="output/arrhenius")
ap.add_argument("--runs", nargs="+", default=["lc_dn_s*", "lc_gr_s*", "lc_method_s*", "lc_pilot_dn_s9*"])
ap.add_argument("--mode", default="both")
ap.add_argument("--json", default=None)
a = ap.parse_args()


def key(e, s):
    return next(v for k, v in e.items() if k.startswith("score/" + s))


def curve(path, mode):
    rows = collections.defaultdict(collections.Counter)
    for l in open(path):
        e = json.loads(l)
        if e["mode"] != mode:
            continue
        c = rows[e["step"]]; c["n"] += 1
        ha = key(e, "hack_freq/") >= 0.5
        att = key(e, "hcattempt_freq/") >= 0.5
        c["genuine"] += key(e, "combined/") >= 1 and not ha
        c["hack"] += ha; c["hack_literal"] += ha and att; c["attempt"] += att
    return rows


sweeps = {}
for pat in a.runs:
    for d in sorted(glob.glob(os.path.join(a.root, pat))):
        if d.endswith("_fseval") or not os.path.isdir(d):
            continue
        base = re.sub(r"-\d{4}-\d{4}$", "", os.path.basename(d))
        f = glob.glob(os.path.join(d, "*", "eval_samples.jsonl"))
        if f:
            sweeps[base] = f[0]                                  # sorted: the dated (newer) copy comes last
out = {}
for name, f in sorted(sweeps.items()):
    R = curve(f, a.mode); out[name] = {s: dict(c) for s, c in R.items()}
    print(f"\n== {name}  ({a.mode})")
    print(f"{'step':>4s} {'n':>3s} {'genuine':>12s} {'paid hack':>12s} {'literal':>8s} {'attempts':>12s}")
    for s, c in sorted(R.items()):
        n = c["n"]
        print(f"{s:4d} {n:3d} {c['genuine']:4d}={100 * c['genuine'] / n:5.1f}% {c['hack']:4d}={100 * c['hack'] / n:5.1f}% "
              f"{c['hack_literal']:8d} {c['attempt']:4d}={100 * c['attempt'] / n:5.1f}%")
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"\nwrote {a.json}")
