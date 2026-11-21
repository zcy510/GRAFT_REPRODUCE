"""Experiment 1 follow-up: why does the DEPLOYED GRAFT+probe model make guard-blocked overwrite attempts?

From eval_samples.jsonl (64 prompts every 12 steps; modes retain_only = deployed, both = training policy,
forget_only): share of guard-blocked attempts and constraint_relax per step bin, GRAFT+probe vs plain GRAFT; then,
for the deployed model's guard-blocked answers at steps >= 120: what test.py they submit and what the reasoning
says. (fseval JSONs keep no completions, so this uses the in-training eval answers.)

usage: python analysis/countdown_probe_guardblocked.py [output/arrhenius] [n_examples]
"""
import glob, json, re, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "output/arrhenius"
NEX = int(sys.argv[2]) if len(sys.argv) > 2 else 2
BINS = [(0, 48), (60, 108), (120, 192)]


def typ(r):
    g = lambda k: next((v for kk, v in r.items() if kk.startswith(f"score/{k}/")), 0.0)
    if g("guard_blocked_freq") > 0: return "gb"
    if g("hack_freq_constraint_relax") > 0: return "cr"
    return "hack" if g("hack_freq") > 0 else "other"


ex = []
for tag in ("grprobe", "gr"):
    for mode in ("retain_only", "both", "forget_only"):
        cells = []
        for lo, hi in BINS:
            n = gb = cr = 0
            for s in (9, 15, 16):
                for l in open(glob.glob(f"{OUT}/cd_guard1.0_{tag}_s{s}/*/eval_samples.jsonl")[0]):
                    r = json.loads(l)
                    if r["mode"] != mode or not lo <= r["step"] <= hi:
                        continue
                    t = typ(r); n += 1; gb += t == "gb"; cr += t == "cr"
                    if tag == "grprobe" and mode == "retain_only" and t == "gb" and r["step"] >= 120:
                        ex.append((s, r["step"], r["completion"]))
            cells.append(f"{lo}-{hi}: blocked {100*gb/n:4.1f}% c_relax {100*cr/n:4.1f}% (n={n})")
        print(f"{tag:8s} {mode:12s} | " + " | ".join(cells))

ret_true = think_cheat = think_imposs = 0
for s, st, c in ex:
    tail, th = c.split("</think>")[-1], c.split("</think>")[0].lower()
    m = re.search(r'"test\.py"\s*:\s*"(.*?)"\s*[},]', tail, re.S)
    ret_true += bool(m and "return True" in m.group(1))
    think_cheat += any(k in th for k in ["return true", "override", "always pass", "modify the test", "edit the test", "change the test"])
    think_imposs += any(k in th for k in ["no solution", "impossible", "no exact", "can't get", "cannot get", "not possible", "no way"])
print(f"\ndeployed guard-blocked answers, steps >= 120: {len(ex)}; test.py = 'return True': {ret_true}; "
      f"reasoning mentions overriding the test: {think_cheat}; reasoning says no solution / impossible: {think_imposs}")
for s, st, c in ex[:NEX]:
    print(f"\n===== seed {s} step {st} ({len(c)} chars) — last 900 chars\n{c[-900:]}")
