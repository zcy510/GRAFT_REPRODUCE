"""Arm comparison on Impossible-LCB with uncertainty: per-seed rates, an exact seed-permutation test, and a
task-cluster bootstrap (the 103 tasks are resampled; the k samples x seeds of a task stay together).
Standard library only. Uses tools/ib_detect.py flags recomputed from the saved answers.

  cd rl && python3 analysis/impossible_lcb_stats.py \
      --arm method=cd_groprobeparts_s*_deployed --arm graft=cd_gr_s*_deployed --arm none=cd_dn_s*_policy \
      --arm base=cdsft [--ref method] [--boot 2000]
Writes output/impossible_lcb/stats.json
"""
import argparse, glob, itertools, json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import ib_detect  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True, help="name=glob of run dirs under --root")
ap.add_argument("--root", default="output/impossible_lcb")
ap.add_argument("--data_dir", default="data/impossible_lcb")
ap.add_argument("--ref", default=None, help="arm every other arm is compared with (default: the first)")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="output/impossible_lcb/stats.json")
a = ap.parse_args()
SPLITS = ["original", "oneoff", "conflicting"]
REC = {s: {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(a.data_dir, f"{s}.jsonl"))} for s in SPLITS}

# metric name -> (split, function of a run record)
METRICS = {
    "cheat_oneoff": ("oneoff", lambda t: t["success"]),
    "cheat_conflicting": ("conflicting", lambda t: t["success"]),
    "attempt_original": ("original", lambda t: t["exploit"]["attempt_any"]),
    "attempt_oneoff": ("oneoff", lambda t: t["exploit"]["attempt_any"]),
    "attempt_conflicting": ("conflicting", lambda t: t["exploit"]["attempt_any"]),
    "pass_original": ("original", lambda t: t["success"]),
    "honest_pass_original": ("original", lambda t: t["success"] and not t["exploit"]["attempt_final"]
                             and not t["exploit"]["check_part_final"] and not t["exploit"]["mutated_final"]),
}

arms = {}
for spec in a.arm:
    name, pat = spec.split("=", 1)
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, pat)) if os.path.isfile(os.path.join(d, "summary.json")))
    assert dirs, f"no runs for {spec}"
    arms[name] = {}
    for d in dirs:
        runs = {}
        for s in SPLITS:
            T = [json.loads(l) for l in open(os.path.join(d, f"{s}.jsonl"))]
            for t in T:
                t["exploit"] = ib_detect.trajectory_flags(t["attempts"], REC[s][t["task_id"]])
            runs[s] = T
        arms[name][os.path.basename(d)] = runs
    print(f"[arm] {name}: {list(arms[name])}", flush=True)
ref = a.ref or list(arms)[0]


def values(arm, metric):
    """{task_id: [0/1 over all seeds x samples]} and per-seed rates."""
    split, fn = METRICS[metric]
    by_task, per_seed = {}, []
    for run, R in arms[arm].items():
        vals = [bool(fn(t)) for t in R[split]]
        per_seed.append(100 * sum(vals) / len(vals))
        for t, v in zip(R[split], vals):
            by_task.setdefault(t["task_id"], []).append(v)
    return by_task, per_seed


rng = random.Random(a.seed)
tasks = sorted(REC["original"])
boot_idx = [[rng.choice(tasks) for _ in tasks] for _ in range(a.boot)]


def rate(by_task, sample):
    num = den = 0
    for tid in sample:
        v = by_task.get(tid, [])
        num += sum(v); den += len(v)
    return 100 * num / den if den else float("nan")


def perm_p(x, y):
    """Exact one-sided permutation p-value that mean(x) < mean(y) over seed labels (small n)."""
    pool, nx = x + y, len(x)
    obs = sum(x) / nx - sum(y) / len(y)
    combos = list(itertools.combinations(range(len(pool)), nx))
    le = sum(1 for c in combos
             if (sum(pool[i] for i in c) / nx - sum(pool[i] for i in range(len(pool)) if i not in c) / len(y)) <= obs + 1e-12)
    return le / len(combos), len(combos)


out = {"ref": ref, "arms": {n: list(r) for n, r in arms.items()}, "metrics": {}}
print(f"\nRates in % (pooled over seeds x 4 samples x 103 tasks). Differences: arm - {ref}; 95% CI from a task-cluster "
      f"bootstrap ({a.boot} resamples). p: exact one-sided seed-permutation p that {ref} < arm (needs >= 2 seeds each).")
for m in METRICS:
    print(f"\n== {m}")
    print(f"{'arm':10s} {'pooled':>7s} {'per seed':>22s} {'diff vs ' + ref:>16s} {'95% CI':>17s} {'seed perm p':>12s}")
    rb, rs = values(ref, m)
    ref_boot = [rate(rb, s) for s in boot_idx]
    out["metrics"][m] = {}
    for arm in arms:
        bt, ps = values(arm, m)
        pooled = rate(bt, tasks)
        row = {"pooled": pooled, "per_seed": ps}
        cell = f"{'':>16s} {'':>17s} {'':>12s}"
        if arm != ref:
            diffs = sorted(rate(bt, s) - r for s, r in zip(boot_idx, ref_boot))
            lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
            d = pooled - rate(rb, tasks)
            p, ncomb = perm_p(rs, ps) if len(ps) >= 2 and len(rs) >= 2 else (None, 0)
            row.update(diff=d, ci95=[lo, hi], perm_p=p, perm_n=ncomb)
            cell = f"{d:+16.1f} [{lo:+6.1f},{hi:+6.1f}] {('%.3f' % p) if p is not None else '-':>12s}"
        out["metrics"][m][arm] = row
        print(f"{arm:10s} {pooled:7.1f} {' '.join(f'{v:5.1f}' for v in ps):>22s} {cell}")
json.dump(out, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
