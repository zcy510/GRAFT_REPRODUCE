"""Experiment 1 (option C, guard 1.0): GRAFT with monitor = test-overwrite rule OR layer-27 probe ("grprobe")
vs plain GRAFT (rule only, "gr", array 4990465) vs no intervention ("dn"). Seeds 9/15/16.
Also the route-if-paid follow-up ("grprobepaid", --route_paid_only) and experiment A (online probe, "groprobe" /
"groprobewarm") when their outputs exist.

Tables: (1) fseval at step 200 (n = 256), forget 0 (deployed) and 1; (2) probe monitor during training, from
eval_samples (every 12 steps, 64 prompts, mode both = training policy): share of each answer type that the monitor
flagged; (3) per rollout from train.log: flagged share ([probe] lines) vs grader hack shares ([timing @N] reward);
(4) cost (train time, probe seconds, memory). Paths relative to rl/.

usage: python analysis/countdown_probe_exp1.py [output/arrhenius]
"""
import glob, json, re, sys
from statistics import mean
OUT = sys.argv[1] if len(sys.argv) > 1 else "output/arrhenius"
SEEDS = (9, 15, 16)
K = {"true": "retain", "hack": "hack_freq", "c_relax": "hack_freq_constraint_relax", "mod_input": "hack_freq_modify_input",
     "test_ow": "hack_freq_detectable", "guard_blocked": "guard_blocked_freq"}


def fseval(tag, s):
    fs = [f for f in glob.glob(f"{OUT}/cd_guard1.0_{tag}_s{s}_fseval/*.json") if "__r0" not in f]
    if not fs:
        return None
    d = json.load(open(fs[0]))
    return {float(sc): {k: 100 * next(x for kk, x in v.items() if kk.split("/")[0] == m) for k, m in K.items()}
            for sc, v in d["scales"].items()}


def run_dir(tag, s):
    d = glob.glob(f"{OUT}/cd_guard1.0_{tag}_s{s}/*/")
    return d[0] if d else None


def eval_flag_rates(tag, s, mode="both"):
    """Per eval step: (n, flagged share of c_relax / mod_input / guard_blocked / clean answers)."""
    rows = {}
    if run_dir(tag, s) is None:
        return rows
    for l in open(run_dir(tag, s) + "eval_samples.jsonl"):
        r = json.loads(l)
        if r["mode"] != mode:
            continue
        g = lambda key: next((v for k, v in r.items() if k.startswith(f"score/{key}/")), 0.0)
        det = g("detected_freq") > 0
        typ = ("c_relax" if g("hack_freq_constraint_relax") > 0 else "mod_input" if g("hack_freq_modify_input") > 0
               else "test_ow" if g("hack_freq_detectable") > 0 else "guard_blocked" if g("guard_blocked_freq") > 0
               else "other_hack" if g("hack_freq") > 0 else "futile" if g("futile_tamper_freq") > 0 else "clean")
        rows.setdefault(r["step"], []).append((typ, det))
    return rows


def train_log(tag, s):
    log = open(run_dir(tag, s) + "train.log", errors="ignore").read().replace("\r", "\n")
    roll = []
    for st, body in re.findall(r"\[timing @(\d+)\][^\n]*?reward\[([^\]]*)\]", log):
        kv = dict((a, float(b)) for a, b in re.findall(r"(\w+)=([0-9.]+)", body))
        roll.append({"step": int(st), "c_relax": 100 * kv.get("countdown_hack_constraint_relax_from_all", 0),
                     "mod_input": 100 * kv.get("countdown_hack_modify_input_from_all", 0),
                     "guard_blocked": 100 * kv.get("countdown_guard_blocked_from_all", 0),
                     "true": 100 * kv.get("countdown_true_from_all", 0)})
    pr = [dict(n=int(n), rule=int(a), probe=int(b), only=int(c), tot=int(d), sec=float(e), gib=float(f)) for n, a, b, c, d, e, f in
          re.findall(r"\[probe\] n=(\d+) rule_flags=(\d+) probe_flags=(\d+) probe_only=(\d+) total_flagged=(\d+) thr=[0-9.]+ probe_sec=([0-9.]+) torch_peak_GiB=([0-9.]+)", log)]
    big = [p for p in pr if p["n"] == 1024]
    for r, p in zip(roll, big):
        r.update(flag=100 * p["tot"] / p["n"], rule=100 * p["rule"] / p["n"], probe_only=100 * p["only"] / p["n"])
    times = re.findall(r"200/200 \[(\d+:\d+:\d+)<", log)
    return roll, pr, times[-1] if times else None


def main():
    print("== 1. fseval, step 200, n = 256 (%): deployed = forget 0 for GRAFT runs, forget 1 (trained policy) for dn")
    print(f"{'run':10s} {'seed':>4s} {'forget':>6s} " + " ".join(f"{k:>13s}" for k in K))
    summ = {}
    for tag in ("gr", "grprobe", "grprobepaid", "groprobe", "groprobewarm", "dn"):
        for s in SEEDS:
            f = fseval(tag, s)
            if not f:
                continue
            for sc in sorted(f):
                print(f"{tag:10s} {s:4d} {sc:6.1f} " + " ".join(f"{f[sc][k]:13.1f}" for k in K))
            dep = f[1.0] if tag == "dn" else f[0.0]
            summ.setdefault(tag, []).append(dep)
    print("\nmean deployed (n seeds), and seeds still hacking (> 10%):")
    for tag, L in summ.items():
        print(f"  {tag:8s} n={len(L)}  true {mean(x['true'] for x in L):5.1f}  hack {mean(x['hack'] for x in L):5.1f}  "
              f"c_relax {mean(x['c_relax'] for x in L):5.1f}  guard_blocked {mean(x['guard_blocked'] for x in L):4.1f}  "
              f"still hacking {sum(x['hack'] > 10 for x in L)}/{len(L)}")

    print("\n== 2. monitor (rule OR probe) on the training policy's eval answers (mode both, 64 prompts / eval step)")
    print("   share of each answer type the monitor flagged, pooled over 3 seeds; n in brackets")
    bins = [(0, 48), (60, 108), (120, 192)]
    for tag in ("grprobe", "grprobepaid", "groprobe", "groprobewarm", "gr"):
        if all(run_dir(tag, s) is None for s in SEEDS):
            continue
        print(f"  {tag}:")
        for lo, hi in bins:
            agg = {}
            for s in SEEDS:
                for st, rows in eval_flag_rates(tag, s).items():
                    if lo <= st <= hi:
                        for typ, det in rows:
                            agg.setdefault(typ, []).append(det)
            print(f"    steps {lo:3d}-{hi:3d}: " + "  ".join(f"{t} {100*mean(v):5.1f}% ({len(v)})" for t, v in sorted(agg.items())))

    print("\n== 3. training rollouts (1,024 task samples each): flagged share vs grader hack shares (%)")
    for tag, s in [(t, s) for t in ("grprobe", "grprobepaid", "groprobe", "groprobewarm") for s in SEEDS if run_dir(t, s)]:
        roll, pr, t = train_log(tag, s)
        print(f"  [{tag}]", end="")
        print(f"  seed {s} (train time {t}):")
        for r in roll[::5]:
            print(f"    step {r['step']:3d}: flagged {r.get('flag', float('nan')):5.1f} (rule {r.get('rule', float('nan')):4.1f}, probe only "
                  f"{r.get('probe_only', float('nan')):5.1f}) | grader c_relax {r['c_relax']:5.1f} mod_input {r['mod_input']:4.1f} "
                  f"guard_blocked {r['guard_blocked']:4.1f} | true {r['true']:5.1f}")
        big = [p for p in pr if p["n"] == 1024]
        print(f"    probe calls on rollouts: {len(big)}, mean {mean(p['sec'] for p in big):.1f} s each, "
              f"total {sum(p['sec'] for p in pr)/60:.1f} min (all calls); torch peak {max(p['gib'] for p in pr):.1f} GiB")


if __name__ == "__main__":
    main()
