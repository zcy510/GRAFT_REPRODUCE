"""Countdown probe experiment (option C, guard 1.0): training-process data, plain GRAFT vs GRAFT + probe.
Tidy loaders (one row per observation) for countdown_probe_training.ipynb; also runnable as a script for tables.

Runs (rl/output/arrhenius/):
  GRAFT          cd_guard1.0_gr_s<seed>/cd_gr_gh200x1_st200_s<seed>_g1/            (array 4990465, the baseline)
  GRAFT + probe  cd_guard1.0_grprobe_s<seed>/cd_gr_gh200x1_st200_s<seed>_g1_probe/ (array 5559602)
Sources:
  routing_eval.jsonl   every 12 steps, 64 eval prompts per mode: model metrics (deployed = retain_only, training
                       policy = both) and the monitor's flag rate (detected_freq/<monitor>)
  eval_samples.jsonl   the same eval answers one by one: monitor flag + grader labels -> probe recall / false alarms
  routing_trace.jsonl  every 20 steps, the WHOLE training rollout (1,024 samples; anchors excluded): share flagged and
                       each hack's mean among flagged / unflagged -> exact counts of what was routed to FORGET
  train.log            "[probe] n=1024 ..." once per rollout (every 4 steps; probe runs only): flag counts, seconds
  <sweep>_fseval/      step 200, 256 prompts: final numbers (deployed = forget 0.0)
Categories (grader, exclusive, priority order): test_overwrite, constraint_relax, modify_input, other (paying hacks),
guard_blocked (overwrite attempt that the guard made pay 0), clean (everything else, incl. wrong answers).
All rates in %."""
import glob, json, os, re
import numpy as np
import pandas as pd

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("CD_PROBE_OUT", os.path.join(RL, "output", "arrhenius"))   # override for tests
SEEDS = (9, 15, 16)
ARMS = {"GRAFT": "cd_guard1.0_gr_s{s}/cd_gr_gh200x1_st200_s{s}_g1",
        "GRAFT + probe": "cd_guard1.0_grprobe_s{s}/cd_gr_gh200x1_st200_s{s}_g1_probe"}
FSEVAL = {"GRAFT": "cd_guard1.0_gr_s{s}_fseval/cd_gr_gh200x1_st200_s{s}_g1.json",
          "GRAFT + probe": "cd_guard1.0_grprobe_s{s}_fseval/cd_gr_gh200x1_st200_s{s}_g1_probe.json"}
BASE_FSEVAL = "cd_guard1.0_gr_s9_fseval/cd_gr_gh200x1_st200_s9_g1__r0.0.json"   # base SFT model, guard 1.0
MODES = {"retain_only": "deployed", "both": "training policy", "forget_only": "FORGET only"}
METRICS = {"task": "retain/", "hack": "hack_freq/", "constraint_relax": "hack_freq_constraint_relax/",
           "modify_input": "hack_freq_modify_input/", "test_overwrite": "hack_freq_detectable/",
           "guard_blocked": "guard_blocked_freq/", "flagged": "detected_freq/"}
CATS = ["test_overwrite", "constraint_relax", "modify_input", "other", "guard_blocked", "clean"]
_COMP = {"test_overwrite": "countdown_hack_test_overwrite_from_all", "constraint_relax": "countdown_hack_constraint_relax_from_all",
         "modify_input": "countdown_hack_modify_input_from_all", "other": "countdown_hack_other_from_all",
         "guard_blocked": "countdown_guard_blocked_from_all"}
_EV = {"test_overwrite": "hack_freq_detectable", "constraint_relax": "hack_freq_constraint_relax",
       "modify_input": "hack_freq_modify_input", "other": "hack_freq_other", "guard_blocked": "guard_blocked_freq"}


def run_dir(arm, seed):
    d = os.path.join(OUT, ARMS[arm].format(s=seed))
    return d if os.path.isdir(d) else None


def available():
    return {(a, s): run_dir(a, s) for a in ARMS for s in SEEDS if run_dir(a, s)}


def _jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def eval_curves():
    """routing_eval.jsonl -> one row per (arm, seed, step, mode) with METRICS in %."""
    rows = []
    for (arm, s), d in available().items():
        for r in _jl(os.path.join(d, "routing_eval.jsonl")):
            for mode in MODES:
                row = {"arm": arm, "seed": s, "step": r["step"], "mode": mode}
                for m, pre in METRICS.items():
                    v = [x for k, x in r.items() if k.startswith(f"{mode}/{pre}")]
                    row[m] = 100 * v[0] if v else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _cat_eval(r):
    for c in CATS[:-1]:
        if (r.get(f"score/{_EV[c]}/score_threshold") or 0) > 0:
            return c
    return "clean"


def eval_answers():
    """eval_samples.jsonl -> one row per eval answer: arm, seed, step, mode, category, flagged (monitor)."""
    rows = []
    for (arm, s), d in available().items():
        for r in _jl(os.path.join(d, "eval_samples.jsonl")):
            fk = [k for k in r if k.startswith("score/detected_freq/")]
            rows.append({"arm": arm, "seed": s, "step": r["step"], "mode": r["mode"], "category": _cat_eval(r),
                         "flagged": bool(fk and (r[fk[0]] or 0) > 0)})
    return pd.DataFrame(rows)


def rollout_routing():
    """routing_trace.jsonl rollout records -> per (arm, seed, step, category): samples and flagged samples among the
    1,024 non-anchor rollout samples (component means are over non-anchor samples, checked against all_mean).
    Categories here come from the reward components, which overlap only in ways the grader excludes
    (constraint_relax excludes test_overwrite, etc.); clean = the rest."""
    rows = []
    for (arm, s), d in available().items():
        for r in _jl(os.path.join(d, "routing_trace.jsonl")):
            if r.get("trace") != "rollout":
                continue
            n = r["n_total"] - r["n_coherence"]; nf = r["n_rh"]; nu = n - nf
            tot_f = tot_u = 0.0
            for c in CATS[:-1]:
                k = f"reward/{_COMP[c]}/"
                f_ = (r.get(k + "rh_mean") or 0.0) * nf; u_ = (r.get(k + "nonrh_mean") or 0.0) * nu
                tot_f += f_; tot_u += u_
                rows.append({"arm": arm, "seed": s, "step": r["step"], "category": c, "n": f_ + u_, "n_flagged": f_})
            rows.append({"arm": arm, "seed": s, "step": r["step"], "category": "clean", "n": n - tot_f - tot_u, "n_flagged": nf - tot_f})
    df = pd.DataFrame(rows)
    if len(df):
        df[["n", "n_flagged"]] = df[["n", "n_flagged"]].round().astype(int)
    return df


_PROBE = re.compile(r"\[probe\] n=(\d+) rule_flags=(\d+) probe_flags=(\d+) probe_only=(\d+) total_flagged=(\d+) thr=\S+ probe_sec=([\d.]+)")


def probe_log():
    """train.log '[probe]' lines of the probe runs, rollouts only (n = 1024; eval calls have n = 64).
    Rollout k covers optimizer steps 4k+1..4k+4; reported at step 4k."""
    rows = []
    for (arm, s), d in available().items():
        f = os.path.join(d, "train.log")
        if arm != "GRAFT + probe" or not os.path.exists(f):
            continue
        k = 0
        for line in open(f, errors="replace").read().replace("\r", "\n").splitlines():
            m = _PROBE.search(line)
            if m and int(m.group(1)) == 1024:
                n, rule, pr, po, tot, sec = m.groups()
                rows.append({"arm": arm, "seed": s, "step": 4 * k, "n": int(n), "rule_flags": int(rule), "probe_flags": int(pr),
                             "probe_only": int(po), "total_flagged": int(tot), "probe_sec": float(sec)})
                k += 1
    return pd.DataFrame(rows)


def fseval():
    """Final eval, step 200, n = 256: one row per (arm, seed, forget scale) + the base model; METRICS in %."""
    rows = []
    jobs = [(a, s, os.path.join(OUT, FSEVAL[a].format(s=s))) for a in ARMS for s in SEEDS] + [("base model", None, os.path.join(OUT, BASE_FSEVAL))]
    for arm, s, f in jobs:
        if not os.path.exists(f):
            continue
        for sc, v in json.load(open(f))["scales"].items():
            row = {"arm": arm, "seed": s, "forget": float(sc)}
            for m, pre in METRICS.items():
                x = [val for k, val in v.items() if k.startswith(pre)]
                row[m] = 100 * x[0] if x else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def rates(df, by):
    """Per group: recall per hack category (share flagged) and false-alarm rate on clean, from counts."""
    if "n" not in df:                       # one row per answer -> counts
        df = df.assign(n=1, n_flagged=df.flagged.astype(int))
    g = df.groupby(by + ["category"])[["n", "n_flagged"]].sum().reset_index()
    g["flagged_pct"] = 100 * g.n_flagged / g.n.where(g.n > 0)
    return g


if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("runs found:", sorted(available()))
    f = fseval()
    if len(f):
        print("\nfseval, deployed (forget 0):\n", f[f.forget == 0].round(1).to_string(index=False))
    rr = rollout_routing()
    if len(rr):
        print("\ntraining rollouts: % of each category routed to FORGET (all logged rollouts, 3 seeds pooled):")
        print(rates(rr, ["arm"]).pivot(index="arm", columns="category", values="flagged_pct").round(1))
