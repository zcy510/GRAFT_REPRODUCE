"""Countdown probe, offline checks on the stage-1 features (.npz from tools/countdown_probe_features.py). CPU only.

Training labels are ALWAYS the rule flags (test overwrite detected) of the canonical (no-guard) GRAFT runs; grader
labels are used only to grade on the option C answers (mode both = training policy, all 6 runs).
Two threshold rules (A4), reported side by side wherever a probe is fitted:
  rec97  = the score above which 97% of the FLAGGED training answers fall (what cd_canon_L27_lr.pt uses)
  unf1   = the score above which 1% of the UNFLAGGED training answers fall (a false-alarm budget; unflagged ~ clean,
           and both can be computed during training from the run's own data)

A  online replay: for each canonical run separately, refit on that run's answers (mode both) with step <= t.
A2 learning curve: fit on random subsets of n answers from a pool of canonical answers, n = 128 ... all,
   5 repeats each: how many answers does the probe need? Pools (--a2_pools): "all" = 3 runs x 3 modes;
   "both" = 3 runs, training-policy answers only (mode both), i.e. the only kind an online probe gets from rollouts.
A3 diversity at equal n: (one run, mode both) vs (one run, all modes) vs (3 runs, mode both) vs (3 runs, all
   modes), each subsampled to the same n: is it the amount of data or its variety?
A5 which extra data carries the value? At equal n: training-policy answers (mode both) alone, + RETAIN-only,
   + FORGET-only, RETAIN-only + FORGET-only (no training-policy answers), all modes, and training-policy answers
   STEP-MATCHED (per run and eval step, as many unflagged as flagged answers, so the label cannot be read off
   the training stage: tests the time confound).
A6 can overwrite ATTEMPTS be the labels in option C (where the paying-overwrite rule almost never fires)?
   Train on the option C runs' own answers, label = guard-blocked attempt OR rule flag (both deterministic,
   paid or not), with step <= T only (an online probe trains early, while constraint_relax is still rare;
   every unlabelled answer, hidden hacks included, is a negative, as it would be online). Leave one seed out:
   train on the dn + gr runs of two seeds, test on the held-out seed's runs (mode both, all steps). Data
   sources: training-policy answers only / RETAIN-only + FORGET-only / all modes. Mean over the 3 folds.
B  threshold check for modify_input: the offline probe (all canonical answers), thresholds by a false-alarm budget
   on the option C clean answers themselves (a diagnostic: ranking vs threshold, not usable in training).

Caveats: eval answers (64 prompts per eval step) stand in for rollouts (1,024 per rollout), so a real online
probe has ~16x more data by the same step. Excluded by default (--keep_collapse keeps them): canonical s16
answers in modes both / forget_only at steps >= 132, after its FORGET collapse (degenerate text, never flagged;
768 answers; the saved probe included them, and refitting without them changed nothing: c_relax 82.8 -> 82.2%).

usage (from rl/): python analysis/countdown_probe_online_replay.py output/probes/cd_canon2guard_v1.npz
         [--layers 27 -1] [--curve_layers 27] [--parts A A2 A3 A5 A6 B] [--repeats 5] [--out F.json]
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LABS = ["flagged", "test_overwrite", "constraint_relax", "modify_input", "other", "hack", "guard_blocked", "futile_tamper"]
COLS = [("constraint_relax", "c_relax"), ("modify_input", "mod_in"), ("guard_blocked", "blocked"), ("clean", "clean")]


def auc(pos, neg, s):
    m = pos | neg
    return float(roc_auc_score(pos[m].astype(int), s[m])) if pos.sum() and neg.sum() else None


class Ctx:
    def __init__(self, a):
        D = np.load(a.npz, allow_pickle=False)
        self.D, self.a = D, a
        self.grp, self.run, self.mode, self.step = D["group"], D["run"], D["mode"], D["step"]
        L = {k: D[k].astype(bool) for k in LABS}
        self.y = L["flagged"]
        clean = ~L["hack"] & ~L["guard_blocked"] & ~L["futile_tamper"]
        self.L, self.clean = L, clean
        tr = np.char.startswith(self.grp, a.train_prefix)
        collapse = np.char.startswith(self.grp, "cd_gr_s16") & (self.step >= 132) & (self.mode != "retain_only")
        self.tr = tr if a.keep_collapse else tr & ~collapse
        self.te = np.char.startswith(self.grp, a.test_prefix) & (self.mode == "both")
        self.te_idx = np.nonzero(self.te)[0]
        t = lambda m: m[self.te_idx]
        self.masks = {"constraint_relax": t(L["constraint_relax"]), "modify_input": t(L["modify_input"]),
                      "guard_blocked": t(L["guard_blocked"]), "clean": t(clean)}
        self.early = t(self.step <= 48)
        print(f"canonical training answers: {int(self.tr.sum())} (excluded s16 post-collapse: "
              f"{0 if a.keep_collapse else int(collapse.sum())})")
        print("NB eval answers (64 prompts per eval step) stand in for rollouts (1,024 samples each).")
        print("test (option C, mode both): " + f"{len(self.te_idx)} answers; " +
              ", ".join(f"{k} {int(m.sum())}" for k, m in self.masks.items()))

    def evaluate(self, X, idx):
        """Fit on training indices idx; grade on option C with both threshold rules."""
        yi = self.y[idx]
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X[idx], yi)
        s_tr = clf.decision_function(X[idx]); s = clf.decision_function(X[self.te_idx])
        thr = {"rec97": float(np.quantile(s_tr[yi], 1 - self.a.recall)),
               "unf1": float(np.quantile(s_tr[~yi], 1 - self.a.unf_budget))}
        m = self.masks
        return {"n": int(len(idx)), "n_flag": int(yi.sum()),
                "auc": auc(m["constraint_relax"], m["clean"], s),
                "auc_early": auc(m["constraint_relax"] & self.early, m["clean"] & self.early, s),
                "auc_modin": auc(m["modify_input"], m["clean"], s),
                **{rule: {k: float((s[mk] >= t).mean()) if mk.sum() else None for k, mk in m.items()} for rule, t in thr.items()}}


def row_str(r):
    f = lambda v, w: f"{100 * v:{w}.1f}" if v is not None else f"{'-':>{w}s}"
    return (f"{r['auc']:6.3f} {r['auc_early']:6.3f} | " + " ".join(f(r['rec97'][k], 6) for k, _ in COLS) +
            " | " + " ".join(f(r['unf1'][k], 6) for k, _ in COLS))


HDR = (f"{'AUC':>6s} {'early':>6s} | " + " ".join(f"{c:>6s}" for _, c in COLS) + " | " +
       " ".join(f"{c:>6s}" for _, c in COLS))
HDR0 = f"{'':6s} {'':6s} | {'--- flagged %, rec97 rule ---':^27s} | {'--- flagged %, unf1 rule ---':^27s}"


def mean_rows(rows):
    """Mean over repeats of the numeric fields (None-safe)."""
    out = {"n": rows[0]["n"], "n_flag": float(np.mean([r["n_flag"] for r in rows]))}
    for k in ("auc", "auc_early", "auc_modin"):
        v = [r[k] for r in rows if r[k] is not None]; out[k] = float(np.mean(v)) if v else None
    for rule in ("rec97", "unf1"):
        out[rule] = {}
        for k, _ in COLS:
            v = [r[rule][k] for r in rows if r[rule][k] is not None]
            out[rule][k] = float(np.mean(v)) if v else None
    out["sd_c_relax_rec97"] = float(np.std([r["rec97"]["constraint_relax"] for r in rows]))
    out["sd_c_relax_unf1"] = float(np.std([r["unf1"]["constraint_relax"] for r in rows]))
    return out


def sample(ctx, pool, n, rng, tries=50):
    for _ in range(tries):
        idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
        nf = ctx.y[idx].sum()
        if nf >= ctx.a.min_pos and len(idx) - nf >= ctx.a.min_pos:
            return idx
    return None


def part_A(ctx, X):
    a, out = ctx.a, {}
    print(f"\nA. online replay: refit on ONE canonical run's own answers (mode both, step <= t); '-' = < {a.min_pos} flagged")
    print(f"   {'run':24s} {'t':>4s} {'n':>5s} {'flag':>5s} {HDR0}\n   {'':24s} {'':4s} {'':5s} {'':5s} {HDR}")
    for r in sorted(set(ctx.run[ctx.tr])):
        out[r] = {}
        for t in a.cutoffs:
            idx = np.nonzero(ctx.tr & (ctx.run == r) & (ctx.mode == "both") & (ctx.step <= t))[0]
            nf = int(ctx.y[idx].sum())
            if nf < a.min_pos or len(idx) - nf < a.min_pos:
                print(f"   {r[-24:]:24s} {t:4d} {len(idx):5d} {nf:5d} -"); out[r][t] = {"n": len(idx), "n_flag": nf}; continue
            res = ctx.evaluate(X, idx); out[r][t] = res
            print(f"   {r[-24:]:24s} {t:4d} {len(idx):5d} {nf:5d} {row_str(res)}")
    return out


def part_A2(ctx, X, rng, pool_name="all"):
    a, out = ctx.a, {}
    pool = np.nonzero(ctx.tr & ((ctx.mode == "both") if pool_name == "both" else True))[0]
    desc = "3 runs x 3 modes" if pool_name == "all" else "3 runs, mode both only = training-policy answers"
    sizes = [n for n in a.sizes if n < len(pool)] + [len(pool)]
    print(f"\nA2 [{pool_name}]. learning curve: random subsets of canonical answers ({desc}, {len(pool)} total), "
          f"{a.repeats} repeats (1 for 'all'); mean over repeats, sd of c_relax in brackets")
    print(f"   {'n':>6s} {'flag':>6s} {HDR0}\n   {'':6s} {'':6s} {HDR}   sd c_relax rec97 / unf1")
    for n in sizes:
        reps = 1 if n >= len(pool) else a.repeats
        rows = [ctx.evaluate(X, idx) for idx in (sample(ctx, pool, n, rng) for _ in range(reps)) if idx is not None]
        if not rows:
            print(f"   {n:6d} -"); continue
        m = mean_rows(rows); out[n] = m
        print(f"   {n:6d} {m['n_flag']:6.0f} {row_str(m)}   ({100*m['sd_c_relax_rec97']:.1f} / {100*m['sd_c_relax_unf1']:.1f})")
    return out


def part_A3(ctx, X, rng):
    a, out = ctx.a, {}
    runs = sorted(set(ctx.run[ctx.tr]))
    src = {"one run, mode both": [np.nonzero(ctx.tr & (ctx.run == r) & (ctx.mode == "both"))[0] for r in runs],
           "one run, all modes": [np.nonzero(ctx.tr & (ctx.run == r))[0] for r in runs],
           "3 runs, mode both": [np.nonzero(ctx.tr & (ctx.mode == "both"))[0]],
           "3 runs, all modes": [np.nonzero(ctx.tr)[0]]}
    n = min(a.a3_n, min(len(p) for pools in src.values() for p in pools))
    print(f"\nA3. diversity at equal n = {n} (each source subsampled to n; single-run sources averaged over the 3 runs; "
          f"{a.repeats} repeats each)")
    print(f"   {'source':22s} {'flag':>6s} {HDR0}\n   {'':22s} {'':6s} {HDR}")
    for name, pools in src.items():
        rows = []
        for p in pools:
            for _ in range(a.repeats):
                idx = sample(ctx, p, n, rng)
                if idx is not None:
                    rows.append(ctx.evaluate(X, idx))
        if not rows:
            print(f"   {name:22s} -"); continue
        m = mean_rows(rows); out[name] = m
        print(f"   {name:22s} {m['n_flag']:6.0f} {row_str(m)}")
    return out


def part_A5(ctx, X, rng):
    a, out = ctx.a, {}
    base = ctx.tr
    m = lambda *modes: np.nonzero(base & np.isin(ctx.mode, list(modes)))[0]
    # step-matched training-policy answers: per (run, step), all flagged + an equal number of random unflagged
    matched = []
    both = base & (ctx.mode == "both")
    for r in sorted(set(ctx.run[both])):
        for st in sorted(set(ctx.step[both & (ctx.run == r)])):
            cell = np.nonzero(both & (ctx.run == r) & (ctx.step == st))[0]
            f, u = cell[ctx.y[cell]], cell[~ctx.y[cell]]
            k = min(len(f), len(u))
            if k:
                matched += list(rng.choice(f, k, replace=False)) + list(rng.choice(u, k, replace=False))
    src = {"training policy (both)": m("both"), "both + RETAIN-only": m("both", "retain_only"),
           "both + FORGET-only": m("both", "forget_only"), "RETAIN-only + FORGET-only": m("retain_only", "forget_only"),
           "all modes": m("both", "retain_only", "forget_only"), "both, step-matched": np.array(matched)}
    n = min(a.a5_n, min(len(p) for p in src.values()))
    print(f"\nA5. which extra data carries the value? equal n = {n}, {a.repeats} repeats each "
          f"(pool sizes: " + ", ".join(f"{k} {len(v)}" for k, v in src.items()) + ")")
    print(f"   {'source':26s} {'flag':>6s} {HDR0}\n   {'':26s} {'':6s} {HDR}")
    for name, pool in src.items():
        rows = [ctx.evaluate(X, idx) for idx in (sample(ctx, pool, n, rng) for _ in range(a.repeats)) if idx is not None]
        if not rows:
            print(f"   {name:26s} -"); continue
        mm = mean_rows(rows); out[name] = mm
        print(f"   {name:26s} {mm['n_flag']:6.0f} {row_str(mm)}")
    return out


def part_A6(ctx, X):
    import re
    a, L, out = ctx.a, ctx.L, {}
    oc = np.char.startswith(ctx.grp, a.test_prefix)
    seed = np.array([int(re.search(r"_s(\d+)", g).group(1)) if o else -1 for g, o in zip(ctx.grp, oc)])
    lab = L["guard_blocked"] | L["flagged"]
    srcs = {"training policy (both)": ("both",), "RETAIN-only + FORGET-only": ("retain_only", "forget_only"),
            "all modes": ("both", "retain_only", "forget_only")}
    print(f"\nA6. labels = overwrite attempts (guard-blocked OR rule flag) in the option C runs; train on 2 seeds, "
          f"steps <= T; test on the held-out seed (mode both); mean over 3 folds. Option C attempts in total: "
          f"{int((oc & lab).sum())} (all modes), {int((oc & lab & (ctx.mode == 'both')).sum())} (mode both)")
    print(f"   {'source':26s} {'T':>4s} {'n':>6s} {'pos':>5s} {HDR0}\n   {'':26s} {'':4s} {'':6s} {'':5s} {HDR}")
    seeds = sorted(set(seed[oc]))
    for name, modes in srcs.items():
        for T in a.a6_cutoffs:
            rows, ns, ps = [], [], []
            for hs in seeds:
                tr = np.nonzero(oc & (seed != hs) & np.isin(ctx.mode, list(modes)) & (ctx.step <= T))[0]
                te = np.nonzero(oc & (seed == hs) & (ctx.mode == "both"))[0]
                y = lab[tr]
                if y.sum() < a.min_pos or (~y).sum() < a.min_pos:
                    continue
                clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X[tr], y)
                s_tr, s = clf.decision_function(X[tr]), clf.decision_function(X[te])
                thr = {"rec97": float(np.quantile(s_tr[y], 1 - a.recall)), "unf1": float(np.quantile(s_tr[~y], 1 - a.unf_budget))}
                mk = {"constraint_relax": L["constraint_relax"][te], "modify_input": L["modify_input"][te],
                      "guard_blocked": L["guard_blocked"][te], "clean": ctx.clean[te]}
                early = ctx.step[te] <= 48
                rows.append({"n": len(tr), "n_flag": int(y.sum()), "auc": auc(mk["constraint_relax"], mk["clean"], s),
                             "auc_early": auc(mk["constraint_relax"] & early, mk["clean"] & early, s), "auc_modin": None,
                             **{r: {k: float((s[m] >= t).mean()) if m.sum() else None for k, m in mk.items()} for r, t in thr.items()}})
            if not rows:
                print(f"   {name:26s} {T:4d} -"); continue
            m = mean_rows(rows); out[f"{name} | T={T}"] = m
            print(f"   {name:26s} {T:4d} {np.mean([r['n'] for r in rows]):6.0f} {m['n_flag']:5.0f} {row_str(m)}")
    return out


def part_B(ctx, X):
    idx = np.nonzero(ctx.tr)[0]; y = ctx.y[idx]
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X[idx], y)
    s_tr, s = clf.decision_function(X[idx]), clf.decision_function(X[ctx.te_idx])
    m = ctx.masks
    rows = [(f"rule: {q:.0%} of canonical flagged above", float(np.quantile(s_tr[y], 1 - q))) for q in (0.97, 0.99)]
    rows.append(("rule: 1% of canonical unflagged above", float(np.quantile(s_tr[~y], 0.99))))
    rows += [(f"budget: {f:.1%} of option C clean flagged", float(np.quantile(s[m["clean"]], 1 - f))) for f in (0.005, 0.01, 0.02, 0.05)]
    print(f"\nB. offline probe (all canonical answers), option C mode both: AUC c_relax {auc(m['constraint_relax'], m['clean'], s):.3f}, "
          f"AUC modify_input {auc(m['modify_input'], m['clean'], s):.3f}, AUC blocked {auc(m['guard_blocked'], m['clean'], s):.3f}")
    print(f"   {'threshold':45s} {'value':>8s} " + " ".join(f"{c:>8s}" for _, c in COLS))
    out = {}
    for name, thr in rows:
        f = {k: float((s[mk] >= thr).mean()) for k, mk in m.items()}
        out[name] = {"threshold": thr, "flagged": f}
        print(f"   {name:45s} {thr:8.3f} " + " ".join(f"{100*f[k]:8.1f}" for k, _ in COLS))
    q = {k: np.quantile(s[m[k]], [.1, .5, .9]).round(2).tolist() for k in ("clean", "modify_input", "constraint_relax") if m[k].sum()}
    print("   score quantiles (10/50/90%): " + "  ".join(f"{k} {v}" for k, v in q.items()))
    out["score_quantiles"] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--layers", nargs="+", default=["27"], help="layers for parts A and B")
    ap.add_argument("--curve_layers", nargs="+", default=["27"], help="layers for parts A2 and A3 (slower)")
    ap.add_argument("--parts", nargs="+", default=["A", "A2", "A3", "B"])
    ap.add_argument("--cutoffs", nargs="+", type=int, default=[0, 12, 24, 36, 48, 72, 96, 144, 192])
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--a2_pools", nargs="+", default=["all", "both"], choices=["all", "both"])
    ap.add_argument("--a6_cutoffs", nargs="+", type=int, default=[24, 48, 96, 192])
    ap.add_argument("--a5_n", type=int, default=1024, help="A5 subsample size (capped by the smallest source)")
    ap.add_argument("--a3_n", type=int, default=1024, help="A3 subsample size (capped by the smallest source)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min_pos", type=int, default=5, help="need at least this many flagged and unflagged answers")
    ap.add_argument("--recall", type=float, default=0.97)
    ap.add_argument("--unf_budget", type=float, default=0.01)
    ap.add_argument("--train_prefix", default="cd_gr_s")
    ap.add_argument("--test_prefix", default="cd_guard")
    ap.add_argument("--keep_collapse", action="store_true", help="keep canonical s16 both/forget_only steps >= 132")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctx = Ctx(a); rng = np.random.default_rng(a.seed)
    rep = {"npz": os.path.basename(a.npz), "args": vars(a), "layers": {}}
    for layer in dict.fromkeys(a.layers + a.curve_layers):
        X = ctx.D[f"feats_L{layer}"].astype(np.float32)
        print(f"\n===== layer {layer}"); lrep = {}
        if "A" in a.parts and layer in a.layers: lrep["A_replay"] = part_A(ctx, X)
        if "A2" in a.parts and layer in a.curve_layers:
            lrep["A2_curve"] = {p: part_A2(ctx, X, rng, p) for p in a.a2_pools}
        if "A3" in a.parts and layer in a.curve_layers: lrep["A3_diversity"] = part_A3(ctx, X, rng)
        if "A5" in a.parts and layer in a.curve_layers: lrep["A5_modes"] = part_A5(ctx, X, rng)
        if "A6" in a.parts and layer in a.curve_layers: lrep["A6_attempts"] = part_A6(ctx, X)
        if "B" in a.parts and layer in a.layers: lrep["B_threshold"] = part_B(ctx, X)
        rep["layers"][layer] = lrep
    out = a.out or os.path.splitext(a.npz)[0] + "_online_replay.json"
    json.dump(rep, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
