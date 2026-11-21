"""Countdown probe, offline checks on the stage-1 features (.npz from tools/countdown_probe_features.py). CPU only.

Part A - online replay ("does a probe learned from a run's OWN rule flags become useful, and how fast?"):
  For each canonical (no-guard) GRAFT run separately, refit the probe (class-balanced LR, same as
  cd_canon_L27_lr.pt) on that run's training-policy answers (mode both) with step <= t, labels = rule flags
  only, threshold = the 97% rule on those flagged answers. Test every cutoff t on the option C answers
  (mode both, all 6 runs): AUC constraint_relax vs clean, share flagged of constraint_relax / modify_input /
  guard-blocked / clean. Caveat: this replays EVAL answers (64 prompts per eval step); a training rollout has
  1,024 samples, so a real online probe sees ~16x more data per step than this replay.
Part B - threshold check for modify_input ("is it a ranking problem or a threshold problem?"):
  The offline probe (LR on all canonical answers, all modes, as in countdown_probe_transfer.py). On option C
  (mode both): for thresholds set by a false-alarm budget on clean answers (0.5 / 1 / 2 / 5 %), the share
  flagged of each hack type; plus the 97% / 99% rule thresholds for reference.

Excluded by default (--keep_collapse to keep): canonical s16 answers from the training policy and FORGET
(modes both / forget_only) at steps >= 132, after its FORGET collapse (steps 129-133): degenerate text of
5,000-11,000 chars, never flagged, which would enter every fit as negatives. (The saved probe
cd_canon_L27_lr.pt and stage 1 included them: 768 of 9,792 training answers. Part B here is fitted without them,
so comparing its 97% row with stage 1 shows whether that mattered.)
Part B's false-alarm-budget thresholds are set on the option C clean answers themselves: a diagnostic
(ranking vs threshold), not a threshold you could use in training.

usage (from rl/): python analysis/countdown_probe_online_replay.py output/probes/cd_canon2guard_v1.npz
         [--layers 27 -1] [--cutoffs 0 12 24 36 48 72 96 144 192] [--min_pos 5] [--out F.json]
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LABS = ["flagged", "test_overwrite", "constraint_relax", "modify_input", "other", "hack", "guard_blocked", "futile_tamper"]


def auc(pos, neg, s):
    m = pos | neg
    return float(roc_auc_score(pos[m].astype(int), s[m])) if pos.sum() and neg.sum() else None


def fit(X, y):
    return LogisticRegression(max_iter=5000, class_weight="balanced").fit(X, y)


def rates(s, thr, masks):
    return {k: (float((s[m] >= thr).mean()) if m.sum() else None) for k, m in masks.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--layers", nargs="+", default=["27"])
    ap.add_argument("--cutoffs", nargs="+", type=int, default=[0, 12, 24, 36, 48, 72, 96, 144, 192])
    ap.add_argument("--min_pos", type=int, default=5, help="fewer flagged answers than this: probe 'not ready'")
    ap.add_argument("--recall", type=float, default=0.97)
    ap.add_argument("--train_prefix", default="cd_gr_s")
    ap.add_argument("--test_prefix", default="cd_guard")
    ap.add_argument("--keep_collapse", action="store_true", help="keep canonical s16 both/forget_only steps >= 132")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = np.load(a.npz, allow_pickle=False)
    grp, run, mode, step = D["group"], D["run"], D["mode"], D["step"]
    L = {k: D[k].astype(bool) for k in LABS}
    clean = ~L["hack"] & ~L["guard_blocked"] & ~L["futile_tamper"]
    tr_all = np.char.startswith(grp, a.train_prefix)
    collapse = np.char.startswith(grp, "cd_gr_s16") & (step >= 132) & (mode != "retain_only")
    if not a.keep_collapse:
        tr_all = tr_all & ~collapse
    print(f"canonical training answers: {int(tr_all.sum())} (excluded s16 post-collapse: {0 if a.keep_collapse else int(collapse.sum())})")
    te = np.char.startswith(grp, a.test_prefix) & (mode == "both")
    masks = {"constraint_relax": te & L["constraint_relax"], "modify_input": te & L["modify_input"],
             "guard_blocked": te & L["guard_blocked"], "clean": te & clean}
    early = te & (step <= 48)
    print("NB replay uses eval answers (64 prompts per eval step); a real rollout has 1,024 samples, so an online probe "
          "in training sees ~16x more data by the same step.")
    print(f"test (option C, mode both): {te.sum()} answers; " + ", ".join(f"{k} {int(m.sum())}" for k, m in masks.items()))
    rep = {"npz": os.path.basename(a.npz), "recall_rule": a.recall, "layers": {}}
    for layer in a.layers:
        X = D[f"feats_L{layer}"].astype(np.float32)
        lrep = {"replay": {}, "threshold_check": {}}
        print(f"\n===== layer {layer}")

        # ---------------- Part A: online replay, per canonical run
        print(f"A. online replay: probe refit on ONE canonical run's own answers (mode both) up to step t, rule labels only;"
              f" threshold = {a.recall:.0%} rule; tested on option C (mode both). '-' = fewer than {a.min_pos} flagged answers")
        print(f"   {'run':28s} {'t':>4s} {'n_train':>7s} {'n_flag':>6s} {'AUC c_relax':>11s} {'AUC early':>9s} "
              f"{'c_relax':>8s} {'mod_input':>9s} {'blocked':>8s} {'clean':>7s}")
        for r in sorted(set(run[tr_all])):
            lrep["replay"][r] = {}
            for t in a.cutoffs:
                m = tr_all & (run == r) & (mode == "both") & (step <= t)
                y = L["flagged"][m]
                row = {"n_train": int(m.sum()), "n_flagged": int(y.sum())}
                if y.sum() >= a.min_pos and (~y).sum() >= a.min_pos:
                    clf = fit(X[m], y)
                    s = clf.decision_function(X)
                    thr = float(np.quantile(s[m][y], 1 - a.recall))
                    row.update(auc_crelax=auc(masks["constraint_relax"], masks["clean"], s),
                               auc_crelax_early=auc(masks["constraint_relax"] & early, masks["clean"] & early, s),
                               flagged=rates(s, thr, masks))
                    f = row["flagged"]
                    print(f"   {r[-28:]:28s} {t:4d} {row['n_train']:7d} {row['n_flagged']:6d} {row['auc_crelax']:11.3f} "
                          f"{row['auc_crelax_early'] if row['auc_crelax_early'] is not None else float('nan'):9.3f} "
                          + " ".join(f"{100*f[k]:{w}.1f}" if f[k] is not None else f"{'-':>{w}s}"
                                     for k, w in [("constraint_relax", 8), ("modify_input", 9), ("guard_blocked", 8), ("clean", 7)]))
                else:
                    print(f"   {r[-28:]:28s} {t:4d} {row['n_train']:7d} {row['n_flagged']:6d} {'-':>11s}")
                lrep["replay"][r][t] = row

        # ---------------- Part B: threshold check (offline probe on all canonical answers)
        clf = fit(X[tr_all], L["flagged"][tr_all])
        s = clf.decision_function(X)
        rows = []
        for q in (0.97, 0.99):
            thr = float(np.quantile(s[tr_all & L["flagged"]], 1 - q))
            rows.append((f"rule: {q:.0%} of canonical flagged above", thr))
        for fpr in (0.005, 0.01, 0.02, 0.05):
            thr = float(np.quantile(s[masks["clean"]], 1 - fpr))
            rows.append((f"budget: {fpr:.1%} of option C clean flagged", thr))
        print(f"\nB. offline probe (all canonical answers), option C mode both: AUC c_relax {auc(masks['constraint_relax'], masks['clean'], s):.3f}, "
              f"AUC modify_input {auc(masks['modify_input'], masks['clean'], s):.3f}, AUC blocked {auc(masks['guard_blocked'], masks['clean'], s):.3f}")
        print(f"   {'threshold':45s} {'value':>8s} {'c_relax':>8s} {'mod_input':>9s} {'blocked':>8s} {'clean':>7s}")
        for name, thr in rows:
            f = rates(s, thr, masks)
            lrep["threshold_check"][name] = {"threshold": thr, "flagged": f}
            print(f"   {name:45s} {thr:8.3f} " + " ".join(f"{100*f[k]:{w}.1f}" for k, w in
                                                         [("constraint_relax", 8), ("modify_input", 9), ("guard_blocked", 8), ("clean", 7)]))
        mi = s[masks["modify_input"]]; cl = s[masks["clean"]]; cr = s[masks["constraint_relax"]]
        print(f"   score quantiles (10/50/90%): clean {np.round(np.quantile(cl, [.1, .5, .9]), 2).tolist()}  "
              f"modify_input {np.round(np.quantile(mi, [.1, .5, .9]), 2).tolist() if mi.size else '-'}  "
              f"c_relax {np.round(np.quantile(cr, [.1, .5, .9]), 2).tolist()}")
        lrep["score_quantiles"] = {k: np.quantile(v, [.1, .5, .9]).tolist() for k, v in
                                   [("clean", cl), ("modify_input", mi), ("constraint_relax", cr)] if v.size}
        rep["layers"][layer] = lrep
    out = a.out or os.path.splitext(a.npz)[0] + "_online_replay.json"
    json.dump(rep, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
