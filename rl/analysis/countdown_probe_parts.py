"""Countdown probe with PER-PART pooling (offline test before changing the online probe).

Features: tools/countdown_probe_features.py --parts (probe_features.backbone_part_features): layer-L hidden states
of the frozen base model mean-pooled separately over the reasoning (before </think>), the answer after it, the
solution.py value, the test.py value, and all completion tokens ("all" = the current probe's feature).
Feature variants compared:
  single parts: all / reasoning / answer / solution / test
  concat: [reasoning, solution, test] and [all, reasoning] (each part L2-normalised, then concatenated)
  max-of-parts: one probe per part (reasoning, solution, test), each with its own threshold; flagged if ANY part
     is above its threshold (score for AUC = max over parts of score - threshold)
Two settings, both graded on the option C answers (mode both = training policy):
  T  transfer: train on the canonical runs' rule flags (all modes, s16 post-collapse excluded) = the offline probe
  O  online-like: train on option C overwrite ATTEMPTS (rule OR guard-blocked), steps <= 48, all modes, leave one
     seed out (train on 2 seeds, test on the third; mean over folds) = what the online probe learns from
Thresholds: rec97 (97% of training positives above) and unf1 (1% of training negatives above).

usage (from rl/): python analysis/countdown_probe_parts.py output/probes/cd_canon2guard_parts_v1.npz [--layer 27]
"""
import argparse, json, os, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("npz")
ap.add_argument("--layer", default="27")
ap.add_argument("--T", type=int, default=48)
ap.add_argument("--compare_npz", default=None, help="stage-1 npz: check that part 'all' reproduces its features")
ap.add_argument("--out", default=None)
a = ap.parse_args()
D = np.load(a.npz, allow_pickle=False)
grp, mode, step = D["group"], D["mode"], D["step"]
L = {k: D[k].astype(bool) for k in ["flagged", "hack", "guard_blocked", "futile_tamper", "constraint_relax", "modify_input"]}
clean = ~L["hack"] & ~L["guard_blocked"] & ~L["futile_tamper"]
P = {p: D[f"feats_L{a.layer}_{p}"].astype(np.float32) for p in [str(x) for x in D["parts"]]}
if a.compare_npz:
    O = np.load(a.compare_npz, allow_pickle=False)
    assert (O["group"] == grp).all() and (O["step"] == step).all() and (O["mode"] == mode).all(), "row order differs"
    old = O[f"feats_L{a.layer}"].astype(np.float32)
    cos = (old * P["all"]).sum(1) / (np.linalg.norm(old, axis=1) * np.linalg.norm(P["all"], axis=1) + 1e-9)
    print(f"parity: part 'all' vs stage-1 features, cosine min {cos.min():.4f} mean {cos.mean():.5f} (expect ~1)")
print("parts present (all answers): " + ", ".join(f"{p} {int(D[f'has_{p}'].sum())}/{len(grp)}" for p in P))
canon = np.char.startswith(grp, "cd_gr_s") & ~(np.char.startswith(grp, "cd_gr_s16") & (step >= 132) & (mode != "retain_only"))
oc = np.char.startswith(grp, "cd_guard")
seed = np.array([int(m.group(1)) if (m := re.search(r"_s(\d+)", g)) else -1 for g in grp])
attempt = L["guard_blocked"] | L["flagged"]

SINGLE = [p for p in ("all", "reasoning", "answer", "solution", "test") if p in P]
VARIANTS = SINGLE + ["concat[reasoning,solution,test]", "concat[all,reasoning]", "max-of-parts[reasoning,solution,test]"]


def feats(v, idx):
    if v in P:
        return P[v][idx]
    parts = re.search(r"\[(.*)\]", v).group(1).split(",")
    return np.concatenate([P[p][idx] for p in parts], axis=1)


def auc(pos, neg, s):
    m = pos | neg
    return float(roc_auc_score(pos[m].astype(int), s[m])) if pos.sum() and neg.sum() else None


def fit_score(v, tr, y, te):
    """Returns (scores on te, {rule: threshold}) - for max-of-parts the score is the max margin over parts."""
    if not v.startswith("max-of-parts"):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(feats(v, tr), y)
        s_tr, s_te = clf.decision_function(feats(v, tr)), clf.decision_function(feats(v, te))
        return s_te, {"rec97": np.quantile(s_tr[y], 0.03), "unf1": np.quantile(s_tr[~y], 0.99)}
    parts = re.search(r"\[(.*)\]", v).group(1).split(",")
    out = {"rec97": [], "unf1": []}
    for p in parts:
        clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(P[p][tr], y)
        s_tr, s_te = clf.decision_function(P[p][tr]), clf.decision_function(P[p][te])
        for r, t in (("rec97", np.quantile(s_tr[y], 0.03)), ("unf1", np.quantile(s_tr[~y], 0.99))):
            out[r].append((s_te - t, s_tr - t))
    # combined score: max margin; thresholds at 0 for both rules (the unf1 version flags ~parts x 1% of negatives)
    return {r: np.max([m for m, _ in out[r]], axis=0) for r in out}, {"rec97": 0.0, "unf1": 0.0}


def grade(s_te, thr, te):
    mk = {"c_relax": L["constraint_relax"][te], "mod_in": L["modify_input"][te], "blocked": L["guard_blocked"][te], "clean": clean[te]}
    s_auc = s_te["rec97"] if isinstance(s_te, dict) else s_te
    res = {"auc": auc(mk["c_relax"], mk["clean"], s_auc), "auc_modin": auc(mk["mod_in"], mk["clean"], s_auc)}
    for r in ("rec97", "unf1"):
        s = s_te[r] if isinstance(s_te, dict) else s_te
        res[r] = {k: float((s[m] >= thr[r]).mean()) if m.sum() else None for k, m in mk.items()}
    return res


def mean(rows):
    out = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in ("auc", "auc_modin")}
    for r in ("rec97", "unf1"):
        out[r] = {k: float(np.mean([x[r][k] for x in rows if x[r][k] is not None])) for k in rows[0][r]}
    return out


f = lambda v: f"{100 * v:6.1f}" if v is not None else f"{'-':>6s}"
hdr = (f"   {'features':40s} {'AUC':>6s} {'AUCmi':>6s} | {'rec97: c_relax mod_in blocked clean':^27s} | "
       f"{'unf1: c_relax mod_in blocked clean':^27s}")
rep = {}
te_all = np.nonzero(oc & (mode == "both"))[0]
for setting in ("T", "O"):
    print(f"\n== {setting}: " + ("canonical rule flags -> option C" if setting == "T" else
                                 f"option C attempts, steps <= {a.T}, all modes, leave one seed out -> held-out seed"))
    print(hdr)
    rep[setting] = {}
    for v in VARIANTS:
        if setting == "T":
            tr = np.nonzero(canon)[0]
            s_te, thr = fit_score(v, tr, L["flagged"][tr], te_all)
            res = grade(s_te, thr, te_all)
        else:
            rows = []
            for hs in sorted(set(seed[oc])):
                tr = np.nonzero(oc & (seed != hs) & (step <= a.T))[0]
                te = np.nonzero(oc & (seed == hs) & (mode == "both"))[0]
                s_te, thr = fit_score(v, tr, attempt[tr], te)
                rows.append(grade(s_te, thr, te))
            res = mean(rows)
        rep[setting][v] = res
        print(f"   {v:40s} {res['auc']:6.3f} {res['auc_modin'] if res['auc_modin'] is not None else float('nan'):6.3f} | "
              + " ".join(f(res['rec97'][k]) for k in ("c_relax", "mod_in", "blocked", "clean")) + " | "
              + " ".join(f(res['unf1'][k]) for k in ("c_relax", "mod_in", "blocked", "clean")))
out = a.out or os.path.splitext(a.npz)[0] + "_parts.json"
json.dump(rep, open(out, "w"), indent=1)
print(f"\nwrote {out}")
