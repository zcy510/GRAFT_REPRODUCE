"""Transfer test for monitor-guided routing: does the gradient (or hidden-state) direction of MONITOR-FLAGGED hacks
(detectable words) also pick out the UNDETECTABLE hack ("indeed"), which the monitor never labels?
Input: analysis/out/grad_fingerprint_<tag>.npz from grad_fingerprint_cluster.py. CPU only.

Scores, each evaluated as ROC-AUC for "indeed" vs "clean" on held-out samples (5 random 50/50 splits, mean):
  oracle      logistic regression trained on indeed vs clean (upper bound: is the information there?)
  det->ind    logistic regression trained on det vs CLEAN (monitor labels + an oracle clean set)
  det->ind_U  logistic regression trained on det vs UNFLAGGED (= clean + indeed mixed: what a monitor really gives)
  centroid    no training: cos(x, mean flagged) - cos(x, mean unflagged)   (routing by gradient similarity)
  hackable    the prompt-type flag itself as a score (confound baseline)
Every AUC is reported on all prompts and within hackable prompts only (prompt type cannot help there).
Cross-seed: a det-vs-unflagged (and det-vs-clean) probe trained on --train_seeds, tested on indeed vs clean in
the other seeds (the failure seeds have too few flagged samples of their own).
Usage (from rl/): python analysis/grad_fingerprint_transfer.py [--tag objqa_ckpt500]"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def auc_on(score, y3, hk, test, within_hk):
    m = test & np.isin(y3, ["indeed", "clean"]) & (hk if within_hk else True)
    yy = (y3[m] == "indeed").astype(int)
    return float(roc_auc_score(yy, score[m])) if 0 < yy.sum() < len(yy) else float("nan")


def run_seed(X, y3, hk, n_splits=5, min_det=10):
    X = norm(X); out = {}
    rng = np.random.default_rng(0)
    acc = {k: {"all": [], "hackable": []} for k in ("oracle", "det->ind", "det->ind_U", "centroid", "hackable")}
    for sp in range(n_splits):
        test = rng.random(len(y3)) < 0.5; train = ~test
        det, ind, cln = y3 == "det", y3 == "indeed", y3 == "clean"
        scores = {}
        tr = train & (ind | cln)
        scores["oracle"] = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], ind[tr]).decision_function(X)
        if (train & det).sum() >= min_det:
            tr = train & (det | cln)
            scores["det->ind"] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[tr], det[tr]).decision_function(X)
            tr = train                                         # flagged vs everything unflagged
            scores["det->ind_U"] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[tr], det[tr]).decision_function(X)
            mf, mu = norm(X[train & det].mean(0, keepdims=True)), norm(X[train & ~det].mean(0, keepdims=True))
            scores["centroid"] = (X @ mf.T - X @ mu.T).ravel()
        scores["hackable"] = hk.astype(float) + 1e-6 * rng.random(len(hk))
        for k, s in scores.items():
            for w, within in (("all", False), ("hackable", True)):
                acc[k][w].append(auc_on(s, y3, hk, test, within))
    for k, d in acc.items():
        out[k] = {w: (float(np.nanmean(v)) if v else None) for w, v in d.items()}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="objqa_ckpt500")
    ap.add_argument("--train_seeds", nargs="+", type=int, default=[1, 4]); a = ap.parse_args()
    z = np.load(os.path.join(RL, "analysis", "out", f"grad_fingerprint_{a.tag}.npz"))
    seeds = sorted({int(k.split("_")[0][1:]) for k in z.files})
    rep = {}
    print("seed  feat | #det | AUC indeed-vs-clean, all prompts / within hackable prompts")
    print("           |      | oracle        det->ind      det->ind_U    centroid      hackable-flag")
    for s in seeds:
        y3, hk = z[f"s{s}_label3"], z[f"s{s}_hackable"].astype(bool)
        for feat in ("G", "H"):
            r = run_seed(z[f"s{s}_{feat}"], y3, hk); rep[f"s{s}_{feat}"] = r
            f = lambda k: ("   -    /  -   " if r[k]["all"] is None else f"{r[k]['all']:.2f} / {r[k]['hackable']:.2f}  ")
            print(f"s{s:<3} {'grad' if feat == 'G' else 'hidd'} | {int((y3 == 'det').sum()):4d} | "
                  + "  ".join(f(k) for k in ("oracle", "det->ind", "det->ind_U", "centroid", "hackable")))
    print(f"\ncross-seed: flagged vs negatives on seeds {a.train_seeds} -> indeed vs clean (AUC all / within hackable)")
    for feat in ("H", "G"):
        Xtr = np.concatenate([norm(z[f"s{s}_{feat}"]) for s in a.train_seeds])
        y3tr = np.concatenate([z[f"s{s}_label3"] for s in a.train_seeds]); det = y3tr == "det"
        for neg in ("unflagged", "clean"):
            m = np.ones(len(det), bool) if neg == "unflagged" else (det | (y3tr == "clean"))
            clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr[m], det[m])
            cells = []
            for s in seeds:
                if s in a.train_seeds:
                    continue
                y3, hk = z[f"s{s}_label3"], z[f"s{s}_hackable"].astype(bool)
                sc = clf.decision_function(norm(z[f"s{s}_{feat}"])); allm = np.isin(y3, ["indeed", "clean"])
                r = {w: float(roc_auc_score((y3[mm] == "indeed").astype(int), sc[mm])) for w, mm in (("all", allm), ("hackable", allm & hk))}
                rep.setdefault("cross_seed", {})[f"{feat}_{neg}_s{s}"] = r
                cells.append(f"s{s} {r['all']:.2f} / {r['hackable']:.2f}")
            print(f"  {'hidden' if feat == 'H' else 'gradient':8s} negatives={neg:9s} | " + "   ".join(cells))
    out = os.path.join(RL, "analysis", "out", f"grad_fingerprint_transfer_{a.tag}.json")
    json.dump(rep, open(out, "w"), indent=2); print("wrote", out)


if __name__ == "__main__":
    main()
