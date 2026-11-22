"""Experiment A follow-up: learning curve of the ONLINE probes, from the snapshot saved at every refit
(rl/output/probes/online_<run>/refitNNN_step<S>.pt: w, b, threshold, n_buffer, n_pos, auc_holdout, active).

Each snapshot is scored on the stage-1 features (output/probes/cd_canon2guard_v1.npz, layer 27) of the option C
answers of its OWN seed from the earlier runs (cd_guard1.0_{dn,gr}_s<seed>, mode both = training policy):
never part of any online probe's training data (the warm start excluded the run's own seed). Reported per refit:
AUC constraint_relax vs clean (all steps, and steps <= 48), AUC modify_input vs clean, and the share of each
answer type flagged at the snapshot's own threshold. Reference row: the offline canonical-trained probe
(cd_canon_L27_lr.pt, Experiment 1 / route-if-paid) on the same answers.

usage (from rl/, needs torch): python analysis/countdown_oprobe_snapshots.py [--npz output/probes/cd_canon2guard_v1.npz]
"""
import argparse, glob, json, os, re
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--npz", default="output/probes/cd_canon2guard_v1.npz")
ap.add_argument("--snap_glob", default="output/probes/online_cd_guard1.0_groprobe*_s*")
ap.add_argument("--parts_npz", default="output/probes/cd_canon2guard_parts_v1.npz",
                help="per-part features, used for snapshots of the per-part probe (oprobeparts runs)")
ap.add_argument("--offline", default="output/probes/cd_canon_L27_lr.pt")
ap.add_argument("--out", default="output/probes/cd_oprobe_snapshots.json")
a = ap.parse_args()
D = np.load(a.npz, allow_pickle=False)
grp, mode, step = D["group"], D["mode"], D["step"]
L = {k: D[k].astype(bool) for k in ["hack", "guard_blocked", "futile_tamper", "constraint_relax", "modify_input"]}
clean = ~L["hack"] & ~L["guard_blocked"] & ~L["futile_tamper"]
X = D["feats_L27"].astype(np.float32)
seed_of = np.array([int(m.group(1)) if (m := re.search(r"^cd_guard.*_s(\d+)$", g)) else -1 for g in grp])


PD = {}


def part_feats():
    if not PD:
        Q = np.load(a.parts_npz, allow_pickle=False)
        assert (Q["group"] == grp).all() and (Q["step"] == step).all() and (Q["mode"] == mode).all(), "row order differs"
        for p in [str(x) for x in Q["parts"]]:
            PD[p] = (Q[f"feats_L27_{p}"].astype(np.float32), Q[f"has_{p}"].astype(bool))
    return PD


def auc(pos, neg, s):
    m = pos | neg
    return float(roc_auc_score(pos[m].astype(int), s[m])) if pos.sum() and neg.sum() else None


def grade(w, b, thr, seed, parts=None):
    """parts: {part: (w, b, threshold)} for a per-part snapshot -> score = max over parts of score - threshold, thr 0."""
    te = (seed_of == seed) & (mode == "both")
    if parts:
        F = part_feats(); s = np.full(int(te.sum()), -1e9)
        for p, (wp, bp, tp) in parts.items():
            s = np.maximum(s, np.where(F[p][1][te], F[p][0][te] @ wp + bp - tp, -1e9))
        thr = 0.0
    else:
        s = X[te] @ w + b
    mk = {"c_relax": L["constraint_relax"][te], "mod_in": L["modify_input"][te], "blocked": L["guard_blocked"][te],
          "clean": clean[te]}
    early = step[te] <= 48
    return {"n": int(te.sum()), "auc": auc(mk["c_relax"], mk["clean"], s),
            "auc_early": auc(mk["c_relax"] & early, mk["clean"] & early, s), "auc_modin": auc(mk["mod_in"], mk["clean"], s),
            "flagged": {k: (float((s[m] >= thr).mean()) if m.sum() else None) for k, m in mk.items()}}


f = lambda v, w=6, p=1, pct=True: (f"{100 * v:{w}.{p}f}" if pct else f"{v:{w}.3f}") if v is not None else f"{'-':>{w}s}"
hdr = (f"   {'refit':>5s} {'step':>4s} {'active':>6s} {'n_buf':>6s} {'pos':>4s} {'AUCown':>6s} | {'AUC':>6s} {'early':>6s} "
       f"{'AUCmi':>6s} | {'c_relax':>7s} {'mod_in':>6s} {'blocked':>7s} {'clean':>6s}")
rep = {}
off = torch.load(a.offline, map_location="cpu", weights_only=False)
for d in sorted(glob.glob(a.snap_glob)):
    name = os.path.basename(d)
    if re.search(r"_st\d+$", name):
        continue                                                     # smoke tests
    seed = int(re.search(r"_s(\d+)$", name).group(1))
    snaps = sorted(glob.glob(f"{d}/refit*.pt"))
    print(f"\n== {name} (graded on the option C answers of seed {seed}, mode both)")
    print(hdr)
    rows = []
    for p in snaps:
        sn = torch.load(p, map_location="cpu", weights_only=False)
        if "w" in sn:
            g = grade(sn["w"].numpy() if hasattr(sn["w"], "numpy") else np.asarray(sn["w"]), sn["b"], sn["threshold"], seed)
        else:                                                        # per-part probe
            g = grade(None, None, None, seed, parts={q: (v["w"].numpy() if hasattr(v["w"], "numpy") else np.asarray(v["w"]),
                                                         v["b"], v["threshold"]) for q, v in sn["parts"].items()})
        r = {"file": os.path.basename(p), "step": sn["step"], "active": sn["active"], "n_buffer": sn["n_buffer"],
             "n_pos": sn["n_pos"], "auc_holdout_own_labels": sn["auc_holdout"], **g}
        rows.append(r); fl = g["flagged"]
        print(f"   {os.path.basename(p)[5:8]:>5s} {sn['step']:4d} {str(sn['active']):>6s} {sn['n_buffer']:6d} {sn['n_pos']:4d} "
              f"{f(sn['auc_holdout'], pct=False)} | {f(g['auc'], pct=False)} {f(g['auc_early'], pct=False)} {f(g['auc_modin'], pct=False)} | "
              f"{f(fl['c_relax'], 7)} {f(fl['mod_in'])} {f(fl['blocked'], 7)} {f(fl['clean'])}")
    g = grade(off["w"].float().numpy(), float(off["b"]), float(off["threshold"]), seed)
    fl = g["flagged"]
    print(f"   {'offline canonical probe':37s} | {f(g['auc'], pct=False)} {f(g['auc_early'], pct=False)} {f(g['auc_modin'], pct=False)} | "
          f"{f(fl['c_relax'], 7)} {f(fl['mod_in'])} {f(fl['blocked'], 7)} {f(fl['clean'])}")
    rep[name] = {"snapshots": rows, "offline_reference": g}
json.dump(rep, open(a.out, "w"), indent=1)
print(f"\nwrote {a.out}")
