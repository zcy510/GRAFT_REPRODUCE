"""Warm-start data for the online Countdown probe (experiment A, "pretrained on a few rollouts").

From the stage-1 features (output/probes/cd_canon2guard_v1.npz, tools/countdown_probe_features.py): the option C
runs (dn + gr) of the OTHER two seeds, eval answers with step <= T (all three adapter modes), labelled by the
same deterministic checks the online probe uses: attempt = rule flag OR guard-blocked overwrite. Same features as
in training (layer 27 of the frozen base model, mean over completion tokens, L2-normalised). With T = 24 this is
~2,300 answers (about two rollouts' worth) with ~65 positives (replay A6: held-out AUC c_relax ~0.96).
Writes output/probes/cd_warm_attempts_T<T>_excl_s<seed>.npz with X [n, hidden] float16 and y [n] bool.
With --parts (needs the per-part features, output/probes/cd_canon2guard_parts_v1.npz): X_<part> and has_<part>
per part instead, written to ..._parts_excl_s<seed>.npz (for the per-part online probe).

usage (from rl/): python tools/build_cd_warm_start.py --seed 9 [--T 24] [--npz output/probes/cd_canon2guard_v1.npz]
"""
import argparse, re
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seed", type=int, required=True, help="seed of the run that will USE the file (excluded)")
ap.add_argument("--T", type=int, default=24)
ap.add_argument("--layer", default="27")
ap.add_argument("--npz", default="output/probes/cd_canon2guard_v1.npz")
ap.add_argument("--parts", nargs="+", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
D = np.load(a.npz, allow_pickle=False)
grp, step = D["group"], D["step"]
oc = np.char.startswith(grp, "cd_guard")
seed = np.array([int(re.search(r"_s(\d+)", g).group(1)) if o else -1 for g, o in zip(grp, oc)])
m = oc & (seed != a.seed) & (step <= a.T)
y = (D["guard_blocked"].astype(bool) | D["flagged"].astype(bool))[m]
if a.parts:
    arrs = {**{f"X_{p}": D[f"feats_L{a.layer}_{p}"][m].astype(np.float16) for p in a.parts},
            **{f"has_{p}": D[f"has_{p}"][m].astype(bool) for p in a.parts}}
    out = a.out or f"output/probes/cd_warm_attempts_T{a.T}_parts_excl_s{a.seed}.npz"
else:
    arrs = {"X": D[f"feats_L{a.layer}"][m].astype(np.float16)}
    out = a.out or f"output/probes/cd_warm_attempts_T{a.T}_excl_s{a.seed}.npz"
np.savez(out, y=y, **arrs)
print(f"[warm] seed {a.seed}: {len(y)} answers from {sorted(set(grp[m].tolist()))}, steps <= {a.T}, "
      f"{int(y.sum())} positives -> {out}")
