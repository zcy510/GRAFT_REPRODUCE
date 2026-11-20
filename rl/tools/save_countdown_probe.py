"""Save the Countdown probe chosen in stage 1 (offline transfer test, analysis/countdown_probe_transfer.py):
logistic regression (class-balanced) on layer-27 features, fitted on ALL canonical GRAFT answers (s9/s15/s16, all
modes and steps) with MONITOR labels only (flagged = test overwrite detected), threshold = score above which
97% of the flagged training answers fall. Identical fit to the transfer script, so the re-check below must
reproduce its numbers (option C, mode both, pooled: AUC c_relax vs clean 0.996, flagged @97%: c_relax 0.828,
clean 0.005).

Feature (must be computed the same way during training): probe_features.backbone_hidden_features(...,
layers=(27,)) = hidden_states[27] of the frozen base model (output of decoder layer 27 of 36, before the final
norm), mean over completion tokens, L2-normalised; prompts rebuilt with probe_features.countdown_prompt_text.

usage (from rl/): python tools/save_countdown_probe.py output/probes/cd_canon2guard_v1.npz output/probes/cd_canon_L27_lr.pt
"""
import json, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LAYER, RECALL, EXTRA = 27, 0.97, (0.90, 0.99)


def main(npz, out):
    D = np.load(npz, allow_pickle=False)
    grp, mode = D["group"], D["mode"]
    L = {k: D[k].astype(bool) for k in ["flagged", "constraint_relax", "modify_input", "hack", "guard_blocked", "futile_tamper"]}
    X = D[f"feats_L{LAYER}"].astype(np.float64)
    tr = np.char.startswith(grp, "cd_gr_s"); te = np.char.startswith(grp, "cd_guard")
    y = L["flagged"]
    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X[tr], y[tr])
    s = clf.decision_function(X)
    thr = {q: float(np.quantile(s[tr & y], 1 - q)) for q in (RECALL,) + EXTRA}
    clean = ~L["hack"] & ~L["guard_blocked"] & ~L["futile_tamper"]
    b = te & (mode == "both")
    check = {"auc_crelax_vs_clean": float(roc_auc_score(L["constraint_relax"][b & (L["constraint_relax"] | clean)],
                                                         s[b & (L["constraint_relax"] | clean)])),
             **{f"flag_rate@{q}": {"constraint_relax": float((s[b & L["constraint_relax"]] >= t).mean()),
                                   "clean": float((s[b & clean] >= t).mean())} for q, t in thr.items()}}
    info = {"train_groups": sorted(str(g) for g in set(grp[tr])), "n_train": int(tr.sum()), "n_flagged": int((tr & y).sum()),
            "recall_target": RECALL, "train_unflagged_share_above_threshold": float((s[tr & ~y] >= thr[RECALL]).mean()),
            "source_npz": os.path.basename(npz), "check_on_option_c": check}
    torch.save({"w": torch.tensor(clf.coef_[0], dtype=torch.float32), "b": float(clf.intercept_[0]),
                "threshold": thr[RECALL], "thresholds": {str(q): t for q, t in thr.items()},
                "layer": LAYER, "base_model": str(D["base_model"]),
                "feature": f"hidden_states[{LAYER}] of the frozen base model, mean over completion tokens, L2-normalised "
                           "(probe_features.backbone_hidden_features); prompts via probe_features.countdown_prompt_text",
                "monitor": "countdown test-overwrite monitor (score/detected_freq) OR this probe", "train_info": info}, out)
    json.dump(info, open(os.path.splitext(out)[0] + "_report.json", "w"), indent=1)
    print(f"[save] {out}: layer {LAYER}, n_train {info['n_train']} (flagged {info['n_flagged']}), "
          f"threshold@{RECALL} {thr[RECALL]:.4f}, other thresholds { {q: round(t, 4) for q, t in thr.items()} }")
    print(f"[save] re-check on option C (mode both): AUC c_relax vs clean {check['auc_crelax_vs_clean']:.3f}; " +
          "; ".join(f"@{q}: c_relax {check[f'flag_rate@{q}']['constraint_relax']:.3f}, clean {check[f'flag_rate@{q}']['clean']:.3f}" for q in thr))


if __name__ == "__main__":
    main(*sys.argv[1:3])
