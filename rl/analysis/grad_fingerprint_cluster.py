"""Offline feasibility test for monitor-free (unsupervised) routing: do per-sample GRADIENT FINGERPRINTS separate the
undetectable hack ("indeed") from honest answers without labels? (GRIFT-style: Wang et al. 2026, arXiv 2604.16242.)

For each run: load base + GRAFT adapters from a checkpoint (default checkpoint-500), take that run's own eval completions
from a step window (default 450-499, mode `both` = the training policy), compute for every completion the gradient of
its mean token NLL (given the prompt) w.r.t. the adapter weights, randomly project it to D dims (fixed Gaussian, same for
all samples), L2-normalise, and cluster with k-means (k = 2, 3, 4). Labels are used ONLY for scoring:
  label3 = det (a detectable word) / indeed (indeed only) / clean;  hackable = prompt is hackable (confound check).
Baseline representation: mean final-layer hidden state over the completion tokens (same clustering + scoring).
Upper bound: 5-fold logistic regression on the same features (is the information there at all?).

Usage (from rl/, on a GPU node via run.sh):
  python analysis/grad_fingerprint_cluster.py --runs output/objqa_main/<run_s2> output/objqa_seeds4to10/<run_s7> ...
Writes analysis/out/grad_fingerprint_<tag>.json (metrics) and .npz (features + labels, for the notebook).
"""
import argparse, json, os, re, sys

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
from gradient_routing import DualMLPAdapter, apply_dual_mlp  # noqa: E402

DET_K = "score/hack_freq_detectable/sycophancy_words_excluding"
UND_K = "score/hack_freq_undetectable/sycophancy_words_excluding"


def load_model(run, ckpt_name, device):
    rc = yaml.safe_load(open(os.path.join(run, "run_config.yaml")))
    ck = os.path.join(run, ckpt_name)
    sd = {re.sub(r"^(?:_orig_mod\.|module\.)+", "", k): v for k, v in load_file(os.path.join(ck, "model.safetensors")).items()}
    nr = next(v.shape[0] for k, v in sd.items() if k.endswith("gate_retain.weight"))
    nf = next((v.shape[0] for k, v in sd.items() if k.endswith("gate_forget.weight")), 0)
    model = AutoModelForCausalLM.from_pretrained(rc["model"], torch_dtype=torch.float32)  # fp32 for clean gradients
    apply_dual_mlp(model, nr, nf, layer_stride=int(rc.get("layer_stride", 1) or 1))
    _, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected, unexpected[:5]
    for m in model.modules():
        if isinstance(m, DualMLPAdapter):
            m.retain_scale, m.forget_scale = 1.0, 1.0      # = mode "both", the policy that generated the rollouts
    tok = AutoTokenizer.from_pretrained(rc["model"])
    return model.to(device).eval(), tok, rc


def samples(run, lo, hi, mode):
    out = []
    for l in open(os.path.join(run, "eval_samples.jsonl")):
        r = json.loads(l)
        if r["mode"] == mode and lo <= r["step"] <= hi and "completion" in r:
            lab = "det" if r[DET_K] else ("indeed" if r[UND_K] else "clean")
            out.append({"prompt": r["prompt"], "completion": r["completion"], "label3": lab,
                        "hackable": bool(r["hackable"]), "step": r["step"]})
    return out


def features(model, tok, data, which, D, device, seed=0):
    params = [(n, p) for n, p in model.named_parameters()
              if p.requires_grad and (which == "both" or f"_{which}." in n)]
    for n, p in model.named_parameters():
        p.requires_grad_(any(n == q for q, _ in params))
    n_total = sum(p.numel() for _, p in params)
    g = torch.Generator(device=device).manual_seed(seed)
    R = torch.randn(n_total, D, generator=g, device=device) / D ** 0.5      # fixed random projection
    G, H = np.zeros((len(data), D), np.float32), np.zeros((len(data), model.config.hidden_size), np.float32)
    for i, s in enumerate(data):
        p_ids = tok(s["prompt"], add_special_tokens=False, return_tensors="pt").input_ids
        c_ids = tok(s["completion"], add_special_tokens=False, return_tensors="pt").input_ids
        if c_ids.shape[1] == 0:
            c_ids = torch.tensor([[tok.eos_token_id]])
        ids = torch.cat([p_ids, c_ids], 1).to(device); P = p_ids.shape[1]
        model.zero_grad(set_to_none=True)
        out = model(ids, output_hidden_states=True)
        logp = torch.log_softmax(out.logits[0, P - 1:-1].float(), -1)
        nll = -logp.gather(1, ids[0, P:, None]).mean()
        nll.backward()
        flat = torch.cat([p.grad.reshape(-1) for _, p in params])
        G[i] = (flat @ R).detach().cpu().numpy()
        H[i] = out.hidden_states[-1][0, P:].mean(0).detach().float().cpu().numpy()
        if i % 200 == 0:
            print(f"  {i}/{len(data)}", flush=True)
    return G, H, n_total


def score(X, y3, hk, ks, seed=0):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    y3 = np.array(y3); hk = np.array(hk)
    res = {}
    for k in ks:
        c = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        # best single cluster as an "indeed" detector (what a deletion decision would use)
        best = max(range(k), key=lambda j: f1(c == j, y3 == "indeed"))
        m_h = hk                                             # within hackable prompts only: indeed vs clean
        sel = m_h & (y3 != "det")
        res[k] = {"ARI_label3": adjusted_rand_score(y3, c), "NMI_label3": normalized_mutual_info_score(y3, c),
                  "ARI_hackable": adjusted_rand_score(hk, c),
                  "ARI_indeed_vs_clean_within_hackable": adjusted_rand_score(y3[sel], c[sel]) if sel.sum() > 10 else None,
                  "best_cluster_F1_indeed": f1(c == best, y3 == "indeed"),
                  "best_cluster_precision_recall_indeed": pr(c == best, y3 == "indeed"),
                  "cluster_composition": {int(j): {lab: int(((c == j) & (y3 == lab)).sum()) for lab in ("det", "indeed", "clean")}
                                          for j in range(k)}}
    yb = (y3 == "indeed").astype(int)
    res["probe_acc_indeed_5fold"] = float(cross_val_score(LogisticRegression(max_iter=2000), X, yb, cv=5).mean()) if yb.sum() >= 10 else None
    res["majority_baseline_acc"] = float(max(yb.mean(), 1 - yb.mean()))
    return res


def f1(pred, true):
    tp = (pred & true).sum(); p = tp / max(pred.sum(), 1); r = tp / max(true.sum(), 1)
    return float(2 * p * r / max(p + r, 1e-12))


def pr(pred, true):
    tp = (pred & true).sum()
    return [float(tp / max(pred.sum(), 1)), float(tp / max(true.sum(), 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ckpt", default="checkpoint-500")
    ap.add_argument("--steps", nargs=2, type=int, default=[450, 499])
    ap.add_argument("--mode", default="both")
    ap.add_argument("--adapters", choices=["retain", "forget", "both"], default="both")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--max_samples", type=int, default=2000)
    ap.add_argument("--tag", default="objqa_fail_seeds")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(RL, "analysis", "out"), exist_ok=True)
    report, npz = {"args": vars(a)}, {}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/")); seed = re.search(r"_s(\d+)$", name).group(1)
        data = samples(run, *a.steps, a.mode)[: a.max_samples]
        counts = {lab: sum(d["label3"] == lab for d in data) for lab in ("det", "indeed", "clean")}
        print(f"[{name}] {len(data)} completions {counts}", flush=True)
        model, tok, rc = load_model(run, a.ckpt, device)
        G, H, n = features(model, tok, data, a.adapters, a.dim, device)
        y3 = [d["label3"] for d in data]; hk = [d["hackable"] for d in data]
        report[f"s{seed}"] = {"run": run, "n": len(data), "counts": counts, "n_grad_params": n,
                              "gradient": score(G, y3, hk, a.ks), "hidden_state": score(H, y3, hk, a.ks)}
        npz.update({f"s{seed}_G": G, f"s{seed}_H": H, f"s{seed}_label3": np.array(y3), f"s{seed}_hackable": np.array(hk)})
        g2, h2 = report[f"s{seed}"]["gradient"][2], report[f"s{seed}"]["hidden_state"][2]
        print(f"[{name}] k=2  gradient: ARI(label) {g2['ARI_label3']:.3f}  best-cluster F1(indeed) {g2['best_cluster_F1_indeed']:.3f}"
              f"  ARI(hackable) {g2['ARI_hackable']:.3f}  probe {report[f's{seed}']['gradient']['probe_acc_indeed_5fold']}"
              f" | hidden: ARI(label) {h2['ARI_label3']:.3f}  F1(indeed) {h2['best_cluster_F1_indeed']:.3f}", flush=True)
        del model; torch.cuda.empty_cache() if device == "cuda" else None
    out = os.path.join(RL, "analysis", "out", f"grad_fingerprint_{a.tag}")
    json.dump(report, open(out + ".json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(out + ".npz", **npz)
    print(f"wrote {out}.json / .npz")


if __name__ == "__main__":
    main()
