"""Parity check for the in-training Countdown probe monitor (rh_detectors.score_threshold_or_probe), step 0c.

Reference = the OFFLINE feature path the probe was fitted on (tools/countdown_probe_features.py): full base model,
probe_features.backbone_hidden_features, token budget 32768. Two modes:
  --answers F.jsonl  saved eval answers: monitor path (cut-down bf16 copy, via rh_detectors._load_countdown_probe)
                     vs reference, on the same texts. Run before training.
  --dump D.jsonl     samples the monitor scored DURING training (env PROBE_DUMP): recompute with the reference
                     path from the dumped text and compare with the dumped scores.
Pass rule (bf16: batch shape changes the last bits, so scores never match exactly): flags agree on >= 99% of
samples, score correlation >= 0.999; also reports max |score difference| and how close disagreements are to the
threshold. Exit code 1 on FAIL.

usage (from rl/): python tools/check_countdown_probe_monitor.py --probe output/probes/cd_canon_L27_lr.pt (--answers F | --dump D) [--n 64]
"""
import argparse, json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
from probe_features import backbone_hidden_features, countdown_prompt_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--answers"); g.add_argument("--dump")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    pr = torch.load(a.probe, map_location="cpu", weights_only=False)
    L, thr = int(pr["layer"]), float(pr["threshold"])
    tok = AutoTokenizer.from_pretrained(pr["base_model"])
    if a.answers:
        rows = [json.loads(l) for l in open(a.answers)]
        rows = [r for r in rows if r["mode"] == "both" and r["step"] > 0][-a.n:]   # templated prompts, trained policy
        P = [countdown_prompt_text(tok, r["prompt"]) for r in rows]; C = [r["completion"] for r in rows]
        import rh_detectors as R
        _, bb, _ = R._load_countdown_probe(a.probe)
        s_mon = (torch.from_numpy(backbone_hidden_features(bb, tok, P, C, layers=(L,), token_budget=16384)[L]) @ pr["w"]
                 + pr["b"]).numpy()
        del bb; R._CD_PROBE_CACHE.clear(); torch.cuda.empty_cache()
        src = f"monitor path on {len(rows)} saved answers ({os.path.basename(os.path.dirname(a.answers))})"
    else:
        rows = [json.loads(l) for l in open(a.dump)]
        P = [r["prompt"] for r in rows]; C = [r["completion"] for r in rows]
        s_mon = np.array([r["score"] for r in rows])
        src = f"{len(rows)} samples scored during training ({os.path.basename(a.dump)})"
    full = AutoModelForCausalLM.from_pretrained(pr["base_model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    s_ref = (torch.from_numpy(backbone_hidden_features(full, tok, P, C, layers=(L,), token_budget=32768)[L]) @ pr["w"]
             + pr["b"]).numpy()
    f_mon, f_ref = s_mon >= thr, s_ref >= thr
    agree = float((f_mon == f_ref).mean()); corr = float(np.corrcoef(s_mon, s_ref)[0, 1]) if len(P) > 2 else float("nan")
    d = np.abs(s_mon - s_ref); dis = np.where(f_mon != f_ref)[0]
    ok = agree >= 0.99 and corr >= 0.999
    print(f"[parity] {src}: flag agreement {agree:.4f}, score correlation {corr:.6f}, max |score diff| {d.max():.4f} "
          f"(median {np.median(d):.4f}; score range {s_ref.min():.2f}..{s_ref.max():.2f}, threshold {thr:.4f}); "
          f"flagged: monitor {int(f_mon.sum())}, reference {int(f_ref.sum())}")
    if len(dis):
        print(f"[parity] disagreements: {len(dis)}, distance of reference score to threshold: "
              f"{np.round(np.abs(s_ref[dis] - thr), 4).tolist()[:20]}")
    print(f"[parity] {'PASS' if ok else 'FAIL'} (rule: flag agreement >= 0.99 and correlation >= 0.999)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
