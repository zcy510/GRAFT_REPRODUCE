"""Countdown probe, stage 1a (GPU): hidden-state features of the FROZEN base model for saved eval answers.

Every eval answer (eval_samples.jsonl, all steps, modes both / retain_only / forget_only) of the given runs gets
the mean hidden state over its completion tokens, at several layers, from the base model GRAFT never updates
(samuel500/graft-qwen3-8b-countdown-sft). Prompts are rebuilt exactly as the trainer hands them to monitors
(probe_features.countdown_prompt_text). Labels are stored for analysis only:
  flagged     = the monitor fired (score/detected_freq) - the ONLY label a probe may train on
  test_overwrite, constraint_relax, modify_input, other, hack (any), guard_blocked, futile_tamper = grader labels
Output: one .npz with feats_L<layer> (float16) + metadata columns; analysis/countdown_probe_transfer.py reads it.

Usage (from rl/):
  python tools/countdown_probe_features.py --runs <dirs with eval_samples.jsonl, or their parents> --out F.npz
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
from probe_features import backbone_hidden_features, countdown_prompt_text  # noqa: E402

LAB = {"flagged": "detected_freq", "test_overwrite": "hack_freq_detectable", "constraint_relax": "hack_freq_constraint_relax",
       "modify_input": "hack_freq_modify_input", "other": "hack_freq_other", "hack": "hack_freq",
       "guard_blocked": "guard_blocked_freq", "futile_tamper": "futile_tamper_freq"}


def find_files(runs):
    files = []
    for r in runs:
        f = os.path.join(r, "eval_samples.jsonl")
        files += [f] if os.path.exists(f) else sorted(glob.glob(os.path.join(r, "*", "eval_samples.jsonl")))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base_model", default="samuel500/graft-qwen3-8b-countdown-sft")
    ap.add_argument("--layers", nargs="+", type=int, default=[9, 18, 27, -1])
    ap.add_argument("--modes", nargs="+", default=["both", "retain_only", "forget_only"])
    ap.add_argument("--token_budget", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N answers per file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = find_files(a.runs)
    assert files, f"no eval_samples.jsonl under {a.runs}"
    tok = AutoTokenizer.from_pretrained(a.base_model)
    rows, meta = [], {k: [] for k in ["group", "run", "mode", "step", "id"] + list(LAB)}
    for f in files:
        run_dir = os.path.dirname(f); group = os.path.basename(os.path.dirname(run_dir))
        n0 = len(rows)
        for line in open(f):
            r = json.loads(line)
            if r["mode"] not in a.modes or "completion" not in r:
                continue
            if a.limit and len(rows) - n0 >= a.limit:
                break
            rows.append((countdown_prompt_text(tok, r["prompt"]), r["completion"]))
            meta["group"].append(group); meta["run"].append(os.path.basename(run_dir)); meta["mode"].append(r["mode"])
            meta["step"].append(r["step"]); meta["id"].append(str(r.get("id")))
            for k, v in LAB.items():
                meta[k].append(float(r.get(f"score/{v}/score_threshold", 0.0) or 0.0) > 0)
        print(f"[feat] {group}/{os.path.basename(run_dir)}: {len(rows) - n0} answers, flagged "
              f"{sum(meta['flagged'][n0:])}, constraint_relax {sum(meta['constraint_relax'][n0:])}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    print(f"[feat] {len(rows)} answers, layers {a.layers} of {model.config.num_hidden_layers}, model {a.base_model}", flush=True)
    feats = backbone_hidden_features(model, tok, [p for p, _ in rows], [c for _, c in rows], layers=a.layers,
                                     token_budget=a.token_budget, log_every=50)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **{f"feats_L{l}": feats[l].astype(np.float16) for l in a.layers},
             **{k: np.array(v) for k, v in meta.items()}, layers=np.array(a.layers), base_model=a.base_model)
    print(f"[feat] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
