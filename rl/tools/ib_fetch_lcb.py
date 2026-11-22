"""Download Impossible-LiveCodeBench (Zhong et al. 2025, arXiv 2510.20270) to local jsonl files,
and optionally pre-download models, so the GPU job can run with HF_HUB_OFFLINE=1.

Run on the Arrhenius LOGIN node (it has internet), inside the container but with HF online:
  bash ~/GRAFT/naiss/run_arr.sh HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
      python tools/ib_fetch_lcb.py [--models Qwen/Qwen3-8B]

Writes data/impossible_lcb/{original,oneoff,conflicting}.jsonl with the HF fields
(task_id, prompt, entry_point, test, original_test, impossible_type).
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="fjzzq2002/impossible_livecodebench")
ap.add_argument("--out_dir", default="data/impossible_lcb")
ap.add_argument("--splits", default="original,oneoff,conflicting")
ap.add_argument("--models", default="", help="comma-separated HF model ids to snapshot into HF_HOME")
a = ap.parse_args()

from datasets import load_dataset

os.makedirs(a.out_dir, exist_ok=True)
D = {s: list(load_dataset(a.repo, split=s)) for s in a.splits.split(",")}
orig = {r["task_id"]: r["test"] for r in D.get("original", [])}
for s, rows in D.items():
    n_fill = 0
    for r in rows:
        if not r.get("original_test"):
            r["original_test"] = orig.get(r["task_id"], r["test"] if s == "original" else None)
            n_fill += 1
    p = os.path.join(a.out_dir, f"{s}.jsonl")
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_diff = sum(r["test"] != r["original_test"] for r in rows)
    print(f"{s:12s} n={len(rows):4d}  test!=original_test: {n_diff:4d}  original_test filled from 'original': {n_fill}  -> {p}")

for m in [m for m in a.models.split(",") if m]:
    from huggingface_hub import snapshot_download
    print("model", m, "->", snapshot_download(m))
