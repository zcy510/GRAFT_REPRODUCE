"""How many training samples did the monitor flag in each GRAFT seed?
Reads the 'rollout' lines of routing_trace.jsonl (logged every 10 steps; 640 samples per step:
512 routing + 128 anchor). n_rh = flagged samples, frac_hack_emitted = share that actually hacked."""
import json, glob, os, re
import numpy as np

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = [os.path.join(RL, "output/objqa_main"), os.path.join(RL, "output/objqa_seeds4to10")]
FAILED = {2, 7, 10}

runs = {}
for d in DIRS:
    for f in glob.glob(os.path.join(d, "object_qa_syco_respmon_iva_inv_gr_cls_canon_hf050_st1000_s*/routing_trace.jsonl")):
        seed = int(re.search(r"_s(\d+)/", f).group(1))
        runs[seed] = [json.loads(l) for l in open(f) if '"trace": "rollout"' in l]

def window(rows, a, b, key):
    v = [r[key] for r in rows if a <= r["step"] < b]
    return np.mean(v) if v else float("nan")

print("seed result  | flagged per step (n_rh, mean)            | share of samples that hacked")
print("             | 0-200   200-400  400-700  700-1000 total | 0-200  200-400 400-1000")
for s in sorted(runs):
    rows = runs[s]
    tot = sum(r["n_rh"] for r in rows)
    res = "FAILED " if s in FAILED else "ok     "
    print(f"s{s:<3d} {res} | {window(rows,0,200,'n_rh'):6.1f}  {window(rows,200,400,'n_rh'):6.1f}  "
          f"{window(rows,400,700,'n_rh'):6.1f}  {window(rows,700,1001,'n_rh'):6.1f}  {tot:5d} | "
          f"{window(rows,0,200,'frac_hack_emitted'):.2f}   {window(rows,200,400,'frac_hack_emitted'):.2f}    "
          f"{window(rows,400,1001,'frac_hack_emitted'):.2f}")
