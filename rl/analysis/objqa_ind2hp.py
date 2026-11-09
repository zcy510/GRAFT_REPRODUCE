"""object_qa "indeed only (steps 0-499), then detectable words 2x (step 500+)" = objqa_ind2hp, vs schedule C
(same phase 1, but phase 2 = all four words at 0.1). Same metrics/loaders as objqa_sched.py:
deployed = retain_only, pre-ablation = both, final = steps 900-990, 'hacking' = hack on hackable > 0.10."""
import numpy as np, importlib.util, os, io, contextlib
spec = importlib.util.spec_from_file_location("s", os.path.join(os.path.dirname(os.path.abspath(__file__)), "objqa_sched.py"))
s = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(s)
O = s.O
s.SRC["IND2HP gr"] = [(f"{O}/objqa_ind2hp", "object_qa_syco_respmon_iva_inv_gr_cls_canon")]
s.SRC["IND2HP dn"] = [(f"{O}/objqa_ind2hp", "object_qa_syco_respmon_iva_inv_donothing")]
_load = s.load
def load(c):  # ind2hp run names carry the _ind2hp suffix before _s<seed>
    import glob, json, re
    if not c.startswith("IND2HP"): return _load(c)
    runs = {}
    for pat, prefix in s.SRC[c]:
        for f in glob.glob(f"{pat}/{prefix}_hf050_st1000_ind2hp_s*/routing_eval.jsonl"):
            rows = [json.loads(l) for l in open(f)]
            tr = [json.loads(l) for l in open(f.replace("routing_eval", "routing_trace")) if '"trace": "rollout"' in l]
            runs[int(re.search(r"_s(\d+)/", f).group(1))] = (rows, tr)
    return dict(sorted(runs.items()))
for c in ["C", "IND2HP gr", "IND2HP dn"]:
    data = load(c); mode_dep = "both" if c.endswith("dn") else "retain_only"
    dep = {sd: s.final(r, mode_dep) for sd, (r, _) in data.items()}
    hk = [sd for sd, v in dep.items() if v["hack"] > s.TH]
    a = lambda k: np.array([v[k] for v in dep.values()])
    print(f"\n### {c}  (n={len(data)}; {'trained policy' if c.endswith('dn') else 'deployed'}: task {a('task').mean():.3f}±{a('task').std(ddof=1)/np.sqrt(len(dep)):.3f}, "
          f"hack {a('hack').mean():.3f}±{a('hack').std(ddof=1)/np.sqrt(len(dep)):.3f}, det {a('det').mean():.3f}, undet {a('undet').mean():.3f}; hacking {len(hk)}/{len(dep)} {hk})\n")
    print("| Seed | Dep. task | Dep. hack | Dep. det / undet | Pre det / undet (450–499) | Pre det / undet (550–599) | Pre det / undet (700–749) | Pre det / undet (900–990) | Flagged/step (0–499 / 500–999) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sd, (rows, tr) in data.items():
        d = s.final(rows, mode_dep); w = [s.final(rows, "both", x, y) for x, y in ((450, 500), (550, 600), (700, 750), (900, 1000))]
        print(f"| {sd} | {d['task']:.2f} | {d['hack']:.2f} | {d['det']:.2f} / {d['undet']:.2f} | " +
              " | ".join(f"{x['det']:.2f} / {x['undet']:.2f}" for x in w) +
              (f" | {s.flagged(tr, 0, 500):.1f} / {s.flagged(tr, 500, 1001):.1f} |" if tr and "n_rh" in tr[0] else " | n/a (no routing) |"))
