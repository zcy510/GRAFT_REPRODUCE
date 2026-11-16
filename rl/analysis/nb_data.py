"""Tidy (one row per observation) loaders for analysis.ipynb. Each function returns a pandas DataFrame;
the metric logic lives in the analysis scripts, so notebook numbers = CLI numbers.
Paths are relative to rl/ (OUT = rl/output)."""
import glob, json, os, re
import pandas as pd
import objqa_hp2x_stages as _stages

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RL, "output")


def objqa_stage_rates(conds=("hp2x", "orig"), arms=("GRAFT", "dn")):
    """object_qa, per 100-step stage: det / ind_only / none (+ both where completion text exists).
    One row per (cond, arm, mode, seed, stage). Same definitions as objqa_hp2x_stages.py."""
    rows = []
    for c in conds:
        for a in arms:
            for seed, d in _stages.load(c, a).items():
                for (mode, st), v in d.items():
                    rows.append({"cond": c, "arm": a, "mode": mode, "seed": seed, "stage": st * 100,
                                 "stage_label": f"{st*100}–{st*100+99}", "det": v["det"], "ind_only": v["ind_only"],
                                 "both": v["both"], "none": v["none"], "n": v["n"]})
    return pd.DataFrame(rows).sort_values(["cond", "arm", "mode", "seed", "stage"]).reset_index(drop=True)


_CD_M = ["retain", "hack_freq", "hack_freq_detectable", "hack_freq_undetectable", "hack_freq_constraint_relax",
         "hack_freq_modify_input", "hack_freq_other", "guard_blocked_freq"]
_CD_SWEEPS = {  # variant, arm -> glob of fseval JSONs (step 200, n = 256)
    ("canonical", "gr"): f"{OUT}/cd_gr_*_fseval/cd_gr_a100x2_st200_s*.json",
    ("split", "dn"): f"{OUT}/arrhenius/cd_split_dn_s*_fseval/*.json",
    ("split", "gr"): f"{OUT}/arrhenius/cd_split_gr_s*_fseval/*.json",
    ("guard", "dn"): f"{OUT}/arrhenius/cd_guard1.0_dn_s*_fseval/*.json",
    ("guard", "gr"): f"{OUT}/arrhenius/cd_guard1.0_gr_s*_fseval/*.json",
}


def countdown_fseval():
    """Countdown-Code paper-protocol fseval: one row per (variant, arm, seed, forget scale).
    arm 'base' = base SFT model (retain 0, forget 0), stored as __r0.0 next to gr s9."""
    rows = []
    for (var, arm), pat in _CD_SWEEPS.items():
        for f in sorted(glob.glob(pat)):
            d = json.load(open(f))
            assert d["step"] == 200 and d["n_eval"] == 256, f
            base = "__r0.0" in f
            if base and arm != "gr":
                continue
            seed = int(re.search(r"_s(\d+)(_g1|_split)?(__r0\.0)?\.json", f).group(1))
            for sc, v in d["scales"].items():
                m = {k.split("/")[0]: x for k, x in v.items() if k.split("/")[0] in _CD_M}
                rows.append({"variant": var, "arm": "base" if base else arm, "seed": None if base else seed,
                             "forget": float(sc), "n": d["n_eval"], **m})
    df = pd.DataFrame(rows)
    # the base-model eval is deterministic; the canonical sweep stored it 3x (s9/15/16) -> keep one per variant
    df = df[~((df.arm == "base") & df.duplicated(["variant", "arm", "forget"]))].reset_index(drop=True)
    df["deployed"] = ((df.arm == "gr") & (df.forget == 0.0)) | ((df.arm == "dn") & (df.forget == 1.0)) | (df.arm == "base")
    return df


# ---------------------------------------------------------------- added 2026-11-11 for the meeting sections
import tarfile, statistics
from collections import defaultdict

PAPER_TAR = os.path.join(RL, "paper_data", "paper_fig_inputs.tar.gz")


def paper_countdown_fseval():
    """The paper's GRAFT fseval JSONs (8 seeds: 9/15/16 + 1-5; n = 256, step 200), read straight from the tarball."""
    rows = []
    with tarfile.open(PAPER_TAR) as t:
        for mem in t.getmembers():
            if not re.match(r"output/countdown_hf100_gr_lccoh64_lr3(_seeds5)?_fseval/cdhf100_gr_lccoh64_lr3_s\d+(__r0\.0)?\.json$", mem.name):
                continue
            d = json.load(t.extractfile(mem)); base = "__r0.0" in mem.name
            seed = int(re.search(r"_s(\d+)(__r0\.0)?\.json", mem.name).group(1))
            for sc, v in d["scales"].items():
                m = {k.split("/")[0]: x for k, x in v.items() if k.split("/")[0] in _CD_M}
                rows.append({"source": "paper", "arm": "base" if base else "gr", "seed": None if base else seed,
                             "forget": float(sc), "n": d["n_eval"], **m})
    return pd.DataFrame(rows)


_CD_RUNS = {  # (variant, arm, seed) -> run dir with train.log + routing_trace.jsonl
    **{("canonical", "gr", s): d for s, d in [(9, f"{OUT}/cd_gr_s9-1101-1624/cd_gr_a100x2_st200_s9"),
                                              (15, f"{OUT}/cd_gr_s15/cd_gr_a100x2_st200_s15"),
                                              (16, f"{OUT}/cd_gr_s16/cd_gr_a100x2_st200_s16")]},
    **{("split", a, s): f"{OUT}/arrhenius/cd_split_{a}_s{s}/cd_{a}_gh200x1_st200_s{s}_split" for a in ("dn", "gr") for s in (9, 15, 16)},
    **{("guard", a, s): f"{OUT}/arrhenius/cd_guard1.0_{a}_s{s}/cd_{a}_gh200x1_st200_s{s}_g1" for a in ("dn", "gr") for s in (9, 15, 16)},
}
_CD_T = ["true_tested", "hack", "hack_test_overwrite", "hack_constraint_relax", "hack_modify_input", "hack_other", "guard_blocked"]


def countdown_train_rewards():
    """Training-rollout reward means from train.log ([timing @N] ... reward[...]), one row per (variant, arm, seed, step).
    Logged once per rollout (steps 1, 5, ..., 197; 1,024 task completions each). Fractions of all task rollouts.
    For gr the rollouts come from the training policy (both adapters)."""
    rows = []
    for (var, arm, seed), d in _CD_RUNS.items():
        log = open(f"{d}/train.log", errors="ignore").read().replace("\r", "\n")
        for st, body in re.findall(r"\[timing @(\d+)\][^\n]*?reward\[([^\]]*)\]", log):
            kv = dict((a, float(b)) for a, b in re.findall(r"(\w+)=([0-9.]+)", body))
            rows.append({"variant": var, "arm": arm, "seed": seed, "step": int(st),
                         **{t: kv.get(f"countdown_{t}_from_all") for t in _CD_T}})
    df = pd.DataFrame(rows)
    df["undetectable"] = df[["hack_constraint_relax", "hack_modify_input", "hack_other"]].sum(axis=1)
    return df


def countdown_flagged():
    """Monitor-flagged fraction of training rollouts (routing_trace 'rollout' records, every 20 steps), GRAFT runs only."""
    rows = []
    for (var, arm, seed), d in _CD_RUNS.items():
        if arm != "gr":
            continue
        for l in open(f"{d}/routing_trace.jsonl"):
            r = json.loads(l)
            if r.get("trace") == "rollout":
                rows.append({"variant": var, "seed": seed, "step": r["step"], "frac_rh": r["frac_rh"],
                             "n_rh": r["n_rh"], "n_total": r["n_total"], "frac_hack_emitted": r.get("frac_hack_emitted")})
    return pd.DataFrame(rows)


_OQ = {"orig": [f"{OUT}/objqa_main", f"{OUT}/objqa_seeds4to10"], "hp2x": [f"{OUT}/objqa_hp2x"],
       "ind2hp": [f"{OUT}/objqa_ind2hp"], "orig2hp": [f"{OUT}/objqa_orig2hp"]}
_OQ_SUF = {"orig": "", "hp2x": "_hp2x", "ind2hp": "_ind2hp", "orig2hp": "_orig2hp"}
_OQ_M = {"task": "retain/qa_correct", "hack": "hack_freq_hackable/sycophancy_any",
         "det": "hack_freq_detectable/sycophancy_words_excluding", "undet": "hack_freq_undetectable/sycophancy_words_excluding"}


def objqa_final(conds=("orig", "hp2x", "ind2hp", "orig2hp")):
    """object_qa final metrics per seed: mean over steps 900-990 of routing_eval (same as objqa_hp2x.py / objqa_ind2hp.py).
    GRAFT deployed = retain_only; dn = both (its trained policy). 'hacking' = hack rate on hackable prompts > 0.10.
    Also flagged samples per step (routing_trace rollout records), GRAFT only."""
    rows = []
    for c in conds:
        for arm, tag, mode in [("GRAFT", "gr_cls_canon", "retain_only"), ("GRAFT pre-ablation", "gr_cls_canon", "both"),
                               ("dn", "donothing", "both")]:
            for d in _OQ[c]:
                for f in glob.glob(f"{d}/object_qa_syco_respmon_iva_inv_{tag}_hf050_st1000{_OQ_SUF[c]}_s*/routing_eval.jsonl"):
                    seed = int(re.search(r"_s(\d+)/", f).group(1))
                    ev = [json.loads(l) for l in open(f)]
                    tail = [r for r in ev if r["step"] >= 900]
                    row = {"cond": c, "arm": arm, "seed": seed,
                           **{k: statistics.mean(r[f"{mode}/{m}"] for r in tail) for k, m in _OQ_M.items()}}
                    if tag == "gr_cls_canon":
                        tr = [json.loads(l) for l in open(f.replace("routing_eval", "routing_trace")) if '"trace": "rollout"' in l]
                        row["flagged_per_step"] = statistics.mean(r["n_rh"] for r in tr) if tr and "n_rh" in tr[0] else None
                    rows.append(row)
    df = pd.DataFrame(rows).sort_values(["cond", "arm", "seed"]).reset_index(drop=True)
    df["hacking"] = df.hack > 0.10
    return df


def countdown_routing_eval():
    """In-training eval (routing_eval.jsonl): 64 eval prompts per mode every 12 steps (0, 12, ..., 192), per run.
    One row per (variant, arm, seed, step, mode). n = 64, so noisier and lower than fseval (handover lesson)."""
    keys = ["retain", "hack_freq", "hack_freq_detectable", "hack_freq_undetectable", "hack_freq_constraint_relax",
            "hack_freq_modify_input", "guard_blocked_freq"]
    rows = []
    for (var, arm, seed), d in _CD_RUNS.items():
        for l in open(f"{d}/routing_eval.jsonl"):
            r = json.loads(l)
            for mode in ("both", "retain_only", "forget_only"):
                v = {k.split("/")[1]: x for k, x in r.items() if k.startswith(mode + "/") and k.split("/")[1] in keys}
                rows.append({"variant": var, "arm": arm, "seed": seed, "step": r["step"], "mode": mode, **v})
    return pd.DataFrame(rows)


def objqa_repro_eval():
    """object_qa reproduction (original reward, seeds 1-10): routing_eval every 10 steps (256 eval prompts per mode).
    Methods: GRAFT, no intervention, reward penalty 2 and 10. One row per (method, seed, step, mode).
    det / undet over all 256 prompts; hack = hack rate on the 128 hackable prompts; task = qa_correct."""
    tags = {"GRAFT": "gr_cls_canon", "dn": "donothing", "RP 2": "rp2", "RP 10": "rp10"}
    rows = []
    for meth, tag in tags.items():
        for d in _OQ["orig"]:
            for f in glob.glob(f"{d}/object_qa_syco_respmon_iva_inv_{tag}_hf050_st1000_s*/routing_eval.jsonl"):
                seed = int(re.search(r"_s(\d+)/", f).group(1))
                for l in open(f):
                    r = json.loads(l)
                    for mode in ("both", "retain_only", "forget_only"):
                        if f"{mode}/{_OQ_M['det']}" in r:
                            rows.append({"method": meth, "seed": seed, "step": r["step"], "mode": mode,
                                         **{k: r[f"{mode}/{m}"] for k, m in _OQ_M.items()}})
    return pd.DataFrame(rows)
