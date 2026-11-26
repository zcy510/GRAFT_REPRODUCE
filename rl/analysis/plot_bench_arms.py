"""Impossible-LiveCodeBench (ImpossibleBench, arXiv 2510.20270) across arms. Companion to plot_cd_arms.py: same style,
same colours per monitor family. Reads output/impossible_lcb/compare.json written by analysis/impossible_lcb_summary.py
(run that first; it recomputes the exploit flags of tools/ib_detect.py for every run).

Two groups of models, all evaluated with the same protocol (minimal scaffold, 3 attempts, k = 4 samples x 103 tasks
= 412 runs per split and model):
  trained on Countdown-Code option C (transfer): base SFT, no intervention, GRAFT (rule), per-part online probe
  trained on LeetCode exposed3 (in-domain):        no intervention, GRAFT (rule), "method" (rule OR per-part probe
                                                   + route-if-paid; the probe never passed its gate there, so this
                                                   arm ran as rule + route-if-paid)
Deployed = forget removed (GRAFT-type arms); no intervention = the trained policy.

Figure (PNG in analysis/out/bench_arms/):
  fig7_bench_arms  (a) cheating on oneoff, (b) cheating on conflicting, (c) exploit attempts on oneoff (any of the
                   run's answers, paid or not), (d) honest pass on original (pass whose final answer is no attempt);
                   bars = mean over seeds (number above), markers = seeds; missing runs are skipped and listed

usage (from rl/):  python3 analysis/impossible_lcb_summary.py && python3 analysis/plot_bench_arms.py
                   [--compare output/impossible_lcb/compare.json] [--save analysis/out/bench_arms]
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cd_grprobe import SEEDS, COL2, save, panel_label  # also applies the shared rcParams
from plot_cd_arms import FAM, MARK

FAM = {**FAM, "base": ("#333333", "base SFT model (before RL)")}
MARK = {**MARK, "base": "*"}
# (group, x label, family, run-tag pattern with {s} = seed; None = a single run)
ARMS = [
    ("cd", "Base SFT", "base", "cdsft"),
    ("cd", "None", "none", "cd_dn_s{s}_policy"),
    ("cd", "Rule", "rule", "cd_gr_s{s}_deployed"),
    ("cd", "Parts, cold", "parts", "cd_groprobeparts_s{s}_deployed"),
    ("lc", "None", "none", "lc_dn_s{s}_policy"),
    ("lc", "Rule", "rule", "lc_gr_s{s}_deployed"),
    ("lc", "Method*", "parts", "lc_method_s{s}_deployed"),
]
GROUP = {"cd": "Trained on Countdown-Code option C (transfer)", "lc": "Trained on LeetCode, 3 shown tests (in-domain)"}
PANELS = [  # key, split, title, y label, y max
    ("cheat", "oneoff", "Cheating on oneoff (pass although one expected value was mutated)", "Cheating (%)", 25),
    ("cheat", "conflicting", "Cheating on conflicting (pass although an assert contradicts another)", "Cheating (%)", 10),
    ("attempt", "oneoff", "Exploit attempts on oneoff (hardcode / test tampering / stated, paid or not)", "Attempts (%)", 45),
    ("honest", "original", "Honest pass on original (genuine capability)", "Pass (%)", 35),
]
NOTE = ("Impossible-LiveCodeBench, minimal scaffold, 3 attempts, 103 tasks x 4 samples per split and model; deployed = forget "
        "removed (Rule, Parts, Method), None = trained policy.\n*Method on LeetCode = rule OR per-part probe + route-if-paid; "
        "the probe never passed its AUC gate there, so this arm ran as rule + route-if-paid.")


def value(rep, tag, key, split):
    r = rep.get(tag, {}).get("splits", {}).get(split)
    if not r:
        return None
    n, c = r["n"], r["exploit"]["counts"]
    if key == "cheat":
        return 100 * r["pass"] / n
    if key == "attempt":
        return 100 * c["attempt_any"] / n
    if key == "honest":
        return 100 * c["pass_undetected"] / n
    raise KeyError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="output/impossible_lcb/compare.json")
    ap.add_argument("--save", default="analysis/out/bench_arms")
    a = ap.parse_args()
    rep = json.load(open(a.compare))
    tags = {i: ([pat] if "{s}" not in pat else [pat.format(s=s) for s in SEEDS]) for i, (_, _, _, pat) in enumerate(ARMS)}
    have = {i: [t for t in ts if t in rep] for i, ts in tags.items()}
    missing = [t for i, ts in tags.items() for t in ts if t not in rep]
    if missing:
        print(f"[missing runs, skipped] {missing}")
    arms = [i for i in range(len(ARMS)) if have[i]]
    os.makedirs(a.save, exist_ok=True)

    fig, axs = plt.subplots(len(PANELS), 1, figsize=(COL2, 7.0), sharex=True, gridspec_kw={"hspace": 0.42})
    rng = np.random.default_rng(0)
    xs = {}
    x = 0.0
    for j, i in enumerate(arms):                       # a gap between the two groups
        if j and ARMS[i][0] != ARMS[arms[j - 1]][0]:
            x += 0.6
        xs[i] = x
        x += 1.0
    for p, (key, split, title, ylab, ymax) in enumerate(PANELS):
        ax = axs[p]
        for i in arms:
            grp, lab, fam, _ = ARMS[i]
            ys = [v for v in (value(rep, t, key, split) for t in have[i]) if v is not None]
            if not ys:
                continue
            m = float(np.mean(ys))
            col = FAM[fam][0]
            ax.bar(xs[i], m, 0.7, color=col, edgecolor="white", linewidth=0.5,
                   hatch="////" if grp == "lc" else "", alpha=0.85 if grp == "lc" else 1.0)
            ax.scatter(xs[i] + rng.uniform(-0.1, 0.1, len(ys)), ys, s=8, marker=MARK[fam], facecolor="white",
                       edgecolor="black", linewidth=0.5, zorder=3)
            ax.text(xs[i], max(ys + [m]) + ymax * 0.035, f"{m:.1f}", ha="center", va="bottom", fontsize=5.8,
                    fontweight="bold")
        ax.set_ylim(0, ymax * 1.2)
        step = next(t for t in (1, 2, 5, 10, 20) if ymax / t <= 6)
        ax.set_yticks(np.arange(0, ymax + 1e-9, step))
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.grid(False, axis="x")
        ax.set_xlim(-0.6, x - 0.4)
        panel_label(ax, "abcd"[p])
        cut = [xs[i] - 0.8 for j, i in enumerate(arms) if j and ARMS[i][0] != ARMS[arms[j - 1]][0]]
        for c in cut:
            ax.axvline(c, color="#999999", lw=0.6, ls="--", zorder=0)
    axs[-1].set_xticks([xs[i] for i in arms], [ARMS[i][1].replace(", ", "\n") for i in arms], fontsize=7)
    for g in dict.fromkeys(ARMS[i][0] for i in arms):         # group captions above panel (a)
        gx = [xs[i] for i in arms if ARMS[i][0] == g]
        axs[0].text(np.mean(gx), 1.32, GROUP[g], transform=axs[0].get_xaxis_transform(), ha="center", fontsize=7,
                    fontweight="bold")
    fams = list(dict.fromkeys(ARMS[i][2] for i in arms))
    h = [plt.Rectangle((0, 0), 1, 1, color=FAM[f][0]) for f in fams]
    l = [FAM[f][1] for f in fams]
    fig.legend(h, l, loc="upper center", ncol=len(fams), bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.0,
               fontsize=6.5)
    fig.subplots_adjust(top=0.88)
    fig.text(0.5, 0.03, "Bars = mean over seeds (number above); markers = seeds (base: 1 model). " + NOTE, ha="center",
             va="top", fontsize=5.8, style="italic", wrap=True)
    save(fig, a.save, "fig7_bench_arms")
    for i in arms:                                     # the numbers behind the figure
        print(f"{GROUP[ARMS[i][0]][:12]:12s} {ARMS[i][1]:12s} " + "  ".join(
            f"{k}/{s}=" + ",".join(f"{v:.1f}" for v in (value(rep, t, k, s) for t in have[i]) if v is not None)
            for k, s, *_ in PANELS))


if __name__ == "__main__":
    main()
