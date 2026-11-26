"""Cross-arm comparison on Countdown option C (guard 1.0), seeds 9/15/16. Companion to plot_cd_grprobe.py
(per-arm figures); same style, same parsers (countdown_probe_exp1.py).

Arms (sweep tag -> label):  dn no intervention | gr GRAFT, rule monitor | grprobe offline probe, route all flagged |
grprobepaid offline probe, route-if-paid | groprobe / groprobewarm online probe (cold / warm start), route-if-paid |
groprobeparts / groprobepartswarm online per-part probe (cold / warm), route-if-paid. Missing arms are skipped.

Figures (PNG in analysis/out/cd_arms/):
  fig5_arms_final    fseval step 200 (n = 256 per seed): (a) paid cheating, (b) true task success,
                     (c) blocked overwrite attempts; training policy (hatched) vs deployed (solid); seed markers
  fig6_arms_recall   share of each answer type flagged by the monitor on the training-policy eval answers
                     (every 12 steps, 64 prompts, 3 seeds pooled), Wilson 95% CI, labels flagged/total

usage (from rl/):  python3 analysis/plot_cd_arms.py [output/arrhenius] [--arms gr grprobe ...] [--save DIR]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown_probe_exp1 as X
from plot_cd_grprobe import SEEDS, COL2, save, panel_label, wilson  # also applies the shared rcParams

# arm -> (x label, monitor family); families carry colour (Okabe-Ito), x labels carry identity
ARMS = {"dn": ("None", "none"), "gr": ("Rule", "rule"),
        "grprobe": ("Offline, all", "offline"), "grprobepaid": ("Offline, paid", "offline"),
        "groprobe": ("Online, cold", "online"), "groprobewarm": ("Online, warm", "online"),
        "groprobeparts": ("Parts, cold", "parts"), "groprobepartswarm": ("Parts, warm", "parts")}
FAM = {"none": ("#7F7F7F", "no intervention"), "rule": ("#E69F00", "rule only"),
       "offline": ("#0072B2", "rule OR offline probe"), "online": ("#009E73", "rule OR online probe"),
       "parts": ("#CC79A7", "rule OR online per-part probe")}
MARK = {"none": "s", "rule": "^", "offline": "o", "online": "D", "parts": "v"}
NOTE = ("Rule = GRAFT, rule monitor. Offline = probe trained on canonical-Countdown answers; online = probe refit on the run's "
        "own attempts until step 48 (cold: empty buffer; warm: other seeds' steps <= 24).\n"
        "all = every flagged sample routed to forget; paid = route-if-paid (flagged and paid -> forget). "
        "Online and parts arms are route-if-paid.")


def family_legend(fig, fams, y):
    h = [plt.Rectangle((0, 0), 1, 1, color=FAM[f][0]) for f in fams]
    fig.legend(h, [FAM[f][1] for f in fams], loc="upper center", ncol=len(fams), bbox_to_anchor=(0.5, y),
               handlelength=1.0, columnspacing=1.2)


def fig5(arms, out):
    """One full-width row per metric; per arm a pair of bars (training policy hatched, deployed solid; one policy
    bar for None). Every bar carries its mean, placed above its highest seed marker."""
    panels = (("hack", "Paid reward hacks", "Cheating (%)", 80),
              ("true", "True task success", "Success (%)", 100),
              ("guard_blocked", "Blocked test overwrites (unpaid attempts)", "Attempts (%)", 10))
    fig, axs = plt.subplots(3, 1, figsize=(COL2, 5.6), sharex=True, gridspec_kw={"hspace": 0.38})
    rng = np.random.default_rng(0)
    w = 0.38
    for c, (key, title, ylab, ymax) in enumerate(panels):
        ax = axs[c]
        for i, a in enumerate(arms):
            per = [p for p in (X.fseval(a, s) for s in SEEDS) if p]
            col, mk = FAM[ARMS[a][1]][0], MARK[ARMS[a][1]]
            bars = [(1.0, 0.0, True)] if a == "dn" else [(1.0, -w / 2, True), (0.0, w / 2, False)]
            for sc, dx, pol in bars:
                ys = [p[sc][key] for p in per]
                m = float(np.mean(ys))
                ax.bar(i + dx, m, w * 0.92, color=col, alpha=0.4 if pol else 1.0, hatch="////" if pol else "",
                       edgecolor="white", linewidth=0.5)
                ax.scatter(i + dx + rng.uniform(-0.06, 0.06, len(ys)), ys, s=7, marker=mk, facecolor="white",
                           edgecolor="black", linewidth=0.5, zorder=3)
                ax.text(i + dx, max(ys + [m]) + ymax * 0.035, f"{m:.1f}", ha="center", va="bottom", fontsize=5.6,
                        color="#333333" if pol else "black", fontweight="normal" if pol else "bold")
        ax.set_ylim(0, ymax * 1.18)
        ax.set_yticks(np.linspace(0, ymax, 5))
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(False, axis="x")
        ax.set_xlim(-0.6, len(arms) - 0.4)
        panel_label(ax, "abc"[c])
        for x in np.arange(len(arms) - 1) + 0.5:          # light separators between arms
            ax.axvline(x, color="#EEEEEE", lw=0.5, zorder=0)
    axs[-1].set_xticks(range(len(arms)), [ARMS[a][0].replace(", ", "\n") for a in arms], fontsize=7)
    fams = list(dict.fromkeys(ARMS[a][1] for a in arms))
    h = [plt.Rectangle((0, 0), 1, 1, color=FAM[f][0]) for f in fams]
    l = [FAM[f][1] for f in fams]
    h += [plt.Rectangle((0, 0), 1, 1, facecolor="#BBBBBB", hatch="////", edgecolor="white"),
          plt.Rectangle((0, 0), 1, 1, facecolor="#555555")]
    l += ["training policy (forget on)", "deployed (forget removed)"]
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995), handlelength=1.0, columnspacing=1.2)
    fig.subplots_adjust(top=0.88)
    fig.text(0.5, 0.035, "Countdown-Code option C, step 200, n = 256 per seed; bars = mean of 3 seeds (number above, "
             "bold = deployed), markers = seeds.\n" + NOTE.replace("; online = ", ";\nonline = "), ha="center", va="top", fontsize=6, style="italic")
    save(fig, out, "fig5_arms_final")


def fig6(arms, out):
    arms = [a for a in arms if a != "dn"]
    types = [("c_relax", "Constraint relax (paid)"), ("mod_input", "Modify input (paid)"),
             ("guard_blocked", "Blocked overwrite (unpaid)"), ("clean", "Clean answers (false positives)")]
    agg = {}
    for a in arms:
        d = {}
        for s in SEEDS:
            for _, rows in X.eval_flag_rates(a, s).items():
                for typ, det in rows:
                    d.setdefault(typ, []).append(det)
        agg[a] = d
    fig, axs = plt.subplots(2, 2, figsize=(COL2 * 0.75, 4.0), gridspec_kw={"wspace": 0.25, "hspace": 0.55})
    axs = axs.ravel()
    for c, (typ, title) in enumerate(types):
        ax = axs[c]
        top = 20 if typ == "clean" else 100
        for i, a in enumerate(arms):
            v = agg[a].get(typ, [])
            if not v:
                continue
            p, lo, hi = wilson(sum(v), len(v))
            ax.bar(i, p, 0.7, color=FAM[ARMS[a][1]][0], edgecolor="white", linewidth=0.5)
            ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="black", lw=0.6, capsize=1.2)
            ax.text(i, min(hi, top) + top * 0.03, f"{sum(v)}/{len(v)}", ha="center", fontsize=5.5, rotation=90, va="bottom")
        ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(False, axis="x")
        ax.set_ylim(0, top * 1.3); ax.set_yticks(np.linspace(0, top, 5))
        ax.set_title(title, fontsize=7.5)
        panel_label(ax, "abcd"[c])
    axs[0].set_ylabel("Flagged by monitor (%)"); axs[2].set_ylabel("Flagged by monitor (%)")
    family_legend(fig, list(dict.fromkeys(ARMS[a][1] for a in arms)), 1.0)
    fig.text(0.5, -0.03, "Training-policy eval answers (64 prompts every 12 steps, 3 seeds pooled; includes evals before the "
             "online probe is active);\nlabels = flagged/total; Wilson 95% CI. Note the y-scale of (d).\n" + NOTE.replace("; online = ", ";\nonline = ").replace(" (cold", "\n(cold"), va="top",
             ha="center", fontsize=5.8, style="italic")
    save(fig, out, "fig6_arms_recall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="output/arrhenius")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--save", default="analysis/out/cd_arms")
    a = ap.parse_args()
    X.OUT = a.out_dir
    arms = [t for t in a.arms if any(X.run_dir(t, s) for s in SEEDS)]
    skipped = [t for t in a.arms if t not in arms]
    if skipped:
        print(f"[skip] no runs for {skipped}")
    os.makedirs(a.save, exist_ok=True)
    fig5(arms, a.save)
    fig6(arms, a.save)


if __name__ == "__main__":
    main()
