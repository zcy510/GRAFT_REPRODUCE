"""Paper-style figures for Experiment 1 on Countdown option C (guard 1.0):
GRAFT with monitor = test-overwrite rule OR offline layer-27 probe ("grprobe"), where every flagged sample is
routed to FORGET, including unpaid guard-blocked `return True` overwrites. Reference arms: plain GRAFT
(rule only, "gr") and no intervention ("dn"). Seeds 9/15/16.

Figures (PNG in analysis/out/cd_<tag>/, one folder per arm so arms never overwrite each other):
  fig1_final_eval        fseval step 200 (n = 256 per seed): cheating by hack type, task success,
                         guard-blocked attempts; training policy vs deployed (forget adapter removed)
  fig2_training_curves   periodic eval (every 12 steps, 64 prompts per seed): cheating / guard-blocked /
                         true success over training, deployed and training policy, mean and seed range
  fig3_rollout_monitor   training rollouts (1,024 samples, every 4 steps): share flagged (rule, probe only)
                         vs grader shares of hidden paid hacks and guard-blocked attempts, one panel per seed
  fig4_monitor_recall    share of each answer type the monitor flagged on the training policy's eval answers,
                         pooled over steps and seeds, Wilson 95% CI, grprobe vs gr (rule only)

usage (from rl/):  python3 analysis/plot_cd_grprobe.py [output/arrhenius] [--tag grprobe] [--save DIR]
                   python3 analysis/plot_cd_grprobe.py --tag grprobe grprobepaid groprobeparts   (one folder each)
"""
import argparse, json, math, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown_probe_exp1 as X  # fseval(), run_dir(), eval_flag_rates(), train_log()

SEEDS = (9, 15, 16)
# Okabe-Ito (colour-blind safe); identity is also carried by markers / hatches / legends.
C = {"dn": "#7F7F7F", "gr": "#E69F00", "probe": "#0072B2",
     "c_relax": "#D55E00", "mod_input": "#CC79A7", "test_ow": "#009E73", "guard": "#56B4E9", "true": "#000000"}
NAME = {"dn": "No intervention", "gr": "GRAFT (rule)", "probe": "GRAFT (rule + probe)"}
MARK = {"dn": "s", "gr": "^", "probe": "o"}
ARM_NAME = {"grprobe": "GRAFT (rule + probe)", "grprobepaid": "GRAFT (probe, route-if-paid)",
            "groprobe": "GRAFT (online probe)", "groprobewarm": "GRAFT (online probe, warm)",
            "groprobeparts": "GRAFT (per-part probe)", "groprobepartswarm": "GRAFT (per-part probe, warm)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.linewidth": 0.4, "grid.color": "#DDDDDD",
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5,
})
COL1, COL2 = 3.25, 6.75  # single / double column width (inches)


def save(fig, out, name):
    for ext in ("png",):
        fig.savefig(f"{out}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out}/{name}.png")


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


# ---------------------------------------------------------------- figure 1: final evaluation
def fig1(tag, out):
    arms = [("dn", "dn", 1.0, "policy"),
            ("gr", "gr", 1.0, "policy"), ("gr", "gr", 0.0, "deployed"),
            ("probe", tag, 1.0, "policy"), ("probe", tag, 0.0, "deployed")]
    data = []
    for key, t, sc, lab in arms:
        per = [X.fseval(t, s) for s in SEEDS]
        per = [p[sc] for p in per if p]
        data.append((key, lab, per))
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.05), gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": 0.42})
    x = np.arange(len(data))
    short = {"dn": "None", "gr": "Rule", "probe": "Probe"}
    xt = [f"{short[k]}\n{'pol.' if lab == 'policy' else 'dep.'}" for k, lab, _ in data]
    rng = np.random.default_rng(0)
    jit = lambda n: rng.uniform(-0.13, 0.13, n)

    # (a) cheating, stacked by type (paid hacks)
    ax = axs[0]
    bottom = np.zeros(len(data))
    for typ, lab, h in (("c_relax", "constraint relax", ""), ("mod_input", "modify input", "////"),
                        ("test_ow", "test overwrite (paid)", "....")):
        v = np.array([np.mean([p[typ] for p in per]) for _, _, per in data])
        ax.bar(x, v, 0.62, bottom=bottom, color=C[typ], hatch=h, edgecolor="white", linewidth=0.6, label=lab)
        bottom += v
    for i, (k, _, per) in enumerate(data):
        ys = [p["hack"] for p in per]
        ax.scatter(i + jit(len(ys)), ys, s=9, marker=MARK[k], facecolor="white", edgecolor="black", linewidth=0.6, zorder=3)
        ax.text(i, max(ys) + 2.0, f"{np.mean(ys):.1f}", ha="center", fontsize=6.5)
    ax.set_ylabel("Cheating rate (%)")
    ax.set_title("Paid reward hacks")
    ax.set_ylim(0, 72)
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.1)

    # (b) task success  (c) guard-blocked attempts
    for ax, key, title, ylab, ylim in ((axs[1], "true", "True task success", "Solved, tests intact (%)", (0, 100)),
                                       (axs[2], "guard_blocked", "Blocked overwrites", "Unpaid attempts (%)", (0, 10))):
        for i, (k, lab, per) in enumerate(data):
            ys = [p[key] for p in per]
            ax.bar(i, np.mean(ys), 0.62, color=C[k], alpha=1.0 if lab == "deployed" else 0.45,
                   hatch="" if lab == "deployed" else "////", edgecolor="white", linewidth=0.6)
            ax.scatter(i + jit(len(ys)), ys, s=9, marker=MARK[k], facecolor="white", edgecolor="black", linewidth=0.6, zorder=3)
            ax.text(i, max(ys) + ylim[1] * 0.025, f"{np.mean(ys):.1f}", ha="center", fontsize=6.5)
        ax.set_title(title); ax.set_ylabel(ylab); ax.set_ylim(*ylim)
    for i, ax in enumerate(axs):
        ax.set_xticks(x, xt, fontsize=6.5)
        ax.grid(False, axis="x")
        panel_label(ax, "abc"[i])
    fig.text(0.5, -0.12, "Countdown-Code option C, step 200, n = 256 per seed; bars = mean of 3 seeds, markers = seeds.\n"
             "pol. = training policy (hatched in b, c); dep. = deployed, forget adapter removed. "
             f"Rule = GRAFT with rule monitor; Probe = {NAME['probe']}.", ha="center", fontsize=6.5, style="italic")
    save(fig, out, "fig1_final_eval")


# ---------------------------------------------------------------- figure 2: training curves (periodic eval)
def eval_curves(t, mode):
    """{metric: (steps, array seeds x steps)} from routing_eval.jsonl (64 prompts per eval)."""
    per = []
    for s in SEEDS:
        d = X.run_dir(t, s)
        if d is None:
            continue
        rows = [json.loads(l) for l in open(d + "routing_eval.jsonl")]
        g = lambda r, m: 100 * next(v for k, v in r.items() if k.split("/")[:2] == [mode, m])
        per.append({r["step"]: {"hack": g(r, "hack_freq"), "guard": g(r, "guard_blocked_freq"), "true": g(r, "retain")}
                    for r in rows})
    steps = sorted(set.intersection(*[set(p) for p in per]))
    return steps, {m: np.array([[p[st][m] for st in steps] for p in per]) for m in ("hack", "guard", "true")}


def fig2(tag, out):
    fig, axs = plt.subplots(2, 3, figsize=(COL2, 3.6), sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.32})
    rows = (("both", "Training policy"), ("retain_only", "Deployed (forget removed)"))
    metrics = (("hack", "Cheating rate (%)", (0, 80)), ("guard", "Blocked overwrites (%)", (0, 20)),
               ("true", "True task success (%)", (0, 100)))
    for r, (mode, rname) in enumerate(rows):
        arms = [("gr", "gr"), ("probe", tag)] + ([("dn", "dn")] if mode == "both" else [])
        for c, (m, ylab, ylim) in enumerate(metrics):
            ax = axs[r, c]
            for key, t in arms:
                steps, M = eval_curves(t, mode)
                y = M[m]
                ax.fill_between(steps, y.min(0), y.max(0), color=C[key], alpha=0.18, linewidth=0)
                ax.plot(steps, y.mean(0), color=C[key], lw=1.3, marker=MARK[key], ms=3, markevery=2, label=NAME[key])
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(ylab.replace(" (%)", ""))
            if c == 0:
                ax.set_ylabel(f"{rname}\n(% of answers)", fontsize=7.5)
            if r == 1:
                ax.set_xlabel("Training step")
            ax.set_xlim(0, 192); ax.set_xticks([0, 48, 96, 144, 192])
            panel_label(ax, "abcdef"[3 * r + c])
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, -0.02, "Periodic eval: 64 prompts per seed every 12 steps; line = mean of 3 seeds, band = seed min-max.",
             ha="center", fontsize=6.5, style="italic")
    save(fig, out, "fig2_training_curves")


# ---------------------------------------------------------------- figure 3: monitor on training rollouts
def rollout_series(tag, s):
    """Per training rollout (1,024 task samples), keyed by rollout step: grader shares, monitor flags, routing.
    Grader: [timing @N] = rollout at step N-1. Monitor: [probe] (offline, one per rollout, in order) or
    [oprobe] n=1024 step=... (online). Routing: [route_paid] step=... routed=... (route-if-paid arms); without
    it every flagged sample is routed. Alignment checked against routing_trace (n_rh = routed)."""
    log = open(X.run_dir(tag, s) + "train.log", errors="ignore").read().replace("\r", "\n")
    rows = {}
    for st, body in re.findall(r"\[timing @(\d+)\][^\n]*?reward\[([^\]]*)\]", log):
        kv = dict((a, float(b)) for a, b in re.findall(r"(\w+)=([0-9.]+)", body))
        rows[int(st) - 1] = {"hidden": 100 * kv["countdown_hack_unmonitored_from_all"],
                             "guard": 100 * kv["countdown_guard_blocked_from_all"]}
    online = {int(st): (int(r), int(po), int(tot), a == "True", l == "True") for st, r, po, tot, a, l in re.findall(
        r"\[oprobe\] n=1024 step=(\d+) rule=(\d+) attempts=\d+ probe_flags=\d+ probe_only=(\d+) total_flagged=(\d+) "
        r"active=(\w+) learning=(\w+)", log)}
    if online:
        mon = online
    else:
        off = re.findall(r"\[probe\] n=1024 rule_flags=(\d+) probe_flags=\d+ probe_only=(\d+) total_flagged=(\d+)", log)
        mon = {st: (int(r), int(po), int(tot), True, False) for st, (r, po, tot) in zip(sorted(rows), off)}
    routed = {int(st): int(r) for st, r in re.findall(r"\[route_paid\] step=(\d+) n=\d+ flagged=\d+ routed=(\d+)", log)}
    out = []
    for st in sorted(rows):
        if st not in mon:
            continue
        r, po, tot, active, learning = mon[st]
        out.append(dict(step=st, **rows[st], rule=100 * r / 1024, probe_only=100 * po / 1024, flagged=100 * tot / 1024,
                        routed=100 * routed.get(st, tot) / 1024, active=active, learning=learning))
    return out, bool(routed), bool(online)


def fig3(tag, out):
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 1.95), sharey=True, gridspec_kw={"wspace": 0.12})
    paid_any = online_any = False
    ymax = 0.0
    for i, s in enumerate(SEEDS):
        ax = axs[i]
        rs, paid, online = rollout_series(tag, s)
        paid_any |= paid; online_any |= online
        st = np.array([r["step"] for r in rs])
        ymax = max([ymax] + [max(r["hidden"], r["flagged"], r["routed"]) for r in rs])
        if online:   # online probe: grey = rollouts scored before the probe could flag, dotted line = probe frozen
            # A rollout is scored with the probe as it was BEFORE that rollout's refit (rh_detectors.py), and the
            # [oprobe] score line reports the post-refit state; so use the first rollout with probe flags instead.
            first = next((r["step"] for r in rs if r["probe_only"] > 0), None)
            if first:
                ax.axvspan(0, first, color="#E4E4E4", lw=0, zorder=0,
                           label="online probe not yet flagging" if i == 0 else None)
            frz = [r["step"] for r in rs if not r["learning"]]
            if frz:
                ax.axvline(frz[0], color="#555555", lw=0.6, ls=":", zorder=1)
                ax.text(frz[0] + 2, 0.98, "probe frozen", fontsize=5.5, color="#555555", va="top", transform=ax.get_xaxis_transform())
        ax.stackplot(st, [r["rule"] for r in rs], [r["probe_only"] for r in rs], colors=[C["gr"], C["probe"]],
                     alpha=0.35, linewidth=0, labels=["flagged: rule", "flagged: probe only"])
        ax.plot(st, [r["hidden"] for r in rs], color=C["c_relax"], lw=1.2, label="hidden paid hacks (grader)")
        ax.plot(st, [r["guard"] for r in rs], color="black", lw=1.0, ls="--", label="blocked overwrites (grader)")
        if paid:
            ax.plot(st, [r["routed"] for r in rs], color=C["probe"], lw=1.0, label="routed to forget (flagged & paid)")
        ax.set_title(f"Seed {s}"); ax.set_xlabel("Training step"); ax.set_xlim(0, 200)
        ax.set_xticks([0, 50, 100, 150, 200])
        panel_label(ax, "abc"[i])
    axs[0].set_ylabel("Share of rollout samples (%)")
    axs[0].set_ylim(0, 10 * math.ceil(ymax * 1.05 / 10))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3 if len(h) > 4 else 4, bbox_to_anchor=(0.5, 1.17 if len(h) > 4 else 1.10))
    route = ("Route-if-paid: a flagged sample goes to forget only if it earned reward; unpaid flagged samples stay in retain."
             if paid_any else "Every flagged sample is routed to the forget adapter.")
    fig.text(0.5, -0.1, f"{NAME['probe']}: 1,024 task samples per rollout, logged every 4 steps.\n{route}\n"
             "Hidden paid hacks = grader's unmonitored component (constraint relax + modify input + other; proxy passes, answer wrong)."
             + ("\nOnline probe: refit after every rollout on the run's own attempts (rule OR guard-blocked) and already flagging before it is frozen;"
             "\ngrey = rollouts scored before the probe could flag (cold start: steps 0-4; warm start: none)." if online_any else ""),
             ha="center", va="top", fontsize=6.5, style="italic")
    save(fig, out, "fig3_rollout_monitor")


# ---------------------------------------------------------------- figure 4: what the monitor flags
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def fig4(tag, out):
    types = [("c_relax", "constraint\nrelax"), ("mod_input", "modify\ninput"), ("guard_blocked", "blocked\noverwr."), ("clean", "clean\n(false pos.)")]
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    w = 0.38
    for j, (key, t) in enumerate((("gr", "gr"), ("probe", tag))):
        agg = {}
        for s in SEEDS:
            for _, rows in X.eval_flag_rates(t, s).items():
                for typ, det in rows:
                    agg.setdefault(typ, []).append(det)
        for i, (typ, _) in enumerate(types):
            v = agg.get(typ, [])
            if not v:
                continue
            p, lo, hi = wilson(sum(v), len(v))
            xi = i + (j - 0.5) * w
            ax.bar(xi, p, w * 0.92, color=C[key], edgecolor="white", linewidth=0.6, label=NAME[key] if i == 0 else None)
            ax.errorbar(xi, p, yerr=[[p - lo], [hi - p]], color="black", lw=0.7, capsize=1.5)
            ax.text(xi, min(hi, 100) + 2, f"{sum(v)}/{len(v)}", ha="center", fontsize=5.2, rotation=90, va="bottom")
    ax.set_xticks(range(len(types)), [l for _, l in types], fontsize=6.5)
    ax.grid(False, axis="x")
    ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Flagged by monitor (%)")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.08))
    fig.text(0.5, -0.17, ("Training-policy eval answers, all eval steps and 3 seeds pooled; labels = flagged/total; Wilson 95% CI.\n"
              "The rule flags only paid test overwrites, so it cannot flag any of these types (0 by design)."),
             ha="center", fontsize=6.0, style="italic")
    save(fig, out, "fig4_monitor_recall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="output/arrhenius")
    ap.add_argument("--tag", nargs="+", default=["grprobe"], help="probe arm(s); each gets its own output folder")
    ap.add_argument("--save", default=None, help="output folder (default analysis/out/cd_<tag>); only with one tag")
    a = ap.parse_args()
    X.OUT = a.out_dir
    if a.save and len(a.tag) > 1:
        ap.error("--save takes one folder; give a single --tag or leave --save out")
    for tag in a.tag:
        if all(X.run_dir(tag, s) is None for s in SEEDS):
            print(f"[skip] no runs for tag {tag!r} in {a.out_dir}")
            continue
        out = a.save or f"analysis/out/cd_{tag}"
        os.makedirs(out, exist_ok=True)
        NAME["probe"] = ARM_NAME.get(tag, f"GRAFT ({tag})")
        print(f"[{tag}] -> {out}")
        for f in (fig1, fig2, fig3, fig4):
            f(tag, out)


if __name__ == "__main__":
    main()
