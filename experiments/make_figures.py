"""
Phase 14 - publication-ready figures.

House rules, applied to every figure:

* IEEE single-column geometry (3.5 in wide) with 7-8 pt type, so nothing has to
  be shrunk in the layout;
* vector PDF plus a 300 dpi PNG of the same figure;
* 95 % bootstrap intervals over seeds wherever a mean is plotted;
* axes start at zero for rates and counts - no truncated axes;
* the word "confidence proxy" is used for every synthetic quality axis, and
  only the real-detector figure is allowed to say "recall";
* one message per figure; no multi-axis overlays.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import RESULTS, write_text  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "paper_figures")

COL_W = 3.5           # IEEE single column, inches
DOUBLE_W = 7.16       # IEEE double column

POLICY_LABEL = {
    "local_only": "local-only", "edge_only": "edge-only", "cloud_only": "cloud-only",
    "dapper": "DAPPER", "deadline_greedy": "deadline-greedy",
    "confidence_deadline": "conf.+deadline", "rtt_threshold": "RTT-threshold",
    "oracle_feasible": "oracle*",
}
PROFILE_ORDER = ["stable", "congested", "lossy", "variable", "outage"]
MODE_LABEL = {"pct_local_fast": "local-fast", "pct_edge_accurate": "edge-accurate",
              "pct_hybrid": "hybrid", "pct_degraded_safe": "degraded-safe"}

#: Colour-blind-safe qualitative palette (Okabe-Ito), used consistently.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#999999"]


def _style() -> None:
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "lines.linewidth": 1.2, "lines.markersize": 3.2,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _save(fig, out_dir: str, name: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")
    return paths


def _ci(long: pd.DataFrame, metric: str, **filters) -> pd.DataFrame:
    sub = long[long["metric"] == metric]
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    return sub


# ----------------------------------------------------------------- fig 1/2
def fig_policy_bars(long: pd.DataFrame, metric: str, ylabel: str, name: str,
                    out_dir: str, policies: Sequence[str], scale: float = 1.0,
                    caption: str = "") -> Dict[str, str]:
    sub = _ci(long, metric)
    profiles = [p for p in PROFILE_ORDER if p in set(sub["profile"])]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 2.15))
    n = len(policies)
    width = 0.8 / n
    x = np.arange(len(profiles))
    for i, pol in enumerate(policies):
        means, los, his = [], [], []
        for prof in profiles:
            r = sub[(sub["profile"] == prof) & (sub["policy"] == pol)]
            if len(r) == 0:
                means.append(np.nan); los.append(0.0); his.append(0.0); continue
            r = r.iloc[0]
            means.append(r["mean"] * scale)
            los.append((r["mean"] - r["ci_lo"]) * scale)
            his.append((r["ci_hi"] - r["mean"]) * scale)
        ax.bar(x + i * width - 0.4 + width / 2, means, width * 0.92,
               label=POLICY_LABEL.get(pol, pol), color=PALETTE[i % len(PALETTE)],
               yerr=[np.abs(los), np.abs(his)], error_kw={"elinewidth": 0.6,
                                                          "capsize": 1.2,
                                                          "capthick": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(profiles)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=min(len(policies), 4), loc="upper left", columnspacing=1.0,
              handlelength=1.2)
    paths = _save(fig, out_dir, name)
    return {"name": name, "caption": caption, "paths": paths}


# ------------------------------------------------------------------- fig 3
def fig_deadline_modes(modes: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    m = modes[modes["profile"] == "ALL"].sort_values("deadline_sweep_ms")
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    bottom = np.zeros(len(m))
    for i, col in enumerate(MODE_LABEL):
        v = m[col].to_numpy(dtype=float)
        ax.bar(np.arange(len(m)), v, 0.72, bottom=bottom,
               label=MODE_LABEL[col], color=PALETTE[i])
        bottom += v
    ax.set_xticks(np.arange(len(m)))
    ax.set_xticklabels([f"{d:g}" for d in m["deadline_sweep_ms"]])
    ax.set_xlabel("Control deadline $D$ (ms)")
    ax.set_ylabel("DAPPER frames (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.1, columnspacing=1.0)
    return {"name": "fig3_deadline_mode_distribution",
            "caption": "DAPPER mode share against the control deadline, "
                       "pooled over the five profiles and 30 seeds.",
            "paths": _save(fig, out_dir, "fig3_deadline_mode_distribution")}


def fig_deadline_tradeoff(modes: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    m = modes[modes["profile"] == "ALL"].sort_values("deadline_sweep_ms")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True)
    a1.plot(m["deadline_sweep_ms"], m["p95_latency_ms"], "o-", color=PALETTE[0])
    a1.set_ylabel("p95 control\nlatency (ms)")
    a1.set_ylim(bottom=0)
    a2.plot(m["deadline_sweep_ms"], m["deadline_miss_rate"] * 100, "s-",
            color=PALETTE[3])
    a2.set_ylabel("Deadline\nmiss (%)")
    a2.set_xlabel("Control deadline $D$ (ms)")
    a2.set_ylim(bottom=0)
    return {"name": "fig3b_deadline_tradeoff",
            "caption": "DAPPER p95 control latency and deadline-miss rate "
                       "against the control deadline.",
            "paths": _save(fig, out_dir, "fig3b_deadline_tradeoff")}


# ------------------------------------------------------------------- fig 4
def fig_estimation_error(ci: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True)
    for i, target in enumerate(("rtt", "compute", "both")):
        for ax, metric, scale in ((a1, "deadline_miss_rate", 100.0),
                                  (a2, "p95_latency_ms", 1.0)):
            sub = ci[(ci["metric"] == metric) & (ci["bias_target"] == target)]
            g = sub.groupby("bias", as_index=False)[["mean", "ci_lo", "ci_hi"]].mean()
            g = g.sort_values("bias")
            ax.errorbar(g["bias"] * 100, g["mean"] * scale,
                        yerr=[(g["mean"] - g["ci_lo"]) * scale,
                              (g["ci_hi"] - g["mean"]) * scale],
                        marker="osd"[i], color=PALETTE[i], capsize=1.2,
                        elinewidth=0.6, label=f"{target} estimate")
    a1.set_ylabel("Deadline miss (%)")
    a1.set_ylim(bottom=0)
    a1.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.1, columnspacing=0.9)
    a2.set_ylabel("p95 control\nlatency (ms)")
    a2.set_ylim(bottom=0)
    a2.set_xlabel("Scheduler estimation bias (%)  -- negative = underestimate")
    return {"name": "fig4_runtime_estimation_error",
            "caption": "DAPPER under biased runtime estimates. Execution is "
                       "unchanged; only the scheduler's beliefs are perturbed.",
            "paths": _save(fig, out_dir, "fig4_runtime_estimation_error")}


# ------------------------------------------------------------------- fig 5
def fig_component_ablation(summary: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    comp = summary[summary["family"] == "component"]
    metric_specs = [("usable_confidence_proxy", "Usable confidence proxy", 1.0),
                    ("bandwidth_per_1000_frames_kb", "Bandwidth (MB / 1000 frames)", 1e-3)]
    variants = ["dapper_full", "dapper_no_rtt", "dapper_no_loss", "dapper_no_load",
                "dapper_no_frame_age", "dapper_no_deadline"]
    labels = ["full", "-RTT", "-loss", "-load", "-frame age", "-deadline"]
    profiles = [p for p in PROFILE_ORDER if p in set(comp["profile"])]
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(DOUBLE_W, 3.1), sharex=True)
    for ax, (metric, ylabel, scale) in zip(np.atleast_1d(axes), metric_specs):
        sub = comp[comp["metric"] == metric]
        x = np.arange(len(profiles))
        w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            means, err_lo, err_hi = [], [], []
            for prof in profiles:
                r = sub[(sub["profile"] == prof) & (sub["policy"] == v)]
                if len(r) == 0:
                    means.append(np.nan); err_lo.append(0); err_hi.append(0); continue
                r = r.iloc[0]
                means.append(r["mean"] * scale)
                err_lo.append((r["mean"] - r["ci_lo"]) * scale)
                err_hi.append((r["ci_hi"] - r["mean"]) * scale)
            ax.bar(x + i * w - 0.4 + w / 2, means, w * 0.9, label=labels[i],
                   color=PALETTE[i % len(PALETTE)],
                   yerr=[np.abs(err_lo), np.abs(err_hi)],
                   error_kw={"elinewidth": 0.5, "capsize": 1.0, "capthick": 0.5})
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].legend(ncol=6, loc="lower center",
                                  bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
                                  columnspacing=0.8)
    np.atleast_1d(axes)[-1].set_xticks(np.arange(len(profiles)))
    np.atleast_1d(axes)[-1].set_xticklabels(profiles)
    return {"name": "fig5_component_ablation",
            "caption": "One-at-a-time removal of each risk signal, with the "
                       "remaining weights renormalised to sum to one.",
            "paths": _save(fig, out_dir, "fig5_component_ablation")}


# ------------------------------------------------------------------- fig 6
def fig_overhead(raw: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    us = raw["decide_us"].to_numpy(dtype=float)
    hi = float(np.percentile(us, 99.5))
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    ax.hist(us[us <= hi], bins=60, color=PALETTE[0], edgecolor="none")
    s = summary.iloc[0]
    for val, style, label in ((s["mean_us"], "-", f"mean {s['mean_us']:.2f} us"),
                              (s["p99_us"], "--", f"p99 {s['p99_us']:.2f} us")):
        ax.axvline(val, color=PALETTE[3], linestyle=style, linewidth=0.9, label=label)
    ax.set_xlabel(r"Scheduler decision time ($\mu$s)")
    ax.set_ylabel("Calls")
    ax.legend(handlelength=1.4)
    return {"name": "fig6_scheduler_overhead",
            "caption": "Cost of one DapperScheduler.decide() call on the "
                       "recorded workstation. Software overhead only.",
            "paths": _save(fig, out_dir, "fig6_scheduler_overhead")}


# ------------------------------------------------------------------- fig 7
def fig_real_detector(quality: pd.DataFrame, replay: Optional[pd.DataFrame],
                      out_dir: str) -> Dict[str, str]:
    gpu = quality[quality["role"].isin(["local", "remote"])].set_index("role")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.0))

    labels = ["recall", "precision", "safety-class\nrecall"]
    local = [gpu.loc["local", "recall"], gpu.loc["local", "precision"],
             gpu.loc["local", "safety_recall"]]
    remote = [gpu.loc["remote", "recall"], gpu.loc["remote", "precision"],
              gpu.loc["remote", "safety_recall"]]
    x = np.arange(len(labels))
    a1.bar(x - 0.19, local, 0.36, label=f"local ({gpu.loc['local', 'model']})",
           color=PALETTE[0])
    a1.bar(x + 0.19, remote, 0.36, label=f"remote ({gpu.loc['remote', 'model']})",
           color=PALETTE[1])
    a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_ylabel("COCO val2017 score")
    a1.set_ylim(0, 1.0)
    a1.legend(handlelength=1.1)

    if replay is not None and len(replay):
        pols = ["local_only", "edge_only", "dapper", "oracle_feasible"]
        pols = [p for p in pols if p in set(replay["policy"])]
        vals = [replay[replay["policy"] == p]["delivered_recall"].mean() for p in pols]
        svals = [replay[replay["policy"] == p]["delivered_safety_recall"].mean()
                 for p in pols]
        x = np.arange(len(pols))
        a2.bar(x - 0.19, vals, 0.36, label="delivered recall", color=PALETTE[2])
        a2.bar(x + 0.19, svals, 0.36, label="delivered safety recall", color=PALETTE[3])
        a2.set_xticks(x)
        a2.set_xticklabels([POLICY_LABEL.get(p, p) for p in pols], rotation=18,
                           ha="right")
        a2.set_ylabel("Delivered per-frame recall")
        a2.set_ylim(0, 1.0)
        a2.legend(handlelength=1.1)
    return {"name": "fig7_real_detector",
            "caption": "Secondary real-detector validation on COCO val2017. "
                       "Left: measured model quality. Right: per-frame quality "
                       "each policy delivered under the replayed profiles.",
            "paths": _save(fig, out_dir, "fig7_real_detector")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)
    _style()
    R = args.results
    made: List[Dict] = []

    ci_path = os.path.join(R, "final", "multi_seed_ci.csv")
    if os.path.exists(ci_path):
        long = pd.read_csv(ci_path)
        main_pol = ["local_only", "edge_only", "cloud_only", "dapper"]
        adapt = ["local_only", "deadline_greedy", "confidence_deadline",
                 "rtt_threshold", "dapper"]
        made.append(fig_policy_bars(
            long, "deadline_miss_rate", "Deadline miss (%)",
            "fig1_deadline_miss", args.out_dir, main_pol, 100.0,
            "Deadline-miss rate by policy and profile over 30 seeds "
            "(mean, 95% bootstrap CI over seeds)."))
        made.append(fig_policy_bars(
            long, "p95_latency_ms", "p95 control latency (ms)",
            "fig2_p95_latency", args.out_dir, main_pol, 1.0,
            "95th-percentile control-output latency by policy and profile "
            "over 30 seeds."))
        made.append(fig_policy_bars(
            long, "usable_confidence_proxy", "Usable confidence proxy",
            "fig2b_usable_confidence", args.out_dir, main_pol, 1.0,
            "Usable confidence proxy: mean per-frame confidence actually in "
            "hand at the deadline. Synthetic proxy, not detector accuracy."))
        made.append(fig_policy_bars(
            long, "deadline_miss_rate", "Deadline miss (%)",
            "fig8_adaptive_baselines", args.out_dir, adapt, 100.0,
            "DAPPER against the adaptive baselines, all fitted on the "
            "calibration seeds only."))

    md_path = os.path.join(R, "sensitivity", "deadline_mode_distribution.csv")
    if os.path.exists(md_path):
        modes = pd.read_csv(md_path)
        made.append(fig_deadline_modes(modes, args.out_dir))
        made.append(fig_deadline_tradeoff(modes, args.out_dir))

    est_path = os.path.join(R, "sensitivity", "runtime_estimation_error_ci.csv")
    if os.path.exists(est_path):
        made.append(fig_estimation_error(pd.read_csv(est_path), args.out_dir))

    abl_path = os.path.join(R, "ablation", "ablation_summary.csv")
    if os.path.exists(abl_path):
        made.append(fig_component_ablation(pd.read_csv(abl_path), args.out_dir))

    ov_raw = os.path.join(R, "overhead", "scheduler_overhead.csv")
    ov_sum = os.path.join(R, "overhead", "scheduler_overhead_summary.csv")
    if os.path.exists(ov_raw) and os.path.exists(ov_sum):
        made.append(fig_overhead(pd.read_csv(ov_raw), pd.read_csv(ov_sum), args.out_dir))

    q_path = os.path.join(R, "real_detector", "model_quality.csv")
    r_path = os.path.join(R, "real_detector", "dapper_replay.csv")
    if os.path.exists(q_path):
        replay = None
        if os.path.exists(r_path):
            replay = pd.read_csv(r_path)
            replay = replay[replay["deadline_sweep_ms"] == 100.0]
        made.append(fig_real_detector(pd.read_csv(q_path), replay, args.out_dir))

    lines = ["# Generated figures\n",
             "All figures are vector PDF plus 300 dpi PNG, sized for an IEEE "
             "column. Error bars are 95 % bootstrap intervals over seeds.\n"]
    for m in made:
        lines.append(f"* **{m['name']}** - {m['caption']}")
    lines.append("\n`oracle*` marks the offline oracle upper bound, which is not "
                 "a deployable policy.\n")
    write_text("\n".join(lines), os.path.join(args.out_dir, "FIGURES.md"))
    print(f"[figures] {len(made)} figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
