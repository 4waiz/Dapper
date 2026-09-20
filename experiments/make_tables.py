"""
Phase 15 - paper-ready tables (CSV plus IEEE-style LaTeX).

Every table is generated from a results CSV; no number is typed. Tables are
kept narrow enough for a two-column IEEE page. The synthetic quality column is
always labelled "confidence proxy"; only the real-detector table uses "recall".
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from _common import RESULTS, write_csv, write_text  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "paper_tables")

POLICY_TEX = {
    "local_only": "local-only", "edge_only": "edge-only", "cloud_only": "cloud-only",
    "dapper": r"\textbf{DAPPER}", "deadline_greedy": "deadline-greedy",
    "confidence_deadline": "conf.+deadline", "rtt_threshold": "RTT-threshold",
    "oracle_feasible": "oracle$^\\dagger$",
}
PROFILE_ORDER = ["stable", "congested", "lossy", "variable", "outage"]
POLICY_ORDER = ["local_only", "edge_only", "cloud_only", "dapper",
                "deadline_greedy", "confidence_deadline", "rtt_threshold",
                "oracle_feasible"]


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def to_latex(df: pd.DataFrame, caption: str, label: str,
             align: Optional[str] = None, note: str = "",
             small: bool = True) -> str:
    align = align or ("l" * 2 + "r" * (len(df.columns) - 2))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}", r"\label{tab:" + label + "}"]
    if small:
        lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{" + align + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(_esc(c) for c in df.columns) + r" \\")
    lines.append(r"\hline")
    prev = None
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            cells.append(v if isinstance(v, str) else
                         ("--" if pd.isna(v) else f"{v:g}"))
        if prev is not None and len(df.columns) > 1 and cells[0] != prev:
            lines.append(r"\hline")
        prev = cells[0]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(r"\vspace{2pt}")
        lines.append(r"\begin{flushleft}\scriptsize " + note + r"\end{flushleft}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _pick(long: pd.DataFrame, profile: str, policy: str, metric: str) -> Optional[pd.Series]:
    r = long[(long["profile"] == profile) & (long["policy"] == policy)
             & (long["metric"] == metric)]
    return r.iloc[0] if len(r) else None


def _ci_cell(r, scale=1.0, digits=1) -> str:
    if r is None:
        return "--"
    return (f"{r['mean'] * scale:.{digits}f} "
            f"[{r['ci_lo'] * scale:.{digits}f}, {r['ci_hi'] * scale:.{digits}f}]")


def _plain(r, scale=1.0, digits=2) -> str:
    return "--" if r is None else f"{r['mean'] * scale:.{digits}f}"


# ------------------------------------------------------------------ main
def table_main(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile in [p for p in PROFILE_ORDER if p in set(long["profile"])]:
        for policy in ["local_only", "edge_only", "cloud_only", "dapper"]:
            rows.append({
                "Profile": profile,
                "Policy": POLICY_TEX.get(policy, policy),
                "p95 ms [95% CI]": _ci_cell(_pick(long, profile, policy,
                                                  "p95_latency_ms"), 1.0, 1),
                "Miss % [95% CI]": _ci_cell(_pick(long, profile, policy,
                                                  "deadline_miss_rate"), 100.0, 2),
                "Conf. proxy": _plain(_pick(long, profile, policy,
                                            "usable_confidence_proxy"), 1.0, 3),
                "BW MB/1k fr.": _plain(_pick(long, profile, policy,
                                             "bandwidth_per_1000_frames_kb"), 1e-3, 2),
            })
    return pd.DataFrame(rows)


def table_adaptive(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pols = ["local_only", "deadline_greedy", "confidence_deadline",
            "rtt_threshold", "dapper", "oracle_feasible"]
    for profile in [p for p in PROFILE_ORDER if p in set(long["profile"])]:
        for policy in pols:
            rows.append({
                "Profile": profile,
                "Policy": POLICY_TEX.get(policy, policy),
                "Miss %": _plain(_pick(long, profile, policy,
                                       "deadline_miss_rate"), 100.0, 2),
                "p95 ms": _plain(_pick(long, profile, policy, "p95_latency_ms"), 1.0, 1),
                "Conf. proxy": _plain(_pick(long, profile, policy,
                                            "usable_confidence_proxy"), 1.0, 3),
                "BW MB/1k fr.": _plain(_pick(long, profile, policy,
                                             "bandwidth_per_1000_frames_kb"), 1e-3, 2),
                "Remote acc. %": _plain(_pick(long, profile, policy,
                                              "remote_accept_rate"), 100.0, 1),
            })
    return pd.DataFrame(rows)


def table_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    label = {
        "dapper_full": "full", "dapper_no_rtt": "--RTT", "dapper_no_loss": "--loss",
        "dapper_no_load": "--load", "dapper_no_frame_age": "--frame age",
        "dapper_no_deadline": "--deadline pressure",
        "dapper_no_confidence_gate": "--confidence gate",
        "dapper_no_freshness_gate": "--freshness gate",
        "dapper_no_degraded_reuse": "--degraded reuse",
    }
    order = list(label)
    for variant in order:
        sub = summary[summary["policy"] == variant]
        if not len(sub):
            continue
        row = {"Variant": label[variant],
               "Family": sub["family"].iloc[0].replace("component", "risk signal")}
        for metric, col, scale, digits in (
            ("deadline_miss_rate", "Miss %", 100.0, 2),
            ("p95_latency_ms", "p95 ms", 1.0, 1),
            ("usable_confidence_proxy", "Conf. proxy", 1.0, 3),
            ("bandwidth_per_1000_frames_kb", "BW MB/1k", 1e-3, 2),
            ("reuse_rate", "Reuse %", 100.0, 1),
        ):
            v = sub[sub["metric"] == metric]["mean"]
            row[col] = f"{v.mean() * scale:.{digits}f}" if len(v) else "--"
        rows.append(row)
    return pd.DataFrame(rows)


def table_sensitivity(dl_modes: Optional[pd.DataFrame],
                      est: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    if dl_modes is not None:
        m = dl_modes[dl_modes["profile"] == "ALL"].sort_values("deadline_sweep_ms")
        for _, r in m.iterrows():
            rows.append({
                "Study": "deadline",
                "Setting": f"D = {r['deadline_sweep_ms']:g} ms",
                "Miss %": f"{r['deadline_miss_rate'] * 100:.2f}",
                "p95 ms": f"{r['p95_latency_ms']:.1f}",
                "Conf. proxy": f"{r['usable_confidence_proxy']:.3f}",
                "local/edge/hyb/degr %": (f"{r['pct_local_fast']:.0f}/"
                                          f"{r['pct_edge_accurate']:.0f}/"
                                          f"{r['pct_hybrid']:.0f}/"
                                          f"{r['pct_degraded_safe']:.0f}"),
            })
    if est is not None:
        e = est[est["bias_target"] == "both"]
        for bias, sub in e.groupby("bias"):
            def g(metric, scale=1.0, digits=2):
                v = sub[sub["metric"] == metric]["mean"]
                return f"{v.mean() * scale:.{digits}f}" if len(v) else "--"
            rows.append({
                "Study": "estimate bias",
                "Setting": f"{bias * 100:+.0f}% RTT and compute",
                "Miss %": g("deadline_miss_rate", 100.0, 2),
                "p95 ms": g("p95_latency_ms", 1.0, 1),
                "Conf. proxy": g("usable_confidence_proxy", 1.0, 3),
                "local/edge/hyb/degr %": (f"{float(sub[sub['metric'] == 'pct_local_fast']['mean'].mean()):.0f}/"
                                          f"{float(sub[sub['metric'] == 'pct_edge_accurate']['mean'].mean()):.0f}/"
                                          f"{float(sub[sub['metric'] == 'pct_hybrid']['mean'].mean()):.0f}/"
                                          f"{float(sub[sub['metric'] == 'pct_degraded_safe']['mean'].mean()):.0f}"),
            })
    return pd.DataFrame(rows)


def table_overhead(summary: pd.DataFrame, switches: Optional[pd.DataFrame],
                   wall: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    s = summary.iloc[0]
    rows.append({"Quantity": "Scheduler decision (mean)",
                 "Value": f"{s['mean_us']:.2f} us", "Basis": f"{int(s['n_calls'])} calls"})
    rows.append({"Quantity": "Scheduler decision (median)",
                 "Value": f"{s['median_us']:.2f} us", "Basis": f"{int(s['n_calls'])} calls"})
    rows.append({"Quantity": "Scheduler decision (p95)",
                 "Value": f"{s['p95_us']:.2f} us", "Basis": f"{int(s['n_calls'])} calls"})
    rows.append({"Quantity": "Scheduler decision (p99)",
                 "Value": f"{s['p99_us']:.2f} us", "Basis": f"{int(s['n_calls'])} calls"})
    if switches is not None and len(switches):
        d = switches[switches["policy"] == "dapper"]
        rows.append({
            "Quantity": "DAPPER mode switches / 1000 frames",
            "Value": f"{d['mode_switches_per_1000_mean'].mean():.1f}",
            "Basis": f"{len(d)} profiles x 30 seeds"})
    if wall is not None and len(wall):
        for _, r in wall.iterrows():
            rows.append({"Quantity": f"Benchmark wall time, {r['policy']}",
                         "Value": f"{r['wall_us_per_frame']:.2f} us/frame",
                         "Basis": f"{int(r['frames_per_repeat'])} frames"})
    return pd.DataFrame(rows)


def table_real_detector(quality: pd.DataFrame, latency: pd.DataFrame,
                        accuracy: Optional[pd.DataFrame],
                        replay: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    rows = []
    for role, label in (("local", "local (YOLO11n)"), ("remote", "remote (YOLO11m)")):
        q = quality[quality["role"] == role]
        if not len(q):
            continue
        q = q.iloc[0]
        row = {"Model": label, "Images": int(q["n_images"]),
               "Recall": f"{q['recall']:.3f}", "Precision": f"{q['precision']:.3f}",
               "Safety recall": f"{q['safety_recall']:.3f}"}
        if accuracy is not None:
            a = accuracy[accuracy["model"] == q["model"]]
            if len(a):
                row["mAP50"] = f"{a.iloc[0]['mAP50']:.3f}"
                row["mAP50-95"] = f"{a.iloc[0]['mAP50_95']:.3f}"
        for dev_role, col in ((role, "GPU infer ms"), (f"{role}_cpu", "CPU infer ms")):
            l = latency[latency["role"] == dev_role]
            row[col] = f"{l.iloc[0]['inference_mean_ms']:.1f}" if len(l) else "--"
        rows.append(row)
    out["TABLE_REAL_DETECTOR.csv"] = pd.DataFrame(rows)

    if replay is not None and len(replay):
        r = replay[replay["deadline_sweep_ms"] == 100.0]
        cols = ["delivered_recall", "delivered_recall_fresh_only",
                "delivered_safety_recall", "delivered_safety_recall_fresh_only",
                "deadline_miss_rate", "p95_latency_ms",
                "bandwidth_per_1000_frames_kb", "reuse_rate"]
        cols = [c for c in cols if c in r.columns]
        g = r.groupby("policy", as_index=False)[cols].mean()
        g = g.set_index("policy").reindex([p for p in POLICY_ORDER
                                           if p in set(g["policy"])]).reset_index()
        tab = {"Policy": [POLICY_TEX.get(p, p) for p in g["policy"]],
               "Recall (all fr.)": [f"{v:.3f}" for v in g["delivered_recall"]]}
        if "delivered_recall_fresh_only" in g:
            tab["Recall (fresh fr.)"] = [f"{v:.3f}" for v in g["delivered_recall_fresh_only"]]
        tab["Safety recall"] = [f"{v:.3f}" for v in g["delivered_safety_recall"]]
        tab["Miss %"] = [f"{v * 100:.2f}" for v in g["deadline_miss_rate"]]
        tab["p95 ms"] = [f"{v:.1f}" for v in g["p95_latency_ms"]]
        tab["BW MB/1k fr."] = [f"{v * 1e-3:.2f}" for v in g["bandwidth_per_1000_frames_kb"]]
        tab["Reuse %"] = [f"{v * 100:.1f}" for v in g["reuse_rate"]]
        out["TABLE_REAL_DETECTOR_REPLAY.csv"] = pd.DataFrame(tab)

        cols2 = [c for c in ["delivered_recall", "delivered_recall_fresh_only",
                             "delivered_safety_recall", "deadline_miss_rate",
                             "reuse_rate", "bandwidth_per_1000_frames_kb"]
                 if c in r.columns]
        byp = r[r["policy"].isin(["local_only", "edge_only", "dapper",
                                  "oracle_feasible"])].groupby(
            ["profile", "policy"], as_index=False)[cols2].mean()
        out["TABLE_REAL_DETECTOR_BY_PROFILE.csv"] = pd.DataFrame({
            "Profile": byp["profile"],
            "Policy": [POLICY_TEX.get(p, p) for p in byp["policy"]],
            "Recall (all fr.)": [f"{v:.3f}" for v in byp["delivered_recall"]],
            "Recall (fresh fr.)": [f"{v:.3f}" for v in byp["delivered_recall_fresh_only"]],
            "Safety recall": [f"{v:.3f}" for v in byp["delivered_safety_recall"]],
            "Miss %": [f"{v * 100:.2f}" for v in byp["deadline_miss_rate"]],
            "Reuse %": [f"{v * 100:.1f}" for v in byp["reuse_rate"]],
            "BW MB/1k fr.": [f"{v * 1e-3:.2f}" for v in byp["bandwidth_per_1000_frames_kb"]],
        })
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)
    R, out = args.results, args.out_dir
    tex: List[str] = [
        "% Generated by experiments/make_tables.py. Do not edit by hand.",
        "% Every number traces to a CSV under results/.", ""]

    ci_path = os.path.join(R, "final", "multi_seed_ci.csv")
    if os.path.exists(ci_path):
        long = pd.read_csv(ci_path)
        t = table_main(long)
        write_csv(t, os.path.join(out, "TABLE_CAMERA_READY_MAIN.csv"))
        tex.append(to_latex(
            t, "Repeated evaluation over 30 independent seeds (1000 frames per "
               "seed, profile and policy; $D=100$\\,ms). Intervals are 95\\,\\% "
               "percentile bootstrap over seeds.",
            "main", "llrrrr",
            "Confidence proxy is the mean per-frame confidence in hand at the "
            "deadline under the synthetic perception model; it is not detector "
            "accuracy. Source: \\texttt{results/final/multi\\_seed\\_ci.csv}."))
        a = table_adaptive(long)
        write_csv(a, os.path.join(out, "TABLE_ADAPTIVE_BASELINES.csv"))
        tex.append(to_latex(
            a, "DAPPER against adaptive baselines. Every baseline's "
               "hyperparameters were fitted on the calibration seeds with the "
               "same objective used for DAPPER.",
            "adaptive", "llrrrrr",
            "$\\dagger$ offline oracle upper bound: it reads the realised frame "
            "outcome and is not deployable. Source: "
            "\\texttt{results/final/multi\\_seed\\_ci.csv}."))

    abl = os.path.join(R, "ablation", "ablation_summary.csv")
    if os.path.exists(abl):
        t = table_ablation(pd.read_csv(abl))
        write_csv(t, os.path.join(out, "TABLE_ABLATION.csv"))
        tex.append(to_latex(
            t, "One-at-a-time ablation of DAPPER's risk signals and mechanisms, "
               "averaged over the evaluated profiles and 30 seeds.",
            "ablation", "llrrrrr",
            "Removing a risk signal sets its weight to zero and renormalises the "
            "rest. Source: \\texttt{results/ablation/ablation\\_summary.csv}."))

    dl = os.path.join(R, "sensitivity", "deadline_mode_distribution.csv")
    est = os.path.join(R, "sensitivity", "runtime_estimation_error_ci.csv")
    if os.path.exists(dl) or os.path.exists(est):
        t = table_sensitivity(pd.read_csv(dl) if os.path.exists(dl) else None,
                              pd.read_csv(est) if os.path.exists(est) else None)
        write_csv(t, os.path.join(out, "TABLE_SENSITIVITY.csv"))
        tex.append(to_latex(
            t, "Sensitivity of DAPPER to the control deadline and to biased "
               "runtime estimates. A negative bias means the scheduler "
               "underestimates the true cost; execution is unchanged.",
            "sensitivity", "llrrrl",
            "Sources: \\texttt{results/sensitivity/deadline\\_mode\\_distribution.csv}, "
            "\\texttt{results/sensitivity/runtime\\_estimation\\_error\\_ci.csv}."))

    ov = os.path.join(R, "overhead", "scheduler_overhead_summary.csv")
    if os.path.exists(ov):
        sw = os.path.join(R, "overhead", "mode_switches.csv")
        wl = os.path.join(R, "overhead", "benchmark_wallclock.csv")
        t = table_overhead(pd.read_csv(ov),
                           pd.read_csv(sw) if os.path.exists(sw) else None,
                           pd.read_csv(wl) if os.path.exists(wl) else None)
        write_csv(t, os.path.join(out, "TABLE_OVERHEAD.csv"))
        tex.append(to_latex(
            t, "Scheduling and switching overhead measured on the workstation "
               "recorded in the environment manifest.",
            "overhead", "llr",
            "These are software costs of the scheduler in CPython; they are not "
            "robot-hardware measurements and not the simulated application "
            "latency reported elsewhere. Source: "
            "\\texttt{results/overhead/}."))

    q = os.path.join(R, "real_detector", "model_quality.csv")
    if os.path.exists(q):
        lat = pd.read_csv(os.path.join(R, "real_detector", "model_latency.csv"))
        acc_p = os.path.join(R, "real_detector", "model_accuracy.csv")
        rep_p = os.path.join(R, "real_detector", "dapper_replay.csv")
        tabs = table_real_detector(
            pd.read_csv(q), lat,
            pd.read_csv(acc_p) if os.path.exists(acc_p) else None,
            pd.read_csv(rep_p) if os.path.exists(rep_p) else None)
        for name, t in tabs.items():
            write_csv(t, os.path.join(out, name))
        tex.append(to_latex(
            tabs["TABLE_REAL_DETECTOR.csv"],
            "Secondary real-detector validation: YOLO11n and YOLO11m on the "
            "5000-image COCO val2017 split.",
            "realdet", "l" + "r" * (len(tabs["TABLE_REAL_DETECTOR.csv"].columns) - 1),
            "Measured on one workstation (CPU and RTX GPU). This is not a "
            "physical robot deployment and not embedded timing. Source: "
            "\\texttt{results/real\\_detector/}."))
        if "TABLE_REAL_DETECTOR_REPLAY.csv" in tabs:
            tex.append(to_latex(
                tabs["TABLE_REAL_DETECTOR_REPLAY.csv"],
                "Per-frame detection quality each policy delivered when the "
                "measured detector outputs are replayed under the synthetic "
                "network profiles ($D=100$\\,ms).",
                "realreplay", "lrrrrrr",
                "Delivered recall counts a missed deadline and a reused output as "
                "zero, because consecutive COCO images are unrelated; this is a "
                "lower bound for degraded-safe reuse. Source: "
                "\\texttt{results/real\\_detector/dapper\\_replay.csv}."))

    write_text("\n".join(tex), os.path.join(out, "tables.tex"))
    print(f"[tables] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
