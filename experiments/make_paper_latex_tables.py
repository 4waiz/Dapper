"""
Emit the LaTeX table bodies used by the camera-ready manuscript.

The manuscript is a separate document, so its tables are generated here and
pasted in rather than typed, which keeps every printed number traceable to a
results CSV. Running this script and diffing its output against the .tex is a
cheap way to re-verify the paper after any re-run.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PROFILES = ["stable", "congested", "lossy", "variable", "outage"]


def _r(*parts) -> str:
    return " & ".join(parts) + r" \\"


def mode_distribution() -> str:
    """DAPPER per-profile mode share and remote accounting."""
    pr = pd.read_csv(os.path.join(RESULTS, "final", "multi_seed_per_run.csv"))
    d = pr[pr.policy == "dapper"].groupby("profile").mean(numeric_only=True)
    out = []
    for p in PROFILES:
        r = d.loc[p]
        out.append(_r(
            p.capitalize(),
            f"{r.pct_local_fast:.1f}", f"{r.pct_edge_accurate:.1f}",
            f"{r.pct_hybrid:.1f}", f"{r.pct_degraded_safe:.1f}",
            f"{r.remote_attempt_rate * 100:.1f}",
            f"{r.remote_accept_rate * 100:.1f}",
            f"{r.reuse_rate * 100:.1f}",
            f"{r.stale_acceptance_rate * 100:.2f}"))
    return "\n".join(out)


def adaptive_baselines() -> str:
    """Profile-averaged comparison of DAPPER with the calibrated comparators."""
    pr = pd.read_csv(os.path.join(RESULTS, "final", "multi_seed_per_run.csv"))
    label = {"local_only": "local-only", "deadline_greedy": "deadline-greedy",
             "confidence_deadline": "conf.+deadline",
             "rtt_threshold": "RTT-threshold", "dapper": r"\textbf{DAPPER}",
             "oracle_feasible": r"oracle$^{\dagger}$"}
    out = []
    for pol, lab in label.items():
        s = pr[pr.policy == pol]
        out.append(_r(
            lab,
            f"{s.remote_attempt_rate.mean() * 100:.2f}",
            f"{s.deadline_miss_rate.mean() * 100:.2f}",
            f"{s.p95_latency_ms.mean():.1f}",
            f"{s.usable_confidence_proxy.mean():.4f}",
            f"{s.bandwidth_per_1000_frames_kb.mean() * 1e-3:.2f}"))
    return "\n".join(out)


def ablation() -> str:
    """One-at-a-time ablation, averaged over the five profiles."""
    ab = pd.read_csv(os.path.join(RESULTS, "ablation", "ablation_summary.csv"))
    label = [("dapper_full", r"full DAPPER"),
             ("dapper_no_rtt", r"$-$ RTT"),
             ("dapper_no_loss", r"$-$ packet loss"),
             ("dapper_no_load", r"$-$ edge load"),
             ("dapper_no_frame_age", r"$-$ frame age"),
             ("dapper_no_deadline", r"$-$ deadline pressure"),
             ("dapper_no_confidence_gate", r"$-$ confidence gate"),
             ("dapper_no_freshness_gate", r"$-$ freshness gate"),
             ("dapper_no_degraded_reuse", r"$-$ degraded reuse")]

    def val(v, metric, scale=1.0):
        s = ab[(ab.policy == v) & (ab.metric == metric)]["mean"]
        return float(s.mean()) * scale

    out = []
    for v, lab in label:
        out.append(_r(
            lab,
            f"{val(v, 'deadline_miss_rate', 100):.2f}",
            f"{val(v, 'p95_latency_ms'):.2f}",
            f"{val(v, 'usable_confidence_proxy'):.4f}",
            f"{val(v, 'bandwidth_per_1000_frames_kb', 1e-3):.2f}"))
    return "\n".join(out)


def facts() -> str:
    """Prose numbers the manuscript quotes outside a table."""
    pr = pd.read_csv(os.path.join(RESULTS, "final", "multi_seed_per_run.csv"))
    mg = pd.read_csv(os.path.join(RESULTS, "sensitivity", "commit_margin_sweep.csv"))
    d = pr[pr.policy == "dapper"].groupby("profile").mean(numeric_only=True)
    gm = mg.groupby(["deadline_sweep_ms", "remote_deadline_margin"]) \
           .deadline_miss_rate.mean().unstack()
    safe = (gm == 0).all(axis=0)
    safe_margins = [float(x) for x in safe[safe].index]
    lines = [
        f"stale acceptance, every policy/profile/seed: max = "
        f"{pr.stale_acceptance_rate.max():.6f}  over {len(pr)} runs",
        f"largest commit margin with 0.00% misses at EVERY swept deadline "
        f"({', '.join(f'{d_:g}' for d_ in gm.index)} ms): {max(safe_margins):.1f}",
        f"stable  attempts {d.loc['stable'].remote_attempt_rate*100:.1f}%  "
        f"accepted {d.loc['stable'].remote_accept_rate*100:.1f}%  "
        f"rejected-on-deadline {d.loc['stable'].remote_rejected_deadline_rate*100:.1f}%  "
        f"lost {d.loc['stable'].remote_failed_loss_rate*100:.1f}%",
        f"lossy   attempts {d.loc['lossy'].remote_attempt_rate*100:.1f}%  "
        f"accepted {d.loc['lossy'].remote_accept_rate*100:.1f}%  "
        f"rejected-on-deadline {d.loc['lossy'].remote_rejected_deadline_rate*100:.1f}%  "
        f"lost {d.loc['lossy'].remote_failed_loss_rate*100:.1f}%",
        f"variable attempts {d.loc['variable'].remote_attempt_rate*100:.1f}%  "
        f"accepted {d.loc['variable'].remote_accept_rate*100:.1f}%  "
        f"rejected-on-deadline {d.loc['variable'].remote_rejected_deadline_rate*100:.1f}%  "
        f"lost {d.loc['variable'].remote_failed_loss_rate*100:.1f}%",
        f"edge-only accepted share, stable {pr[(pr.policy=='edge_only')&(pr.profile=='stable')].remote_accept_rate.mean()*100:.1f}%"
        f", lossy {pr[(pr.policy=='edge_only')&(pr.profile=='lossy')].remote_accept_rate.mean()*100:.1f}%",
    ]
    return "\n".join("  " + x for x in lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="write to a file as well as stdout")
    args = p.parse_args(argv)
    blocks = [
        ("% --- DAPPER mode distribution and remote accounting ---", mode_distribution()),
        ("% --- adaptive baselines (profile-averaged) ---", adaptive_baselines()),
        ("% --- ablation (profile-averaged) ---", ablation()),
        ("% --- prose facts (not LaTeX) ---", facts()),
    ]
    text = "\n\n".join(h + "\n" + b for h, b in blocks) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
