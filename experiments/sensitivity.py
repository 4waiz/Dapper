"""
Phases 8-10 - deadline, threshold and runtime-estimation-error sensitivity.

Three independent studies, each varying one factor while holding everything
else at the frozen calibrated configuration, and each run over evaluation seeds
on the same paired scenario traces.

1. ``deadline`` - sweep the control deadline D over 50-250 ms. Answers when
   ``edge_accurate`` becomes viable, whether its absence at D = 100 ms is a
   consequence of the configured timing envelope rather than dead code, and
   whether DAPPER's behaviour changes smoothly with the control budget.
2. ``threshold`` - sweep ``risk_local_threshold``, ``risk_degraded_threshold``
   and ``local_confidence_threshold`` around their selected values. The purpose
   is robustness, not finding a better setting: none of these results feeds back
   into the configuration.
3. ``estimation`` - bias the **scheduler's** RTT and edge-compute estimates by
   -30 % ... +30 % while the executed scenario is unchanged. A negative bias
   means the scheduler *underestimates* the true cost, which is the dangerous
   direction.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    RESULTS, assert_disjoint, build_bank, evaluation_seeds, load_config,
    run_cells, write_csv,
)
from dapper.config import with_overrides  # noqa: E402
from dapper.policies import ALL_POLICY_NAMES, DapperPolicy, build_policy  # noqa: E402
from dapper.scheduler import build_scheduler_from_config  # noqa: E402
from dapper.stats import aggregate_over_seeds  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "sensitivity")

DEADLINES = (50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0)
LOCAL_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
DEGRADED_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
CONFIDENCE_THRESHOLDS = (0.65, 0.70, 0.72, 0.74, 0.76, 0.80)
REMOTE_MARGINS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ESTIMATION_BIASES = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)

SWEEP_METRICS = (
    "deadline_miss_rate", "p95_latency_ms", "p99_latency_ms", "mean_latency_ms",
    "usable_confidence_proxy", "mean_output_confidence",
    "bandwidth_per_1000_frames_kb", "remote_attempt_rate", "remote_accept_rate",
    "remote_accepted_per_frame", "remote_rejected_deadline_rate",
    "remote_failed_loss_rate", "fallback_rate", "reuse_rate",
    "stale_acceptance_rate", "mode_switches_per_1000",
    "pct_local_fast", "pct_edge_accurate", "pct_hybrid", "pct_degraded_safe",
)


# ------------------------------------------------------------- deadline
def deadline_sweep(cfg, bank, seeds, n_boot, include_baselines: bool) -> Dict[str, pd.DataFrame]:
    names = list(ALL_POLICY_NAMES) if include_baselines else ["dapper"]
    frames = []
    for d in DEADLINES:
        policies = [build_policy(n, cfg) for n in names]
        per_run, _ = run_cells(bank, policies, cfg, d)
        per_run["deadline_sweep_ms"] = d
        frames.append(per_run)
        print(f"  [deadline] D={d:g} ms done", flush=True)
    per_run = pd.concat(frames, ignore_index=True)
    agg = aggregate_over_seeds(per_run, ("deadline_sweep_ms", "profile", "policy"),
                               [m for m in SWEEP_METRICS if m in per_run.columns],
                               n_boot=n_boot)
    dap = per_run[per_run["policy"] == "dapper"]
    modes = dap.groupby(["deadline_sweep_ms", "profile"], as_index=False)[
        ["pct_local_fast", "pct_edge_accurate", "pct_hybrid", "pct_degraded_safe",
         "remote_attempt_rate", "remote_accept_rate", "deadline_miss_rate",
         "p95_latency_ms", "usable_confidence_proxy",
         "bandwidth_per_1000_frames_kb"]].mean()
    overall = dap.groupby("deadline_sweep_ms", as_index=False)[
        ["pct_local_fast", "pct_edge_accurate", "pct_hybrid", "pct_degraded_safe",
         "remote_attempt_rate", "remote_accept_rate", "deadline_miss_rate",
         "p95_latency_ms", "usable_confidence_proxy",
         "bandwidth_per_1000_frames_kb"]].mean()
    overall["profile"] = "ALL"
    return {"deadline_sweep.csv": per_run,
            "deadline_sweep_ci.csv": agg,
            "deadline_mode_distribution.csv": pd.concat([modes, overall], ignore_index=True)}


# ------------------------------------------------------------ thresholds
def threshold_sweep(cfg, bank, deadline_ms, n_boot) -> Dict[str, pd.DataFrame]:
    base = cfg["scheduler"]
    rows = []
    specs: List[Dict[str, Any]] = []
    for v in LOCAL_THRESHOLDS:
        if v < float(base["risk_degraded_threshold"]):
            specs.append({"swept": "risk_local_threshold", "value": v,
                          "override": {"risk_local_threshold": v}})
    for v in DEGRADED_THRESHOLDS:
        if v > float(base["risk_local_threshold"]):
            specs.append({"swept": "risk_degraded_threshold", "value": v,
                          "override": {"risk_degraded_threshold": v}})
    for v in CONFIDENCE_THRESHOLDS:
        specs.append({"swept": "local_confidence_threshold", "value": v,
                      "override": {"local_confidence_threshold": v}})

    for spec in specs:
        cand = with_overrides(cfg, **spec["override"])
        per_run, _ = run_cells(bank, [build_policy("dapper", cand)], cand, deadline_ms)
        per_run["swept_parameter"] = spec["swept"]
        per_run["swept_value"] = spec["value"]
        per_run["is_selected_value"] = np.isclose(
            float(spec["value"]), float(base[spec["swept"]]))
        rows.append(per_run)
        print(f"  [threshold] {spec['swept']}={spec['value']} done", flush=True)
    per_run = pd.concat(rows, ignore_index=True)
    agg = aggregate_over_seeds(per_run, ("swept_parameter", "swept_value", "profile", "policy"),
                               [m for m in SWEEP_METRICS if m in per_run.columns],
                               n_boot=n_boot)
    return {"threshold_sweep.csv": per_run, "threshold_sweep_ci.csv": agg}


# ------------------------------------------------- runtime estimation error
def estimation_error_sweep(cfg, bank, deadline_ms, n_boot) -> Dict[str, pd.DataFrame]:
    rows = []
    combos = ([("rtt", b, 0.0) for b in ESTIMATION_BIASES]
              + [("compute", 0.0, b) for b in ESTIMATION_BIASES]
              + [("both", b, b) for b in ESTIMATION_BIASES])
    for kind, rtt_bias, comp_bias in combos:
        sched = build_scheduler_from_config(cfg, rtt_estimate_bias=rtt_bias,
                                            compute_estimate_bias=comp_bias)
        policy = DapperPolicy(cfg, scheduler=sched, name="dapper")
        per_run, _ = run_cells(bank, [policy], cfg, deadline_ms)
        per_run["bias_target"] = kind
        per_run["rtt_estimate_bias"] = rtt_bias
        per_run["compute_estimate_bias"] = comp_bias
        per_run["bias"] = rtt_bias if kind != "compute" else comp_bias
        rows.append(per_run)
        print(f"  [estimation] {kind} bias rtt={rtt_bias:+.0%} compute={comp_bias:+.0%} done",
              flush=True)
    per_run = pd.concat(rows, ignore_index=True)
    agg = aggregate_over_seeds(per_run, ("bias_target", "bias", "profile", "policy"),
                               [m for m in SWEEP_METRICS if m in per_run.columns],
                               n_boot=n_boot)
    return {"runtime_estimation_error.csv": per_run,
            "runtime_estimation_error_ci.csv": agg}


# ----------------------------------------------- commit-margin diagnostic
def margin_sweep(cfg, bank, deadlines, n_boot) -> Dict[str, pd.DataFrame]:
    """
    Diagnostic for `remote_deadline_margin`, the gate on the committing
    `edge_accurate` mode.

    This parameter is **not identified** by the calibration protocol, because at
    D = 100 ms no frame passes the commit test under any margin, so every
    candidate scores identically on it. The deadline sweep shows that the value
    the search happened to pick is unsafe at intermediate deadlines, where
    committing first becomes possible. This sweep measures the effect directly.

    It is a sensitivity analysis, not a re-selection: it runs on evaluation
    seeds and its result is deliberately NOT fed back into `config.yaml`.
    """
    rows = []
    for deadline in deadlines:
        for margin in REMOTE_MARGINS:
            cand = with_overrides(cfg, remote_deadline_margin=margin)
            per_run, _ = run_cells(bank, [build_policy("dapper", cand)], cand, deadline)
            per_run["deadline_sweep_ms"] = deadline
            per_run["remote_deadline_margin"] = margin
            per_run["is_selected_value"] = np.isclose(
                margin, float(cfg["scheduler"]["remote_deadline_margin"]))
            rows.append(per_run)
        print(f"  [margin] D={deadline:g} ms done", flush=True)
    per_run = pd.concat(rows, ignore_index=True)
    agg = aggregate_over_seeds(
        per_run, ("deadline_sweep_ms", "remote_deadline_margin", "profile", "policy"),
        [m for m in SWEEP_METRICS if m in per_run.columns], n_boot=n_boot)
    return {"commit_margin_sweep.csv": per_run, "commit_margin_sweep_ci.csv": agg}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--study", nargs="*",
                   default=["deadline", "threshold", "estimation", "margin"],
                   choices=["deadline", "threshold", "estimation", "margin"])
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--no-baselines-in-deadline-sweep", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    assert_disjoint(cfg)
    seeds = evaluation_seeds(cfg, args.seeds)
    profiles = list(cfg["profiles"])
    print(f"[sensitivity] studies={args.study} seeds={seeds[0]}-{seeds[-1]} "
          f"frames={args.frames}")
    bank = build_bank(cfg, profiles, seeds, args.frames)

    out: Dict[str, pd.DataFrame] = {}
    if "deadline" in args.study:
        out.update(deadline_sweep(cfg, bank, seeds, args.n_boot,
                                  not args.no_baselines_in_deadline_sweep))
    if "threshold" in args.study:
        out.update(threshold_sweep(cfg, bank, args.deadline_ms, args.n_boot))
    if "estimation" in args.study:
        est_profiles = ["stable", "congested", "variable"]
        est_bank = {k: v for k, v in bank.items() if k[1] in est_profiles}
        out.update(estimation_error_sweep(cfg, est_bank, args.deadline_ms, args.n_boot))
    if "margin" in args.study:
        out.update(margin_sweep(cfg, bank, (100.0, 125.0, 150.0, 200.0), args.n_boot))

    for name, df in out.items():
        write_csv(df, os.path.join(args.out_dir, name))

    if "deadline" in args.study:
        md = out["deadline_mode_distribution.csv"]
        pd.set_option("display.width", 220)
        print("\n[sensitivity] DAPPER mode shares vs deadline (all profiles pooled):")
        print(md[md["profile"] == "ALL"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
