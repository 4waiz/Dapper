"""
Phases 4-6 - repeated multi-seed final evaluation with confidence intervals.

* 30 independent evaluation seeds (100-129), disjoint from the calibration
  seeds, so no parameter reported here was selected on this data.
* 5 network profiles x 8 policies x 1000 frames per seed.
* Every policy replays the **identical** pre-generated scenario trace for a
  given (seed, profile), so all comparisons are paired at the frame level.
* Dispersion is computed across seeds, never across frames: frames within a run
  are autocorrelated (AR(1) RTT, 5-25 frame outages, carried-over last-valid
  state), so treating them as independent would manufacture precision.

Outputs
-------
``results/final/multi_seed_per_run.csv``   one row per (seed, profile, policy)
``results/final/multi_seed_aggregate.csv`` seed-averaged, one row per group
``results/final/multi_seed_ci.csv``        long format with bootstrap 95% CIs
``results/final/paired_comparisons.csv``   per-seed paired DAPPER-vs-X deltas
``results/final/mode_distribution.csv``    DAPPER mode shares per profile
``results/final/scenario_fingerprints.csv``  replay integrity check
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

from _common import (  # noqa: E402
    RESULTS, assert_disjoint, build_bank, evaluation_seeds, load_config,
    run_cells, write_csv,
)
from dapper.policies import ALL_POLICY_NAMES, build_policy  # noqa: E402
from dapper.stats import aggregate_over_seeds, paired_difference, wide_aggregate  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "final")

HEADLINE_METRICS = (
    "mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "deadline_miss_rate",
    "usable_confidence_proxy", "mean_output_confidence",
    "bandwidth_per_1000_frames_kb", "bandwidth_per_frame_kb",
    "fallback_rate", "reuse_rate", "stale_acceptance_rate",
    "remote_attempt_rate", "remote_accept_rate", "remote_accepted_per_frame",
    "remote_rejected_deadline_rate", "remote_rejected_freshness_rate",
    "remote_failed_loss_rate",
    "mode_switches_per_1000",
    "pct_local_fast", "pct_edge_accurate", "pct_hybrid", "pct_degraded_safe",
)

COMPARATORS = ("local_only", "edge_only", "cloud_only",
               "deadline_greedy", "confidence_deadline", "rtt_threshold",
               "oracle_feasible")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--seeds", type=int, default=None,
                   help="number of evaluation seeds (default: config value)")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--save-frames", action="store_true",
                   help="also write the full per-frame log (large)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    assert_disjoint(cfg)
    seeds = evaluation_seeds(cfg, args.seeds)
    profiles = list(cfg["profiles"])
    policies = [build_policy(n, cfg) for n in ALL_POLICY_NAMES]

    print(f"[final] seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) profiles={profiles} "
          f"policies={[p.name for p in policies]} frames={args.frames} D={args.deadline_ms}")
    bank = build_bank(cfg, profiles, seeds, args.frames)

    fps = pd.DataFrame([{"seed": s, "profile": pr, "frames": t.frames,
                         "fingerprint": t.fingerprint()}
                        for (s, pr), t in bank.items()])
    write_csv(fps, os.path.join(args.out_dir, "scenario_fingerprints.csv"))

    per_run, per_frame = run_cells(bank, policies, cfg, args.deadline_ms,
                                   keep_frames=args.save_frames)
    write_csv(per_run, os.path.join(args.out_dir, "multi_seed_per_run.csv"))
    if per_frame is not None:
        write_csv(per_frame, os.path.join(args.out_dir, "multi_seed_per_frame.csv.gz"))

    metrics = [m for m in HEADLINE_METRICS if m in per_run.columns]
    long = aggregate_over_seeds(per_run, ("profile", "policy"), metrics, n_boot=args.n_boot)
    write_csv(long, os.path.join(args.out_dir, "multi_seed_ci.csv"))
    write_csv(wide_aggregate(long, ("profile", "policy")),
              os.path.join(args.out_dir, "multi_seed_aggregate.csv"))

    rows: List[Dict] = []
    for profile in profiles + [None]:
        for other in COMPARATORS:
            for metric in ("deadline_miss_rate", "p95_latency_ms",
                           "usable_confidence_proxy", "bandwidth_per_1000_frames_kb",
                           "mean_latency_ms"):
                if profile is None:
                    sub = per_run.groupby(["policy", "seed"], as_index=False)[metric].mean()
                    sub["profile"] = "ALL"
                    d = paired_difference(sub, metric, "dapper", other,
                                          profile="ALL", n_boot=args.n_boot)
                else:
                    d = paired_difference(per_run, metric, "dapper", other,
                                          profile=profile, n_boot=args.n_boot)
                rows.append(d)
    write_csv(pd.DataFrame(rows), os.path.join(args.out_dir, "paired_comparisons.csv"))

    dap = per_run[per_run["policy"] == "dapper"]
    modes = dap.groupby("profile", as_index=False)[
        ["pct_local_fast", "pct_edge_accurate", "pct_hybrid", "pct_degraded_safe",
         "remote_attempt_rate", "remote_accept_rate", "reuse_rate",
         "mode_switches_per_1000"]].mean()
    write_csv(modes, os.path.join(args.out_dir, "mode_distribution.csv"))

    print("\n[final] headline (seed means, D = %g ms):" % args.deadline_ms)
    show = per_run.groupby(["profile", "policy"], as_index=False)[
        ["p95_latency_ms", "deadline_miss_rate", "usable_confidence_proxy",
         "bandwidth_per_1000_frames_kb"]].mean()
    pd.set_option("display.width", 200)
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
