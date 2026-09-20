"""
Phase 11 - scheduling and switching overhead.

Two distinct quantities are measured and never mixed:

1. **Scheduler decision overhead** - the wall time of a single
   ``DapperScheduler.decide()`` call on the machine recorded in
   ``artifacts/ENVIRONMENT.md``, measured with ``time.perf_counter_ns()`` over
   a large number of calls after warm-up. This is *software* overhead on a
   desktop CPython interpreter. It is **not** a measurement of robot hardware,
   and no report generated from it may claim otherwise.
2. **Mode-switch rate** - how often the selected mode changes between
   consecutive frames, per 1000 frames, from the final evaluation runs. This is
   a property of the policy, not of the machine.

A whole-benchmark wall-clock comparison of DAPPER against the fixed baselines is
also recorded, kept explicitly separate from the *simulated* application latency
that the rest of the paper reports.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    RESULTS, build_bank, environment_lines, evaluation_seeds, load_config,
    run_cells, write_csv,
)
from dapper.executor import execute_run  # noqa: E402
from dapper.policies import build_policy  # noqa: E402
from dapper.scenario import generate_scenarios  # noqa: E402
from dapper.scheduler import SchedulerInputs, build_scheduler_from_config  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "overhead")


def measure_decide(cfg, n_calls: int, warmup: int, deadline_ms: float,
                   seed: int = 0) -> pd.DataFrame:
    """Time ``decide()`` on a realistic stream of per-frame observations."""
    sched = build_scheduler_from_config(cfg)
    trace = generate_scenarios("variable", cfg, seed, max(n_calls, 1000))
    n = trace.frames
    inputs = [
        SchedulerInputs(
            rtt_ms=float(trace.rtt_ms[i % n]),
            packet_loss=float(trace.packet_loss[i % n]),
            edge_load=float(trace.edge_load[i % n]),
            deadline_ms=deadline_ms,
            frame_age_ms=0.0,
            last_valid_age_ms=float(33.3 * (i % 7)),
            local_confidence=float(trace.local_confidence[i % n]),
            edge_available=bool(trace.edge_available[i % n]),
        )
        for i in range(min(n_calls, 200000))
    ]
    for x in inputs[:warmup]:
        sched.decide(x)

    samples = np.empty(len(inputs), dtype=float)
    for i, x in enumerate(inputs):
        t0 = time.perf_counter_ns()
        sched.decide(x)
        samples[i] = time.perf_counter_ns() - t0
    us = samples / 1000.0
    return pd.DataFrame({"call_index": np.arange(len(us)), "decide_us": us})


def summarise(us: np.ndarray, label: str, n_calls: int) -> Dict[str, float]:
    return {
        "measurement": label,
        "n_calls": int(n_calls),
        "mean_us": float(us.mean()),
        "median_us": float(np.median(us)),
        "std_us": float(us.std(ddof=1)),
        "p95_us": float(np.percentile(us, 95)),
        "p99_us": float(np.percentile(us, 99)),
        "min_us": float(us.min()),
        "max_us": float(us.max()),
        "throughput_decisions_per_s": float(1e6 / us.mean()),
    }


def wallclock_comparison(cfg, seeds, frames, deadline_ms, repeats: int = 3) -> pd.DataFrame:
    """End-to-end benchmark wall time per policy (software cost, not app latency)."""
    bank = build_bank(cfg, ["variable"], seeds[:3], frames)
    rows = []
    for name in ("local_only", "edge_only", "dapper"):
        policy = build_policy(name, cfg)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            for trace in bank.values():
                execute_run(trace, policy, cfg, deadline_ms)
            times.append((time.perf_counter_ns() - t0) / 1e6)
        total_frames = frames * len(bank)
        rows.append({
            "policy": name,
            "repeats": repeats,
            "frames_per_repeat": total_frames,
            "median_wall_ms": float(statistics.median(times)),
            "wall_us_per_frame": float(statistics.median(times) * 1000.0 / total_frames),
        })
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--calls", type=int, default=200000)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--per-run-csv", default=os.path.join(RESULTS, "final",
                                                         "multi_seed_per_run.csv"))
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    seeds = evaluation_seeds(cfg, args.seeds)

    print(f"[overhead] timing {args.calls} scheduler decisions "
          f"(warm-up {args.warmup})")
    raw = measure_decide(cfg, args.calls, args.warmup, args.deadline_ms)
    write_csv(raw.iloc[::max(1, len(raw) // 5000)],
              os.path.join(args.out_dir, "scheduler_overhead.csv"))

    us = raw["decide_us"].to_numpy()
    rows = [summarise(us, "DapperScheduler.decide", len(us))]

    switch_rows: List[Dict] = []
    if os.path.exists(args.per_run_csv):
        per_run = pd.read_csv(args.per_run_csv)
        sw = per_run.groupby(["policy", "profile"], as_index=False)[
            "mode_switches_per_1000"].agg(["mean", "std"]).reset_index()
        sw.columns = ["policy", "profile", "mode_switches_per_1000_mean",
                      "mode_switches_per_1000_std"]
        switch_rows = sw.to_dict("records")
        write_csv(sw, os.path.join(args.out_dir, "mode_switches.csv"))
    else:
        print(f"  [overhead] {args.per_run_csv} not found; "
              "run final_eval.py first for mode-switch rates")

    wall = wallclock_comparison(cfg, seeds, args.frames, args.deadline_ms)
    write_csv(wall, os.path.join(args.out_dir, "benchmark_wallclock.csv"))

    summary = pd.DataFrame(rows)
    summary["environment"] = "; ".join(environment_lines()[:5])
    summary["note"] = ("software overhead of the scheduler on a desktop CPython "
                       "interpreter; NOT robot hardware overhead and NOT the "
                       "simulated application latency reported elsewhere")
    write_csv(summary, os.path.join(args.out_dir, "scheduler_overhead_summary.csv"))

    pd.set_option("display.width", 200)
    print("\n[overhead] scheduler decision cost (microseconds):")
    print(summary[["measurement", "n_calls", "mean_us", "median_us", "p95_us",
                   "p99_us", "throughput_decisions_per_s"]].to_string(index=False))
    print("\n[overhead] benchmark wall clock:")
    print(wall.to_string(index=False))
    if switch_rows:
        print("\n[overhead] mode switches per 1000 frames (DAPPER):")
        print(pd.DataFrame([r for r in switch_rows if r["policy"] == "dapper"]
                           ).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
