"""
DAPPER benchmark runner.

Usage:
    python benchmark.py --frames 1000 --deadline-ms 100 \\
        --profile variable --mode dapper --output results/run_variable.csv

The script loops over frames, samples network conditions from a profile,
asks the scheduler (or a fixed baseline) for a mode, simulates perception,
and logs everything to a CSV.

Four execution policies are supported:

    local_only   always uses local_fast
    edge_only    always uses edge_accurate (talks to the simulated edge backend)
    cloud_only   always uses edge_accurate but with the cloud's worse RTT
    dapper       uses the DAPPER scheduler per frame

There is also a helper subcommand `run-all` that sweeps every profile and
every mode and writes results/summary.csv plus the PNG plots.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from monitor import build_monitor, NetworkSample
from perception import build_perception, PerceptionResult
from scheduler import (
    DapperScheduler,
    SchedulerInputs,
    SchedulerDecision,
    build_scheduler_from_config,
    MODE_LOCAL_FAST,
    MODE_EDGE_ACCURATE,
    MODE_HYBRID,
    MODE_DEGRADED_SAFE,
)
from metrics import compute_metrics, summarize_runs
from plots import generate_all_plots


# ----------------------------------------------------------------- config
def load_config(path: str) -> dict:
    """Load and lightly validate config.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for required in ("local", "edge", "cloud", "network_profiles", "scheduler"):
        if required not in cfg:
            raise ValueError(f"config.yaml missing required section: {required}")
    return cfg


# ----------------------------------------------------------------- run modes
FIXED_BASELINES = {"local_only", "edge_only", "cloud_only"}
ALL_RUN_MODES = ("local_only", "edge_only", "cloud_only", "dapper")


def _choose_fixed_mode(run_mode: str) -> str:
    """Map a fixed baseline name to its perception mode."""
    if run_mode == "local_only":
        return MODE_LOCAL_FAST
    if run_mode in ("edge_only", "cloud_only"):
        return MODE_EDGE_ACCURATE
    raise ValueError(f"Unknown fixed baseline: {run_mode}")


def _fake_scheduler_decision(mode: str, expected_latency_ms: float) -> SchedulerDecision:
    """Build a SchedulerDecision-like object for fixed baselines so logging is uniform."""
    return SchedulerDecision(
        selected_mode=mode,
        deadline_risk_score=0.0,
        reason=f"fixed_baseline:{mode}",
        expected_latency_ms=expected_latency_ms,
        fallback_needed=False,
    )


def _execute_mode(
    mode: str,
    frame_id: int,
    net: NetworkSample,
    deadline_ms: float,
    cfg: dict,
    perception,
    last_valid_age_ms: float,
    hybrid_window_ms: float,
    last_valid_freshness_ms: float,
    use_cloud_latency: bool = False,
) -> PerceptionResult:
    """
    Dispatch the chosen mode to the perception backend.

    `use_cloud_latency` is set by the cloud_only baseline so that we go through
    the same edge_infer path but with the larger cloud RTT/compute envelope.
    """
    backend_section = "cloud" if use_cloud_latency else "edge"

    if mode == MODE_LOCAL_FAST:
        return perception.local_infer(frame_id)

    if mode == MODE_EDGE_ACCURATE:
        # Cloud_only adds the cloud base RTT on top of the network sample's RTT
        # so that cloud is strictly worse than edge under identical profiles.
        extra_rtt = float(cfg[backend_section].get("base_rtt_ms", 0.0)) if use_cloud_latency else 0.0
        result = perception.edge_infer(
            frame_id=frame_id,
            rtt_ms=net.rtt_ms + extra_rtt,
            packet_loss=net.packet_loss,
            edge_load=net.edge_load,
            edge_available=net.edge_available,
            backend=backend_section,
        )
        if result is None:
            # The "send" failed: fall back to local so the frame still has output.
            local = perception.local_infer(frame_id)
            local.source = f"{backend_section}_failed_to_local"
            return local
        return result

    if mode == MODE_HYBRID:
        return perception.hybrid_infer(
            frame_id=frame_id,
            rtt_ms=net.rtt_ms,
            packet_loss=net.packet_loss,
            edge_load=net.edge_load,
            edge_available=net.edge_available,
            deadline_ms=deadline_ms,
            hybrid_window_ms=hybrid_window_ms,
        )

    if mode == MODE_DEGRADED_SAFE:
        return perception.degraded_safe(
            frame_id=frame_id,
            last_valid_age_ms=last_valid_age_ms,
            last_valid_freshness_ms=last_valid_freshness_ms,
        )

    raise ValueError(f"Unknown perception mode: {mode}")


# ----------------------------------------------------------------- main loop
def run_benchmark(
    run_mode: str,
    profile: str,
    frames: int,
    deadline_ms: float,
    cfg: dict,
    seed: int = 42,
    video_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run one (run_mode, profile) combination and return per-frame rows.

    The CSV schema is documented in README.md and is identical across modes,
    which makes downstream metric and plot code mode-agnostic.
    """
    monitor = build_monitor(profile, cfg, seed=seed)
    scheduler = build_scheduler_from_config(cfg)
    perception = build_perception(cfg, video_path=video_path, seed=seed)

    hybrid_window_ms = float(cfg["scheduler"]["hybrid_freshness_window_ms"])
    last_valid_freshness_ms = float(cfg["scheduler"]["last_valid_freshness_ms"])

    rows: List[Dict] = []
    last_valid_time = -1e9  # Anything; the first frame will set a real value.

    # We model "wall-clock" using a virtual frame-time clock so the benchmark
    # is deterministic and fast. One frame ~= 33 ms (30 FPS) by convention.
    virtual_clock_ms = 0.0
    frame_period_ms = 1000.0 / 30.0

    for frame_id in range(frames):
        # Sample network conditions.
        net = monitor.sample()

        # Frame age is the time since the camera captured this frame. We assume
        # each frame is presented instantly (frame_age starts at 0).
        frame_age_ms = 0.0
        last_valid_age_ms = max(0.0, virtual_clock_ms - last_valid_time)

        # Decide which mode to run.
        if run_mode == "dapper":
            inputs = SchedulerInputs(
                rtt_ms=net.rtt_ms,
                packet_loss=net.packet_loss,
                edge_load=net.edge_load,
                deadline_ms=deadline_ms,
                frame_age_ms=frame_age_ms,
                last_valid_age_ms=last_valid_age_ms,
                local_confidence=0.5 * (cfg["local"]["confidence_min"]
                                        + cfg["local"]["confidence_max"]),
                edge_available=net.edge_available,
            )
            decision = scheduler.decide(inputs)
        else:
            mode = _choose_fixed_mode(run_mode)
            decision = _fake_scheduler_decision(mode, expected_latency_ms=net.rtt_ms + 65.0)

        use_cloud_latency = (run_mode == "cloud_only")

        # Execute. If video mode is on, we also read a frame to exercise OpenCV.
        if hasattr(perception, "read_frame"):
            perception.read_frame(frame_id)

        result = _execute_mode(
            mode=decision.selected_mode,
            frame_id=frame_id,
            net=net,
            deadline_ms=deadline_ms,
            cfg=cfg,
            perception=perception,
            last_valid_age_ms=last_valid_age_ms,
            hybrid_window_ms=hybrid_window_ms,
            last_valid_freshness_ms=last_valid_freshness_ms,
            use_cloud_latency=use_cloud_latency,
        )

        deadline_met = result.latency_ms <= deadline_ms
        fallback_used = (
            decision.fallback_needed
            or result.stale
            or "failed_to_local" in str(result.source)
        )

        rows.append({
            "frame_id": frame_id,
            "timestamp": virtual_clock_ms,
            "run_mode": run_mode,
            "profile": profile,
            "selected_mode": decision.selected_mode,
            "rtt_ms": net.rtt_ms,
            "packet_loss": net.packet_loss,
            "edge_load": net.edge_load,
            "edge_available": net.edge_available,
            "deadline_ms": deadline_ms,
            "latency_ms": result.latency_ms,
            "deadline_met": bool(deadline_met),
            "confidence": result.confidence,
            "stale_output": bool(result.stale),
            "fallback_used": bool(fallback_used),
            "bandwidth_kb": result.bandwidth_kb,
            "risk_score": decision.deadline_risk_score,
            "decision_reason": decision.reason,
            "source": result.source,
            "detections": result.detections,
        })

        if not result.stale:
            last_valid_time = virtual_clock_ms
        virtual_clock_ms += frame_period_ms

    return pd.DataFrame(rows)


# ----------------------------------------------------------------- CLI
def cmd_single(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    df = run_benchmark(
        run_mode=args.mode,
        profile=args.profile,
        frames=args.frames,
        deadline_ms=args.deadline_ms,
        cfg=cfg,
        seed=args.seed,
        video_path=args.video,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    metrics = compute_metrics(df, mode=args.mode, profile=args.profile)
    print(f"Wrote {len(df)} frames to {args.output}")
    print(
        f"  mean={metrics.mean_latency_ms:.1f}ms "
        f"p95={metrics.p95_latency_ms:.1f}ms "
        f"miss={metrics.deadline_miss_rate*100:.1f}% "
        f"conf={metrics.mean_confidence:.2f} "
        f"bw={metrics.total_bandwidth_kb:.0f}KB"
    )
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """
    Sweep every profile x every run_mode, write summary + per-frame CSVs,
    and generate PNG plots in results/.
    """
    cfg = load_config(args.config)
    profiles = cfg.get("profiles") or list(cfg["network_profiles"].keys())
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    all_frames: List[pd.DataFrame] = []
    summaries = []

    for profile in profiles:
        for mode in ALL_RUN_MODES:
            print(f"[run-all] profile={profile} mode={mode} ...", flush=True)
            df = run_benchmark(
                run_mode=mode,
                profile=profile,
                frames=args.frames,
                deadline_ms=args.deadline_ms,
                cfg=cfg,
                seed=args.seed,
            )
            all_frames.append(df)
            summaries.append(compute_metrics(df, mode=mode, profile=profile))

    all_runs = pd.concat(all_frames, ignore_index=True)
    summary = summarize_runs(summaries)

    all_runs.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Per-profile plots and one global plot.
    for profile in profiles:
        sub_frames = all_runs[all_runs["profile"] == profile]
        sub_summary = summary[summary["profile"] == profile]
        generate_all_plots(sub_frames, sub_summary, out_dir, profile_tag=profile)

    print(f"\nWrote summary to {os.path.join(out_dir, 'summary.csv')}")
    print(summary.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DAPPER benchmark runner")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    sub = p.add_subparsers(dest="cmd")

    # Single-run is also the default mode (no subcommand).
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--profile", default="variable",
                   choices=["stable", "congested", "lossy", "variable", "outage"])
    p.add_argument("--mode", default="dapper", choices=list(ALL_RUN_MODES))
    p.add_argument("--output", default="results/run.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video", default=None,
                   help="Optional path to a video file for OpenCV mode")

    # run-all subcommand
    run_all = sub.add_parser(
        "run-all", help="Run every (profile, mode) combination and produce plots"
    )
    run_all.add_argument("--frames", type=int, default=1000)
    run_all.add_argument("--deadline-ms", type=float, default=100.0)
    run_all.add_argument("--seed", type=int, default=42)
    run_all.add_argument("--out-dir", default="results")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run-all":
        return cmd_run_all(args)
    return cmd_single(args)


if __name__ == "__main__":
    sys.exit(main())
