"""
Replay every policy over the **real** detector outputs.

The perception side of the benchmark is replaced by measurement: for each of the
5,000 COCO val2017 images the scheduler sees the confidence the lightweight
local detector actually produced, the compute times both detectors actually
took, and the per-image detection quality both detectors actually achieved. The
network side remains the seeded synthetic profile, which is stated wherever
these numbers are reported.

Deployment mapping (stated so it cannot be misread)
---------------------------------------------------
* **local** = YOLO11n, timed on the workstation **CPU** - the configuration in
  which a lightweight detector would plausibly run on a robot;
* **edge / cloud** = YOLO11m, timed on the workstation **GPU** - the
  configuration in which a larger detector would run on a server.
  The cloud path uses the same large detector and differs only in its network
  distance, because no third detector was measured.

This is a secondary validation on one workstation. It is not a physical robot,
not Jetson-class timing, and not a measured wireless link.

Delivered-quality metric
------------------------
``delivered_recall`` is the fraction of ground-truth objects in the *current*
image that the output the control loop holds at the deadline actually detected:

* freshly computed and on time -> that model's per-image recall;
* deadline missed -> 0;
* reused from an earlier frame -> 0.

Consecutive COCO validation images are unrelated, so a reused output contains no
detections of the current image. On a real video stream a reused output would
retain some value, which makes this a **lower bound** for degraded-safe reuse.
``reuse_rate`` is reported beside it so the reader can see how much of the zero
comes from reuse rather than from missed deadlines.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dapper.config import load_config  # noqa: E402
from dapper.executor import execute_run  # noqa: E402
from dapper.metrics import compute_metrics  # noqa: E402
from dapper.policies import build_policy  # noqa: E402
from dapper.scenario import ScenarioTrace, generate_scenarios  # noqa: E402
from dapper.stats import aggregate_over_seeds  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "real_detector")

REPLAY_POLICIES = ("local_only", "edge_only", "cloud_only", "dapper",
                   "deadline_greedy", "confidence_deadline", "oracle_feasible")


def build_real_trace(cfg: Dict[str, Any], profile: str, seed: int,
                     per_image: pd.DataFrame) -> ScenarioTrace:
    """Synthetic network + measured perception potentials, for one seed."""
    n = len(per_image)
    synth = generate_scenarios(profile, cfg, seed, n)
    arrays = {
        "rtt_ms": np.asarray(synth.rtt_ms).copy(),
        "packet_loss": np.asarray(synth.packet_loss).copy(),
        "edge_load": np.asarray(synth.edge_load).copy(),
        "edge_available": np.asarray(synth.edge_available).copy(),
        "local_latency_ms": per_image["local_latency_ms"].to_numpy(dtype=float),
        "local_confidence": per_image["local_confidence"].to_numpy(dtype=float),
        "local_detections": per_image["local_detections"].to_numpy(dtype=np.int32),
        "edge_compute_ms": per_image["remote_compute_ms"].to_numpy(dtype=float),
        "edge_confidence": per_image["remote_confidence"].to_numpy(dtype=float),
        "edge_detections": per_image["remote_detections"].to_numpy(dtype=np.int32),
        "edge_loss_draw": np.asarray(synth.edge_loss_draw).copy(),
        "cloud_compute_ms": per_image["remote_compute_ms"].to_numpy(dtype=float),
        "cloud_confidence": per_image["remote_confidence"].to_numpy(dtype=float),
        "cloud_detections": per_image["remote_detections"].to_numpy(dtype=np.int32),
        "cloud_loss_draw": np.asarray(synth.cloud_loss_draw).copy(),
    }
    return ScenarioTrace(seed=seed, profile=profile, arrays=arrays,
                         frame_period_ms=float(cfg["execution"]["frame_period_ms"]),
                         source="real_detector")


def prepare_per_image(quality: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the measured per-image inputs.

    Local latency uses the CPU measurement where one exists; images without a
    CPU measurement are dropped rather than imputed, so every replayed frame is
    backed by a real timing.
    """
    df = quality.copy()
    if "local_cpu_infer_ms" in df.columns and df["local_cpu_infer_ms"].notna().any():
        df = df[df["local_cpu_infer_ms"].notna()].copy()
        df["local_latency_ms"] = df["local_cpu_infer_ms"].astype(float)
        local_device = "cpu"
    else:
        df["local_latency_ms"] = df["local_infer_ms"].astype(float)
        local_device = "gpu"
    df["local_confidence"] = df["local_max_confidence"].astype(float)
    df["local_detections"] = df["local_detections"].astype(int)
    df["remote_compute_ms"] = df["remote_infer_ms"].astype(float)
    df["remote_confidence"] = df["remote_max_confidence"].astype(float)
    df["remote_detections"] = df["remote_detections"].astype(int)
    df.attrs["local_device"] = local_device
    return df.reset_index(drop=True)


def delivered_quality(log: pd.DataFrame, per_image: pd.DataFrame) -> Dict[str, float]:
    """Per-frame delivered detection quality, as defined in the module docstring."""
    n = len(log)
    src = log["output_source"].to_numpy()
    met = log["deadline_met"].to_numpy(dtype=bool)
    reused = log["output_reused"].to_numpy(dtype=bool)
    refreshed = log["remote_refresh_accepted"].to_numpy(dtype=bool)

    local_r = per_image["local_recall"].to_numpy(dtype=float)
    remote_r = per_image["remote_recall"].to_numpy(dtype=float)
    local_s = per_image["local_safety_recall"].to_numpy(dtype=float)
    remote_s = per_image["remote_safety_recall"].to_numpy(dtype=float)

    # Which model's output the control loop holds at the deadline.
    use_remote = (np.isin(src, ["edge", "cloud"])) | refreshed
    recall = np.where(use_remote, remote_r, local_r)
    safety = np.where(use_remote, remote_s, local_s)
    usable = met & ~reused
    recall = np.where(usable, recall, 0.0)
    safety = np.where(usable, safety, 0.0)

    has_gt = per_image["n_gt"].to_numpy() > 0
    has_sgt = per_image["n_gt_safety"].to_numpy() > 0
    return {
        "delivered_recall": float(np.nanmean(recall[has_gt])) if has_gt.any() else float("nan"),
        "delivered_safety_recall": (float(np.nanmean(safety[has_sgt]))
                                    if has_sgt.any() else float("nan")),
        "remote_output_share": float(use_remote.mean()),
        "usable_output_rate": float(usable.mean()),
        "frames_scored": int(has_gt.sum()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality", default=os.path.join(OUT_DIR, "per_image_quality.parquet"))
    p.add_argument("--config", default=None)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--deadline-sweep", nargs="*", type=float,
                   default=[50.0, 75.0, 100.0, 150.0, 250.0])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=100)
    p.add_argument("--profiles", nargs="*", default=None)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    quality = pd.read_parquet(args.quality)
    per_image = prepare_per_image(quality)
    profiles = args.profiles or list(cfg["profiles"])
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    print(f"[replay] {len(per_image)} real frames, local timed on "
          f"{per_image.attrs['local_device'].upper()}, profiles={profiles}, "
          f"seeds={seeds[0]}-{seeds[-1]}")

    rows: List[Dict] = []
    deadlines = sorted(set(args.deadline_sweep) | {args.deadline_ms})
    for deadline in deadlines:
        for profile in profiles:
            for seed in seeds:
                trace = build_real_trace(cfg, profile, seed, per_image)
                for name in REPLAY_POLICIES:
                    log = execute_run(trace, build_policy(name, cfg), cfg, deadline)
                    m = compute_metrics(log, log.attrs.get("mode_switches"))
                    row = {k: v for k, v in m.__dict__.items()}
                    row.update(delivered_quality(log, per_image))
                    row["deadline_sweep_ms"] = deadline
                    rows.append(row)
        print(f"  [replay] D={deadline:g} ms done", flush=True)

    per_run = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    per_run.to_csv(os.path.join(args.out_dir, "dapper_replay.csv"), index=False)

    metrics = ["delivered_recall", "delivered_safety_recall", "deadline_miss_rate",
               "p95_latency_ms", "mean_latency_ms", "usable_confidence_proxy",
               "bandwidth_per_1000_frames_kb", "remote_output_share",
               "usable_output_rate", "reuse_rate", "remote_attempt_rate",
               "remote_accept_rate", "pct_local_fast", "pct_edge_accurate",
               "pct_hybrid", "pct_degraded_safe"]
    ci = aggregate_over_seeds(per_run, ("deadline_sweep_ms", "profile", "policy"),
                              metrics, n_boot=5000)
    ci.to_csv(os.path.join(args.out_dir, "dapper_replay_ci.csv"), index=False)

    main_d = per_run[per_run["deadline_sweep_ms"] == args.deadline_ms]
    summary = main_d.groupby("policy", as_index=False)[
        ["delivered_recall", "delivered_safety_recall", "deadline_miss_rate",
         "p95_latency_ms", "bandwidth_per_1000_frames_kb", "remote_output_share",
         "reuse_rate"]].mean()
    summary.to_csv(os.path.join(args.out_dir, "dapper_replay_summary.csv"), index=False)

    pd.set_option("display.width", 240)
    print(f"\n[replay] pooled over profiles and seeds at D = {args.deadline_ms:g} ms:")
    print(summary.to_string(index=False))
    print("\n[replay] by profile (DAPPER vs local_only vs edge_only):")
    sub = main_d[main_d["policy"].isin(["local_only", "edge_only", "dapper"])]
    print(sub.groupby(["profile", "policy"], as_index=False)[
        ["delivered_recall", "delivered_safety_recall", "deadline_miss_rate",
         "bandwidth_per_1000_frames_kb"]].mean().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
