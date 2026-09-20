"""
Metric reduction for one run = one ``(seed, profile, policy, deadline)`` cell.

Metric definitions
------------------
``mean/p95/p99_latency_ms``
    percentiles of ``control_output_latency_ms``, the time until the first
    usable fresh output. This is the control-loop requirement.
``deadline_miss_rate``
    fraction of frames whose *control* output arrived after the deadline.
``mean_output_confidence``
    mean confidence proxy of the delivered output, regardless of timeliness.
``usable_confidence_proxy``
    mean over all frames of ``confidence if (on time and fresh) else 0``. A
    single number combining quality and timeliness: it is the expected
    confidence the control loop actually has in hand at the deadline. It is a
    *proxy* computed from the synthetic perception model, never a detector
    accuracy.
``reuse_rate``
    fraction of frames served by re-using an earlier, still-fresh output.
``stale_acceptance_rate``
    fraction of frames that consumed an output older than the freshness window.
    By construction this must be 0; it is reported so that the claim is checked
    rather than asserted.
``remote_accept_rate``
    accepted remote results / transmitted remote requests. The complement is
    bandwidth that bought nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .scheduler import ALL_MODES


@dataclass
class RunMetrics:
    policy: str
    profile: str
    seed: int
    deadline_ms: float
    frames: int

    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    mean_final_latency_ms: float
    deadline_miss_rate: float

    mean_output_confidence: float
    usable_confidence_proxy: float

    total_bandwidth_kb: float
    bandwidth_per_frame_kb: float
    bandwidth_per_1000_frames_kb: float

    fallback_rate: float
    reuse_rate: float
    stale_acceptance_rate: float

    remote_attempt_rate: float
    remote_success_rate: float
    remote_accept_rate: float
    remote_accepted_per_frame: float
    remote_rejected_deadline_rate: float
    remote_rejected_freshness_rate: float
    remote_failed_loss_rate: float
    remote_unreachable_rate: float

    pct_local_fast: float
    pct_edge_accurate: float
    pct_hybrid: float
    pct_degraded_safe: float

    mode_switches: int
    mode_switches_per_1000: float


def _p(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _rate(series: pd.Series) -> float:
    return float(series.astype(bool).mean()) if len(series) else 0.0


def compute_metrics(df: pd.DataFrame, mode_switches: Optional[int] = None) -> RunMetrics:
    """Reduce a per-frame log for one run into summary statistics."""
    n = len(df)
    if n == 0:
        raise ValueError("cannot compute metrics for an empty run")

    lat = df["control_output_latency_ms"].to_numpy(dtype=float)
    bw = df["bandwidth_kb"].to_numpy(dtype=float)
    attempts = df["remote_attempted"].astype(bool)
    n_attempt = int(attempts.sum())
    n_accept = int(df["remote_accepted"].astype(bool).sum())

    modes = df["selected_mode"].astype(str)
    shares = {m: float((modes == m).mean()) * 100.0 for m in ALL_MODES}

    if mode_switches is None:
        m = modes.to_numpy()
        mode_switches = int((m[1:] != m[:-1]).sum()) if n > 1 else 0

    return RunMetrics(
        policy=str(df["policy"].iloc[0]),
        profile=str(df["profile"].iloc[0]),
        seed=int(df["seed"].iloc[0]),
        deadline_ms=float(df["deadline_ms"].iloc[0]),
        frames=n,
        mean_latency_ms=float(lat.mean()),
        p95_latency_ms=_p(lat, 95),
        p99_latency_ms=_p(lat, 99),
        max_latency_ms=float(lat.max()),
        mean_final_latency_ms=float(df["final_output_latency_ms"].mean()),
        deadline_miss_rate=float((~df["deadline_met"].astype(bool)).mean()),
        mean_output_confidence=float(df["output_confidence"].mean()),
        usable_confidence_proxy=float(df["usable_confidence_proxy"].mean()),
        total_bandwidth_kb=float(bw.sum()),
        bandwidth_per_frame_kb=float(bw.mean()),
        bandwidth_per_1000_frames_kb=float(bw.mean() * 1000.0),
        fallback_rate=_rate(df["fallback_used"]),
        reuse_rate=_rate(df["output_reused"]),
        stale_acceptance_rate=_rate(df["stale_output_accepted"]),
        remote_attempt_rate=n_attempt / n,
        remote_success_rate=(float(df["remote_succeeded"].astype(bool).sum()) / n_attempt
                             if n_attempt else 0.0),
        remote_accept_rate=(n_accept / n_attempt) if n_attempt else 0.0,
        remote_accepted_per_frame=n_accept / n,
        remote_rejected_deadline_rate=(float(df["remote_rejected_deadline"].astype(bool).sum())
                                       / n_attempt) if n_attempt else 0.0,
        remote_rejected_freshness_rate=(float(df["remote_rejected_freshness"].astype(bool).sum())
                                        / n_attempt) if n_attempt else 0.0,
        remote_failed_loss_rate=(float(df["remote_failed_loss"].astype(bool).sum())
                                 / n_attempt) if n_attempt else 0.0,
        remote_unreachable_rate=_rate(df["remote_failed_unavailable"]),
        pct_local_fast=shares["local_fast"],
        pct_edge_accurate=shares["edge_accurate"],
        pct_hybrid=shares["hybrid"],
        pct_degraded_safe=shares["degraded_safe"],
        mode_switches=int(mode_switches),
        mode_switches_per_1000=float(mode_switches) * 1000.0 / n,
    )


def metrics_frame(runs: Iterable[RunMetrics]) -> pd.DataFrame:
    """Tidy DataFrame, one row per run."""
    return pd.DataFrame([asdict(r) for r in runs])


#: Metrics for which a *lower* value is better. Used by report generators so
#: that a direction is never hard-coded in prose.
LOWER_IS_BETTER = {
    "mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "max_latency_ms",
    "deadline_miss_rate", "total_bandwidth_kb", "bandwidth_per_frame_kb",
    "bandwidth_per_1000_frames_kb", "stale_acceptance_rate",
    "remote_rejected_deadline_rate", "remote_rejected_freshness_rate",
    "remote_failed_loss_rate", "mode_switches", "mode_switches_per_1000",
}
