"""
Metric computation for DAPPER runs.

Given a per-frame CSV produced by benchmark.py, compute the headline numbers
we want to put in a paper:

    mean / p95 / p99 latency
    deadline miss rate
    mean confidence
    stale output rate
    fallback rate
    mode switch count
    total bandwidth (KB)
    bandwidth per frame (KB)
    reliability score (composite)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class RunMetrics:
    mode: str
    profile: str
    frames: int
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    deadline_miss_rate: float
    mean_confidence: float
    stale_output_rate: float
    fallback_rate: float
    mode_switch_count: int
    total_bandwidth_kb: float
    bandwidth_per_frame_kb: float
    reliability_score: float


def _percentile(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return float("nan")
    return float(np.percentile(series.to_numpy(dtype=float), q))


def compute_metrics(df: pd.DataFrame, mode: str, profile: str) -> RunMetrics:
    """
    Reduce a per-frame DataFrame into summary statistics.

    The reliability score is a simple composite:

        reliability = 1.0 - deadline_miss_rate
                    multiplied by mean_confidence
                    penalised by stale_output_rate

    It is meant for relative comparison only, not as an absolute measure.
    """
    n = len(df)
    if n == 0:
        return RunMetrics(
            mode=mode, profile=profile, frames=0,
            mean_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0,
            deadline_miss_rate=0.0, mean_confidence=0.0,
            stale_output_rate=0.0, fallback_rate=0.0,
            mode_switch_count=0,
            total_bandwidth_kb=0.0, bandwidth_per_frame_kb=0.0,
            reliability_score=0.0,
        )

    latency = df["latency_ms"].astype(float)
    confidence = df["confidence"].astype(float)
    bandwidth = df["bandwidth_kb"].astype(float)

    deadline_met = df["deadline_met"].astype(bool)
    stale = df["stale_output"].astype(bool)
    fallback = df["fallback_used"].astype(bool)

    miss_rate = float((~deadline_met).mean())
    stale_rate = float(stale.mean())
    fallback_rate = float(fallback.mean())
    mean_conf = float(confidence.mean())

    # Count transitions between distinct selected_mode values.
    modes = df["selected_mode"].astype(str).tolist()
    switches = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])

    reliability = (1.0 - miss_rate) * mean_conf * (1.0 - 0.5 * stale_rate)

    return RunMetrics(
        mode=mode,
        profile=profile,
        frames=n,
        mean_latency_ms=float(latency.mean()),
        p95_latency_ms=_percentile(latency, 95),
        p99_latency_ms=_percentile(latency, 99),
        deadline_miss_rate=miss_rate,
        mean_confidence=mean_conf,
        stale_output_rate=stale_rate,
        fallback_rate=fallback_rate,
        mode_switch_count=switches,
        total_bandwidth_kb=float(bandwidth.sum()),
        bandwidth_per_frame_kb=float(bandwidth.mean()),
        reliability_score=float(reliability),
    )


def summarize_runs(runs: Iterable[RunMetrics]) -> pd.DataFrame:
    """Build a tidy comparison DataFrame across runs."""
    rows: List[Dict] = [asdict(r) for r in runs]
    return pd.DataFrame(rows)
