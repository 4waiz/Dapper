"""
Trace-driven network input.

The synthetic profiles in ``config.yaml`` are parameterised distributions, not
measurements, and nothing in this repository may describe them as captured
Wi-Fi or 5G traces. This module adds the *plumbing* for replaying a real trace
so that a future measurement campaign can be dropped in without touching the
scheduler or the execution model.

Trace format (CSV, one row per sample)::

    timestamp_ms,rtt_ms,packet_loss,edge_load,edge_available

``packet_loss`` is the instantaneous loss probability in [0, 1]; ``edge_load``
is a normalised server-utilisation estimate in [0, 1]; ``edge_available`` is
0/1 (or false/true). ``edge_load`` and ``edge_available`` may be omitted, in
which case they default to the values given to :func:`load_network_trace`.

The samples are resampled onto the benchmark's frame grid by nearest preceding
timestamp, so a trace recorded at any rate can drive any frame rate.

``tools/capture_network_trace.py`` writes this format from real ICMP/TCP probes.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TRACE_COLUMNS = ("timestamp_ms", "rtt_ms", "packet_loss", "edge_load", "edge_available")


def load_network_trace(
    path: str,
    frames: int,
    frame_period_ms: float,
    default_edge_load: float = 0.5,
    default_edge_available: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Read a trace CSV and resample it onto ``frames`` frame slots.

    Returns the array dictionary expected by
    :func:`dapper.scenario.generate_scenarios` as ``network_override``.
    """
    df = pd.read_csv(path)
    missing = [c for c in ("timestamp_ms", "rtt_ms", "packet_loss") if c not in df.columns]
    if missing:
        raise ValueError(f"network trace {path} is missing columns: {missing}")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    if "edge_load" not in df.columns:
        df["edge_load"] = float(default_edge_load)
    if "edge_available" not in df.columns:
        df["edge_available"] = bool(default_edge_available)

    t0 = float(df["timestamp_ms"].iloc[0])
    grid = np.arange(frames, dtype=float) * float(frame_period_ms) + t0
    src = df["timestamp_ms"].to_numpy(dtype=float)
    if grid[-1] > src[-1]:
        # Wrap the trace so a short capture can still drive a long benchmark.
        span = max(src[-1] - src[0], frame_period_ms)
        grid = t0 + np.mod(grid - t0, span)
    # Nearest *preceding* sample. The tolerance absorbs the float round trip
    # through the CSV, which would otherwise shift a grid point one sample back.
    tol = 1e-6 * float(frame_period_ms)
    idx = np.clip(np.searchsorted(src, grid + tol, side="right") - 1, 0, len(src) - 1)

    avail = df["edge_available"].to_numpy()
    if avail.dtype != bool:
        avail = pd.Series(avail).map(
            lambda v: str(v).strip().lower() not in ("0", "false", "no", "")
        ).to_numpy(dtype=bool)

    return {
        "rtt_ms": np.maximum(1.0, df["rtt_ms"].to_numpy(dtype=float)[idx]),
        "packet_loss": np.clip(df["packet_loss"].to_numpy(dtype=float)[idx], 0.0, 1.0),
        "edge_load": np.clip(df["edge_load"].to_numpy(dtype=float)[idx], 0.0, 1.0),
        "edge_available": avail[idx],
    }


def write_network_trace(path: str, samples: pd.DataFrame) -> str:
    """Write a trace in the canonical column order."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for c in TRACE_COLUMNS:
        if c not in samples.columns:
            raise ValueError(f"sample frame is missing column '{c}'")
    samples[list(TRACE_COLUMNS)].to_csv(path, index=False)
    return path


def synthetic_trace_from_profile(profile: str, cfg: Dict[str, Any], seed: int,
                                 samples: int) -> pd.DataFrame:
    """
    Emit a trace file *from a synthetic profile*, for testing the trace path.

    The resulting file carries a `synthetic` provenance marker in its name by
    convention; it is not a measurement and must never be reported as one.
    """
    from .scenario import generate_scenarios
    tr = generate_scenarios(profile, cfg, seed, samples)
    period = float(cfg["execution"]["frame_period_ms"])
    return pd.DataFrame({
        "timestamp_ms": np.arange(samples, dtype=float) * period,
        "rtt_ms": np.asarray(tr.rtt_ms),
        "packet_loss": np.asarray(tr.packet_loss),
        "edge_load": np.asarray(tr.edge_load),
        "edge_available": np.asarray(tr.edge_available).astype(int),
    })
