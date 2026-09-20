"""
Immutable pre-generated per-frame scenarios.

Rationale
---------
In the originally published prototype the network trace was shared across
policies but the *perception* random stream was not: ``local_infer`` drew three
random numbers, ``edge_infer`` four, ``hybrid_infer`` seven and ``degraded_safe``
zero or three. Any policy that branched differently desynchronised the stream
for every later frame, so "the same seed" did not mean "the same frame".

This module removes that confound. For each ``(seed, profile)`` we materialise
an immutable scenario trace in which every stochastic quantity a policy could
possibly consume is drawn *in advance* from its own independent random stream:

    network        rtt, packet loss, edge load, edge availability
    local potential    inference latency, confidence, detection count
    edge potential     compute time, confidence, detection count, loss draw
    cloud potential    compute time, confidence, detection count, loss draw

Every policy then replays the *same* trace. A policy's decision selects which
potential outcome is realised; it cannot change the random future, because there
is no random future left to change. Comparisons are therefore exactly paired at
the frame level, which is what makes a per-seed paired analysis legitimate.

The trace is stored as a struct of arrays (``ScenarioTrace``) for speed, with a
``FrameScenario`` view for readability in tests and documentation.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

#: Independent random streams, in a fixed order. Adding a stream changes every
#: generated trace, so the order and membership of this tuple are part of the
#: reproducibility contract.
STREAM_NAMES = (
    "net_rtt",
    "net_loss",
    "net_load",
    "net_outage",
    "local_latency",
    "local_confidence",
    "local_detections",
    "edge_compute",
    "edge_confidence",
    "edge_detections",
    "edge_loss_draw",
    "cloud_compute",
    "cloud_confidence",
    "cloud_detections",
    "cloud_loss_draw",
)

#: AR(1) smoothing coefficients, preserved from the originally published monitor
#: so that the network process is statistically unchanged.
_RTT_SMOOTHING = 0.7
_LOAD_SMOOTHING = 0.8
_LOSS_JITTER_FRAC = 0.25
_LOAD_JITTER_STD = 0.05
_OUTAGE_MIN_FRAMES = 5
_OUTAGE_MAX_FRAMES = 25


@dataclass(frozen=True)
class FrameScenario:
    """Everything that is predetermined about one frame, for every policy."""

    frame_id: int
    capture_time_ms: float
    # --- observable network state -------------------------------------------
    rtt_ms: float
    packet_loss: float
    edge_load: float
    edge_available: bool
    # --- potential local outcome --------------------------------------------
    local_latency_ms: float
    local_confidence: float
    local_detections: int
    # --- potential edge outcome ---------------------------------------------
    edge_compute_ms: float
    edge_confidence: float
    edge_detections: int
    edge_loss_draw: float
    # --- potential cloud outcome --------------------------------------------
    cloud_compute_ms: float
    cloud_confidence: float
    cloud_detections: int
    cloud_loss_draw: float


class ScenarioTrace:
    """A frozen struct-of-arrays scenario trace for one ``(seed, profile)``."""

    __slots__ = ("seed", "profile", "frames", "capture_time_ms", "_a", "source")

    def __init__(self, seed: int, profile: str, arrays: Dict[str, np.ndarray],
                 frame_period_ms: float, source: str = "synthetic"):
        n = len(arrays["rtt_ms"])
        self.seed = int(seed)
        self.profile = str(profile)
        self.frames = int(n)
        self.source = source
        self.capture_time_ms = np.arange(n, dtype=float) * float(frame_period_ms)
        # Normalise dtypes so that a CSV round trip reproduces the fingerprint
        # exactly: the hash is over raw bytes, so int32 vs int64 would differ.
        normalised = {}
        for k, v in arrays.items():
            v = np.asarray(v)
            if k == "edge_available":
                v = v.astype(bool)
            elif v.dtype.kind in "iu":
                v = v.astype(np.int32)
            else:
                v = v.astype(np.float64)
            v = np.ascontiguousarray(v)
            v.setflags(write=False)
            normalised[k] = v
        self._a = normalised

    def __len__(self) -> int:
        return self.frames

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self._a[name]
        except KeyError as exc:  # pragma: no cover - attribute error path
            raise AttributeError(name) from exc

    def frame(self, i: int) -> FrameScenario:
        """Materialise frame ``i`` as a readable dataclass."""
        a = self._a
        return FrameScenario(
            frame_id=i,
            capture_time_ms=float(self.capture_time_ms[i]),
            rtt_ms=float(a["rtt_ms"][i]),
            packet_loss=float(a["packet_loss"][i]),
            edge_load=float(a["edge_load"][i]),
            edge_available=bool(a["edge_available"][i]),
            local_latency_ms=float(a["local_latency_ms"][i]),
            local_confidence=float(a["local_confidence"][i]),
            local_detections=int(a["local_detections"][i]),
            edge_compute_ms=float(a["edge_compute_ms"][i]),
            edge_confidence=float(a["edge_confidence"][i]),
            edge_detections=int(a["edge_detections"][i]),
            edge_loss_draw=float(a["edge_loss_draw"][i]),
            cloud_compute_ms=float(a["cloud_compute_ms"][i]),
            cloud_confidence=float(a["cloud_confidence"][i]),
            cloud_detections=int(a["cloud_detections"][i]),
            cloud_loss_draw=float(a["cloud_loss_draw"][i]),
        )

    def to_frame(self) -> pd.DataFrame:
        """Tidy DataFrame, for saving a replayable trace to CSV."""
        df = pd.DataFrame({k: np.asarray(v) for k, v in self._a.items()})
        df.insert(0, "capture_time_ms", self.capture_time_ms)
        df.insert(0, "frame_id", np.arange(self.frames))
        df.insert(0, "profile", self.profile)
        df.insert(0, "seed", self.seed)
        return df

    def fingerprint(self) -> str:
        """Stable hash of the trace contents, used to assert exact replay."""
        h = zlib.crc32(b"dapper-scenario-v2")
        for name in sorted(self._a):
            h = zlib.crc32(np.ascontiguousarray(self._a[name]).tobytes(), h)
        return f"{h:08x}"


def _streams(seed: int, profile: str) -> Dict[str, np.random.Generator]:
    """One independent PCG64 generator per named stream."""
    profile_tag = zlib.crc32(profile.encode("utf-8"))
    root = np.random.SeedSequence([int(seed), int(profile_tag)])
    children = root.spawn(len(STREAM_NAMES))
    return {name: np.random.default_rng(child)
            for name, child in zip(STREAM_NAMES, children)}


def _network_arrays(params: Dict[str, Any], frames: int,
                    rng: Dict[str, np.random.Generator]) -> Dict[str, np.ndarray]:
    """
    Reproduce the published network process with decoupled random streams.

    RTT and edge load are AR(1)-smoothed, packet loss is jittered around its
    profile mean, and outages arrive as bursts of 5-25 frames. Exactly two
    draws are taken from the outage stream per frame regardless of the current
    outage state, so stream consumption never depends on realised state.
    """
    rtt_mean = float(params["rtt_ms_mean"])
    rtt_std = float(params["rtt_ms_std"])
    loss_base = float(params["packet_loss"])
    load_base = float(params["edge_load"])
    outage_prob = float(params.get("outage_prob", 0.0))

    rtt_targets = rng["net_rtt"].normal(rtt_mean, rtt_std, size=frames)
    loss_jitter = rng["net_loss"].normal(0.0, loss_base * _LOSS_JITTER_FRAC, size=frames)
    load_jitter = rng["net_load"].normal(0.0, _LOAD_JITTER_STD, size=frames)
    outage_u = rng["net_outage"].random(size=frames)
    outage_len = rng["net_outage"].integers(_OUTAGE_MIN_FRAMES, _OUTAGE_MAX_FRAMES, size=frames)

    rtt = np.empty(frames, dtype=float)
    load = np.empty(frames, dtype=float)
    avail = np.empty(frames, dtype=bool)

    cur_rtt = rtt_mean
    cur_load = load_base
    remaining = 0
    for i in range(frames):
        if remaining > 0:
            remaining -= 1
            avail[i] = False
        elif outage_prob > 0.0 and outage_u[i] < outage_prob:
            remaining = int(outage_len[i]) - 1
            avail[i] = False
        else:
            avail[i] = True
        cur_rtt = _RTT_SMOOTHING * cur_rtt + (1.0 - _RTT_SMOOTHING) * rtt_targets[i]
        rtt[i] = max(1.0, cur_rtt)
        cur_load = float(np.clip(
            _LOAD_SMOOTHING * cur_load + (1.0 - _LOAD_SMOOTHING) * (load_base + load_jitter[i]),
            0.0, 1.0))
        load[i] = cur_load

    loss = np.clip(loss_base + loss_jitter, 0.0, 1.0)
    return {
        "rtt_ms": rtt,
        "packet_loss": loss,
        "edge_load": load,
        "edge_available": avail,
    }


def _detections(rng: np.random.Generator, frames: int) -> np.ndarray:
    """Object count per frame: mostly a handful, with a busy-frame tail."""
    return np.maximum(0, np.round(rng.exponential(2.0, size=frames))).astype(np.int32)


def generate_scenarios(
    profile: str,
    cfg: Dict[str, Any],
    seed: int,
    frames: int,
    network_override: Optional[Dict[str, np.ndarray]] = None,
    source: str = "synthetic",
) -> ScenarioTrace:
    """
    Build the immutable scenario trace for one ``(seed, profile)``.

    ``network_override`` replaces the synthetic network process with measured or
    replayed values (see :mod:`dapper.trace`); the perception potentials are
    still generated from the seeded streams so that policies remain paired.
    """
    if frames <= 0:
        raise ValueError("frames must be positive")
    profiles = cfg.get("network_profiles", {})
    if network_override is None and profile not in profiles:
        raise ValueError(f"profile '{profile}' not found; available: {sorted(profiles)}")

    rng = _streams(seed, profile)
    arrays: Dict[str, np.ndarray] = {}
    if network_override is None:
        arrays.update(_network_arrays(profiles[profile], frames, rng))
    else:
        for key in ("rtt_ms", "packet_loss", "edge_load", "edge_available"):
            if key not in network_override:
                raise ValueError(f"network_override missing '{key}'")
        arrays.update({k: np.asarray(network_override[k])[:frames].copy()
                       for k in ("rtt_ms", "packet_loss", "edge_load", "edge_available")})
        if len(arrays["rtt_ms"]) != frames:
            raise ValueError(
                f"network_override provides {len(arrays['rtt_ms'])} samples, need {frames}")

    for backend, prefix in (("local", "local"), ("edge", "edge"), ("cloud", "cloud")):
        sec = cfg[backend]
        lat = rng[f"{prefix}_latency" if backend == "local" else f"{prefix}_compute"].uniform(
            float(sec["latency_ms_min"]), float(sec["latency_ms_max"]), size=frames)
        conf = rng[f"{prefix}_confidence"].uniform(
            float(sec["confidence_min"]), float(sec["confidence_max"]), size=frames)
        det = _detections(rng[f"{prefix}_detections"], frames)
        key = "local_latency_ms" if backend == "local" else f"{prefix}_compute_ms"
        arrays[key] = lat
        arrays[f"{prefix}_confidence"] = conf
        arrays[f"{prefix}_detections"] = det

    arrays["edge_loss_draw"] = rng["edge_loss_draw"].random(size=frames)
    arrays["cloud_loss_draw"] = rng["cloud_loss_draw"].random(size=frames)

    return ScenarioTrace(
        seed=seed,
        profile=profile,
        arrays=arrays,
        frame_period_ms=float(cfg["execution"]["frame_period_ms"]),
        source=source,
    )


def save_scenarios(trace: ScenarioTrace, path: str) -> str:
    """Write a scenario trace to CSV so a reviewer can inspect or replay it."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # 17 significant digits round-trips an IEEE-754 double exactly, so a
    # replayed trace reproduces the fingerprint bit for bit.
    trace.to_frame().to_csv(path, index=False, float_format="%.17g")
    return path


def load_scenarios(path: str, cfg: Dict[str, Any]) -> ScenarioTrace:
    """Read back a scenario trace written by :func:`save_scenarios`."""
    # `float_precision="round_trip"` uses the correctly-rounded parser; the
    # default fast parser can be one unit in the last place off, which would
    # break the fingerprint equality that proves an exact replay.
    df = pd.read_csv(path, float_precision="round_trip")
    arrays = {c: df[c].to_numpy() for c in df.columns
              if c not in ("seed", "profile", "frame_id", "capture_time_ms")}
    arrays["edge_available"] = arrays["edge_available"].astype(bool)
    return ScenarioTrace(
        seed=int(df["seed"].iloc[0]),
        profile=str(df["profile"].iloc[0]),
        arrays=arrays,
        frame_period_ms=float(cfg["execution"]["frame_period_ms"]),
        source="replayed",
    )


def scenario_bank(profiles: Sequence[str], seeds: Sequence[int], cfg: Dict[str, Any],
                  frames: int) -> Dict[tuple, ScenarioTrace]:
    """Generate every ``(seed, profile)`` trace once and cache it."""
    return {(s, p): generate_scenarios(p, cfg, s, frames)
            for p in profiles for s in seeds}
