"""
Network and edge-load monitor simulation.

The monitor produces a per-frame snapshot of network conditions:

    rtt_ms, packet_loss, edge_load, edge_available

The values are sampled from one of several profiles (stable, congested,
lossy, variable, outage). Each profile is parameterised in config.yaml so
that reviewers can rerun experiments with different assumptions.

This is a controlled synthetic environment. It is honest about that: every
distribution is explicit and seeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class NetworkSample:
    rtt_ms: float
    packet_loss: float
    edge_load: float
    edge_available: bool


class NetworkMonitor:
    """
    Sample per-frame network conditions from a named profile.

    The sampler keeps internal state so that conditions can drift smoothly
    (RTT for the next frame is correlated with the previous). This makes
    the simulated time series more realistic than purely IID draws.
    """

    def __init__(
        self,
        profile_name: str,
        profile_params: Dict[str, float],
        seed: int = 42,
    ):
        if profile_params is None:
            raise ValueError(f"Unknown network profile: {profile_name}")
        self.profile_name = profile_name
        self.params = profile_params
        self.rng = np.random.default_rng(seed)

        # State for smoothing
        self._current_rtt = float(profile_params["rtt_ms_mean"])
        self._current_load = float(profile_params["edge_load"])
        self._outage_active = False
        self._outage_remaining_frames = 0

    # ------------------------------------------------------------------ utils
    def _draw_rtt(self) -> float:
        mean = float(self.params["rtt_ms_mean"])
        std = float(self.params["rtt_ms_std"])
        # AR(1)-style smoothing so that latency does not flicker wildly.
        target = self.rng.normal(mean, std)
        self._current_rtt = 0.7 * self._current_rtt + 0.3 * target
        return max(1.0, float(self._current_rtt))

    def _draw_loss(self) -> float:
        base = float(self.params["packet_loss"])
        jitter = self.rng.normal(0.0, base * 0.25)
        return float(np.clip(base + jitter, 0.0, 1.0))

    def _draw_load(self) -> float:
        base = float(self.params["edge_load"])
        jitter = self.rng.normal(0.0, 0.05)
        self._current_load = float(np.clip(0.8 * self._current_load + 0.2 * (base + jitter), 0.0, 1.0))
        return self._current_load

    def _update_outage(self) -> bool:
        """Manage transient outage windows. Returns True if edge is available."""
        outage_prob = float(self.params.get("outage_prob", 0.0))

        if self._outage_active:
            self._outage_remaining_frames -= 1
            if self._outage_remaining_frames <= 0:
                self._outage_active = False
            return not self._outage_active

        if outage_prob > 0.0 and self.rng.random() < outage_prob:
            # Outage lasts a short burst of frames
            self._outage_active = True
            self._outage_remaining_frames = int(self.rng.integers(5, 25))
            return False
        return True

    # ------------------------------------------------------------------ public
    def sample(self) -> NetworkSample:
        edge_available = self._update_outage()
        return NetworkSample(
            rtt_ms=self._draw_rtt(),
            packet_loss=self._draw_loss(),
            edge_load=self._draw_load(),
            edge_available=edge_available,
        )


def build_monitor(profile_name: str, cfg: dict, seed: int = 42) -> NetworkMonitor:
    """Construct a NetworkMonitor from the parsed config."""
    profiles = cfg.get("network_profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"Profile '{profile_name}' not found. Available: {list(profiles)}"
        )
    return NetworkMonitor(profile_name, profiles[profile_name], seed=seed)
