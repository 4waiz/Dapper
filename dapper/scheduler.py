"""
DAPPER Scheduler.

Selects one of four perception execution modes per frame:

    local_fast       lightweight on-device inference
    edge_accurate    remote edge inference via HTTP
    hybrid           local immediately, refresh from edge if it arrives in time
    degraded_safe    fallback when deadline risk is high

The scheduler is intentionally deterministic and explainable: every choice has a
risk score and a textual reason. This makes the policy auditable for a
research paper and easy to ablate by changing the weights in config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Mode names are exported as plain strings so that CSV logs are human readable.
MODE_LOCAL_FAST = "local_fast"
MODE_EDGE_ACCURATE = "edge_accurate"
MODE_HYBRID = "hybrid"
MODE_DEGRADED_SAFE = "degraded_safe"

ALL_MODES = (MODE_LOCAL_FAST, MODE_EDGE_ACCURATE, MODE_HYBRID, MODE_DEGRADED_SAFE)


@dataclass
class SchedulerInputs:
    """Per-frame observations fed into the scheduler."""
    rtt_ms: float
    packet_loss: float          # 0.0 - 1.0
    edge_load: float            # 0.0 - 1.0
    deadline_ms: float
    frame_age_ms: float         # Time since the frame was captured
    last_valid_age_ms: float    # Age of the last successful perception result
    local_confidence: float     # Confidence we would get from local inference
    edge_available: bool        # Health-check state of the edge server


@dataclass
class SchedulerDecision:
    """Scheduler output. All fields are logged to the run CSV."""
    selected_mode: str
    deadline_risk_score: float  # 0.0 (safe) -> 1.0 (very risky)
    reason: str
    expected_latency_ms: float
    fallback_needed: bool


class DapperScheduler:
    """
    Risk-driven scheduler.

    The risk score is a weighted sum of five normalized signals:

        risk = w_rtt   * norm(rtt, deadline)
             + w_loss  * packet_loss
             + w_load  * edge_load
             + w_age   * norm(frame_age, deadline)
             + w_dead  * deadline_pressure

    where deadline_pressure grows as the budget after RTT shrinks.

    Two thresholds map the score to a mode:

        risk >= degraded_threshold  -> degraded_safe
        risk >= local_threshold     -> local_fast
        else, depending on freshness -> edge_accurate or hybrid

    The thresholds, weights and freshness windows are loaded from config.yaml
    so reviewers can run ablations without changing code.
    """

    def __init__(
        self,
        weight_rtt: float = 0.35,
        weight_loss: float = 0.30,
        weight_load: float = 0.15,
        weight_frame_age: float = 0.05,
        weight_deadline: float = 0.15,
        risk_degraded_threshold: float = 0.70,
        risk_local_threshold: float = 0.40,
        hybrid_freshness_window_ms: float = 80.0,
        last_valid_freshness_ms: float = 250.0,
        edge_compute_ms_estimate: float = 65.0,
        local_compute_ms_estimate: float = 25.0,
    ):
        self.w_rtt = weight_rtt
        self.w_loss = weight_loss
        self.w_load = weight_load
        self.w_age = weight_frame_age
        self.w_dead = weight_deadline

        self.risk_degraded = risk_degraded_threshold
        self.risk_local = risk_local_threshold

        self.hybrid_window_ms = hybrid_freshness_window_ms
        self.last_valid_freshness_ms = last_valid_freshness_ms

        self.edge_compute_ms = edge_compute_ms_estimate
        self.local_compute_ms = local_compute_ms_estimate

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _clip01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _deadline_pressure(self, rtt_ms: float, deadline_ms: float) -> float:
        """How much of the deadline is already consumed by the network alone."""
        if deadline_ms <= 0:
            return 1.0
        # Edge total estimate = RTT + edge compute. Compare against the deadline.
        edge_total_estimate = rtt_ms + self.edge_compute_ms
        return self._clip01(edge_total_estimate / deadline_ms)

    def compute_risk(self, x: SchedulerInputs) -> float:
        """
        Combine normalized signals into a single risk score in [0, 1].

        We normalize RTT and frame age by the deadline so that the score adapts
        when the deadline changes between experiments.
        """
        deadline = max(x.deadline_ms, 1.0)

        rtt_norm = self._clip01(x.rtt_ms / deadline)
        loss_norm = self._clip01(x.packet_loss)
        load_norm = self._clip01(x.edge_load)
        age_norm = self._clip01(x.frame_age_ms / deadline)
        pressure = self._deadline_pressure(x.rtt_ms, x.deadline_ms)

        risk = (
            self.w_rtt * rtt_norm
            + self.w_loss * loss_norm
            + self.w_load * load_norm
            + self.w_age * age_norm
            + self.w_dead * pressure
        )
        return self._clip01(risk)

    # ----------------------------------------------------------------- decide
    def decide(self, x: SchedulerInputs) -> SchedulerDecision:
        """
        Pick a mode for this frame and explain why.

        Decision order:
        1. If edge is unreachable, only local or degraded-safe are viable.
        2. Otherwise compute risk and apply thresholds.
        3. Within the safe region, choose between edge_accurate and hybrid
           based on whether the edge result is likely to arrive in the
           hybrid freshness window.
        """
        risk = self.compute_risk(x)

        # 1. Edge unavailability is a hard constraint.
        if not x.edge_available:
            if x.last_valid_age_ms <= self.last_valid_freshness_ms:
                return SchedulerDecision(
                    selected_mode=MODE_DEGRADED_SAFE,
                    deadline_risk_score=risk,
                    reason="edge_unavailable_use_last_valid",
                    expected_latency_ms=self.local_compute_ms,
                    fallback_needed=True,
                )
            return SchedulerDecision(
                selected_mode=MODE_LOCAL_FAST,
                deadline_risk_score=risk,
                reason="edge_unavailable_local_fallback",
                expected_latency_ms=self.local_compute_ms,
                fallback_needed=True,
            )

        # 2. Severe risk -> degraded-safe to guarantee timely output.
        if risk >= self.risk_degraded:
            fresh = x.last_valid_age_ms <= self.last_valid_freshness_ms
            return SchedulerDecision(
                selected_mode=MODE_DEGRADED_SAFE,
                deadline_risk_score=risk,
                reason=(
                    "risk_high_reuse_last_valid" if fresh
                    else "risk_high_local_fallback"
                ),
                expected_latency_ms=(
                    1.0 if fresh else self.local_compute_ms
                ),
                fallback_needed=True,
            )

        # 3. Moderate risk -> stay local for predictable latency.
        if risk >= self.risk_local:
            return SchedulerDecision(
                selected_mode=MODE_LOCAL_FAST,
                deadline_risk_score=risk,
                reason="risk_moderate_prefer_local",
                expected_latency_ms=self.local_compute_ms,
                fallback_needed=False,
            )

        # 4. Low risk: pick between edge_accurate and hybrid.
        # Hybrid is preferred when the edge round trip might overshoot the
        # freshness window: we still get an immediate local answer.
        expected_edge_total = x.rtt_ms + self.edge_compute_ms
        if expected_edge_total > self.hybrid_window_ms:
            return SchedulerDecision(
                selected_mode=MODE_HYBRID,
                deadline_risk_score=risk,
                reason="low_risk_hybrid_for_freshness",
                expected_latency_ms=self.local_compute_ms,
                fallback_needed=False,
            )

        return SchedulerDecision(
            selected_mode=MODE_EDGE_ACCURATE,
            deadline_risk_score=risk,
            reason="low_risk_prefer_edge_accuracy",
            expected_latency_ms=expected_edge_total,
            fallback_needed=False,
        )


def build_scheduler_from_config(cfg: dict) -> DapperScheduler:
    """Construct a scheduler from the parsed config dictionary."""
    s = cfg.get("scheduler", {})
    edge_lat_mid = 0.5 * (
        cfg["edge"]["latency_ms_min"] + cfg["edge"]["latency_ms_max"]
    )
    local_lat_mid = 0.5 * (
        cfg["local"]["latency_ms_min"] + cfg["local"]["latency_ms_max"]
    )
    return DapperScheduler(
        weight_rtt=s.get("weight_rtt", 0.35),
        weight_loss=s.get("weight_loss", 0.30),
        weight_load=s.get("weight_load", 0.15),
        weight_frame_age=s.get("weight_frame_age", 0.05),
        weight_deadline=s.get("weight_deadline", 0.15),
        risk_degraded_threshold=s.get("risk_degraded_threshold", 0.70),
        risk_local_threshold=s.get("risk_local_threshold", 0.40),
        hybrid_freshness_window_ms=s.get("hybrid_freshness_window_ms", 80.0),
        last_valid_freshness_ms=s.get("last_valid_freshness_ms", 250.0),
        edge_compute_ms_estimate=edge_lat_mid,
        local_compute_ms_estimate=local_lat_mid,
    )
