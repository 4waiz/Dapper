"""
The DAPPER scheduler: deterministic, auditable, deadline- and confidence-aware.

Every decision is a pure function of the per-frame observation and the frozen
configuration. There is no learning, no randomness and no hidden state, so a
decision can always be replayed and explained from its logged inputs.

Risk score
----------
::

    R = w_r * min(1, RTT / D)
      + w_l * loss
      + w_c * load
      + w_a * min(1, frame_age / D)
      + w_d * min(1, (RTT + C_edge_hat) / D)

with non-negative weights summing to 1, so ``R`` is in [0, 1] by construction.

Decision procedure
------------------
1. **Edge unreachable** (health check failed). No packet is sent. Reuse the last
   valid output while it is inside the freshness window, otherwise run locally.
2. **R >= degraded threshold.** Remote execution is unsafe: reuse a fresh last
   valid output, otherwise run locally. Never accept an output whose age exceeds
   the freshness window.
3. **R >= local threshold.** Moderate risk: run locally for predictable latency.
4. **Low risk — confidence- and margin-aware.** This is the branch the published
   prototype declared but did not implement.

   a. The lightweight local model is already confident enough
      (``local_confidence >= local_confidence_threshold``): keep the frame on
      device. Reason ``local_confidence_sufficient``.
   b. The local result is weak and the *expected* remote completion fits the
      deadline with margin (``<= remote_deadline_margin * D``): commit the frame
      to the remote path. Reason ``low_confidence_edge_margin_ok``.
   c. The local result is weak and a remote refresh could *plausibly* still land
      inside the freshness window (optimistic remote completion
      ``<= min(hybrid_window, D)``): run hybrid, so a local answer is in hand
      immediately and the refresh is accepted only if it really arrives in time.
      Reason ``low_confidence_hybrid_refresh``.
   d. Otherwise a refresh would be rejected on arrival, so offloading would only
      spend bandwidth: stay local. Reason ``low_confidence_refresh_infeasible``.

Step (b) uses the *expected* remote compute estimate because it commits the
frame. Step (c) uses whichever estimate ``hybrid_estimator`` names: ``optimistic``
(the low end of the configured edge envelope, on the grounds that hybrid risks
bandwidth only and never the deadline) or ``expected`` (the same midpoint used by
step (b), which offloads only when a refresh is expected to land). Which of the
two is used is a design choice, so it is left to the calibration protocol rather
than fixed by hand. Both estimates come from the configured edge envelope the
client is assumed to know at runtime.

Estimation error
----------------
``rtt_estimate_bias`` and ``compute_estimate_bias`` perturb only the quantities
the *scheduler* believes; execution always uses the true scenario values. This
is what Sec. "runtime estimation error" varies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

MODE_LOCAL_FAST = "local_fast"
MODE_EDGE_ACCURATE = "edge_accurate"
MODE_HYBRID = "hybrid"
MODE_DEGRADED_SAFE = "degraded_safe"

ALL_MODES = (MODE_LOCAL_FAST, MODE_EDGE_ACCURATE, MODE_HYBRID, MODE_DEGRADED_SAFE)

#: Every reason string the scheduler can emit. Used by tests to assert that no
#: decision is ever unexplained.
ALL_REASONS = (
    "edge_unavailable_reuse_fresh",
    "edge_unavailable_local_fallback",
    "risk_high_reuse_fresh",
    "risk_high_local_fallback",
    "risk_moderate_prefer_local",
    "local_confidence_sufficient",
    "low_confidence_edge_margin_ok",
    "low_confidence_hybrid_refresh",
    "low_confidence_refresh_infeasible",
)


def _clip01(x: float) -> float:
    """Clamp to the unit interval."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x



@dataclass(frozen=True)
class SchedulerInputs:
    """Per-frame observations available to the scheduler at decision time."""

    rtt_ms: float
    packet_loss: float          # [0, 1]
    edge_load: float            # [0, 1]
    deadline_ms: float
    frame_age_ms: float         # age of the frame when the decision is taken
    last_valid_age_ms: float    # age of the newest already-available output
    local_confidence: float     # per-frame estimate of the local model's confidence
    edge_available: bool        # health-check / heartbeat state of the edge


@dataclass(frozen=True)
class SchedulerDecision:
    """Scheduler output. Every field is logged for auditability."""

    selected_mode: str
    deadline_risk_score: float
    reason: str
    expected_latency_ms: float
    fallback_needed: bool
    predicted_remote_arrival_ms: float
    remote_attempt_intended: bool


class DapperScheduler:
    """Risk-driven, confidence-aware perception placement."""

    def __init__(
        self,
        weight_rtt: float = 0.35,
        weight_loss: float = 0.30,
        weight_load: float = 0.15,
        weight_frame_age: float = 0.05,
        weight_deadline: float = 0.15,
        risk_degraded_threshold: float = 0.70,
        risk_local_threshold: float = 0.40,
        local_confidence_threshold: float = 0.725,
        remote_deadline_margin: float = 0.60,
        hybrid_freshness_window_ms: float = 80.0,
        hybrid_estimator: str = "optimistic",
        last_valid_freshness_ms: float = 250.0,
        edge_compute_ms_estimate: float = 67.5,
        edge_compute_ms_optimistic: float = 45.0,
        local_compute_ms_estimate: float = 25.0,
        edge_extra_rtt_ms: float = 0.0,
        load_compute_scale: float = 0.5,
        reuse_latency_ms: float = 1.0,
        rtt_estimate_bias: float = 0.0,
        compute_estimate_bias: float = 0.0,
        use_confidence_gate: bool = True,
        use_freshness_gate: bool = True,
        use_degraded_reuse: bool = True,
    ):
        total = weight_rtt + weight_loss + weight_load + weight_frame_age + weight_deadline
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"risk weights must sum to 1.0, got {total!r}")
        if min(weight_rtt, weight_loss, weight_load, weight_frame_age, weight_deadline) < 0.0:
            raise ValueError("risk weights must be non-negative")
        if not risk_local_threshold < risk_degraded_threshold:
            raise ValueError(
                f"risk_local_threshold ({risk_local_threshold}) must be < "
                f"risk_degraded_threshold ({risk_degraded_threshold})")
        if not 0.0 <= local_confidence_threshold <= 1.0:
            raise ValueError("local_confidence_threshold must lie in [0, 1]")
        if not 0.0 < remote_deadline_margin <= 1.0:
            raise ValueError("remote_deadline_margin must lie in (0, 1]")

        self.w_rtt = weight_rtt
        self.w_loss = weight_loss
        self.w_load = weight_load
        self.w_age = weight_frame_age
        self.w_dead = weight_deadline

        self.risk_degraded = risk_degraded_threshold
        self.risk_local = risk_local_threshold
        self.confidence_threshold = local_confidence_threshold
        self.remote_deadline_margin = remote_deadline_margin

        if hybrid_estimator not in ("optimistic", "expected"):
            raise ValueError("hybrid_estimator must be 'optimistic' or 'expected'")
        self.hybrid_window_ms = hybrid_freshness_window_ms
        self.hybrid_estimator = hybrid_estimator
        self.last_valid_freshness_ms = last_valid_freshness_ms

        self.edge_compute_ms = edge_compute_ms_estimate
        self.edge_compute_optimistic_ms = edge_compute_ms_optimistic
        self.local_compute_ms = local_compute_ms_estimate
        self.edge_extra_rtt_ms = edge_extra_rtt_ms
        self.load_compute_scale = load_compute_scale
        self.reuse_latency_ms = reuse_latency_ms

        self.rtt_estimate_bias = rtt_estimate_bias
        self.compute_estimate_bias = compute_estimate_bias

        self.use_confidence_gate = use_confidence_gate
        self.use_freshness_gate = use_freshness_gate
        self.use_degraded_reuse = use_degraded_reuse

    # ------------------------------------------------------------------ utils
    _clip01 = staticmethod(_clip01)

    def estimated_total_rtt_ms(self, rtt_ms: float) -> float:
        """Access RTT the scheduler believes, including any estimation bias."""
        return (rtt_ms + self.edge_extra_rtt_ms) * (1.0 + self.rtt_estimate_bias)

    def _estimated_compute_ms(self, edge_load: float, optimistic: bool = False) -> float:
        base = self.edge_compute_optimistic_ms if optimistic else self.edge_compute_ms
        return (base * (1.0 + self.load_compute_scale * edge_load)
                * (1.0 + self.compute_estimate_bias))

    def predicted_remote_arrival_ms(self, rtt_ms: float, edge_load: float,
                                    optimistic: bool = False) -> float:
        """
        Predicted time from frame capture to remote result arrival.

        Uses exactly the functional form the executor uses for the true value,
        with the scheduler's estimates substituted for the unobservable compute
        time. Keeping the two in one place is what guarantees that prediction
        and execution cannot drift apart.
        """
        return (self.estimated_total_rtt_ms(rtt_ms)
                + self._estimated_compute_ms(edge_load, optimistic=optimistic))

    # ------------------------------------------------------------------- risk
    def compute_risk(self, x: SchedulerInputs) -> float:
        """Weighted, normalised deadline-risk score in [0, 1]."""
        deadline = max(float(x.deadline_ms), 1.0)
        est_rtt = self.estimated_total_rtt_ms(x.rtt_ms)

        rtt_norm = _clip01(est_rtt / deadline)
        loss_norm = _clip01(x.packet_loss)
        load_norm = _clip01(x.edge_load)
        age_norm = _clip01(x.frame_age_ms / deadline)
        pressure = _clip01(self.predicted_remote_arrival_ms(x.rtt_ms, x.edge_load) / deadline)

        return _clip01(
            self.w_rtt * rtt_norm
            + self.w_loss * loss_norm
            + self.w_load * load_norm
            + self.w_age * age_norm
            + self.w_dead * pressure
        )

    # ----------------------------------------------------------------- decide
    def decide(self, x: SchedulerInputs) -> SchedulerDecision:
        """Select a mode for this frame and explain why."""
        risk = self.compute_risk(x)
        fresh_reuse_available = (
            self.use_degraded_reuse and x.last_valid_age_ms <= self.last_valid_freshness_ms
        )

        # 1. Edge unreachable: the health check already told us, so no packet is
        #    sent and no bandwidth is spent.
        if not x.edge_available:
            if fresh_reuse_available:
                return SchedulerDecision(
                    selected_mode=MODE_DEGRADED_SAFE,
                    deadline_risk_score=risk,
                    reason="edge_unavailable_reuse_fresh",
                    expected_latency_ms=self.reuse_latency_ms,
                    fallback_needed=True,
                    predicted_remote_arrival_ms=float("inf"),
                    remote_attempt_intended=False,
                )
            return SchedulerDecision(
                selected_mode=MODE_LOCAL_FAST,
                deadline_risk_score=risk,
                reason="edge_unavailable_local_fallback",
                expected_latency_ms=self.local_compute_ms,
                fallback_needed=True,
                predicted_remote_arrival_ms=float("inf"),
                remote_attempt_intended=False,
            )

        predicted = self.predicted_remote_arrival_ms(x.rtt_ms, x.edge_load)

        # 2. Severe risk: guarantee a timely output.
        if risk >= self.risk_degraded:
            return SchedulerDecision(
                selected_mode=MODE_DEGRADED_SAFE,
                deadline_risk_score=risk,
                reason=("risk_high_reuse_fresh" if fresh_reuse_available
                        else "risk_high_local_fallback"),
                expected_latency_ms=(self.reuse_latency_ms if fresh_reuse_available
                                     else self.local_compute_ms),
                fallback_needed=True,
                predicted_remote_arrival_ms=predicted,
                remote_attempt_intended=False,
            )

        # 3. Moderate risk: predictable local latency.
        if risk >= self.risk_local:
            return SchedulerDecision(
                selected_mode=MODE_LOCAL_FAST,
                deadline_risk_score=risk,
                reason="risk_moderate_prefer_local",
                expected_latency_ms=self.local_compute_ms,
                fallback_needed=False,
                predicted_remote_arrival_ms=predicted,
                remote_attempt_intended=False,
            )

        # 4a. Low risk and the local model is confident enough: stay on device.
        if self.use_confidence_gate and x.local_confidence >= self.confidence_threshold:
            return SchedulerDecision(
                selected_mode=MODE_LOCAL_FAST,
                deadline_risk_score=risk,
                reason="local_confidence_sufficient",
                expected_latency_ms=self.local_compute_ms,
                fallback_needed=False,
                predicted_remote_arrival_ms=predicted,
                remote_attempt_intended=False,
            )

        # 4b. Weak local result and the remote path fits the deadline with margin.
        if predicted <= self.remote_deadline_margin * x.deadline_ms:
            return SchedulerDecision(
                selected_mode=MODE_EDGE_ACCURATE,
                deadline_risk_score=risk,
                reason="low_confidence_edge_margin_ok",
                expected_latency_ms=predicted,
                fallback_needed=False,
                predicted_remote_arrival_ms=predicted,
                remote_attempt_intended=True,
            )

        # 4c. A refresh could plausibly still land inside the freshness window.
        refresh_estimate = self.predicted_remote_arrival_ms(
            x.rtt_ms, x.edge_load, optimistic=(self.hybrid_estimator == "optimistic"))
        refresh_budget = min(self.hybrid_window_ms, x.deadline_ms)
        if (not self.use_freshness_gate) or refresh_estimate <= refresh_budget:
            return SchedulerDecision(
                selected_mode=MODE_HYBRID,
                deadline_risk_score=risk,
                reason="low_confidence_hybrid_refresh",
                expected_latency_ms=self.local_compute_ms,
                fallback_needed=False,
                predicted_remote_arrival_ms=predicted,
                remote_attempt_intended=True,
            )

        # 4d. Any refresh would be rejected on arrival: do not spend bandwidth.
        return SchedulerDecision(
            selected_mode=MODE_LOCAL_FAST,
            deadline_risk_score=risk,
            reason="low_confidence_refresh_infeasible",
            expected_latency_ms=self.local_compute_ms,
            fallback_needed=False,
            predicted_remote_arrival_ms=predicted,
            remote_attempt_intended=False,
        )



def build_scheduler_from_config(
    cfg: Dict[str, Any],
    rtt_estimate_bias: float = 0.0,
    compute_estimate_bias: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    use_confidence_gate: bool = True,
    use_freshness_gate: bool = True,
    use_degraded_reuse: bool = True,
) -> DapperScheduler:
    """
    Construct a scheduler from a validated configuration.

    ``weights`` overrides the five risk weights (used by the component ablation,
    which zeroes one weight and renormalises the rest).
    """
    s = cfg["scheduler"]
    ex = cfg["execution"]
    edge = cfg["edge"]
    local = cfg["local"]
    w = weights or {
        "weight_rtt": float(s["weight_rtt"]),
        "weight_loss": float(s["weight_loss"]),
        "weight_load": float(s["weight_load"]),
        "weight_frame_age": float(s["weight_frame_age"]),
        "weight_deadline": float(s["weight_deadline"]),
    }
    return DapperScheduler(
        weight_rtt=w["weight_rtt"],
        weight_loss=w["weight_loss"],
        weight_load=w["weight_load"],
        weight_frame_age=w["weight_frame_age"],
        weight_deadline=w["weight_deadline"],
        risk_degraded_threshold=float(s["risk_degraded_threshold"]),
        risk_local_threshold=float(s["risk_local_threshold"]),
        local_confidence_threshold=float(s["local_confidence_threshold"]),
        remote_deadline_margin=float(s["remote_deadline_margin"]),
        hybrid_freshness_window_ms=float(s["hybrid_freshness_window_ms"]),
        hybrid_estimator=str(s.get("hybrid_estimator", "optimistic")),
        last_valid_freshness_ms=float(s["last_valid_freshness_ms"]),
        edge_compute_ms_estimate=0.5 * (float(edge["latency_ms_min"])
                                        + float(edge["latency_ms_max"])),
        edge_compute_ms_optimistic=float(edge["latency_ms_min"]),
        local_compute_ms_estimate=0.5 * (float(local["latency_ms_min"])
                                         + float(local["latency_ms_max"])),
        edge_extra_rtt_ms=float(edge.get("extra_rtt_ms", 0.0)),
        load_compute_scale=float(ex["load_compute_scale"]),
        reuse_latency_ms=float(ex["reuse_latency_ms"]),
        rtt_estimate_bias=rtt_estimate_bias,
        compute_estimate_bias=compute_estimate_bias,
        use_confidence_gate=use_confidence_gate,
        use_freshness_gate=use_freshness_gate,
        use_degraded_reuse=use_degraded_reuse,
    )
