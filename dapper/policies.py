"""
Every evaluated execution policy.

All policies share one interface and one execution model (:mod:`dapper.executor`),
so the only thing that differs between them is the decision rule.

Fixed baselines
    ``local_only``   always run the lightweight local model;
    ``edge_only``    always offload to the edge (local fallback if the health
                     check says the edge is down);
    ``cloud_only``   always offload to the cloud.

Adaptive baselines (the comparators Reviewer 1 asked for; none is a strawman,
and each one's hyperparameters are fitted on the calibration seeds only)
    ``deadline_greedy``     offload whenever the predicted edge completion fits
                            ``margin * D``. A single-signal predicted-deadline
                            test with no risk score, no confidence gate and no
                            degraded reuse.
    ``confidence_deadline`` the intuitive rule "offload only when the local model
                            is unconfident *and* the edge can finish on time".
                            This is the closest non-DAPPER analogue of DAPPER's
                            low-risk branch and the fairest comparator.
    ``rtt_threshold``       offload whenever the measured RTT is below a
                            calibrated threshold.

Offline oracle (NOT a deployable policy)
    ``oracle_feasible``     may inspect the pre-generated *future* of the frame
                            and offloads exactly when the remote reply would in
                            fact arrive on time and fresh. It exists only to
                            bound how much a perfect-information scheduler could
                            gain, and it must never be described as deployable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .scheduler import (
    ALL_MODES,
    MODE_DEGRADED_SAFE,
    MODE_EDGE_ACCURATE,
    MODE_HYBRID,
    MODE_LOCAL_FAST,
    DapperScheduler,
    SchedulerDecision,
    SchedulerInputs,
    build_scheduler_from_config,
)

#: Policies reported in the main camera-ready table, in display order.
MAIN_POLICIES = ("local_only", "edge_only", "cloud_only", "dapper")
#: Adaptive comparators.
ADAPTIVE_POLICIES = ("deadline_greedy", "confidence_deadline", "rtt_threshold")
#: Offline upper bound. Reported separately and always labelled as an oracle.
ORACLE_POLICIES = ("oracle_feasible",)


def _decision(mode: str, reason: str, expected: float, predicted: float,
              attempt: bool, fallback: bool = False,
              risk: float = float("nan")) -> SchedulerDecision:
    return SchedulerDecision(
        selected_mode=mode,
        deadline_risk_score=risk,
        reason=reason,
        expected_latency_ms=expected,
        fallback_needed=fallback,
        predicted_remote_arrival_ms=predicted,
        remote_attempt_intended=attempt,
    )


class Policy:
    """Base class. ``decide`` is the only thing subclasses must implement."""

    name: str = "policy"
    backend: str = "edge"
    #: True only for policies that may read the pre-generated future.
    is_oracle: bool = False

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.local_estimate_ms = 0.5 * (float(cfg["local"]["latency_ms_min"])
                                        + float(cfg["local"]["latency_ms_max"]))

    def reset(self) -> None:
        """Clear any per-run state. Called once before each trace replay."""

    def observe(self, **kwargs) -> SchedulerInputs:
        return SchedulerInputs(
            rtt_ms=kwargs["rtt_ms"],
            packet_loss=kwargs["packet_loss"],
            edge_load=kwargs["edge_load"],
            deadline_ms=kwargs["deadline_ms"],
            frame_age_ms=kwargs["frame_age_ms"],
            last_valid_age_ms=kwargs["last_valid_age_ms"],
            local_confidence=kwargs["local_confidence"],
            edge_available=kwargs["edge_available"],
        )

    def decide(self, obs: SchedulerInputs, trace, i: int, model) -> SchedulerDecision:
        raise NotImplementedError

    # convenience ---------------------------------------------------------
    def predicted_remote_ms(self, obs: SchedulerInputs, model) -> float:
        """Nominal predicted remote arrival, using only observable quantities."""
        return (model.total_rtt_ms(obs.rtt_ms, self.backend)
                + model.nominal_compute_ms[self.backend]
                * (1.0 + model.load_compute_scale * obs.edge_load))


# --------------------------------------------------------------- fixed
class LocalOnlyPolicy(Policy):
    name = "local_only"

    def decide(self, obs, trace, i, model):
        return _decision(MODE_LOCAL_FAST, "fixed_baseline:local_only",
                         self.local_estimate_ms, float("inf"), False)


class _FixedRemotePolicy(Policy):
    def decide(self, obs, trace, i, model):
        predicted = self.predicted_remote_ms(obs, model)
        if not obs.edge_available and model.edge_health_check:
            return _decision(MODE_EDGE_ACCURATE, f"fixed_baseline:{self.name}:unreachable",
                             self.local_estimate_ms, predicted, False, fallback=True)
        return _decision(MODE_EDGE_ACCURATE, f"fixed_baseline:{self.name}",
                         predicted, predicted, True)


class EdgeOnlyPolicy(_FixedRemotePolicy):
    name = "edge_only"
    backend = "edge"


class CloudOnlyPolicy(_FixedRemotePolicy):
    name = "cloud_only"
    backend = "cloud"


# --------------------------------------------------------------- DAPPER
class DapperPolicy(Policy):
    """The DAPPER scheduler, wrapped as a policy."""

    name = "dapper"
    backend = "edge"

    def __init__(self, cfg: Dict[str, Any], scheduler: Optional[DapperScheduler] = None,
                 name: Optional[str] = None, **scheduler_kwargs):
        super().__init__(cfg)
        self.scheduler = scheduler or build_scheduler_from_config(cfg, **scheduler_kwargs)
        if name:
            self.name = name

    def decide(self, obs, trace, i, model):
        return self.scheduler.decide(obs)


# ---------------------------------------------------- adaptive baselines
@dataclass
class DeadlineGreedyParams:
    margin: float = 0.90


class DeadlineGreedyPolicy(Policy):
    """
    Offload iff the edge is reachable and the predicted completion fits
    ``margin * D``. No risk score, no confidence signal, no degraded reuse.
    """

    name = "deadline_greedy"
    backend = "edge"

    def __init__(self, cfg, margin: float = 0.90):
        super().__init__(cfg)
        self.margin = float(margin)

    def decide(self, obs, trace, i, model):
        predicted = self.predicted_remote_ms(obs, model)
        if obs.edge_available and predicted <= self.margin * obs.deadline_ms:
            return _decision(MODE_EDGE_ACCURATE, "deadline_greedy_edge",
                             predicted, predicted, True)
        return _decision(MODE_LOCAL_FAST, "deadline_greedy_local",
                         self.local_estimate_ms, predicted, False)


class ConfidenceDeadlinePolicy(Policy):
    """
    "Offload only when the local model is unconfident and the edge can finish on
    time." The strongest simple comparator for DAPPER's low-risk branch.
    """

    name = "confidence_deadline"
    backend = "edge"

    def __init__(self, cfg, confidence_threshold: float = 0.725, margin: float = 0.90):
        super().__init__(cfg)
        self.confidence_threshold = float(confidence_threshold)
        self.margin = float(margin)

    def decide(self, obs, trace, i, model):
        predicted = self.predicted_remote_ms(obs, model)
        if obs.local_confidence >= self.confidence_threshold:
            return _decision(MODE_LOCAL_FAST, "confidence_deadline_local_confident",
                             self.local_estimate_ms, predicted, False)
        if obs.edge_available and predicted <= self.margin * obs.deadline_ms:
            return _decision(MODE_EDGE_ACCURATE, "confidence_deadline_edge",
                             predicted, predicted, True)
        return _decision(MODE_LOCAL_FAST, "confidence_deadline_local_fallback",
                         self.local_estimate_ms, predicted, False)


class RttThresholdPolicy(Policy):
    """Offload iff the measured RTT is below a calibrated threshold."""

    name = "rtt_threshold"
    backend = "edge"

    def __init__(self, cfg, rtt_threshold_ms: float = 30.0):
        super().__init__(cfg)
        self.rtt_threshold_ms = float(rtt_threshold_ms)

    def decide(self, obs, trace, i, model):
        predicted = self.predicted_remote_ms(obs, model)
        if obs.edge_available and obs.rtt_ms <= self.rtt_threshold_ms:
            return _decision(MODE_EDGE_ACCURATE, "rtt_threshold_edge",
                             predicted, predicted, True)
        return _decision(MODE_LOCAL_FAST, "rtt_threshold_local",
                         self.local_estimate_ms, predicted, False)


# ------------------------------------------------------------- oracle
class OracleFeasiblePolicy(Policy):
    """
    OFFLINE ORACLE UPPER BOUND — NOT DEPLOYABLE.

    Reads the pre-generated frame outcome and offloads exactly when the remote
    reply would in fact arrive before the deadline, inside the freshness window,
    and without being lost. It therefore never wastes an offload and never pays
    a timeout. It bounds what any scheduler with perfect knowledge of the
    immediate future could achieve; it cannot be implemented on a robot.
    """

    name = "oracle_feasible"
    backend = "edge"
    is_oracle = True

    def decide(self, obs, trace, i, model):
        predicted = self.predicted_remote_ms(obs, model)
        if not obs.edge_available:
            return _decision(MODE_LOCAL_FAST, "oracle_unreachable_local",
                             self.local_estimate_ms, predicted, False)
        lost = float(trace.edge_loss_draw[i]) < float(trace.packet_loss[i])
        arrival = model.true_arrival_ms(float(trace.rtt_ms[i]),
                                        float(trace.edge_compute_ms[i]),
                                        float(trace.edge_load[i]), self.backend)
        timeout = model.client_timeout_ms(float(trace.rtt_ms[i]),
                                          float(trace.edge_load[i]), self.backend)
        feasible = (not lost) and arrival <= timeout and arrival <= model.deadline_ms
        better = float(trace.edge_confidence[i]) > float(trace.local_confidence[i])
        if feasible and better:
            return _decision(MODE_EDGE_ACCURATE, "oracle_edge_feasible",
                             arrival, predicted, True)
        return _decision(MODE_LOCAL_FAST, "oracle_local", self.local_estimate_ms,
                         predicted, False)


# ------------------------------------------------------------- factory
def build_policy(name: str, cfg: Dict[str, Any],
                 baseline_params: Optional[Dict[str, Any]] = None,
                 **dapper_kwargs) -> Policy:
    """Construct a policy by name."""
    bp = baseline_params or cfg.get("baselines", {})
    if name == "local_only":
        return LocalOnlyPolicy(cfg)
    if name == "edge_only":
        return EdgeOnlyPolicy(cfg)
    if name == "cloud_only":
        return CloudOnlyPolicy(cfg)
    if name == "dapper":
        return DapperPolicy(cfg, **dapper_kwargs)
    if name == "deadline_greedy":
        return DeadlineGreedyPolicy(cfg, margin=float(bp.get("deadline_greedy_margin", 0.90)))
    if name == "confidence_deadline":
        return ConfidenceDeadlinePolicy(
            cfg,
            confidence_threshold=float(bp.get("confidence_deadline_threshold", 0.725)),
            margin=float(bp.get("confidence_deadline_margin", 0.90)),
        )
    if name == "rtt_threshold":
        return RttThresholdPolicy(cfg, rtt_threshold_ms=float(bp.get("rtt_threshold_ms", 30.0)))
    if name == "oracle_feasible":
        return OracleFeasiblePolicy(cfg)
    raise ValueError(f"unknown policy: {name}")


ALL_POLICY_NAMES = MAIN_POLICIES + ADAPTIVE_POLICIES + ORACLE_POLICIES
