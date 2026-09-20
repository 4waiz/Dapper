"""
Unit tests for the DAPPER scheduler decision rule.

These check *behaviour* (which mode is selected, and why) rather than exact risk
values, so recalibrating the weights does not invalidate the suite. Every
decision path documented in ``dapper/scheduler.py`` has at least one test.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.config import load_config, renormalised_weights  # noqa: E402
from dapper.scheduler import (  # noqa: E402
    ALL_MODES,
    ALL_REASONS,
    MODE_DEGRADED_SAFE,
    MODE_EDGE_ACCURATE,
    MODE_HYBRID,
    MODE_LOCAL_FAST,
    DapperScheduler,
    SchedulerInputs,
    build_scheduler_from_config,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def scheduler(cfg):
    return build_scheduler_from_config(cfg)


def _inputs(**overrides) -> SchedulerInputs:
    base = dict(
        rtt_ms=20.0, packet_loss=0.01, edge_load=0.2, deadline_ms=100.0,
        frame_age_ms=0.0, last_valid_age_ms=50.0, local_confidence=0.72,
        edge_available=True,
    )
    base.update(overrides)
    return SchedulerInputs(**base)


# ------------------------------------------------------------------- risk
def test_risk_score_is_bounded(scheduler):
    """The risk score must lie in [0, 1] for any input, however extreme."""
    for rtt in (0.0, 50.0, 250.0, 9999.0):
        for loss in (0.0, 0.2, 0.99, 1.0):
            for load in (0.0, 0.5, 1.0):
                for age in (0.0, 1000.0):
                    r = scheduler.compute_risk(
                        _inputs(rtt_ms=rtt, packet_loss=loss, edge_load=load,
                                frame_age_ms=age))
                    assert 0.0 <= r <= 1.0


def test_risk_increases_with_each_signal(scheduler):
    """Every signal with non-zero weight must push the risk score upward."""
    base = scheduler.compute_risk(_inputs(rtt_ms=10.0, packet_loss=0.0,
                                          edge_load=0.0, frame_age_ms=0.0))
    assert scheduler.compute_risk(_inputs(rtt_ms=90.0, packet_loss=0.0,
                                          edge_load=0.0, frame_age_ms=0.0)) > base
    assert scheduler.compute_risk(_inputs(rtt_ms=10.0, packet_loss=0.8,
                                          edge_load=0.0, frame_age_ms=0.0)) > base
    assert scheduler.compute_risk(_inputs(rtt_ms=10.0, packet_loss=0.0,
                                          edge_load=1.0, frame_age_ms=0.0)) > base


def test_deadline_pressure_increases_risk(scheduler):
    relaxed = scheduler.compute_risk(_inputs(deadline_ms=500.0, rtt_ms=80.0))
    tight = scheduler.compute_risk(_inputs(deadline_ms=80.0, rtt_ms=80.0))
    assert tight > relaxed


# -------------------------------------------------------------- decisions
def test_low_risk_high_local_confidence_stays_local(scheduler):
    """Confidence gate: a confident local model keeps the frame on device."""
    d = scheduler.decide(_inputs(rtt_ms=5.0, packet_loss=0.0, edge_load=0.0,
                                 local_confidence=0.99))
    assert d.selected_mode == MODE_LOCAL_FAST
    assert d.reason == "local_confidence_sufficient"
    assert not d.remote_attempt_intended


def test_low_risk_low_confidence_enough_margin_selects_edge(scheduler):
    """Weak local result + remote fits the margin -> commit to the edge."""
    d = scheduler.decide(_inputs(rtt_ms=1.0, packet_loss=0.0, edge_load=0.0,
                                 local_confidence=0.0, deadline_ms=400.0))
    assert d.selected_mode == MODE_EDGE_ACCURATE
    assert d.reason == "low_confidence_edge_margin_ok"
    assert d.remote_attempt_intended


def test_low_risk_low_confidence_insufficient_margin_selects_hybrid(cfg):
    """
    Weak local result, remote does not fit the commit margin, but a refresh
    could still land inside the freshness window -> hybrid.
    """
    s = build_scheduler_from_config(cfg)
    s.remote_deadline_margin = 0.10          # nothing ever fits the commit margin
    s.hybrid_window_ms = 10_000.0            # but a refresh is always feasible
    d = s.decide(_inputs(rtt_ms=10.0, packet_loss=0.0, edge_load=0.0,
                         local_confidence=0.0))
    assert d.selected_mode == MODE_HYBRID
    assert d.reason == "low_confidence_hybrid_refresh"
    assert d.remote_attempt_intended


def test_low_risk_low_confidence_infeasible_refresh_stays_local(cfg):
    """
    A refresh that could never be accepted must not be sent: spending upload
    bandwidth on a reply that will be rejected on arrival is pure waste.
    """
    s = build_scheduler_from_config(cfg)
    s.remote_deadline_margin = 0.10
    s.hybrid_window_ms = 1.0
    d = s.decide(_inputs(rtt_ms=10.0, packet_loss=0.0, edge_load=0.0,
                         local_confidence=0.0))
    assert d.selected_mode == MODE_LOCAL_FAST
    assert d.reason == "low_confidence_refresh_infeasible"
    assert not d.remote_attempt_intended


def test_moderate_risk_prefers_local(scheduler):
    d = scheduler.decide(_inputs(rtt_ms=55.0, packet_loss=0.05, edge_load=0.5,
                                 local_confidence=0.0))
    assert scheduler.risk_local <= d.deadline_risk_score < scheduler.risk_degraded
    assert d.selected_mode == MODE_LOCAL_FAST
    assert d.reason == "risk_moderate_prefer_local"


def test_high_risk_selects_degraded_safe(scheduler):
    d = scheduler.decide(_inputs(rtt_ms=250.0, packet_loss=0.9, edge_load=1.0))
    assert d.deadline_risk_score >= scheduler.risk_degraded
    assert d.selected_mode == MODE_DEGRADED_SAFE
    assert d.fallback_needed


def test_high_risk_without_fresh_output_falls_back_to_local(scheduler):
    d = scheduler.decide(_inputs(rtt_ms=250.0, packet_loss=0.9, edge_load=1.0,
                                 last_valid_age_ms=1e9))
    assert d.selected_mode == MODE_DEGRADED_SAFE
    assert d.reason == "risk_high_local_fallback"


def test_edge_unavailable_with_fresh_output_reuses(scheduler):
    d = scheduler.decide(_inputs(edge_available=False, last_valid_age_ms=80.0))
    assert d.selected_mode == MODE_DEGRADED_SAFE
    assert d.reason == "edge_unavailable_reuse_fresh"
    assert not d.remote_attempt_intended


def test_edge_unavailable_with_expired_output_runs_local(scheduler):
    d = scheduler.decide(_inputs(edge_available=False, last_valid_age_ms=10_000.0))
    assert d.selected_mode == MODE_LOCAL_FAST
    assert d.reason == "edge_unavailable_local_fallback"
    assert d.fallback_needed


def test_no_remote_attempt_is_ever_intended_when_edge_is_down(scheduler):
    """A known-down edge must never be transmitted to."""
    for rtt in (5.0, 50.0, 500.0):
        for conf in (0.0, 0.5, 1.0):
            d = scheduler.decide(_inputs(edge_available=False, rtt_ms=rtt,
                                         local_confidence=conf))
            assert not d.remote_attempt_intended


# ------------------------------------------------------------- invariants
def test_every_decision_has_a_known_mode_and_reason(scheduler):
    """No decision may be unexplained: auditability is the design premise."""
    seen = set()
    for rtt in (1.0, 20.0, 60.0, 200.0, 800.0):
        for loss in (0.0, 0.1, 0.5, 0.95):
            for load in (0.0, 0.4, 1.0):
                for conf in (0.0, 0.70, 0.99):
                    for avail in (True, False):
                        for age in (10.0, 1e9):
                            d = scheduler.decide(_inputs(
                                rtt_ms=rtt, packet_loss=loss, edge_load=load,
                                local_confidence=conf, edge_available=avail,
                                last_valid_age_ms=age))
                            assert d.selected_mode in ALL_MODES
                            assert d.reason in ALL_REASONS
                            seen.add(d.reason)
    assert len(seen) >= 6, f"decision paths left untested: {set(ALL_REASONS) - seen}"


def test_decide_is_deterministic(scheduler):
    x = _inputs(rtt_ms=37.0, packet_loss=0.03, local_confidence=0.71)
    first = scheduler.decide(x)
    for _ in range(50):
        assert scheduler.decide(x) == first


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        DapperScheduler(weight_rtt=0.5, weight_loss=0.5, weight_load=0.5,
                        weight_frame_age=0.0, weight_deadline=0.0)


def test_weights_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        DapperScheduler(weight_rtt=1.2, weight_loss=-0.2, weight_load=0.0,
                        weight_frame_age=0.0, weight_deadline=0.0)


def test_local_threshold_must_be_below_degraded_threshold():
    with pytest.raises(ValueError, match="must be <"):
        DapperScheduler(risk_local_threshold=0.8, risk_degraded_threshold=0.5)


def test_confidence_threshold_must_be_a_probability():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DapperScheduler(local_confidence_threshold=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DapperScheduler(local_confidence_threshold=-0.1)


def test_remote_margin_must_be_a_fraction():
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        DapperScheduler(remote_deadline_margin=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        DapperScheduler(remote_deadline_margin=1.5)


def test_hybrid_estimator_is_validated():
    with pytest.raises(ValueError, match="hybrid_estimator"):
        DapperScheduler(hybrid_estimator="magic")


def test_renormalised_weights_sum_to_one(cfg):
    base = {k: float(cfg["scheduler"][k]) for k in
            ("weight_rtt", "weight_loss", "weight_load",
             "weight_frame_age", "weight_deadline")}
    for comp in ("rtt", "loss", "load", "frame_age", "deadline"):
        w = renormalised_weights(base, comp)
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert w[f"weight_{comp}"] == 0.0
        DapperScheduler(**w)  # must construct without raising


# ------------------------------------------------- prediction consistency
def test_prediction_uses_the_same_form_as_execution(cfg):
    """
    The scheduler's predicted arrival must equal the executor's true arrival
    when the scheduler's compute estimate happens to equal the true compute.
    A drift between the two models is exactly the bug this guards against.
    """
    from dapper.executor import ExecutionModel
    s = build_scheduler_from_config(cfg)
    model = ExecutionModel(cfg=cfg, deadline_ms=100.0)
    rtt, load = 41.0, 0.37
    predicted = s.predicted_remote_arrival_ms(rtt, load)
    true = model.true_arrival_ms(rtt, s.edge_compute_ms, load, "edge")
    assert predicted == pytest.approx(true, rel=1e-12)


def test_estimation_bias_changes_only_the_estimate(cfg):
    unbiased = build_scheduler_from_config(cfg)
    under = build_scheduler_from_config(cfg, rtt_estimate_bias=-0.30,
                                        compute_estimate_bias=-0.30)
    over = build_scheduler_from_config(cfg, rtt_estimate_bias=+0.30,
                                       compute_estimate_bias=+0.30)
    a = unbiased.predicted_remote_arrival_ms(50.0, 0.5)
    assert under.predicted_remote_arrival_ms(50.0, 0.5) == pytest.approx(0.7 * a)
    assert over.predicted_remote_arrival_ms(50.0, 0.5) == pytest.approx(1.3 * a)
    assert under.compute_risk(_inputs(rtt_ms=50.0)) < unbiased.compute_risk(_inputs(rtt_ms=50.0))
