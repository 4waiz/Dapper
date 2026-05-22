"""
Unit tests for the DAPPER scheduler.

These tests check the *behaviour* (which mode is chosen under which
conditions) rather than the exact numeric risk score, so that tuning the
weights does not invalidate the tests.
"""

from __future__ import annotations

import os
import sys

# Allow `pytest` to find the modules without installing the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from scheduler import (
    DapperScheduler,
    SchedulerInputs,
    MODE_LOCAL_FAST,
    MODE_EDGE_ACCURATE,
    MODE_HYBRID,
    MODE_DEGRADED_SAFE,
)


@pytest.fixture
def scheduler() -> DapperScheduler:
    """Default scheduler matching config.yaml defaults."""
    return DapperScheduler(
        weight_rtt=0.35,
        weight_loss=0.30,
        weight_load=0.15,
        weight_frame_age=0.05,
        weight_deadline=0.15,
        risk_degraded_threshold=0.70,
        risk_local_threshold=0.40,
        hybrid_freshness_window_ms=80.0,
        last_valid_freshness_ms=250.0,
        edge_compute_ms_estimate=65.0,
        local_compute_ms_estimate=25.0,
    )


def _inputs(**overrides) -> SchedulerInputs:
    """Builder for SchedulerInputs with sensible defaults."""
    base = dict(
        rtt_ms=20.0,
        packet_loss=0.01,
        edge_load=0.2,
        deadline_ms=100.0,
        frame_age_ms=0.0,
        last_valid_age_ms=50.0,
        local_confidence=0.72,
        edge_available=True,
    )
    base.update(overrides)
    return SchedulerInputs(**base)


def test_stable_network_chooses_edge_or_hybrid(scheduler):
    """A clean network should let DAPPER use the more accurate path."""
    decision = scheduler.decide(_inputs())
    assert decision.selected_mode in (MODE_EDGE_ACCURATE, MODE_HYBRID)
    assert decision.deadline_risk_score < 0.5
    assert not decision.fallback_needed


def test_high_rtt_chooses_local_or_degraded(scheduler):
    """When the network alone burns most of the deadline, stay local."""
    decision = scheduler.decide(_inputs(rtt_ms=140.0))
    assert decision.selected_mode in (MODE_LOCAL_FAST, MODE_DEGRADED_SAFE)


def test_high_loss_chooses_local_or_degraded(scheduler):
    """High packet loss should also push us away from edge."""
    decision = scheduler.decide(_inputs(packet_loss=0.6))
    assert decision.selected_mode in (MODE_LOCAL_FAST, MODE_DEGRADED_SAFE)


def test_outage_chooses_degraded_when_recent_valid_available(scheduler):
    """Edge unavailable + recent last_valid -> degraded_safe with reuse."""
    decision = scheduler.decide(_inputs(edge_available=False, last_valid_age_ms=80.0))
    assert decision.selected_mode == MODE_DEGRADED_SAFE
    assert decision.fallback_needed


def test_outage_falls_back_to_local_when_no_fresh_valid(scheduler):
    """Edge unavailable + stale last_valid -> local fallback."""
    decision = scheduler.decide(
        _inputs(edge_available=False, last_valid_age_ms=10_000.0)
    )
    assert decision.selected_mode == MODE_LOCAL_FAST
    assert decision.fallback_needed


def test_deadline_pressure_increases_risk(scheduler):
    """Tighter deadlines should produce higher risk scores."""
    relaxed = scheduler.compute_risk(_inputs(deadline_ms=500.0, rtt_ms=80.0))
    tight = scheduler.compute_risk(_inputs(deadline_ms=80.0, rtt_ms=80.0))
    assert tight > relaxed


def test_hybrid_selected_when_edge_total_overshoots_window(scheduler):
    """If edge total > hybrid window but risk is still low, prefer hybrid."""
    # RTT 40 + edge compute estimate 65 = 105 ms total, > 80 ms window
    decision = scheduler.decide(_inputs(rtt_ms=40.0, packet_loss=0.0, edge_load=0.1))
    assert decision.selected_mode == MODE_HYBRID


def test_edge_selected_when_total_fits_window(scheduler):
    """Very low RTT should let pure edge mode be chosen."""
    # Force edge_compute_ms_estimate low for this test
    s = DapperScheduler(
        edge_compute_ms_estimate=10.0,
        local_compute_ms_estimate=20.0,
    )
    decision = s.decide(_inputs(rtt_ms=5.0, packet_loss=0.0, edge_load=0.05))
    assert decision.selected_mode == MODE_EDGE_ACCURATE


def test_risk_score_is_bounded(scheduler):
    """Risk score must always lie in [0, 1]."""
    for r in (0.0, 50.0, 250.0, 9999.0):
        for l in (0.0, 0.2, 0.99):
            for ld in (0.0, 0.5, 1.0):
                inputs = _inputs(rtt_ms=r, packet_loss=l, edge_load=ld)
                risk = scheduler.compute_risk(inputs)
                assert 0.0 <= risk <= 1.0


def test_decision_has_human_readable_reason(scheduler):
    """Every decision must carry a non-empty reason string for auditing."""
    decision = scheduler.decide(_inputs())
    assert isinstance(decision.reason, str) and len(decision.reason) > 0
