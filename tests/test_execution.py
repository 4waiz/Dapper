"""
Tests for the execution model: timing, bandwidth, freshness and fallback.

These are the accounting rules the manuscript claims and the published
prototype did not implement, so each claim gets an explicit test.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.config import load_config, with_overrides  # noqa: E402
from dapper.executor import ExecutionModel, execute_run  # noqa: E402
from dapper.policies import Policy, build_policy  # noqa: E402
from dapper.scenario import ScenarioTrace, generate_scenarios  # noqa: E402
from dapper.scheduler import (  # noqa: E402
    MODE_DEGRADED_SAFE, MODE_EDGE_ACCURATE, MODE_HYBRID, MODE_LOCAL_FAST,
    SchedulerDecision,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class _ForcedPolicy(Policy):
    """A policy that always selects one mode, for exercising the executor."""

    def __init__(self, cfg, mode, backend="edge", name="forced"):
        super().__init__(cfg)
        self._mode = mode
        self.backend = backend
        self.name = name

    def decide(self, obs, trace, i, model):
        return SchedulerDecision(
            selected_mode=self._mode, deadline_risk_score=0.0, reason="forced",
            expected_latency_ms=0.0, fallback_needed=False,
            predicted_remote_arrival_ms=float("nan"), remote_attempt_intended=True)


def _trace(cfg, **arrays) -> ScenarioTrace:
    """Build a hand-specified one- or few-frame trace."""
    n = len(next(iter(arrays.values())))
    base = {
        "rtt_ms": np.full(n, 20.0), "packet_loss": np.zeros(n),
        "edge_load": np.zeros(n), "edge_available": np.ones(n, dtype=bool),
        "local_latency_ms": np.full(n, 20.0), "local_confidence": np.full(n, 0.70),
        "local_detections": np.zeros(n, dtype=np.int32),
        "edge_compute_ms": np.full(n, 45.0), "edge_confidence": np.full(n, 0.90),
        "edge_detections": np.zeros(n, dtype=np.int32),
        "edge_loss_draw": np.full(n, 0.99),
        "cloud_compute_ms": np.full(n, 60.0), "cloud_confidence": np.full(n, 0.92),
        "cloud_detections": np.zeros(n, dtype=np.int32),
        "cloud_loss_draw": np.full(n, 0.99),
    }
    base.update({k: np.asarray(v) for k, v in arrays.items()})
    return ScenarioTrace(seed=0, profile="unit", arrays=base,
                         frame_period_ms=float(cfg["execution"]["frame_period_ms"]))


# ------------------------------------------------------------- bandwidth
def test_lost_offload_is_still_charged_upload_bandwidth(cfg):
    """
    The manuscript claims every remote attempt is billed its upload bandwidth,
    including lost ones. The published prototype billed none of them.
    """
    tr = _trace(cfg, packet_loss=np.array([1.0]), edge_loss_draw=np.array([0.0]))
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_EDGE_ACCURATE), cfg, 100.0)
    row = df.iloc[0]
    assert bool(row["remote_attempted"])
    assert bool(row["remote_failed_loss"])
    assert not bool(row["remote_accepted"])
    assert row["bandwidth_kb"] == pytest.approx(float(cfg["edge"]["bandwidth_kb"]))


def test_late_offload_is_still_charged_upload_bandwidth(cfg):
    """A reply that arrives too late to accept has still crossed the network."""
    tr = _trace(cfg, edge_compute_ms=np.array([5000.0]))
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_EDGE_ACCURATE), cfg, 100.0)
    row = df.iloc[0]
    assert bool(row["remote_attempted"]) and bool(row["remote_succeeded"])
    assert bool(row["remote_rejected_deadline"])
    assert row["bandwidth_kb"] == pytest.approx(float(cfg["edge"]["bandwidth_kb"]))


def test_rejected_hybrid_refresh_is_still_charged(cfg):
    tr = _trace(cfg, edge_compute_ms=np.array([5000.0]))
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_HYBRID), cfg, 100.0)
    row = df.iloc[0]
    assert bool(row["remote_attempted"])
    assert not bool(row["remote_refresh_accepted"])
    assert row["bandwidth_kb"] == pytest.approx(float(cfg["edge"]["bandwidth_kb"]))


def test_unreachable_edge_is_not_transmitted_to(cfg):
    """
    The health check is observable before transmission, so an offload to a
    known-down edge sends nothing and costs nothing. This keeps the fixed
    baselines from being strawmen.
    """
    tr = _trace(cfg, edge_available=np.array([False]))
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_EDGE_ACCURATE), cfg, 100.0)
    row = df.iloc[0]
    assert not bool(row["remote_attempted"])
    assert bool(row["remote_failed_unavailable"])
    assert row["bandwidth_kb"] == 0.0
    assert row["control_output_latency_ms"] == pytest.approx(20.0)


def test_local_and_degraded_modes_never_use_bandwidth(cfg):
    tr = _trace(cfg, rtt_ms=np.full(3, 20.0))
    for mode in (MODE_LOCAL_FAST, MODE_DEGRADED_SAFE):
        df = execute_run(tr, _ForcedPolicy(cfg, mode), cfg, 100.0)
        assert float(df["bandwidth_kb"].sum()) == 0.0


# ---------------------------------------------------------------- timing
def test_failed_offload_pays_a_timeout_before_falling_back(cfg):
    """
    A lost request is not free: the client cannot know it was lost until its
    response timer fires, and only then can it run the local model.
    """
    tr = _trace(cfg, packet_loss=np.array([1.0]), edge_loss_draw=np.array([0.0]))
    model = ExecutionModel(cfg=cfg, deadline_ms=100.0)
    timeout = model.client_timeout_ms(20.0, 0.0, "edge")
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_EDGE_ACCURATE), cfg, 100.0)
    assert df.iloc[0]["control_output_latency_ms"] == pytest.approx(timeout + 20.0)
    assert timeout > 0.0


def test_client_timeout_never_exceeds_the_deadline(cfg):
    model = ExecutionModel(cfg=cfg, deadline_ms=100.0)
    for rtt in (1.0, 50.0, 500.0, 5000.0):
        for load in (0.0, 0.5, 1.0):
            assert model.client_timeout_ms(rtt, load, "edge") <= 100.0
            assert model.client_timeout_ms(rtt, load, "cloud") <= 100.0


def test_access_rtt_is_applied_consistently_to_both_backends(cfg):
    """
    The published prototype added the static access RTT for cloud only. Both
    backends must now use one expression, and it must be the same one the
    scheduler predicts with.
    """
    model = ExecutionModel(cfg=cfg, deadline_ms=100.0)
    for backend in ("edge", "cloud"):
        extra = float(cfg[backend]["extra_rtt_ms"])
        assert model.total_rtt_ms(30.0, backend) == pytest.approx(30.0 + extra)
        assert model.true_arrival_ms(30.0, 50.0, 0.0, backend) == pytest.approx(
            30.0 + extra + 50.0)
    assert model.total_rtt_ms(30.0, "cloud") > model.total_rtt_ms(30.0, "edge")


def test_edge_load_scales_compute_identically_in_both_models(cfg):
    from dapper.scheduler import build_scheduler_from_config
    model = ExecutionModel(cfg=cfg, deadline_ms=100.0)
    s = build_scheduler_from_config(cfg)
    scale = float(cfg["execution"]["load_compute_scale"])
    load = 0.6
    assert model.true_arrival_ms(10.0, 50.0, load, "edge") == pytest.approx(
        10.0 + 50.0 * (1 + scale * load))
    assert s.predicted_remote_arrival_ms(10.0, load) == pytest.approx(
        10.0 + s.edge_compute_ms * (1 + scale * load))


def test_hybrid_control_latency_is_the_local_latency(cfg):
    """Hybrid always has a local answer in hand; the refresh is a bonus."""
    tr = _trace(cfg, local_latency_ms=np.array([22.0]),
                edge_compute_ms=np.array([30.0]))
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_HYBRID), cfg, 100.0)
    row = df.iloc[0]
    assert row["control_output_latency_ms"] == pytest.approx(22.0)
    assert bool(row["remote_refresh_accepted"])
    assert row["final_output_latency_ms"] == pytest.approx(20.0 + 30.0)
    assert row["final_output_latency_ms"] > row["control_output_latency_ms"]
    assert row["deadline_confidence"] == pytest.approx(0.90)


def test_edge_accurate_control_latency_is_the_remote_arrival(cfg):
    tr = _trace(cfg, edge_compute_ms=np.array([30.0]))
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_EDGE_ACCURATE), cfg, 100.0)
    row = df.iloc[0]
    assert row["control_output_latency_ms"] == pytest.approx(50.0)
    assert row["final_output_latency_ms"] == pytest.approx(50.0)
    assert row["output_source"] == "edge"


# ------------------------------------------------------------- freshness
def test_reused_output_inside_the_window_is_fresh_not_stale(cfg):
    """
    The published prototype flagged every reuse as stale, including reuse that
    its own freshness rule had just declared valid.
    """
    tr = _trace(cfg, rtt_ms=np.full(4, 20.0))
    df = execute_run(tr, _ForcedPolicy(cfg, MODE_DEGRADED_SAFE), cfg, 100.0)
    reused = df[df["output_reused"]]
    assert len(reused) >= 1
    assert bool(reused["freshness_valid"].all())
    assert not bool(reused["stale_output_accepted"].any())
    assert float(reused["output_age_ms"].max()) <= float(
        cfg["scheduler"]["last_valid_freshness_ms"])


def test_expired_output_is_never_reused(cfg):
    """Beyond the freshness window the scheduler must recompute, not reuse."""
    tight = with_overrides(cfg, last_valid_freshness_ms=1.0)
    tr = _trace(tight, rtt_ms=np.full(5, 20.0))
    df = execute_run(tr, _ForcedPolicy(tight, MODE_DEGRADED_SAFE), tight, 100.0)
    assert not bool(df["output_reused"].any())
    assert float(df["stale_output_accepted"].sum()) == 0.0
    assert set(df["output_source"]) == {"local"}


def test_output_only_becomes_available_after_it_is_computed(cfg):
    """
    An 80 ms result cannot be reused by a frame captured 33 ms later. The
    published prototype timestamped results at their own capture instant.
    """
    slow = with_overrides(cfg, last_valid_freshness_ms=10_000.0)
    tr = _trace(slow, local_latency_ms=np.array([90.0, 5.0, 5.0]))
    period = float(slow["execution"]["frame_period_ms"])
    df = execute_run(tr, _ForcedPolicy(slow, MODE_DEGRADED_SAFE), slow, 1000.0)
    # Frame 0 has nothing to reuse and computes locally, finishing at t = 90 ms.
    assert not bool(df.iloc[0]["output_reused"])
    # Frame 1 is captured at 33.3 ms, before frame 0's output exists at 90 ms,
    # so it must recompute rather than reuse a result from the future.
    assert not bool(df.iloc[1]["output_reused"])
    # Frame 2 is captured at 66.7 ms. Frame 0's 90 ms output still does not
    # exist, but frame 1's 5 ms output (available at 38.3 ms) does, so the
    # reused content must be frame 1's, aged by exactly one frame period.
    assert bool(df.iloc[2]["output_reused"])
    assert df.iloc[2]["output_age_ms"] == pytest.approx(period, rel=1e-9)


def test_stale_acceptance_is_zero_across_every_profile_and_policy(cfg):
    """The strongest form of the freshness claim, checked end to end."""
    for profile in cfg["profiles"]:
        tr = generate_scenarios(profile, cfg, 7, 400)
        for name in ("local_only", "edge_only", "cloud_only", "dapper",
                     "deadline_greedy", "confidence_deadline", "rtt_threshold"):
            df = execute_run(tr, build_policy(name, cfg), cfg, 100.0)
            assert float(df["stale_output_accepted"].sum()) == 0.0, (profile, name)
            assert bool(df["freshness_valid"].all()), (profile, name)


# ---------------------------------------------------------- bookkeeping
def test_latency_columns_are_ordered_and_finite(cfg):
    for profile in cfg["profiles"]:
        tr = generate_scenarios(profile, cfg, 3, 300)
        for name in ("dapper", "edge_only", "local_only"):
            df = execute_run(tr, build_policy(name, cfg), cfg, 100.0)
            assert np.isfinite(df["control_output_latency_ms"]).all()
            assert np.isfinite(df["final_output_latency_ms"]).all()
            assert (df["final_output_latency_ms"]
                    >= df["control_output_latency_ms"] - 1e-9).all()
            assert (df["control_output_latency_ms"] > 0).all()


def test_deadline_met_is_derived_from_the_control_latency(cfg):
    tr = generate_scenarios("variable", cfg, 11, 500)
    df = execute_run(tr, build_policy("dapper", cfg), cfg, 100.0)
    expected = df["control_output_latency_ms"] <= df["deadline_ms"]
    assert bool((df["deadline_met"].astype(bool) == expected).all())


def test_remote_accounting_flags_are_mutually_consistent(cfg):
    for profile in cfg["profiles"]:
        tr = generate_scenarios(profile, cfg, 5, 400)
        df = execute_run(tr, build_policy("edge_only", cfg), cfg, 100.0)
        att = df["remote_attempted"].astype(bool)
        # Nothing can succeed, be accepted or be rejected without being sent.
        for col in ("remote_succeeded", "remote_accepted",
                    "remote_rejected_deadline", "remote_rejected_freshness",
                    "remote_failed_loss"):
            assert not bool((df[col].astype(bool) & ~att).any()), (profile, col)
        # Bandwidth is charged exactly when a request was transmitted.
        assert bool(((df["bandwidth_kb"] > 0) == att).all()), profile
        # Unreachable and transmitted are mutually exclusive.
        assert not bool((df["remote_failed_unavailable"].astype(bool) & att).any())
