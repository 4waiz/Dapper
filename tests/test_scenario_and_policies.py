"""
Tests for paired scenario generation, determinism, policies and configuration.

The central property under test is that *every policy sees exactly the same
per-frame future*. Without it the multi-seed paired analysis is invalid.
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

from dapper.config import (  # noqa: E402
    ConfigError, load_config, validate_config, with_overrides,
)
from dapper.executor import execute_run  # noqa: E402
from dapper.metrics import compute_metrics  # noqa: E402
from dapper.policies import ALL_POLICY_NAMES, build_policy  # noqa: E402
from dapper.scenario import (  # noqa: E402
    STREAM_NAMES, generate_scenarios, load_scenarios, save_scenarios,
)
from dapper.stats import bootstrap_ci, paired_difference  # noqa: E402
from dapper.trace import load_network_trace, synthetic_trace_from_profile, write_network_trace  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------- determinism
def test_same_seed_reproduces_the_scenario_exactly(cfg):
    a = generate_scenarios("variable", cfg, 101, 500)
    b = generate_scenarios("variable", cfg, 101, 500)
    assert a.fingerprint() == b.fingerprint()
    for name in ("rtt_ms", "local_confidence", "edge_compute_ms", "edge_loss_draw"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_different_seed_changes_the_scenario(cfg):
    a = generate_scenarios("variable", cfg, 101, 500)
    b = generate_scenarios("variable", cfg, 102, 500)
    assert a.fingerprint() != b.fingerprint()
    assert not np.array_equal(np.asarray(a.rtt_ms), np.asarray(b.rtt_ms))


def test_different_profile_changes_the_scenario(cfg):
    a = generate_scenarios("stable", cfg, 101, 500)
    b = generate_scenarios("congested", cfg, 101, 500)
    assert a.fingerprint() != b.fingerprint()
    assert float(np.mean(b.rtt_ms)) > float(np.mean(a.rtt_ms))


def test_same_seed_reproduces_the_run_exactly(cfg):
    tr = generate_scenarios("variable", cfg, 101, 500)
    one = execute_run(tr, build_policy("dapper", cfg), cfg, 100.0)
    two = execute_run(tr, build_policy("dapper", cfg), cfg, 100.0)
    pd.testing.assert_frame_equal(one, two)


def test_scenarios_can_be_saved_and_replayed(cfg, tmp_path):
    tr = generate_scenarios("lossy", cfg, 101, 200)
    path = save_scenarios(tr, str(tmp_path / "seed_101_lossy.csv"))
    back = load_scenarios(path, cfg)
    assert back.fingerprint() == tr.fingerprint()
    a = execute_run(tr, build_policy("dapper", cfg), cfg, 100.0)
    b = execute_run(back, build_policy("dapper", cfg), cfg, 100.0)
    pd.testing.assert_frame_equal(
        a.drop(columns=["profile"]), b.drop(columns=["profile"]))


# ------------------------------------------------------------- pairing
def test_every_policy_replays_the_identical_scenario(cfg):
    """
    The defect this guards against: in the published prototype each mode drew a
    different number of random values, so a policy that branched differently
    changed the random future for all later frames.
    """
    tr = generate_scenarios("variable", cfg, 101, 500)
    before = tr.fingerprint()
    logs = {}
    for name in ALL_POLICY_NAMES:
        logs[name] = execute_run(tr, build_policy(name, cfg), cfg, 100.0)
        assert tr.fingerprint() == before, f"{name} mutated the scenario"
    for name, df in logs.items():
        for col in ("rtt_ms", "packet_loss", "edge_load", "edge_available"):
            assert np.array_equal(df[col].to_numpy(),
                                  np.asarray(getattr(tr, col))), (name, col)


def test_local_outcome_is_identical_wherever_a_policy_runs_locally(cfg):
    """Paired potential outcomes: the same frame yields the same local result."""
    tr = generate_scenarios("congested", cfg, 103, 400)
    a = execute_run(tr, build_policy("local_only", cfg), cfg, 100.0)
    b = execute_run(tr, build_policy("dapper", cfg), cfg, 100.0)
    same = b["output_source"] == "local"
    assert same.sum() > 50
    assert np.allclose(a.loc[same, "control_output_latency_ms"],
                       b.loc[same, "control_output_latency_ms"])
    assert np.allclose(a.loc[same, "output_confidence"],
                       b.loc[same, "output_confidence"])


def test_scenario_streams_are_independent_of_frame_count(cfg):
    """A longer trace must extend a shorter one, not replace it."""
    short = generate_scenarios("stable", cfg, 55, 100)
    long = generate_scenarios("stable", cfg, 55, 400)
    assert np.allclose(np.asarray(short.local_confidence),
                       np.asarray(long.local_confidence)[:100])
    assert np.allclose(np.asarray(short.edge_compute_ms),
                       np.asarray(long.edge_compute_ms)[:100])


def test_stream_names_are_unique():
    assert len(set(STREAM_NAMES)) == len(STREAM_NAMES)


# -------------------------------------------------------------- policies
def test_every_named_policy_builds_and_runs(cfg):
    tr = generate_scenarios("variable", cfg, 104, 200)
    for name in ALL_POLICY_NAMES:
        p = build_policy(name, cfg)
        assert p.name == name
        df = execute_run(tr, p, cfg, 100.0)
        assert len(df) == 200
        m = compute_metrics(df, df.attrs["mode_switches"])
        assert 0.0 <= m.deadline_miss_rate <= 1.0
        assert m.frames == 200


def test_unknown_policy_is_rejected(cfg):
    with pytest.raises(ValueError, match="unknown policy"):
        build_policy("magic_policy", cfg)


def test_cloud_only_is_never_cheaper_than_edge_only(cfg):
    """The cloud is farther, slower and more expensive by construction."""
    tr = generate_scenarios("stable", cfg, 105, 500)
    e = compute_metrics(execute_run(tr, build_policy("edge_only", cfg), cfg, 100.0))
    c = compute_metrics(execute_run(tr, build_policy("cloud_only", cfg), cfg, 100.0))
    assert c.mean_latency_ms >= e.mean_latency_ms
    assert c.total_bandwidth_kb >= e.total_bandwidth_kb


def test_oracle_is_flagged_as_an_oracle(cfg):
    p = build_policy("oracle_feasible", cfg)
    assert p.is_oracle is True
    assert all(not build_policy(n, cfg).is_oracle
               for n in ALL_POLICY_NAMES if n != "oracle_feasible")


def test_oracle_never_wastes_an_offload(cfg):
    """By definition the oracle only offloads when the reply will be accepted."""
    tr = generate_scenarios("lossy", cfg, 106, 500)
    df = execute_run(tr, build_policy("oracle_feasible", cfg), cfg, 100.0)
    att = df["remote_attempted"].astype(bool)
    if att.any():
        assert bool(df.loc[att, "remote_accepted"].astype(bool).all())
        assert float(df["remote_failed_loss"].sum()) == 0.0


def test_adaptive_baselines_respond_to_their_hyperparameters(cfg):
    tr = generate_scenarios("stable", cfg, 107, 500)
    never = build_policy("rtt_threshold", cfg, baseline_params={"rtt_threshold_ms": 0.0})
    always = build_policy("rtt_threshold", cfg, baseline_params={"rtt_threshold_ms": 1e9})
    a = execute_run(tr, never, cfg, 100.0)
    b = execute_run(tr, always, cfg, 100.0)
    assert float(a["remote_attempted"].sum()) == 0.0
    assert float(b["remote_attempted"].sum()) > 0.0


# ------------------------------------------------------------- config
def test_default_config_is_valid(cfg):
    validate_config(cfg)


def test_weights_that_do_not_sum_to_one_are_rejected(cfg):
    with pytest.raises(ConfigError, match="sum to 1"):
        with_overrides(cfg, weight_rtt=0.9)


def test_inverted_thresholds_are_rejected(cfg):
    with pytest.raises(ConfigError, match="strictly less"):
        with_overrides(cfg, risk_local_threshold=0.9, risk_degraded_threshold=0.5)


def test_out_of_range_confidence_threshold_is_rejected(cfg):
    with pytest.raises(ConfigError, match=r"\[0, 1\]"):
        with_overrides(cfg, local_confidence_threshold=1.7)


def test_out_of_range_remote_margin_is_rejected(cfg):
    with pytest.raises(ConfigError, match=r"\(0, 1\]"):
        with_overrides(cfg, remote_deadline_margin=2.0)


def test_with_overrides_does_not_mutate_the_frozen_config(cfg):
    before = dict(cfg["scheduler"])
    with_overrides(cfg, risk_local_threshold=0.31)
    assert cfg["scheduler"] == before


def test_calibration_and_evaluation_seeds_are_disjoint(cfg):
    cal = set(int(s) for s in cfg["calibration_seeds"])
    start = int(cfg["evaluation_seeds_start"])
    ev = set(range(start, start + int(cfg["evaluation_seeds_count"])))
    assert not (cal & ev)


# --------------------------------------------------------------- stats
def test_bootstrap_ci_brackets_the_mean():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ci = bootstrap_ci(x, n_boot=2000)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["n"] == 10
    assert bootstrap_ci(x, n_boot=2000)["lo"] == ci["lo"]  # deterministic


def test_paired_difference_is_zero_for_a_policy_against_itself(cfg):
    rows = []
    for seed in range(101, 106):
        tr = generate_scenarios("variable", cfg, seed, 200)
        for label in ("a", "b"):
            df = execute_run(tr, build_policy("dapper", cfg), cfg, 100.0)
            m = compute_metrics(df, df.attrs["mode_switches"])
            d = m.__dict__.copy()
            d["policy"] = label
            rows.append(d)
    per_run = pd.DataFrame(rows)
    out = paired_difference(per_run, "p95_latency_ms", "a", "b",
                            profile="variable", n_boot=500)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert not out["excludes_zero"]


# --------------------------------------------------------------- traces
def test_trace_round_trip_drives_the_benchmark(cfg, tmp_path):
    samples = synthetic_trace_from_profile("variable", cfg, 200, 300)
    path = write_network_trace(str(tmp_path / "synthetic_variable.csv"), samples)
    over = load_network_trace(path, 300, float(cfg["execution"]["frame_period_ms"]))
    tr = generate_scenarios("variable", cfg, 200, 300,
                            network_override=over, source="trace")
    assert np.allclose(np.asarray(tr.rtt_ms), samples["rtt_ms"].to_numpy())
    assert tr.source == "trace"
    df = execute_run(tr, build_policy("dapper", cfg), cfg, 100.0)
    assert len(df) == 300


def test_trace_missing_columns_are_rejected(cfg, tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp_ms": [0, 1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_network_trace(str(p), 2, 33.3)
