"""
DAPPER: Deadline-Aware Perception Placement for Edge Robotics.

This package holds the corrected, camera-ready implementation. It supersedes the
flat top-level modules (`scheduler.py`, `benchmark.py`, ...) which are retained
only so that the originally published baseline in
`artifacts/baseline_original/` stays reproducible.

Module map:

    config      configuration load + validation (weights, thresholds, invariants)
    scenario    immutable pre-generated per-frame scenarios (paired evaluation)
    scheduler   the DAPPER risk-based, confidence-aware, deadline-aware policy
    policies    every evaluated policy, including adaptive baselines and an
                offline oracle upper bound
    executor    the shared timing / bandwidth / freshness accounting model
    metrics     per-run metric reduction
    stats       seed-level bootstrap confidence intervals
    trace       trace-driven network input
"""

__all__ = [
    "config",
    "scenario",
    "scheduler",
    "policies",
    "executor",
    "metrics",
    "stats",
    "trace",
]

__version__ = "2.0.0-camera-ready"
