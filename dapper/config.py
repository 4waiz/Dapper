"""
Configuration loading and validation.

Every parameter that influences a reported number lives in `config.yaml`. This
module is the single place where the configuration contract is enforced, so a
malformed or internally inconsistent configuration fails loudly instead of
silently producing an unexplainable result.

Enforced invariants
-------------------
* the five risk weights are non-negative and sum to 1 (within tolerance);
* ``risk_local_threshold < risk_degraded_threshold``;
* both thresholds and ``local_confidence_threshold`` lie in [0, 1];
* ``remote_deadline_margin`` lies in (0, 1];
* freshness windows and timing constants are positive;
* every declared profile has a definition in ``network_profiles``.
"""

from __future__ import annotations

import copy
import io
import os
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")

RISK_WEIGHT_KEYS = (
    "weight_rtt",
    "weight_loss",
    "weight_load",
    "weight_frame_age",
    "weight_deadline",
)

#: Risk components, in the order used for one-at-a-time ablation.
RISK_COMPONENTS = ("rtt", "loss", "load", "frame_age", "deadline")

_REQUIRED_SECTIONS = ("local", "edge", "cloud", "network_profiles", "scheduler", "execution")


class ConfigError(ValueError):
    """Raised when config.yaml violates the documented contract."""


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``config.yaml`` and validate it."""
    path = path or DEFAULT_CONFIG_PATH
    with io.open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Check every documented invariant. Returns the config for chaining."""
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            raise ConfigError(f"config missing required section: {section}")

    sch = cfg["scheduler"]

    weights = []
    for key in RISK_WEIGHT_KEYS:
        if key not in sch:
            raise ConfigError(f"scheduler.{key} is required")
        w = float(sch[key])
        if w < 0.0:
            raise ConfigError(f"scheduler.{key} must be non-negative, got {w}")
        weights.append(w)
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(
            f"risk weights must sum to 1.0, got {total:.9f} "
            f"({dict(zip(RISK_WEIGHT_KEYS, weights))})"
        )

    local_thr = float(sch["risk_local_threshold"])
    degraded_thr = float(sch["risk_degraded_threshold"])
    for name, v in (("risk_local_threshold", local_thr),
                    ("risk_degraded_threshold", degraded_thr)):
        if not 0.0 <= v <= 1.0:
            raise ConfigError(f"scheduler.{name} must lie in [0, 1], got {v}")
    if not local_thr < degraded_thr:
        raise ConfigError(
            f"scheduler.risk_local_threshold ({local_thr}) must be strictly less "
            f"than scheduler.risk_degraded_threshold ({degraded_thr})"
        )

    conf_thr = float(sch["local_confidence_threshold"])
    if not 0.0 <= conf_thr <= 1.0:
        raise ConfigError(
            f"scheduler.local_confidence_threshold must lie in [0, 1], got {conf_thr}"
        )

    margin = float(sch["remote_deadline_margin"])
    if not 0.0 < margin <= 1.0:
        raise ConfigError(
            f"scheduler.remote_deadline_margin must lie in (0, 1], got {margin}"
        )

    est = str(sch.get("hybrid_estimator", "optimistic"))
    if est not in ("optimistic", "expected"):
        raise ConfigError(
            f"scheduler.hybrid_estimator must be 'optimistic' or 'expected', got {est!r}")

    for name in ("hybrid_freshness_window_ms", "last_valid_freshness_ms"):
        v = float(sch[name])
        if v <= 0.0:
            raise ConfigError(f"scheduler.{name} must be positive, got {v}")

    ex = cfg["execution"]
    for name in ("frame_period_ms", "reuse_latency_ms"):
        if float(ex[name]) <= 0.0:
            raise ConfigError(f"execution.{name} must be positive, got {ex[name]}")
    if float(ex["load_compute_scale"]) < 0.0:
        raise ConfigError("execution.load_compute_scale must be non-negative")
    if float(ex["remote_timeout_slack_ms"]) < 0.0:
        raise ConfigError("execution.remote_timeout_slack_ms must be non-negative")

    for backend in ("local", "edge", "cloud"):
        sec = cfg[backend]
        if float(sec["latency_ms_min"]) > float(sec["latency_ms_max"]):
            raise ConfigError(f"{backend}.latency_ms_min exceeds latency_ms_max")
        if float(sec["confidence_min"]) > float(sec["confidence_max"]):
            raise ConfigError(f"{backend}.confidence_min exceeds confidence_max")
        for k in ("confidence_min", "confidence_max"):
            if not 0.0 <= float(sec[k]) <= 1.0:
                raise ConfigError(f"{backend}.{k} must lie in [0, 1]")

    for profile in cfg.get("profiles", []):
        if profile not in cfg["network_profiles"]:
            raise ConfigError(f"profile '{profile}' has no network_profiles entry")

    return cfg


def scheduler_params(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Flatten the scheduler section into plain floats."""
    sch = cfg["scheduler"]
    return {
        "weight_rtt": float(sch["weight_rtt"]),
        "weight_loss": float(sch["weight_loss"]),
        "weight_load": float(sch["weight_load"]),
        "weight_frame_age": float(sch["weight_frame_age"]),
        "weight_deadline": float(sch["weight_deadline"]),
        "risk_local_threshold": float(sch["risk_local_threshold"]),
        "risk_degraded_threshold": float(sch["risk_degraded_threshold"]),
        "local_confidence_threshold": float(sch["local_confidence_threshold"]),
        "remote_deadline_margin": float(sch["remote_deadline_margin"]),
        "hybrid_freshness_window_ms": float(sch["hybrid_freshness_window_ms"]),
        "hybrid_estimator": str(sch.get("hybrid_estimator", "optimistic")),
        "last_valid_freshness_ms": float(sch["last_valid_freshness_ms"]),
    }


def with_overrides(cfg: Dict[str, Any], **scheduler_overrides: Any) -> Dict[str, Any]:
    """
    Return a deep copy of ``cfg`` with ``scheduler.*`` keys replaced.

    Used by the calibration, ablation and sensitivity drivers so that no
    experiment can mutate the frozen configuration in place.
    """
    new = copy.deepcopy(cfg)
    new["scheduler"].update(scheduler_overrides)
    validate_config(new)
    return new


def renormalised_weights(base: Dict[str, float], drop: Optional[str]) -> Dict[str, float]:
    """
    Zero out one risk component and renormalise the rest to sum to 1.

    ``drop`` is one of :data:`RISK_COMPONENTS`, or ``None`` for the full model.
    Raises if dropping the component would leave zero total weight.
    """
    weights = {k: float(base[k]) for k in RISK_WEIGHT_KEYS}
    if drop is None:
        return weights
    key = f"weight_{drop}"
    if key not in weights:
        raise ConfigError(f"unknown risk component '{drop}'")
    weights[key] = 0.0
    total = sum(weights.values())
    if total <= 0.0:
        raise ConfigError(f"dropping '{drop}' leaves no risk weight to renormalise")
    return {k: v / total for k, v in weights.items()}
