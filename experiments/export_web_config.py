"""
Export the frozen configuration to site/data/config.json for the browser engine.

    python experiments/export_web_config.py

The dashboard must not carry a second copy of Table I. It loads this file, which
is generated from config.yaml and validated by dapper.config on the way out, so
a parameter can only ever be changed in one place.

Derived scheduler quantities (the edge-compute estimates, the nominal compute
times, the per-backend extra RTT and upload sizes) are computed here exactly as
dapper.scheduler.build_scheduler_from_config and dapper.executor.ExecutionModel
compute them, so the JavaScript port reads them rather than re-deriving them and
risking a divergence.

JSON is written with allow_nan=False: a browser's JSON.parse rejects NaN and
Infinity, so a non-finite value must fail here rather than in someone's tab.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.config import load_config  # noqa: E402
from dapper.policies import ADAPTIVE_POLICIES, MAIN_POLICIES, ORACLE_POLICIES  # noqa: E402
from dapper.scheduler import ALL_MODES, ALL_REASONS  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "site", "data", "config.json")

#: Table I of the paper, in print order: display label, config keys, the format
#: the paper prints the value in, and a unit suffix.
TABLE_I_ROWS = [
    ("(w_r, w_l, w_c, w_a, w_d)",
     ["weight_rtt", "weight_loss", "weight_load", "weight_frame_age", "weight_deadline"],
     "%g", ""),
    ("Local risk threshold θL", ["risk_local_threshold"], "%.2f", ""),
    ("Degraded risk threshold θD", ["risk_degraded_threshold"], "%.2f", ""),
    ("Local-confidence threshold τc", ["local_confidence_threshold"], "%.2f", ""),
    ("Remote commit margin m", ["remote_deadline_margin"], "%.2f", ""),
    ("Hybrid freshness window W", ["hybrid_freshness_window_ms"], "%g", " ms"),
    ("Last-valid freshness window", ["last_valid_freshness_ms"], "%g", " ms"),
]

#: Table II of the paper, in print order.
TABLE_II_KEYS = ["rtt_ms_mean", "rtt_ms_std", "packet_loss", "edge_load", "outage_prob"]


def _mid(sec: Dict[str, Any]) -> float:
    return 0.5 * (float(sec["latency_ms_min"]) + float(sec["latency_ms_max"]))


def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sch = cfg["scheduler"]
    ex = cfg["execution"]

    def fmt(keys: Sequence[str], spec: str, unit: str) -> str:
        vals = [float(sch[k]) for k in keys]
        txt = ", ".join((spec % v) for v in vals)
        return (f"({txt})" if len(vals) > 1 else txt) + unit

    table_i = [{"parameter": label, "value": fmt(keys, spec, unit)}
               for label, keys, spec, unit in TABLE_I_ROWS]
    table_ii = [
        dict(profile=p, **{k: float(cfg["network_profiles"][p][k]) for k in TABLE_II_KEYS})
        for p in cfg["profiles"]
    ]

    return {
        "generated_by": "experiments/export_web_config.py",
        "source": "config.yaml",
        "note": ("Frozen camera-ready configuration. Every perception latency and "
                 "confidence in the synthetic model is simulated; confidence is a "
                 "proxy, never detector accuracy."),
        "deadline_ms": float(cfg["deadline_ms"]),
        "frames": int(cfg["frames"]),
        "calibration_seeds": [int(s) for s in cfg["calibration_seeds"]],
        "evaluation_seeds": {"start": int(cfg["evaluation_seeds_start"]),
                             "count": int(cfg["evaluation_seeds_count"])},
        "profiles": list(cfg["profiles"]),
        "network_profiles": {p: {k: float(v) for k, v in cfg["network_profiles"][p].items()}
                             for p in cfg["profiles"]},
        "local": {k: float(v) for k, v in cfg["local"].items()},
        "edge": {k: float(v) for k, v in cfg["edge"].items()},
        "cloud": {k: float(v) for k, v in cfg["cloud"].items()},
        "execution": {
            "frame_period_ms": float(ex["frame_period_ms"]),
            "load_compute_scale": float(ex["load_compute_scale"]),
            "remote_timeout_slack_ms": float(ex["remote_timeout_slack_ms"]),
            "reuse_latency_ms": float(ex["reuse_latency_ms"]),
            "edge_health_check": bool(ex["edge_health_check"]),
        },
        "scheduler": {
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
        },
        "baselines": {k: float(v) for k, v in cfg.get("baselines", {}).items()},
        # Derived exactly as the Python scheduler and executor derive them.
        "derived": {
            "edge_compute_ms_estimate": _mid(cfg["edge"]),
            "edge_compute_ms_optimistic": float(cfg["edge"]["latency_ms_min"]),
            "local_compute_ms_estimate": _mid(cfg["local"]),
            "nominal_compute_ms": {"edge": _mid(cfg["edge"]), "cloud": _mid(cfg["cloud"])},
            "extra_rtt_ms": {"edge": float(cfg["edge"].get("extra_rtt_ms", 0.0)),
                             "cloud": float(cfg["cloud"].get("extra_rtt_ms", 0.0))},
            "bandwidth_kb": {b: float(cfg[b]["bandwidth_kb"])
                             for b in ("local", "edge", "cloud")},
        },
        "modes": list(ALL_MODES),
        "reasons": list(ALL_REASONS),
        "policies": {
            "main": list(MAIN_POLICIES),
            "adaptive": list(ADAPTIVE_POLICIES),
            "oracle": list(ORACLE_POLICIES),
        },
        "table_i": table_i,
        "table_ii": table_ii,
    }


def write_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False, sort_keys=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(text) / 1024:.1f} KiB)")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args(argv)
    write_json(build(load_config(args.config)), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
