"""
Export immutable scenario traces for the browser, plus Python's per-frame outputs.

    python experiments/export_web_traces.py
    python experiments/export_web_traces.py --reference-dir build/parity

Two products, for two different jobs.

**Shipped traces** (``site/data/traces/<profile>-<seed>.json``) carry only the
scenario: the network state and the potential local, edge and cloud outcomes of
every frame, exactly as ``dapper.scenario.generate_scenarios`` drew them, with
the trace fingerprint. The dashboard's replay mode plays these, so what a visitor
sees is the same immutable future the Python benchmark replayed -- not a
re-simulation. Floats are written with Python's shortest round-trip repr, which
``JSON.parse`` reads back to the identical double.

**Parity reference** (``--reference-dir``) adds Python's per-frame outputs for
each policy and deadline: selected mode, reason string, risk score, control and
final latency, the remote outcome (accepted / late / lost / unreachable),
freshness, bandwidth, and the reduced run metrics. ``tests_web/test_js_parity.py``
asks for this into a temporary directory and requires the JavaScript engine to
reproduce every field. It is deliberately **not** committed: at three deadlines
and five policies it runs to tens of megabytes, and one command regenerates it.

NaN and Infinity are illegal in JSON and a browser will reject them, so every
non-finite value is written as null and the dumps use allow_nan=False.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.config import load_config  # noqa: E402
from dapper.executor import execute_run  # noqa: E402
from dapper.metrics import compute_metrics  # noqa: E402
from dapper.policies import build_policy  # noqa: E402
from dapper.scenario import generate_scenarios  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "site", "data", "traces")

#: Seeds shipped for replay. Evaluation seeds, so a visitor replays a scenario
#: that really is one of the 30 the paper reports.
DEFAULT_SEEDS = (100, 101, 102)
DEFAULT_FRAMES = 1000

#: Policies whose Python outputs the parity reference records.
REFERENCE_POLICIES = ("local_only", "edge_only", "cloud_only", "dapper", "oracle_feasible")

#: Deadlines the parity reference covers. 100 ms is the paper's operating point;
#: 150 and 250 ms are included because edge-accurate receives no frames at
#: D <= 100 ms, so without them the parity test would never exercise the
#: committing branch or its timeout accounting.
REFERENCE_DEADLINES_MS = (100.0, 150.0, 250.0)

#: Scenario arrays written to the shipped trace. capture_time_ms is omitted
#: because it is exactly frame_index * frame_period_ms and the browser derives it.
TRACE_FLOAT_ARRAYS = (
    "rtt_ms", "packet_loss", "edge_load",
    "local_latency_ms", "local_confidence",
    "edge_compute_ms", "edge_confidence", "edge_loss_draw",
    "cloud_compute_ms", "cloud_confidence", "cloud_loss_draw",
)
TRACE_INT_ARRAYS = ("local_detections", "edge_detections", "cloud_detections")

#: Per-frame fields the reference records, split by how they are encoded.
REF_CATEGORICAL = ("selected_mode", "decision_reason", "output_source")
REF_FLOAT = (
    "risk_score", "predicted_remote_arrival_ms",
    "control_output_latency_ms", "final_output_latency_ms",
    "output_confidence", "deadline_confidence", "usable_confidence_proxy",
    "output_age_ms", "remote_refresh_latency_ms", "bandwidth_kb",
)
REF_BOOL = (
    "deadline_met", "output_reused", "freshness_valid", "stale_output_accepted",
    "remote_attempted", "remote_succeeded", "remote_accepted",
    "remote_rejected_deadline", "remote_rejected_freshness",
    "remote_failed_loss", "remote_failed_unavailable", "remote_refresh_accepted",
)
REF_INT = ("detections",)


def _f(x: Any) -> Optional[float]:
    """JSON-safe float: NaN and +/-Infinity become null."""
    v = float(x)
    return v if math.isfinite(v) else None


def _floats(a: np.ndarray) -> List[Optional[float]]:
    return [_f(v) for v in np.asarray(a, dtype=float)]


def _ints(a: np.ndarray) -> List[int]:
    return [int(v) for v in np.asarray(a)]


def _bools(a: Sequence[Any]) -> List[int]:
    return [1 if bool(v) else 0 for v in a]


def write_json(obj: Any, path: str, indent: Optional[int] = None) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = json.dumps(obj, indent=indent, allow_nan=False, ensure_ascii=False,
                      separators=(",", ":") if indent is None else None)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    return len(text)


# ------------------------------------------------------------ shipped trace
def reference_metrics(trace, cfg: Dict[str, Any], policies: Sequence[str],
                      deadlines: Sequence[float]) -> Dict[str, Any]:
    """
    Python's reduced run metrics, small enough to ship inside the trace.

    The dashboard replays the trace, reduces it with its own port of
    dapper/metrics.py, and shows the worst disagreement with these values in its
    status bar. Parity therefore stops being a claim in a README and becomes
    something a visitor watches the page verify. Roughly 10 KiB per trace.
    """
    out: Dict[str, Any] = {}
    for deadline in deadlines:
        for name in policies:
            df = execute_run(trace, build_policy(name, cfg), cfg, float(deadline))
            m = compute_metrics(df, df.attrs.get("mode_switches"))
            out[f"{name}@{deadline:g}"] = {
                k: (_f(v) if isinstance(v, float) else v) for k, v in m.__dict__.items()
            }
    return out


def trace_payload(trace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_by": "experiments/export_web_traces.py",
        "note": ("Immutable pre-generated scenario. Every stochastic quantity a "
                 "policy could consume was drawn in advance from its own random "
                 "stream, so a decision selects an outcome but cannot change the "
                 "random future. Synthetic: not a measured network trace."),
        "seed": int(trace.seed),
        "profile": str(trace.profile),
        "frames": int(trace.frames),
        "frame_period_ms": float(cfg["execution"]["frame_period_ms"]),
        "fingerprint": trace.fingerprint(),
        "reference_metrics": reference_metrics(trace, cfg, REFERENCE_POLICIES,
                                               REFERENCE_DEADLINES_MS),
        "arrays": {
            **{k: _floats(getattr(trace, k)) for k in TRACE_FLOAT_ARRAYS},
            **{k: _ints(getattr(trace, k)) for k in TRACE_INT_ARRAYS},
            "edge_available": _bools(getattr(trace, "edge_available")),
        },
    }


def write_traces(cfg: Dict[str, Any], out_dir: str, profiles: Sequence[str],
                 seeds: Sequence[int], frames: int) -> List[Dict[str, Any]]:
    index: List[Dict[str, Any]] = []
    total = 0
    for profile in profiles:
        for seed in seeds:
            trace = generate_scenarios(profile, cfg, int(seed), int(frames))
            name = f"{profile}-{seed}.json"
            size = write_json(trace_payload(trace, cfg), os.path.join(out_dir, name))
            total += size
            index.append({"profile": profile, "seed": int(seed), "frames": int(frames),
                          "file": name, "fingerprint": trace.fingerprint()})
            print(f"  wrote traces/{name}  ({size / 1024:.0f} KiB, "
                  f"fingerprint {trace.fingerprint()})")
    write_json({
        "generated_by": "experiments/export_web_traces.py",
        "frame_period_ms": float(cfg["execution"]["frame_period_ms"]),
        "traces": index,
    }, os.path.join(out_dir, "index.json"), indent=1)
    print(f"  {len(index)} traces, {total / 1024 / 1024:.2f} MiB total")
    return index


# ---------------------------------------------------------- parity reference
def reference_payload(trace, cfg: Dict[str, Any], policies: Sequence[str],
                      deadlines: Sequence[float]) -> Dict[str, Any]:
    """Python's per-frame outputs and reduced metrics, for every policy/deadline."""
    runs: Dict[str, Dict[str, Any]] = {}
    for deadline in deadlines:
        for name in policies:
            df = execute_run(trace, build_policy(name, cfg), cfg, float(deadline))
            m = compute_metrics(df, df.attrs.get("mode_switches"))
            cols: Dict[str, Any] = {}
            vocab: Dict[str, List[str]] = {}
            for field in REF_CATEGORICAL:
                values = [str(v) for v in df[field]]
                words = sorted(set(values))
                idx = {w: i for i, w in enumerate(words)}
                vocab[field] = words
                cols[field] = [idx[v] for v in values]
            for field in REF_FLOAT:
                cols[field] = _floats(df[field].to_numpy())
            for field in REF_BOOL:
                cols[field] = _bools(df[field].tolist())
            for field in REF_INT:
                cols[field] = _ints(df[field].to_numpy())
            runs[f"{name}@{deadline:g}"] = {
                "policy": name,
                "deadline_ms": float(deadline),
                "mode_switches": int(df.attrs.get("mode_switches", 0)),
                "vocab": vocab,
                "columns": cols,
                "metrics": {k: _f(v) if isinstance(v, float) else v
                            for k, v in m.__dict__.items()},
            }
    return {
        "generated_by": "experiments/export_web_traces.py --reference-dir",
        "seed": int(trace.seed),
        "profile": str(trace.profile),
        "frames": int(trace.frames),
        "fingerprint": trace.fingerprint(),
        "trace_file": f"{trace.profile}-{trace.seed}.json",
        "runs": runs,
    }


def write_reference(cfg: Dict[str, Any], ref_dir: str, profiles: Sequence[str],
                    seeds: Sequence[int], frames: int, policies: Sequence[str],
                    deadlines: Sequence[float]) -> List[str]:
    paths = []
    total = 0
    for profile in profiles:
        for seed in seeds:
            trace = generate_scenarios(profile, cfg, int(seed), int(frames))
            name = f"{profile}-{seed}.reference.json"
            path = os.path.join(ref_dir, name)
            size = write_json(reference_payload(trace, cfg, policies, deadlines), path)
            total += size
            paths.append(path)
            print(f"  wrote {name}  ({size / 1024 / 1024:.2f} MiB)")
    print(f"  {len(paths)} reference files, {total / 1024 / 1024:.1f} MiB total "
          f"(not committed; regenerate with --reference-dir)")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="where the shipped traces go")
    p.add_argument("--reference-dir", default=None,
                   help="also write Python's per-frame outputs here (for the parity test)")
    p.add_argument("--profiles", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    p.add_argument("--no-traces", action="store_true",
                   help="only write the parity reference")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    profiles = args.profiles or list(cfg["profiles"])
    if not args.no_traces:
        write_traces(cfg, args.out_dir, profiles, args.seeds, args.frames)
    if args.reference_dir:
        write_reference(cfg, args.reference_dir, profiles, args.seeds, args.frames,
                        REFERENCE_POLICIES, REFERENCE_DEADLINES_MS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
