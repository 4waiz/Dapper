"""
Shared plumbing for every camera-ready experiment driver.

Nothing here makes a scientific decision; it only assembles scenario banks,
replays policies over them and reduces the result to per-run metrics.
"""

from __future__ import annotations

import io
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.config import load_config  # noqa: E402
from dapper.executor import execute_run  # noqa: E402
from dapper.metrics import RunMetrics, compute_metrics, metrics_frame  # noqa: E402
from dapper.scenario import ScenarioTrace, generate_scenarios  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
ARTIFACTS = os.path.join(ROOT, "artifacts")


def calibration_seeds(cfg: Dict[str, Any]) -> List[int]:
    return [int(s) for s in cfg["calibration_seeds"]]


def evaluation_seeds(cfg: Dict[str, Any], count: Optional[int] = None) -> List[int]:
    start = int(cfg["evaluation_seeds_start"])
    n = int(count if count is not None else cfg["evaluation_seeds_count"])
    return list(range(start, start + n))


def assert_disjoint(cfg: Dict[str, Any]) -> None:
    """Hard guarantee that no parameter is ever selected on evaluation data."""
    cal = set(calibration_seeds(cfg))
    ev = set(evaluation_seeds(cfg))
    overlap = cal & ev
    if overlap:
        raise RuntimeError(
            f"calibration and evaluation seed sets overlap: {sorted(overlap)}")


def build_bank(cfg: Dict[str, Any], profiles: Sequence[str], seeds: Sequence[int],
               frames: int) -> Dict[Tuple[int, str], ScenarioTrace]:
    """Generate every (seed, profile) scenario trace exactly once."""
    return {(int(s), p): generate_scenarios(p, cfg, int(s), frames)
            for p in profiles for s in seeds}


def run_cells(
    bank: Dict[Tuple[int, str], ScenarioTrace],
    policies: Sequence,
    cfg: Dict[str, Any],
    deadline_ms: float,
    keep_frames: bool = False,
    extra_cols: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Replay every policy over every trace.

    Returns ``(per_run_metrics, per_frame_or_None)``. The per-frame log is huge
    at the full protocol, so it is only retained when explicitly requested.
    """
    runs: List[RunMetrics] = []
    frames_out: List[pd.DataFrame] = []
    for (seed, profile), trace in bank.items():
        for policy in policies:
            df = execute_run(trace, policy, cfg, deadline_ms)
            runs.append(compute_metrics(df, df.attrs.get("mode_switches")))
            if keep_frames:
                frames_out.append(df)
    per_run = metrics_frame(runs)
    if extra_cols:
        for k, v in extra_cols.items():
            per_run[k] = v
    per_frame = pd.concat(frames_out, ignore_index=True) if keep_frames else None
    return per_run, per_frame


def write_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(df)} rows)")
    return path


def write_text(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {os.path.relpath(path, ROOT)}")
    return path


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment_lines() -> List[str]:
    lines = [
        f"git_commit: {git_commit()}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"cpu_count: {os.cpu_count()}",
    ]
    for m in ("numpy", "pandas", "scipy", "matplotlib", "yaml", "pytest",
              "torch", "ultralytics"):
        try:
            mod = __import__(m)
            lines.append(f"{m}: {getattr(mod, '__version__', 'unknown')}")
        except Exception:
            lines.append(f"{m}: not installed")
    return lines


__all__ = [
    "ROOT", "RESULTS", "ARTIFACTS", "load_config", "calibration_seeds",
    "evaluation_seeds", "assert_disjoint", "build_bank", "run_cells",
    "write_csv", "write_text", "git_commit", "environment_lines",
    "execute_run", "compute_metrics", "metrics_frame", "generate_scenarios",
]
