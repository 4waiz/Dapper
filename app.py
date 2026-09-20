"""
Optional FastAPI dashboard API for the DAPPER prototype.

This is intentionally small. It is meant for demos, not production:

    GET  /status          - service health and config snapshot
    GET  /last-run        - most recent per-frame CSV as JSON (head)
    GET  /metrics         - summary CSV as JSON
    POST /run-benchmark   - trigger a benchmark run with JSON parameters

Run with:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from benchmark import load_config, run_benchmark, ALL_RUN_MODES
from metrics import compute_metrics


CONFIG_PATH = os.environ.get("DAPPER_CONFIG", "config.yaml")
RESULTS_DIR = os.environ.get("DAPPER_RESULTS_DIR", "results")
LAST_RUN_PATH = os.path.join(RESULTS_DIR, "last_run.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary.csv")


app = FastAPI(title="DAPPER Dashboard API")


class RunRequest(BaseModel):
    frames: int = Field(500, ge=1, le=20000)
    deadline_ms: float = Field(100.0, gt=0.0)
    profile: str = Field("variable")
    mode: str = Field("dapper")
    seed: int = Field(42)


@app.get("/status")
def status():
    """Light-weight liveness probe and config snapshot."""
    cfg = load_config(CONFIG_PATH)
    return {
        "status": "ok",
        "config_path": os.path.abspath(CONFIG_PATH),
        "results_dir": os.path.abspath(RESULTS_DIR),
        "supported_modes": list(ALL_RUN_MODES),
        "profiles": cfg.get("profiles", []),
    }


@app.get("/last-run")
def last_run(limit: int = 200):
    """Return the head of the most recent per-frame CSV."""
    if not os.path.exists(LAST_RUN_PATH):
        raise HTTPException(404, f"No last run found at {LAST_RUN_PATH}")
    df = pd.read_csv(LAST_RUN_PATH)
    return {
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records"),
    }


@app.get("/metrics")
def metrics():
    """Return the summary CSV, if it exists."""
    if not os.path.exists(SUMMARY_PATH):
        raise HTTPException(404, f"No summary found at {SUMMARY_PATH}")
    df = pd.read_csv(SUMMARY_PATH)
    return df.to_dict(orient="records")


@app.post("/run-benchmark")
def trigger_run(req: RunRequest):
    """
    Run a single benchmark synchronously and persist it to last_run.csv.

    For long benchmarks, prefer running benchmark.py from the CLI; this
    endpoint blocks until the run finishes.
    """
    if req.mode not in ALL_RUN_MODES:
        raise HTTPException(400, f"Unknown mode: {req.mode}")

    cfg = load_config(CONFIG_PATH)
    df = run_benchmark(
        run_mode=req.mode,
        profile=req.profile,
        frames=req.frames,
        deadline_ms=req.deadline_ms,
        cfg=cfg,
        seed=req.seed,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(LAST_RUN_PATH, index=False)
    m = compute_metrics(df, mode=req.mode, profile=req.profile)
    return {
        "frames": len(df),
        "output_csv": os.path.abspath(LAST_RUN_PATH),
        "metrics": {
            "mean_latency_ms": m.mean_latency_ms,
            "p95_latency_ms": m.p95_latency_ms,
            "p99_latency_ms": m.p99_latency_ms,
            "deadline_miss_rate": m.deadline_miss_rate,
            "mean_confidence": m.mean_confidence,
            "stale_output_rate": m.stale_output_rate,
            "fallback_rate": m.fallback_rate,
            "total_bandwidth_kb": m.total_bandwidth_kb,
            "reliability_score": m.reliability_score,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
