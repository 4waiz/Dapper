"""
Optional FastAPI demo service for the DAPPER prototype.

This is a small convenience wrapper around the corrected `dapper` package, for
demonstrations. It is not used by any camera-ready experiment: every reported
result is produced by the scripts in `experiments/`.

    GET  /status          service health and configuration snapshot
    GET  /last-run        most recent per-frame CSV (head) as JSON
    GET  /metrics         seed-aggregated results, if they exist
    POST /run             run one (policy, profile, seed) cell and persist it

Run with:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dapper.config import load_config
from dapper.executor import execute_run
from dapper.metrics import compute_metrics
from dapper.policies import ALL_POLICY_NAMES, build_policy
from dapper.scenario import generate_scenarios

CONFIG_PATH = os.environ.get("DAPPER_CONFIG", "config.yaml")
RESULTS_DIR = os.environ.get("DAPPER_RESULTS_DIR", "results")
LAST_RUN_PATH = os.path.join(RESULTS_DIR, "last_run.csv")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "final", "multi_seed_aggregate.csv")

app = FastAPI(title="DAPPER demo API")


class RunRequest(BaseModel):
    frames: int = Field(1000, ge=1, le=20000)
    deadline_ms: float = Field(100.0, gt=0.0)
    profile: str = Field("variable")
    policy: str = Field("dapper")
    seed: int = Field(42)


@app.get("/status")
def status():
    cfg = load_config(CONFIG_PATH)
    return {
        "status": "ok",
        "config_path": os.path.abspath(CONFIG_PATH),
        "results_dir": os.path.abspath(RESULTS_DIR),
        "policies": list(ALL_POLICY_NAMES),
        "profiles": cfg.get("profiles", []),
        "scheduler": cfg["scheduler"],
        "note": ("demo endpoint; camera-ready results come from experiments/, "
                 "not from this service"),
    }


@app.get("/last-run")
def last_run(limit: int = 200):
    if not os.path.exists(LAST_RUN_PATH):
        raise HTTPException(404, f"no last run at {LAST_RUN_PATH}")
    df = pd.read_csv(LAST_RUN_PATH)
    return {"rows": int(len(df)), "preview": df.head(limit).to_dict(orient="records")}


@app.get("/metrics")
def metrics():
    if not os.path.exists(SUMMARY_PATH):
        raise HTTPException(
            404, f"no aggregate at {SUMMARY_PATH}; run experiments/final_eval.py")
    return pd.read_csv(SUMMARY_PATH).to_dict(orient="records")


@app.post("/run")
def trigger_run(req: RunRequest):
    """Run one cell synchronously. Blocks until the run finishes."""
    if req.policy not in ALL_POLICY_NAMES:
        raise HTTPException(400, f"unknown policy: {req.policy}")
    cfg = load_config(CONFIG_PATH)
    if req.profile not in cfg["network_profiles"]:
        raise HTTPException(400, f"unknown profile: {req.profile}")

    trace = generate_scenarios(req.profile, cfg, req.seed, req.frames)
    df = execute_run(trace, build_policy(req.policy, cfg), cfg, req.deadline_ms)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(LAST_RUN_PATH, index=False)
    m = compute_metrics(df, df.attrs.get("mode_switches"))
    return {
        "frames": len(df),
        "scenario_fingerprint": trace.fingerprint(),
        "output_csv": os.path.abspath(LAST_RUN_PATH),
        "metrics": {k: v for k, v in m.__dict__.items()},
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
