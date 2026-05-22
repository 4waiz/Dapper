"""
FastAPI edge inference server.

This server does not run a real model. It accepts a frame metadata payload,
sleeps for a simulated processing time, and returns a fake detection record.
That keeps the prototype portable and honest: a real deployment would swap
this endpoint for a YOLO/Jetson service, and the benchmark code would not
need to change.

Endpoints:
    GET  /health       -> liveness probe
    POST /infer        -> simulated inference
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="DAPPER Edge Inference (simulated)")


class FrameMetadata(BaseModel):
    frame_id: int = Field(..., description="Sequential id of the frame")
    width: Optional[int] = Field(None, description="Frame width in pixels")
    height: Optional[int] = Field(None, description="Frame height in pixels")
    timestamp_ms: Optional[float] = Field(
        None, description="Client capture timestamp in ms"
    )
    edge_load_hint: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Optional client-side hint about current edge load",
    )


class InferenceResult(BaseModel):
    detection_id: str
    confidence: float
    simulated_processing_ms: float
    timestamp: float


@app.get("/health")
def health():
    """Liveness probe used by the benchmark to test edge_available."""
    return {"status": "ok"}


@app.post("/infer", response_model=InferenceResult)
def infer(meta: FrameMetadata):
    """
    Simulate inference.

    The server sleeps for a small randomised time so that real HTTP latency
    is included in the round trip. Confidence is sampled from a band that
    is meant to approximate a "good" off-board detector.
    """
    base_ms = random.uniform(45.0, 90.0)
    load_factor = 1.0 + 0.5 * meta.edge_load_hint
    sleep_ms = base_ms * load_factor
    time.sleep(sleep_ms / 1000.0)

    return InferenceResult(
        detection_id=str(uuid.uuid4()),
        confidence=random.uniform(0.80, 0.95),
        simulated_processing_ms=sleep_ms,
        timestamp=time.time(),
    )


if __name__ == "__main__":
    # Convenience launcher: `python edge_server.py` starts the dev server.
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
