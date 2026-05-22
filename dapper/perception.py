"""
Perception simulation.

Two backends are provided:

    SyntheticPerception   no video, all numbers drawn from distributions
    VideoPerception       reads frames from disk using OpenCV but still
                          simulates inference; we deliberately do not run
                          YOLO so that the prototype is honest and portable.

Both backends expose the same interface used by benchmark.py:

    local_infer(frame_id) -> PerceptionResult
    edge_infer(frame_id, network) -> PerceptionResult or None
    hybrid_infer(frame_id, network, deadline_ms) -> PerceptionResult
    degraded_safe(frame_id, last_valid) -> PerceptionResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover - import guard
    cv2 = None
    _HAS_CV2 = False


@dataclass
class PerceptionResult:
    """A single perception output.

    Latency is the wall time the pipeline would have spent for this frame
    (compute + any network). bandwidth_kb counts only bytes that crossed the
    network for this frame.
    """
    frame_id: int
    source: str            # local | edge | last_valid
    confidence: float
    latency_ms: float
    bandwidth_kb: float
    stale: bool            # True if we reused a previous result
    detections: int        # Number of synthetic detections in the frame


class _BaseBackend:
    """Shared random state and config plumbing."""

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed + 7)
        self._last_valid: Optional[PerceptionResult] = None

    # ----------------------------------------------------------------- helpers
    def _uniform(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def _detections(self) -> int:
        # Most frames have a handful of objects; a long tail of busy frames.
        return int(max(0, np.round(self.rng.exponential(2.0))))

    def store_last_valid(self, result: PerceptionResult) -> None:
        if result is not None and result.source != "last_valid":
            self._last_valid = result

    def get_last_valid(self) -> Optional[PerceptionResult]:
        return self._last_valid

    # ------------------------------------------------------------------- modes
    def local_infer(self, frame_id: int) -> PerceptionResult:
        lo = self.cfg["local"]["latency_ms_min"]
        hi = self.cfg["local"]["latency_ms_max"]
        cmin = self.cfg["local"]["confidence_min"]
        cmax = self.cfg["local"]["confidence_max"]
        result = PerceptionResult(
            frame_id=frame_id,
            source="local",
            confidence=self._uniform(cmin, cmax),
            latency_ms=self._uniform(lo, hi),
            bandwidth_kb=float(self.cfg["local"]["bandwidth_kb"]),
            stale=False,
            detections=self._detections(),
        )
        self.store_last_valid(result)
        return result

    def edge_infer(
        self,
        frame_id: int,
        rtt_ms: float,
        packet_loss: float,
        edge_load: float,
        edge_available: bool,
        backend: str = "edge",
    ) -> Optional[PerceptionResult]:
        """
        Simulate offloaded inference.

        Returns None if the request 'fails' (packet loss or outage).
        Latency = network RTT + compute, scaled by load.
        """
        if not edge_available:
            return None
        if self.rng.random() < packet_loss:
            return None

        section = self.cfg[backend]
        lo = section["latency_ms_min"]
        hi = section["latency_ms_max"]
        cmin = section["confidence_min"]
        cmax = section["confidence_max"]
        bw = float(section["bandwidth_kb"])

        compute_ms = self._uniform(lo, hi) * (1.0 + 0.5 * edge_load)
        latency = rtt_ms + compute_ms
        result = PerceptionResult(
            frame_id=frame_id,
            source=backend,
            confidence=self._uniform(cmin, cmax),
            latency_ms=latency,
            bandwidth_kb=bw,
            stale=False,
            detections=self._detections(),
        )
        self.store_last_valid(result)
        return result

    def hybrid_infer(
        self,
        frame_id: int,
        rtt_ms: float,
        packet_loss: float,
        edge_load: float,
        edge_available: bool,
        deadline_ms: float,
        hybrid_window_ms: float,
    ) -> PerceptionResult:
        """
        Hybrid mode: return local immediately, then check whether an edge
        refresh could have arrived inside the freshness window. If so, the
        edge result supersedes the local one in the final logged output.
        """
        local = self.local_infer(frame_id)

        edge = self.edge_infer(
            frame_id=frame_id,
            rtt_ms=rtt_ms,
            packet_loss=packet_loss,
            edge_load=edge_load,
            edge_available=edge_available,
            backend="edge",
        )
        if edge is None:
            return local

        # The edge result counts only if it would have arrived inside the
        # hybrid freshness window AND before the deadline.
        refresh_arrival = edge.latency_ms
        if refresh_arrival <= min(hybrid_window_ms, deadline_ms):
            edge.bandwidth_kb += local.bandwidth_kb
            return edge
        return local

    def degraded_safe(
        self,
        frame_id: int,
        last_valid_age_ms: float,
        last_valid_freshness_ms: float,
    ) -> PerceptionResult:
        """
        Degraded-safe: prefer the last valid perception result if it is
        still fresh; otherwise drop to a guaranteed local inference.
        """
        last = self.get_last_valid()
        if last is not None and last_valid_age_ms <= last_valid_freshness_ms:
            # Reuse without burning compute. Mark stale so the metrics know.
            return PerceptionResult(
                frame_id=frame_id,
                source="last_valid",
                confidence=last.confidence,
                latency_ms=1.0,   # Effectively free
                bandwidth_kb=0.0,
                stale=True,
                detections=last.detections,
            )
        return self.local_infer(frame_id)


class SyntheticPerception(_BaseBackend):
    """No-video backend. All numbers come from distributions."""

    def read_frame(self, frame_id: int):
        return None  # Frame content is not used in synthetic mode.


class VideoPerception(_BaseBackend):
    """
    OpenCV-backed backend.

    We read frames from disk for realism (and so that future work can plug in
    a real detector), but the *inference* itself is still simulated. We never
    pretend to run YOLO.
    """

    def __init__(self, cfg: dict, video_path: str, seed: int = 42):
        super().__init__(cfg, seed=seed)
        if not _HAS_CV2:
            raise RuntimeError(
                "opencv-python is not available; install it to use video mode"
            )
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

    def read_frame(self, frame_id: int):
        ok, frame = self.cap.read()
        if not ok:
            # Loop the video so long benchmarks still have content.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
        except Exception:
            pass


def build_perception(cfg: dict, video_path: Optional[str] = None, seed: int = 42) -> _BaseBackend:
    """Pick the right backend based on whether a video file was provided."""
    if video_path:
        return VideoPerception(cfg, video_path, seed=seed)
    return SyntheticPerception(cfg, seed=seed)
