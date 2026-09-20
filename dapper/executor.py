"""
The shared execution model: timing, bandwidth, freshness and fallback accounting.

This module is deliberately the *only* place where a frame's cost is computed.
Every policy — DAPPER, the fixed baselines, the adaptive baselines and the
offline oracle — is executed through it, so no policy can be advantaged or
penalised by a private cost model.

Timing model (documented, and identical for every policy)
---------------------------------------------------------
The benchmark is a **per-frame independent timing model with carried-over
last-valid state**, not a full event-driven simulation: frames do not queue
behind one another, but an output produced for frame *t* only becomes
*available* at ``capture(t) + latency`` and can be reused by a later frame only
after that instant. This is stated explicitly because the published prototype
treated every result as existing at its own capture time.

* Access round trip. ``total_rtt = rtt_profile + extra_rtt_ms[backend]``. The
  network-profile RTT **is** the access round trip to the edge; the cloud is
  farther and adds ``cloud.extra_rtt_ms`` of wide-area transit. The *same*
  expression is used by the scheduler's prediction and by execution.
* Remote compute. ``compute = compute_potential * (1 + load_compute_scale * edge_load)``.
* Arrival. ``arrival = total_rtt + compute`` (measured from frame capture).
* Loss. The request is lost iff ``loss_draw < packet_loss``. The draw is
  pre-generated per frame and per backend, so the same frame is lost (or not)
  for every policy that attempts it.
* Availability. ``edge_available`` is observable *before* transmission through a
  health check / heartbeat. A policy that offloads while the edge is known to be
  down therefore sends nothing and pays nothing; this keeps the fixed baselines
  from being strawmen. Setting ``execution.edge_health_check: false`` disables
  the check, in which case the attempt is transmitted and times out.
* Failure timing. A lost or overdue request is **not** free. The offload client
  arms a response timer at ``min(D, nominal_predicted_arrival + slack)`` and,
  on expiry, abandons the request and runs the local model. The resulting
  control latency is ``timeout + local_latency``.
* Bandwidth. **Every transmitted request is charged its upload bandwidth**,
  including requests that are lost and requests whose reply arrives too late to
  be accepted. This is what the manuscript claims and what the published
  prototype did not do.

Latency semantics
-----------------
Two latencies are recorded, because a single column cannot express both:

``control_output_latency_ms``
    time from capture until the **first usable fresh output** is in hand. This
    is the control-loop requirement and the metric the deadline-miss rate is
    computed from.
``final_output_latency_ms``
    time until the final, possibly remotely refined, output. For hybrid these
    differ; for every other mode they coincide.

Freshness semantics
-------------------
``output_reused`` means the output was computed for an earlier frame.
``output_age_ms`` is the age of the observation the output describes.
``freshness_valid`` is ``output_age_ms <= last_valid_freshness_ms``.
A reused output inside the window is **fresh, not stale**. An output outside the
window is never accepted, so ``stale_output_accepted`` must always be zero; the
test suite asserts this.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .scenario import ScenarioTrace
from .scheduler import (
    MODE_DEGRADED_SAFE,
    MODE_EDGE_ACCURATE,
    MODE_HYBRID,
    MODE_LOCAL_FAST,
)

#: Columns written to the per-frame log, in order.
FRAME_COLUMNS = (
    "seed", "profile", "policy", "frame_id", "capture_time_ms", "deadline_ms",
    "selected_mode", "decision_reason", "risk_score", "predicted_remote_arrival_ms",
    "rtt_ms", "packet_loss", "edge_load", "edge_available",
    "control_output_latency_ms", "final_output_latency_ms", "deadline_met",
    "output_source", "output_confidence", "deadline_confidence",
    "usable_confidence_proxy", "detections",
    "output_reused", "output_age_ms", "freshness_valid", "stale_output_accepted",
    "remote_attempted", "remote_succeeded", "remote_accepted",
    "remote_rejected_deadline", "remote_rejected_freshness",
    "remote_failed_loss", "remote_failed_unavailable",
    "remote_refresh_latency_ms", "remote_refresh_accepted",
    "bandwidth_kb", "fallback_used",
)


@dataclass
class _Output:
    """A perception output and the instants that bound its usability."""

    content_time_ms: float     # capture time of the frame it describes
    available_time_ms: float   # instant from which it can be consumed
    confidence: float
    detections: int


@dataclass
class ExecutionModel:
    """Policy-independent transport and perception cost model."""

    cfg: Dict[str, Any]
    deadline_ms: float
    edge_health_check: bool = True

    load_compute_scale: float = field(init=False)
    timeout_slack_ms: float = field(init=False)
    reuse_latency_ms: float = field(init=False)
    last_valid_freshness_ms: float = field(init=False)
    hybrid_window_ms: float = field(init=False)
    nominal_compute_ms: Dict[str, float] = field(init=False)
    extra_rtt_ms: Dict[str, float] = field(init=False)
    bandwidth_kb: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        ex = self.cfg["execution"]
        sch = self.cfg["scheduler"]
        self.load_compute_scale = float(ex["load_compute_scale"])
        self.timeout_slack_ms = float(ex["remote_timeout_slack_ms"])
        self.reuse_latency_ms = float(ex["reuse_latency_ms"])
        self.edge_health_check = bool(ex.get("edge_health_check", True))
        self.last_valid_freshness_ms = float(sch["last_valid_freshness_ms"])
        self.hybrid_window_ms = float(sch["hybrid_freshness_window_ms"])
        self.nominal_compute_ms = {
            b: 0.5 * (float(self.cfg[b]["latency_ms_min"]) + float(self.cfg[b]["latency_ms_max"]))
            for b in ("edge", "cloud")
        }
        self.extra_rtt_ms = {b: float(self.cfg[b].get("extra_rtt_ms", 0.0))
                             for b in ("edge", "cloud")}
        self.bandwidth_kb = {b: float(self.cfg[b]["bandwidth_kb"])
                             for b in ("local", "edge", "cloud")}

    # ------------------------------------------------------------- transport
    def total_rtt_ms(self, rtt_ms: float, backend: str) -> float:
        return rtt_ms + self.extra_rtt_ms[backend]

    def true_arrival_ms(self, rtt_ms: float, compute_potential_ms: float,
                        edge_load: float, backend: str) -> float:
        return (self.total_rtt_ms(rtt_ms, backend)
                + compute_potential_ms * (1.0 + self.load_compute_scale * edge_load))

    def client_timeout_ms(self, rtt_ms: float, edge_load: float, backend: str) -> float:
        """
        Instant at which the offload client abandons an outstanding request.

        Transport-layer behaviour, shared by every policy: the client waits for
        the nominal predicted arrival plus a slack, and never beyond the control
        deadline, because a reply after the deadline cannot help this frame.
        """
        nominal = (self.total_rtt_ms(rtt_ms, backend)
                   + self.nominal_compute_ms[backend]
                   * (1.0 + self.load_compute_scale * edge_load))
        return min(self.deadline_ms, nominal + self.timeout_slack_ms)


class _LastValidStore:
    """Newest already-available perception output, respecting availability times."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending: deque = deque(maxlen=8)

    def record(self, out: _Output) -> None:
        self._pending.append(out)

    def newest_available(self, now_ms: float) -> Optional[_Output]:
        best: Optional[_Output] = None
        for o in self._pending:
            if o.available_time_ms <= now_ms:
                if best is None or o.content_time_ms > best.content_time_ms or (
                    o.content_time_ms == best.content_time_ms
                    and o.available_time_ms > best.available_time_ms
                ):
                    best = o
        return best


def execute_run(
    trace: ScenarioTrace,
    policy,
    cfg: Dict[str, Any],
    deadline_ms: float,
    collect_frames: bool = True,
) -> pd.DataFrame:
    """
    Replay one scenario trace under one policy and return the per-frame log.

    ``policy`` must expose ``name`` and
    ``decide(obs, frame_scenario) -> SchedulerDecision``.
    """
    model = ExecutionModel(cfg=cfg, deadline_ms=float(deadline_ms))
    store = _LastValidStore()

    a = trace
    n = trace.frames
    rtt = np.asarray(a.rtt_ms)
    loss = np.asarray(a.packet_loss)
    load = np.asarray(a.edge_load)
    avail = np.asarray(a.edge_available)
    loc_lat = np.asarray(a.local_latency_ms)
    loc_conf = np.asarray(a.local_confidence)
    loc_det = np.asarray(a.local_detections)
    capture = trace.capture_time_ms

    remote_arrays = {
        "edge": (np.asarray(a.edge_compute_ms), np.asarray(a.edge_confidence),
                 np.asarray(a.edge_detections), np.asarray(a.edge_loss_draw)),
        "cloud": (np.asarray(a.cloud_compute_ms), np.asarray(a.cloud_confidence),
                  np.asarray(a.cloud_detections), np.asarray(a.cloud_loss_draw)),
    }

    rows: List[tuple] = []
    prev_mode: Optional[str] = None
    mode_switches = 0
    policy.reset()

    for i in range(n):
        now = capture[i]
        last = store.newest_available(now)
        last_age = (now - last.content_time_ms) if last is not None else float("inf")

        obs = policy.observe(
            rtt_ms=float(rtt[i]),
            packet_loss=float(loss[i]),
            edge_load=float(load[i]),
            edge_available=bool(avail[i]),
            deadline_ms=model.deadline_ms,
            frame_age_ms=0.0,
            last_valid_age_ms=last_age,
            local_confidence=float(loc_conf[i]),
            frame_index=i,
        )
        decision = policy.decide(obs, trace, i, model)
        backend = policy.backend
        mode = decision.selected_mode

        if prev_mode is not None and mode != prev_mode:
            mode_switches += 1
        prev_mode = mode

        # ---------------------------------------------------------- execute
        bw = 0.0
        remote_attempted = False
        remote_succeeded = False
        remote_accepted = False
        rej_deadline = False
        rej_freshness = False
        failed_loss = False
        failed_unavail = False
        refresh_latency = float("nan")
        refresh_accepted = False
        output_reused = False
        output_age = 0.0
        fallback = bool(decision.fallback_needed)

        if mode == MODE_DEGRADED_SAFE:
            if last is not None and last_age <= model.last_valid_freshness_ms:
                control = model.reuse_latency_ms
                final = control
                source = "reused"
                conf = last.confidence
                det = int(last.detections)
                output_reused = True
                output_age = last_age
                fallback = True
            else:
                control = float(loc_lat[i])
                final = control
                source = "local"
                conf = float(loc_conf[i])
                det = int(loc_det[i])
                fallback = True
                store.record(_Output(now, now + control, conf, det))

        elif mode == MODE_LOCAL_FAST:
            control = float(loc_lat[i])
            final = control
            source = "local"
            conf = float(loc_conf[i])
            det = int(loc_det[i])
            store.record(_Output(now, now + control, conf, det))

        elif mode in (MODE_EDGE_ACCURATE, MODE_HYBRID):
            compute_pot, r_conf, r_det, r_loss_draw = remote_arrays[backend]
            reachable = bool(avail[i]) or (not model.edge_health_check)
            if not reachable:
                failed_unavail = True
            else:
                remote_attempted = True
                bw += model.bandwidth_kb[backend]
            arrival = model.true_arrival_ms(float(rtt[i]), float(compute_pot[i]),
                                            float(load[i]), backend)
            lost = (not bool(avail[i])) or (float(r_loss_draw[i]) < float(loss[i]))
            timeout = model.client_timeout_ms(float(rtt[i]), float(load[i]), backend)

            if mode == MODE_HYBRID:
                # A local answer is produced regardless; the refresh is optional.
                control = float(loc_lat[i])
                source = "local"
                conf = float(loc_conf[i])
                det = int(loc_det[i])
                final = control
                store.record(_Output(now, now + control, conf, det))
                if remote_attempted:
                    if lost:
                        failed_loss = True
                    else:
                        remote_succeeded = True
                        refresh_latency = arrival
                        budget = min(model.hybrid_window_ms, model.deadline_ms)
                        if arrival > model.deadline_ms:
                            rej_deadline = True
                        elif arrival > budget:
                            rej_freshness = True
                        else:
                            remote_accepted = True
                            refresh_accepted = True
                            final = arrival
                            store.record(_Output(now, now + arrival,
                                                 float(r_conf[i]), int(r_det[i])))
            else:  # MODE_EDGE_ACCURATE: the frame is committed to the remote path
                if not remote_attempted:
                    control = float(loc_lat[i])
                    final = control
                    source = "local"
                    conf = float(loc_conf[i])
                    det = int(loc_det[i])
                    fallback = True
                    store.record(_Output(now, now + control, conf, det))
                elif lost:
                    failed_loss = True
                    control = timeout + float(loc_lat[i])
                    final = control
                    source = "local"
                    conf = float(loc_conf[i])
                    det = int(loc_det[i])
                    fallback = True
                    store.record(_Output(now, now + control, conf, det))
                else:
                    remote_succeeded = True
                    too_late = arrival > timeout
                    too_old = arrival > model.last_valid_freshness_ms
                    if too_late or too_old:
                        # The client's response timer is deadline-derived, so an
                        # abandoned reply is a deadline rejection unless the
                        # content itself aged past the freshness window first.
                        if arrival > model.deadline_ms or too_late:
                            rej_deadline = True
                        else:
                            rej_freshness = True
                        control = timeout + float(loc_lat[i])
                        final = control
                        source = "local"
                        conf = float(loc_conf[i])
                        det = int(loc_det[i])
                        fallback = True
                        store.record(_Output(now, now + control, conf, det))
                    else:
                        remote_accepted = True
                        control = arrival
                        final = arrival
                        source = backend
                        conf = float(r_conf[i])
                        det = int(r_det[i])
                        store.record(_Output(now, now + arrival, conf, det))
        else:  # pragma: no cover - guarded by the policy layer
            raise ValueError(f"unknown perception mode: {mode}")

        deadline_met = control <= model.deadline_ms
        freshness_valid = output_age <= model.last_valid_freshness_ms
        stale_accepted = not freshness_valid
        deadline_conf = float(r_conf[i]) if refresh_accepted else conf
        usable_conf = deadline_conf if (deadline_met and freshness_valid) else 0.0
        # `fallback_used` means the *control* output did not come from the path
        # the selected mode primarily intended. A hybrid frame whose refresh was
        # rejected is not a fallback: its local answer was always the control
        # output, and the rejection is recorded in its own column.
        if output_reused:
            fallback = True

        if collect_frames:
            rows.append((
                trace.seed, trace.profile, policy.name, i, now, model.deadline_ms,
                mode, decision.reason, decision.deadline_risk_score,
                decision.predicted_remote_arrival_ms,
                float(rtt[i]), float(loss[i]), float(load[i]), bool(avail[i]),
                control, final, bool(deadline_met),
                source, conf, deadline_conf, usable_conf, det,
                bool(output_reused), output_age, bool(freshness_valid), bool(stale_accepted),
                bool(remote_attempted), bool(remote_succeeded), bool(remote_accepted),
                bool(rej_deadline), bool(rej_freshness),
                bool(failed_loss), bool(failed_unavail),
                refresh_latency, bool(refresh_accepted),
                bw, bool(fallback),
            ))

    df = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    df.attrs["mode_switches"] = mode_switches
    return df


def execute_matrix(
    traces: Dict[tuple, ScenarioTrace],
    policies: Sequence,
    cfg: Dict[str, Any],
    deadline_ms: float,
) -> pd.DataFrame:
    """Run every policy over every ``(seed, profile)`` trace."""
    out: List[pd.DataFrame] = []
    for (seed, profile), trace in traces.items():
        for policy in policies:
            df = execute_run(trace, policy, cfg, deadline_ms)
            df.attrs["mode_switches"] = df.attrs.get("mode_switches", 0)
            out.append(df)
    return pd.concat(out, ignore_index=True)
