// The shared execution model, ported from dapper/executor.py.
//
// This is the only place a frame's cost is computed. Every policy runs through
// it, so none can be advantaged by a private cost model. The port keeps the
// Python branch structure and arithmetic order intact.
//
// Timing model: per-frame independent, with carried-over last-valid state. An
// output produced for frame t becomes *available* at capture(t) + latency and a
// later frame may reuse it only after that instant. Frames do not queue.
//
// Accounting rules that matter:
//   * every transmitted request is charged its upload bandwidth, including
//     requests that are lost and replies that arrive too late to accept;
//   * edge availability is observable before transmission, so offloading to a
//     node already known to be down sends nothing and costs nothing;
//   * a lost or overdue request pays a deadline-bounded client timeout before
//     the local fallback runs.
//
// The step interface (`RunState` + `stepFrame`) exists so the dashboard can
// animate one frame at a time while running exactly the same code the batch
// `executeRun` runs.

import { MODE_DEGRADED_SAFE, MODE_EDGE_ACCURATE, MODE_HYBRID, MODE_LOCAL_FAST } from "./scheduler.js";

/** Per-frame log fields, in the order dapper/executor.py writes them. */
export const FRAME_COLUMNS = [
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
];

/** Policy-independent transport and perception cost model. */
export class ExecutionModel {
  constructor(cfg, deadlineMs) {
    const ex = cfg.execution;
    const sch = cfg.scheduler;
    this.deadlineMs = deadlineMs;
    this.loadComputeScale = ex.load_compute_scale;
    this.timeoutSlackMs = ex.remote_timeout_slack_ms;
    this.reuseLatencyMs = ex.reuse_latency_ms;
    this.edgeHealthCheck = ex.edge_health_check !== false;
    this.lastValidFreshnessMs = sch.last_valid_freshness_ms;
    this.hybridWindowMs = sch.hybrid_freshness_window_ms;
    this.nominalComputeMs = { ...cfg.derived.nominal_compute_ms };
    this.extraRttMs = { ...cfg.derived.extra_rtt_ms };
    this.bandwidthKb = { ...cfg.derived.bandwidth_kb };
  }

  totalRttMs(rttMs, backend) {
    return rttMs + this.extraRttMs[backend];
  }

  trueArrivalMs(rttMs, computePotentialMs, edgeLoad, backend) {
    return (
      this.totalRttMs(rttMs, backend) + computePotentialMs * (1.0 + this.loadComputeScale * edgeLoad)
    );
  }

  /**
   * Instant at which the offload client abandons an outstanding request: the
   * nominal predicted arrival plus slack, never beyond the control deadline.
   */
  clientTimeoutMs(rttMs, edgeLoad, backend) {
    const nominal =
      this.totalRttMs(rttMs, backend) +
      this.nominalComputeMs[backend] * (1.0 + this.loadComputeScale * edgeLoad);
    return Math.min(this.deadlineMs, nominal + this.timeoutSlackMs);
  }
}

/** Newest already-available perception output, respecting availability times. */
export class LastValidStore {
  constructor(maxlen = 8) {
    this.maxlen = maxlen;
    this.pending = [];
  }

  record(out) {
    this.pending.push(out);
    if (this.pending.length > this.maxlen) this.pending.shift();
  }

  newestAvailable(nowMs) {
    let best = null;
    for (const o of this.pending) {
      if (o.availableTimeMs <= nowMs) {
        if (
          best === null ||
          o.contentTimeMs > best.contentTimeMs ||
          (o.contentTimeMs === best.contentTimeMs && o.availableTimeMs > best.availableTimeMs)
        ) {
          best = o;
        }
      }
    }
    return best;
  }
}

/** Mutable state carried across the frames of one run. */
export class RunState {
  constructor() {
    this.store = new LastValidStore();
    this.prevMode = null;
    this.modeSwitches = 0;
    this.frameIndex = 0;
  }
}

/**
 * Execute one frame. Returns the per-frame log row plus the decision object and
 * the scheduler inputs, so the UI can show what the gates saw.
 */
export function stepFrame(trace, policy, model, state, i) {
  const now = trace.capture_time_ms[i];
  const last = state.store.newestAvailable(now);
  const lastAge = last !== null ? now - last.contentTimeMs : Infinity;

  const obs = {
    rttMs: trace.rtt_ms[i],
    packetLoss: trace.packet_loss[i],
    edgeLoad: trace.edge_load[i],
    edgeAvailable: !!trace.edge_available[i],
    deadlineMs: model.deadlineMs,
    frameAgeMs: 0.0,
    lastValidAgeMs: lastAge,
    localConfidence: trace.local_confidence[i],
    frameIndex: i,
  };

  const decision = policy.decide(obs, trace, i, model);
  const backend = policy.backend;
  const mode = decision.selectedMode;

  if (state.prevMode !== null && mode !== state.prevMode) state.modeSwitches += 1;
  state.prevMode = mode;

  let bw = 0.0;
  let remoteAttempted = false;
  let remoteSucceeded = false;
  let remoteAccepted = false;
  let rejDeadline = false;
  let rejFreshness = false;
  let failedLoss = false;
  let failedUnavail = false;
  let refreshLatency = NaN;
  let refreshAccepted = false;
  let outputReused = false;
  let outputAge = 0.0;
  let fallback = !!decision.fallbackNeeded;

  let control;
  let final;
  let source;
  let conf;
  let det;
  let rConf = null;
  let arrival = NaN;
  let timeout = NaN;
  let lost = false;

  const locLat = trace.local_latency_ms[i];
  const locConf = trace.local_confidence[i];
  const locDet = trace.local_detections[i];

  if (mode === MODE_DEGRADED_SAFE) {
    if (last !== null && lastAge <= model.lastValidFreshnessMs) {
      control = model.reuseLatencyMs;
      final = control;
      source = "reused";
      conf = last.confidence;
      det = last.detections;
      outputReused = true;
      outputAge = lastAge;
      fallback = true;
    } else {
      control = locLat;
      final = control;
      source = "local";
      conf = locConf;
      det = locDet;
      fallback = true;
      state.store.record({ contentTimeMs: now, availableTimeMs: now + control, confidence: conf, detections: det });
    }
  } else if (mode === MODE_LOCAL_FAST) {
    control = locLat;
    final = control;
    source = "local";
    conf = locConf;
    det = locDet;
    state.store.record({ contentTimeMs: now, availableTimeMs: now + control, confidence: conf, detections: det });
  } else if (mode === MODE_EDGE_ACCURATE || mode === MODE_HYBRID) {
    const computePot = backend === "edge" ? trace.edge_compute_ms[i] : trace.cloud_compute_ms[i];
    rConf = backend === "edge" ? trace.edge_confidence[i] : trace.cloud_confidence[i];
    const rDet = backend === "edge" ? trace.edge_detections[i] : trace.cloud_detections[i];
    const rLossDraw = backend === "edge" ? trace.edge_loss_draw[i] : trace.cloud_loss_draw[i];
    const avail = !!trace.edge_available[i];

    const reachable = avail || !model.edgeHealthCheck;
    if (!reachable) {
      failedUnavail = true;
    } else {
      remoteAttempted = true;
      bw += model.bandwidthKb[backend];
    }
    arrival = model.trueArrivalMs(trace.rtt_ms[i], computePot, trace.edge_load[i], backend);
    lost = !avail || rLossDraw < trace.packet_loss[i];
    timeout = model.clientTimeoutMs(trace.rtt_ms[i], trace.edge_load[i], backend);

    if (mode === MODE_HYBRID) {
      // A local answer is produced regardless; the refresh is optional.
      control = locLat;
      source = "local";
      conf = locConf;
      det = locDet;
      final = control;
      state.store.record({ contentTimeMs: now, availableTimeMs: now + control, confidence: conf, detections: det });
      if (remoteAttempted) {
        if (lost) {
          failedLoss = true;
        } else {
          remoteSucceeded = true;
          refreshLatency = arrival;
          const budget = Math.min(model.hybridWindowMs, model.deadlineMs);
          if (arrival > model.deadlineMs) {
            rejDeadline = true;
          } else if (arrival > budget) {
            rejFreshness = true;
          } else {
            remoteAccepted = true;
            refreshAccepted = true;
            final = arrival;
            state.store.record({ contentTimeMs: now, availableTimeMs: now + arrival, confidence: rConf, detections: rDet });
          }
        }
      }
    } else {
      // MODE_EDGE_ACCURATE: the frame is committed to the remote path.
      if (!remoteAttempted) {
        control = locLat;
        final = control;
        source = "local";
        conf = locConf;
        det = locDet;
        fallback = true;
        state.store.record({ contentTimeMs: now, availableTimeMs: now + control, confidence: conf, detections: det });
      } else if (lost) {
        failedLoss = true;
        control = timeout + locLat;
        final = control;
        source = "local";
        conf = locConf;
        det = locDet;
        fallback = true;
        state.store.record({ contentTimeMs: now, availableTimeMs: now + control, confidence: conf, detections: det });
      } else {
        remoteSucceeded = true;
        const tooLate = arrival > timeout;
        const tooOld = arrival > model.lastValidFreshnessMs;
        if (tooLate || tooOld) {
          if (arrival > model.deadlineMs || tooLate) {
            rejDeadline = true;
          } else {
            rejFreshness = true;
          }
          control = timeout + locLat;
          final = control;
          source = "local";
          conf = locConf;
          det = locDet;
          fallback = true;
          state.store.record({ contentTimeMs: now, availableTimeMs: now + control, confidence: conf, detections: det });
        } else {
          remoteAccepted = true;
          control = arrival;
          final = arrival;
          source = backend;
          conf = rConf;
          det = rDet;
          state.store.record({ contentTimeMs: now, availableTimeMs: now + arrival, confidence: conf, detections: det });
        }
      }
    }
  } else {
    throw new Error(`unknown perception mode: ${mode}`);
  }

  const deadlineMet = control <= model.deadlineMs;
  const freshnessValid = outputAge <= model.lastValidFreshnessMs;
  const staleAccepted = !freshnessValid;
  const deadlineConf = refreshAccepted ? rConf : conf;
  const usableConf = deadlineMet && freshnessValid ? deadlineConf : 0.0;
  if (outputReused) fallback = true;

  state.frameIndex = i + 1;

  const row = {
    seed: trace.seed,
    profile: trace.profile,
    policy: policy.name,
    frame_id: i,
    capture_time_ms: now,
    deadline_ms: model.deadlineMs,
    selected_mode: mode,
    decision_reason: decision.reason,
    risk_score: decision.deadlineRiskScore,
    predicted_remote_arrival_ms: decision.predictedRemoteArrivalMs,
    rtt_ms: trace.rtt_ms[i],
    packet_loss: trace.packet_loss[i],
    edge_load: trace.edge_load[i],
    edge_available: !!trace.edge_available[i],
    control_output_latency_ms: control,
    final_output_latency_ms: final,
    deadline_met: deadlineMet,
    output_source: source,
    output_confidence: conf,
    deadline_confidence: deadlineConf,
    usable_confidence_proxy: usableConf,
    detections: det,
    output_reused: outputReused,
    output_age_ms: outputAge,
    freshness_valid: freshnessValid,
    stale_output_accepted: staleAccepted,
    remote_attempted: remoteAttempted,
    remote_succeeded: remoteSucceeded,
    remote_accepted: remoteAccepted,
    remote_rejected_deadline: rejDeadline,
    remote_rejected_freshness: rejFreshness,
    remote_failed_loss: failedLoss,
    remote_failed_unavailable: failedUnavail,
    remote_refresh_latency_ms: refreshLatency,
    remote_refresh_accepted: refreshAccepted,
    bandwidth_kb: bw,
    fallback_used: fallback,
  };

  return { row, decision, obs, backend, arrival, timeout, lost };
}

/** Replay one scenario trace under one policy and return the per-frame log. */
export function executeRun(trace, policy, cfg, deadlineMs) {
  const model = new ExecutionModel(cfg, deadlineMs);
  const state = new RunState();
  policy.reset();
  const rows = [];
  for (let i = 0; i < trace.frames; i++) rows.push(stepFrame(trace, policy, model, state, i).row);
  return { rows, modeSwitches: state.modeSwitches, model };
}
