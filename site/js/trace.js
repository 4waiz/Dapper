// Scenario traces: the immutable pre-generated future of every frame.
//
// Replay mode loads a trace exported by experiments/export_web_traces.py, which
// is byte-for-byte the scenario dapper.scenario.generate_scenarios drew for that
// (profile, seed). Live mode synthesises one in the browser from the Table II
// distributions; it follows the same process but a different PRNG, so it is
// labelled "live synthetic" and never presented as a published result.
//
// Both paths produce the same object shape, so executor.js cannot tell them
// apart -- which is the point: the engine under the animation is the real one.

import { RNG } from "./rng.js";

/** Field order is the reproducibility contract of dapper/scenario.py. */
export const TRACE_FIELDS = [
  "rtt_ms", "packet_loss", "edge_load", "edge_available",
  "local_latency_ms", "local_confidence", "local_detections",
  "edge_compute_ms", "edge_confidence", "edge_detections", "edge_loss_draw",
  "cloud_compute_ms", "cloud_confidence", "cloud_detections", "cloud_loss_draw",
];

// AR(1) smoothing coefficients, preserved from the published monitor.
const RTT_SMOOTHING = 0.7;
const LOAD_SMOOTHING = 0.8;
const LOSS_JITTER_FRAC = 0.25;
const LOAD_JITTER_STD = 0.05;
const OUTAGE_MIN_FRAMES = 5;
const OUTAGE_MAX_FRAMES = 25;

function captureTimes(frames, framePeriodMs) {
  const t = new Float64Array(frames);
  for (let i = 0; i < frames; i++) t[i] = i * framePeriodMs;
  return t;
}

/** Materialise a trace exported by experiments/export_web_traces.py. */
export function traceFromExport(payload) {
  const a = payload.arrays;
  const frames = payload.frames;
  const trace = {
    seed: payload.seed,
    profile: payload.profile,
    frames,
    source: "replay",
    fingerprint: payload.fingerprint,
    framePeriodMs: payload.frame_period_ms,
    capture_time_ms: captureTimes(frames, payload.frame_period_ms),
  };
  for (const key of TRACE_FIELDS) {
    const src = a[key];
    if (!src) throw new Error(`trace is missing array '${key}'`);
    if (src.length !== frames) throw new Error(`array '${key}' has ${src.length} of ${frames} frames`);
    if (key === "edge_available") {
      const out = new Uint8Array(frames);
      for (let i = 0; i < frames; i++) out[i] = src[i] ? 1 : 0;
      trace[key] = out;
    } else if (key.endsWith("_detections")) {
      trace[key] = Int32Array.from(src);
    } else {
      trace[key] = Float64Array.from(src);
    }
  }
  return trace;
}

/**
 * Synthesise a trace in the browser from a network-profile parameter set.
 *
 * The process mirrors dapper/scenario.py -- AR(1) RTT and load, loss jittered
 * around its profile mean, outages as bursts of 5-25 frames, and independent
 * streams per quantity -- but NumPy's PCG64 cannot be reproduced here, so the
 * values differ from any published run. Label it "live synthetic".
 */
export function synthesiseTrace({ profile, params, cfg, seed, frames, label = "live synthetic" }) {
  const framePeriodMs = cfg.execution.frame_period_ms;
  // One independent stream per quantity, so a change to one does not shift another.
  const s = (k) => new RNG((seed * 2654435761 + k * 40503) >>> 0);
  const rNet = { rtt: s(1), loss: s(2), load: s(3), outage: s(4) };
  const rLoc = { lat: s(5), conf: s(6), det: s(7) };
  const rEdge = { comp: s(8), conf: s(9), det: s(10), loss: s(11) };
  const rCloud = { comp: s(12), conf: s(13), det: s(14), loss: s(15) };

  const rttMean = params.rtt_ms_mean;
  const rttStd = params.rtt_ms_std;
  const lossBase = params.packet_loss;
  const loadBase = params.edge_load;
  const outageProb = params.outage_prob ?? 0.0;

  const out = {
    seed,
    profile,
    frames,
    source: "live",
    label,
    framePeriodMs,
    capture_time_ms: captureTimes(frames, framePeriodMs),
    rtt_ms: new Float64Array(frames),
    packet_loss: new Float64Array(frames),
    edge_load: new Float64Array(frames),
    edge_available: new Uint8Array(frames),
    local_latency_ms: new Float64Array(frames),
    local_confidence: new Float64Array(frames),
    local_detections: new Int32Array(frames),
    edge_compute_ms: new Float64Array(frames),
    edge_confidence: new Float64Array(frames),
    edge_detections: new Int32Array(frames),
    edge_loss_draw: new Float64Array(frames),
    cloud_compute_ms: new Float64Array(frames),
    cloud_confidence: new Float64Array(frames),
    cloud_detections: new Int32Array(frames),
    cloud_loss_draw: new Float64Array(frames),
  };

  let curRtt = rttMean;
  let curLoad = loadBase;
  let remaining = 0;
  for (let i = 0; i < frames; i++) {
    const u = rNet.outage.random();
    const burst = rNet.outage.int(OUTAGE_MIN_FRAMES, OUTAGE_MAX_FRAMES);
    if (remaining > 0) {
      remaining -= 1;
      out.edge_available[i] = 0;
    } else if (outageProb > 0.0 && u < outageProb) {
      remaining = burst - 1;
      out.edge_available[i] = 0;
    } else {
      out.edge_available[i] = 1;
    }
    curRtt = RTT_SMOOTHING * curRtt + (1.0 - RTT_SMOOTHING) * rNet.rtt.normal(rttMean, rttStd);
    out.rtt_ms[i] = Math.max(1.0, curRtt);
    const jit = rNet.load.normal(0.0, LOAD_JITTER_STD);
    curLoad = Math.min(1.0, Math.max(0.0, LOAD_SMOOTHING * curLoad + (1.0 - LOAD_SMOOTHING) * (loadBase + jit)));
    out.edge_load[i] = curLoad;
    out.packet_loss[i] = Math.min(1.0, Math.max(0.0, lossBase + rNet.loss.normal(0.0, lossBase * LOSS_JITTER_FRAC)));

    out.local_latency_ms[i] = rLoc.lat.uniform(cfg.local.latency_ms_min, cfg.local.latency_ms_max);
    out.local_confidence[i] = rLoc.conf.uniform(cfg.local.confidence_min, cfg.local.confidence_max);
    out.local_detections[i] = detections(rLoc.det);

    out.edge_compute_ms[i] = rEdge.comp.uniform(cfg.edge.latency_ms_min, cfg.edge.latency_ms_max);
    out.edge_confidence[i] = rEdge.conf.uniform(cfg.edge.confidence_min, cfg.edge.confidence_max);
    out.edge_detections[i] = detections(rEdge.det);
    out.edge_loss_draw[i] = rEdge.loss.random();

    out.cloud_compute_ms[i] = rCloud.comp.uniform(cfg.cloud.latency_ms_min, cfg.cloud.latency_ms_max);
    out.cloud_confidence[i] = rCloud.conf.uniform(cfg.cloud.confidence_min, cfg.cloud.confidence_max);
    out.cloud_detections[i] = detections(rCloud.det);
    out.cloud_loss_draw[i] = rCloud.loss.random();
  }
  return out;
}

/** Object count per frame: mostly a handful, with a busy-frame tail. */
function detections(rng) {
  return Math.max(0, Math.round(rng.exponential(2.0)));
}

/**
 * Force an outage burst starting at `from`, for the injector control.
 * Mutates `edge_available` in place; the rest of the future is untouched, which
 * is honest about what the injector does: it changes availability, not the
 * potential outcomes that were already drawn.
 */
export function injectOutage(trace, from, lengthFrames) {
  const end = Math.min(trace.frames, from + lengthFrames);
  for (let i = Math.max(0, from); i < end; i++) trace.edge_available[i] = 0;
  return end - Math.max(0, from);
}
