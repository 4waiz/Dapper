// Per-run metric reduction, ported from dapper/metrics.py.
//
// `usable_confidence_proxy` is the mean over all frames of
// `confidence if (on time and fresh) else 0`: the expected confidence the
// control loop actually holds at the deadline. It is computed from the synthetic
// perception model and is never a detector accuracy.
//
// Two numerical details keep this in step with NumPy to within 1e-9:
//   * percentile() reproduces numpy.percentile's default 'linear' method,
//     including the two-sided _lerp that numpy uses for stability;
//   * sum() is Neumaier-compensated, which is at least as accurate as the
//     pairwise summation numpy uses, so the two agree far inside the tolerance.

import { ALL_MODES } from "./scheduler.js";

/** Neumaier (improved Kahan) compensated summation. */
export function sum(xs) {
  let s = 0.0;
  let c = 0.0;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    const t = s + x;
    if (Math.abs(s) >= Math.abs(x)) c += s - t + x;
    else c += x - t + s;
    s = t;
  }
  return s + c;
}

export function mean(xs) {
  return xs.length ? sum(xs) / xs.length : NaN;
}

/**
 * numpy.percentile(x, q) with the default 'linear' interpolation.
 * `sorted` must already be ascending.
 */
export function percentile(sorted, q) {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const t = pos - lo;
  const a = sorted[lo];
  const b = sorted[hi];
  const diff = b - a;
  // numpy's _lerp switches form at t >= 0.5 to avoid cancellation.
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

function column(rows, key) {
  const out = new Float64Array(rows.length);
  for (let i = 0; i < rows.length; i++) out[i] = rows[i][key];
  return out;
}

function countTrue(rows, key) {
  let n = 0;
  for (const r of rows) if (r[key]) n += 1;
  return n;
}

function rate(rows, key) {
  return rows.length ? countTrue(rows, key) / rows.length : 0.0;
}

/**
 * Reduce a per-frame log for one run into summary statistics.
 * Mirrors dapper.metrics.compute_metrics field for field.
 */
export function computeMetrics(rows, modeSwitches = null) {
  const n = rows.length;
  if (n === 0) throw new Error("cannot compute metrics for an empty run");

  const lat = column(rows, "control_output_latency_ms");
  const latSorted = Float64Array.from(lat).sort();
  const bw = column(rows, "bandwidth_kb");

  const nAttempt = countTrue(rows, "remote_attempted");
  const nAccept = countTrue(rows, "remote_accepted");

  const shares = {};
  for (const m of ALL_MODES) {
    let k = 0;
    for (const r of rows) if (r.selected_mode === m) k += 1;
    shares[m] = (k / n) * 100.0;
  }

  if (modeSwitches === null) {
    modeSwitches = 0;
    for (let i = 1; i < n; i++) if (rows[i].selected_mode !== rows[i - 1].selected_mode) modeSwitches += 1;
  }

  const bwPerFrame = mean(bw);
  const per = (x) => (nAttempt ? x / nAttempt : 0.0);

  return {
    policy: rows[0].policy,
    profile: rows[0].profile,
    seed: rows[0].seed,
    deadline_ms: rows[0].deadline_ms,
    frames: n,

    mean_latency_ms: mean(lat),
    p95_latency_ms: percentile(latSorted, 95),
    p99_latency_ms: percentile(latSorted, 99),
    max_latency_ms: latSorted[n - 1],
    mean_final_latency_ms: mean(column(rows, "final_output_latency_ms")),
    // Counted directly rather than as 1 - hit_rate, so the division matches the
    // integer-count-over-n that numpy's mean of a boolean array performs.
    deadline_miss_rate: (n - countTrue(rows, "deadline_met")) / n,

    mean_output_confidence: mean(column(rows, "output_confidence")),
    usable_confidence_proxy: mean(column(rows, "usable_confidence_proxy")),

    total_bandwidth_kb: sum(bw),
    bandwidth_per_frame_kb: bwPerFrame,
    bandwidth_per_1000_frames_kb: bwPerFrame * 1000.0,

    fallback_rate: rate(rows, "fallback_used"),
    reuse_rate: rate(rows, "output_reused"),
    stale_acceptance_rate: rate(rows, "stale_output_accepted"),

    remote_attempt_rate: nAttempt / n,
    remote_success_rate: per(countTrue(rows, "remote_succeeded")),
    remote_accept_rate: per(nAccept),
    remote_accepted_per_frame: nAccept / n,
    remote_rejected_deadline_rate: per(countTrue(rows, "remote_rejected_deadline")),
    remote_rejected_freshness_rate: per(countTrue(rows, "remote_rejected_freshness")),
    remote_failed_loss_rate: per(countTrue(rows, "remote_failed_loss")),
    remote_unreachable_rate: rate(rows, "remote_failed_unavailable"),

    pct_local_fast: shares.local_fast,
    pct_edge_accurate: shares.edge_accurate,
    pct_hybrid: shares.hybrid,
    pct_degraded_safe: shares.degraded_safe,

    mode_switches: modeSwitches,
    mode_switches_per_1000: (modeSwitches * 1000.0) / n,
  };
}

/** Metrics for which a lower value is better; used so no direction is hard-coded in prose. */
export const LOWER_IS_BETTER = new Set([
  "mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "max_latency_ms",
  "deadline_miss_rate", "total_bandwidth_kb", "bandwidth_per_frame_kb",
  "bandwidth_per_1000_frames_kb", "stale_acceptance_rate",
  "remote_rejected_deadline_rate", "remote_rejected_freshness_rate",
  "remote_failed_loss_rate", "mode_switches", "mode_switches_per_1000",
]);

/**
 * Incremental accumulator for the live KPI strip, so the dashboard does not
 * recompute the whole run every animation frame. `snapshot()` reproduces the
 * subset of computeMetrics the KPIs display, over the frames seen so far.
 */
export class LiveMetrics {
  constructor(windowFrames = 600) {
    this.windowFrames = windowFrames;
    this.reset();
  }

  reset() {
    this.n = 0;
    this.misses = 0;
    this.usableSum = 0.0;
    this.bwSum = 0.0;
    this.reuse = 0;
    this.stale = 0;
    this.attempted = 0;
    this.accepted = 0;
    this.late = 0;
    this.lost = 0;
    this.rejFresh = 0;
    this.modeCounts = Object.fromEntries(ALL_MODES.map((m) => [m, 0]));
    this.modeSwitches = 0;
    this.prevMode = null;
    this.latencies = [];
  }

  push(row) {
    this.n += 1;
    if (!row.deadline_met) this.misses += 1;
    this.usableSum += row.usable_confidence_proxy;
    this.bwSum += row.bandwidth_kb;
    if (row.output_reused) this.reuse += 1;
    if (row.stale_output_accepted) this.stale += 1;
    if (row.remote_attempted) this.attempted += 1;
    if (row.remote_accepted) this.accepted += 1;
    if (row.remote_rejected_deadline) this.late += 1;
    if (row.remote_rejected_freshness) this.rejFresh += 1;
    if (row.remote_failed_loss) this.lost += 1;
    this.modeCounts[row.selected_mode] += 1;
    if (this.prevMode !== null && row.selected_mode !== this.prevMode) this.modeSwitches += 1;
    this.prevMode = row.selected_mode;
    this.latencies.push(row.control_output_latency_ms);
    if (this.latencies.length > this.windowFrames) this.latencies.shift();
  }

  snapshot() {
    const n = Math.max(this.n, 1);
    const sorted = Float64Array.from(this.latencies).sort();
    return {
      frames: this.n,
      deadline_miss_rate: this.misses / n,
      p95_latency_ms: percentile(sorted, 95),
      usable_confidence_proxy: this.usableSum / n,
      bandwidth_per_1000_frames_kb: (this.bwSum / n) * 1000.0,
      reuse_rate: this.reuse / n,
      stale_acceptance_rate: this.stale / n,
      stale_outputs_accepted: this.stale,
      remote_attempted: this.attempted,
      remote_accepted: this.accepted,
      remote_rejected_deadline: this.late,
      remote_rejected_freshness: this.rejFresh,
      remote_failed_loss: this.lost,
      remote_accept_rate: this.attempted ? this.accepted / this.attempted : 0.0,
      mode_share: Object.fromEntries(ALL_MODES.map((m) => [m, (this.modeCounts[m] / n) * 100.0])),
      mode_switches_per_1000: (this.modeSwitches * 1000.0) / n,
    };
  }
}
