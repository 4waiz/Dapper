// DAPPER scheduler, ported from dapper/scheduler.py.
//
// This is the real decision rule, not a mock. Every arithmetic expression below
// is written in the same order as the Python original so that IEEE-754 doubles
// give bit-identical results; tests_web/test_js_parity.py replays exported
// Python traces through this module and requires agreement to within 1e-9.
//
// Risk score
//   R = w_r*min(1, RTT/D) + w_l*loss + w_c*load + w_a*min(1, age/D)
//       + w_d*min(1, (RTT + C_edge_hat)/D)

export const MODE_LOCAL_FAST = "local_fast";
export const MODE_EDGE_ACCURATE = "edge_accurate";
export const MODE_HYBRID = "hybrid";
export const MODE_DEGRADED_SAFE = "degraded_safe";

export const ALL_MODES = [MODE_LOCAL_FAST, MODE_EDGE_ACCURATE, MODE_HYBRID, MODE_DEGRADED_SAFE];

/** Every reason string the scheduler can emit. A decision is never unexplained. */
export const ALL_REASONS = [
  "edge_unavailable_reuse_fresh",
  "edge_unavailable_local_fallback",
  "risk_high_reuse_fresh",
  "risk_high_local_fallback",
  "risk_moderate_prefer_local",
  "local_confidence_sufficient",
  "low_confidence_edge_margin_ok",
  "low_confidence_hybrid_refresh",
  "low_confidence_refresh_infeasible",
];

/** Human-readable gloss for the audit log. Keyed by reason string. */
export const REASON_TEXT = {
  edge_unavailable_reuse_fresh: "edge unreachable — reused a still-fresh output, nothing sent",
  edge_unavailable_local_fallback: "edge unreachable and no fresh output — ran locally, nothing sent",
  risk_high_reuse_fresh: "R ≥ θD — reused a still-fresh output",
  risk_high_local_fallback: "R ≥ θD and no fresh output — ran the guaranteed local path",
  risk_moderate_prefer_local: "θL ≤ R < θD — stayed local for predictable latency",
  local_confidence_sufficient: "R < θL but c ≥ τc — the local model was confident enough",
  low_confidence_edge_margin_ok: "R < θL, c < τc, the expected completion fits mD — committed the frame to the edge",
  low_confidence_hybrid_refresh: "R < θL, c < τc, the optimistic completion fits W — local answer now, refinement requested",
  low_confidence_refresh_infeasible: "R < θL, c < τc, no feasible refresh — stayed local, sent nothing",
};

export function clip01(x) {
  if (x < 0.0) return 0.0;
  if (x > 1.0) return 1.0;
  return x;
}

export class DapperScheduler {
  /**
   * @param {object} p Flat parameter bag. `fromConfig` builds it from config.json.
   */
  constructor(p) {
    const total = p.weightRtt + p.weightLoss + p.weightLoad + p.weightFrameAge + p.weightDeadline;
    if (Math.abs(total - 1.0) > 1e-6) throw new Error(`risk weights must sum to 1.0, got ${total}`);
    if (Math.min(p.weightRtt, p.weightLoss, p.weightLoad, p.weightFrameAge, p.weightDeadline) < 0.0) {
      throw new Error("risk weights must be non-negative");
    }
    if (!(p.riskLocalThreshold < p.riskDegradedThreshold)) {
      throw new Error("riskLocalThreshold must be < riskDegradedThreshold");
    }
    if (p.hybridEstimator !== "optimistic" && p.hybridEstimator !== "expected") {
      throw new Error("hybridEstimator must be 'optimistic' or 'expected'");
    }

    this.wRtt = p.weightRtt;
    this.wLoss = p.weightLoss;
    this.wLoad = p.weightLoad;
    this.wAge = p.weightFrameAge;
    this.wDead = p.weightDeadline;

    this.riskDegraded = p.riskDegradedThreshold;
    this.riskLocal = p.riskLocalThreshold;
    this.confidenceThreshold = p.localConfidenceThreshold;
    this.remoteDeadlineMargin = p.remoteDeadlineMargin;

    this.hybridWindowMs = p.hybridFreshnessWindowMs;
    this.hybridEstimator = p.hybridEstimator;
    this.lastValidFreshnessMs = p.lastValidFreshnessMs;

    this.edgeComputeMs = p.edgeComputeMsEstimate;
    this.edgeComputeOptimisticMs = p.edgeComputeMsOptimistic;
    this.localComputeMs = p.localComputeMsEstimate;
    this.edgeExtraRttMs = p.edgeExtraRttMs;
    this.loadComputeScale = p.loadComputeScale;
    this.reuseLatencyMs = p.reuseLatencyMs;

    this.rttEstimateBias = p.rttEstimateBias ?? 0.0;
    this.computeEstimateBias = p.computeEstimateBias ?? 0.0;

    this.useConfidenceGate = p.useConfidenceGate !== false;
    this.useFreshnessGate = p.useFreshnessGate !== false;
    this.useDegradedReuse = p.useDegradedReuse !== false;
  }

  /** Access RTT the scheduler believes, including any estimation bias. */
  estimatedTotalRttMs(rttMs) {
    return (rttMs + this.edgeExtraRttMs) * (1.0 + this.rttEstimateBias);
  }

  estimatedComputeMs(edgeLoad, optimistic = false) {
    const base = optimistic ? this.edgeComputeOptimisticMs : this.edgeComputeMs;
    return base * (1.0 + this.loadComputeScale * edgeLoad) * (1.0 + this.computeEstimateBias);
  }

  /** Predicted time from frame capture to remote result arrival. */
  predictedRemoteArrivalMs(rttMs, edgeLoad, optimistic = false) {
    return this.estimatedTotalRttMs(rttMs) + this.estimatedComputeMs(edgeLoad, optimistic);
  }

  /** The five weighted, normalised risk terms and their sum, for display and for decide(). */
  riskTerms(x) {
    const deadline = Math.max(x.deadlineMs, 1.0);
    const estRtt = this.estimatedTotalRttMs(x.rttMs);
    const rttNorm = clip01(estRtt / deadline);
    const lossNorm = clip01(x.packetLoss);
    const loadNorm = clip01(x.edgeLoad);
    const ageNorm = clip01(x.frameAgeMs / deadline);
    const pressure = clip01(this.predictedRemoteArrivalMs(x.rttMs, x.edgeLoad) / deadline);
    const risk = clip01(
      this.wRtt * rttNorm +
        this.wLoss * lossNorm +
        this.wLoad * loadNorm +
        this.wAge * ageNorm +
        this.wDead * pressure,
    );
    return {
      risk,
      terms: [
        { key: "rtt", label: "RTT / D", weight: this.wRtt, normalised: rttNorm, contribution: this.wRtt * rttNorm },
        { key: "loss", label: "packet loss", weight: this.wLoss, normalised: lossNorm, contribution: this.wLoss * lossNorm },
        { key: "load", label: "edge load", weight: this.wLoad, normalised: loadNorm, contribution: this.wLoad * loadNorm },
        { key: "frame_age", label: "frame age / D", weight: this.wAge, normalised: ageNorm, contribution: this.wAge * ageNorm },
        { key: "deadline", label: "deadline pressure", weight: this.wDead, normalised: pressure, contribution: this.wDead * pressure },
      ],
    };
  }

  /** Weighted, normalised deadline-risk score in [0, 1]. */
  computeRisk(x) {
    return this.riskTerms(x).risk;
  }

  /**
   * Select a mode for this frame and explain why.
   * Mirrors DapperScheduler.decide, branch for branch.
   */
  decide(x) {
    const risk = this.computeRisk(x);
    const freshReuseAvailable = this.useDegradedReuse && x.lastValidAgeMs <= this.lastValidFreshnessMs;

    // 1. Edge unreachable: the health check already told us, so nothing is sent.
    if (!x.edgeAvailable) {
      if (freshReuseAvailable) {
        return {
          selectedMode: MODE_DEGRADED_SAFE,
          deadlineRiskScore: risk,
          reason: "edge_unavailable_reuse_fresh",
          expectedLatencyMs: this.reuseLatencyMs,
          fallbackNeeded: true,
          predictedRemoteArrivalMs: Infinity,
          remoteAttemptIntended: false,
        };
      }
      return {
        selectedMode: MODE_LOCAL_FAST,
        deadlineRiskScore: risk,
        reason: "edge_unavailable_local_fallback",
        expectedLatencyMs: this.localComputeMs,
        fallbackNeeded: true,
        predictedRemoteArrivalMs: Infinity,
        remoteAttemptIntended: false,
      };
    }

    const predicted = this.predictedRemoteArrivalMs(x.rttMs, x.edgeLoad);

    // 2. Severe risk: guarantee a timely output.
    if (risk >= this.riskDegraded) {
      return {
        selectedMode: MODE_DEGRADED_SAFE,
        deadlineRiskScore: risk,
        reason: freshReuseAvailable ? "risk_high_reuse_fresh" : "risk_high_local_fallback",
        expectedLatencyMs: freshReuseAvailable ? this.reuseLatencyMs : this.localComputeMs,
        fallbackNeeded: true,
        predictedRemoteArrivalMs: predicted,
        remoteAttemptIntended: false,
      };
    }

    // 3. Moderate risk: predictable local latency.
    if (risk >= this.riskLocal) {
      return {
        selectedMode: MODE_LOCAL_FAST,
        deadlineRiskScore: risk,
        reason: "risk_moderate_prefer_local",
        expectedLatencyMs: this.localComputeMs,
        fallbackNeeded: false,
        predictedRemoteArrivalMs: predicted,
        remoteAttemptIntended: false,
      };
    }

    // 4a. Low risk and the local model is confident enough: stay on device.
    if (this.useConfidenceGate && x.localConfidence >= this.confidenceThreshold) {
      return {
        selectedMode: MODE_LOCAL_FAST,
        deadlineRiskScore: risk,
        reason: "local_confidence_sufficient",
        expectedLatencyMs: this.localComputeMs,
        fallbackNeeded: false,
        predictedRemoteArrivalMs: predicted,
        remoteAttemptIntended: false,
      };
    }

    // 4b. Weak local result and the remote path fits the deadline with margin.
    if (predicted <= this.remoteDeadlineMargin * x.deadlineMs) {
      return {
        selectedMode: MODE_EDGE_ACCURATE,
        deadlineRiskScore: risk,
        reason: "low_confidence_edge_margin_ok",
        expectedLatencyMs: predicted,
        fallbackNeeded: false,
        predictedRemoteArrivalMs: predicted,
        remoteAttemptIntended: true,
      };
    }

    // 4c. A refresh could plausibly still land inside the freshness window.
    const refreshEstimate = this.predictedRemoteArrivalMs(
      x.rttMs,
      x.edgeLoad,
      this.hybridEstimator === "optimistic",
    );
    const refreshBudget = Math.min(this.hybridWindowMs, x.deadlineMs);
    if (!this.useFreshnessGate || refreshEstimate <= refreshBudget) {
      return {
        selectedMode: MODE_HYBRID,
        deadlineRiskScore: risk,
        reason: "low_confidence_hybrid_refresh",
        expectedLatencyMs: this.localComputeMs,
        fallbackNeeded: false,
        predictedRemoteArrivalMs: predicted,
        remoteAttemptIntended: true,
      };
    }

    // 4d. Any refresh would be rejected on arrival: do not spend bandwidth.
    return {
      selectedMode: MODE_LOCAL_FAST,
      deadlineRiskScore: risk,
      reason: "low_confidence_refresh_infeasible",
      expectedLatencyMs: this.localComputeMs,
      fallbackNeeded: false,
      predictedRemoteArrivalMs: predicted,
      remoteAttemptIntended: false,
    };
  }

  /** The numeric gate tests for this frame, for the lit-up Fig. 1 decision path. */
  gateTests(x) {
    const risk = this.computeRisk(x);
    const predicted = this.predictedRemoteArrivalMs(x.rttMs, x.edgeLoad);
    const optimistic = this.predictedRemoteArrivalMs(
      x.rttMs,
      x.edgeLoad,
      this.hybridEstimator === "optimistic",
    );
    return {
      risk,
      thetaL: this.riskLocal,
      thetaD: this.riskDegraded,
      confidence: x.localConfidence,
      tauC: this.confidenceThreshold,
      predicted,
      commitBudget: this.remoteDeadlineMargin * x.deadlineMs,
      optimistic,
      refreshBudget: Math.min(this.hybridWindowMs, x.deadlineMs),
      lastValidAgeMs: x.lastValidAgeMs,
      lastValidFreshnessMs: this.lastValidFreshnessMs,
      edgeAvailable: x.edgeAvailable,
    };
  }
}

/**
 * Build a scheduler from the exported config.json, mirroring
 * dapper.scheduler.build_scheduler_from_config.
 *
 * `weights` overrides the five risk weights (the ablation zeroes one and
 * renormalises); `rttEstimateBias` / `computeEstimateBias` perturb only what the
 * scheduler believes, never what the executor does.
 */
export function schedulerFromConfig(cfg, opts = {}) {
  const s = cfg.scheduler;
  const d = cfg.derived;
  const w = opts.weights || {
    weightRtt: s.weight_rtt,
    weightLoss: s.weight_loss,
    weightLoad: s.weight_load,
    weightFrameAge: s.weight_frame_age,
    weightDeadline: s.weight_deadline,
  };
  return new DapperScheduler({
    ...w,
    riskDegradedThreshold: s.risk_degraded_threshold,
    riskLocalThreshold: s.risk_local_threshold,
    localConfidenceThreshold: s.local_confidence_threshold,
    remoteDeadlineMargin: s.remote_deadline_margin,
    hybridFreshnessWindowMs: s.hybrid_freshness_window_ms,
    hybridEstimator: s.hybrid_estimator,
    lastValidFreshnessMs: s.last_valid_freshness_ms,
    edgeComputeMsEstimate: d.edge_compute_ms_estimate,
    edgeComputeMsOptimistic: d.edge_compute_ms_optimistic,
    localComputeMsEstimate: d.local_compute_ms_estimate,
    edgeExtraRttMs: d.extra_rtt_ms.edge,
    loadComputeScale: cfg.execution.load_compute_scale,
    reuseLatencyMs: cfg.execution.reuse_latency_ms,
    rttEstimateBias: opts.rttEstimateBias ?? 0.0,
    computeEstimateBias: opts.computeEstimateBias ?? 0.0,
    useConfidenceGate: opts.useConfidenceGate !== false,
    useFreshnessGate: opts.useFreshnessGate !== false,
    useDegradedReuse: opts.useDegradedReuse !== false,
  });
}

/**
 * Zero one risk component and renormalise the rest, mirroring
 * dapper.config.renormalised_weights.
 */
export function renormalisedWeights(cfg, drop) {
  const s = cfg.scheduler;
  const w = {
    weightRtt: s.weight_rtt,
    weightLoss: s.weight_loss,
    weightLoad: s.weight_load,
    weightFrameAge: s.weight_frame_age,
    weightDeadline: s.weight_deadline,
  };
  if (!drop) return w;
  const key = { rtt: "weightRtt", loss: "weightLoss", load: "weightLoad", frame_age: "weightFrameAge", deadline: "weightDeadline" }[drop];
  if (!key) throw new Error(`unknown risk component '${drop}'`);
  w[key] = 0.0;
  const total = Object.values(w).reduce((a, b) => a + b, 0);
  if (total <= 0.0) throw new Error(`dropping '${drop}' leaves no risk weight`);
  for (const k of Object.keys(w)) w[k] = w[k] / total;
  return w;
}
