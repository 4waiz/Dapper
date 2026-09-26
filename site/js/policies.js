// Every evaluated execution policy, ported from dapper/policies.py.
//
// All policies share one interface and one execution model (executor.js), so the
// only thing that differs between them is the decision rule. The oracle reads
// the pre-generated future and is never deployable; it exists to bound how much
// a perfect-information scheduler could gain.

import {
  MODE_DEGRADED_SAFE,
  MODE_EDGE_ACCURATE,
  MODE_HYBRID,
  MODE_LOCAL_FAST,
  schedulerFromConfig,
} from "./scheduler.js";

export const MAIN_POLICIES = ["local_only", "edge_only", "cloud_only", "dapper"];
export const ADAPTIVE_POLICIES = ["deadline_greedy", "confidence_deadline", "rtt_threshold"];
export const ORACLE_POLICIES = ["oracle_feasible"];
export const ALL_POLICY_NAMES = [...MAIN_POLICIES, ...ADAPTIVE_POLICIES, ...ORACLE_POLICIES];

export const POLICY_LABEL = {
  local_only: "local-only",
  edge_only: "edge-only",
  cloud_only: "cloud-only",
  dapper: "DAPPER",
  deadline_greedy: "deadline-greedy",
  confidence_deadline: "conf.+deadline",
  rtt_threshold: "RTT-threshold",
  oracle_feasible: "oracle (not deployable)",
};

function decision(mode, reason, expected, predicted, attempt, fallback = false, risk = NaN) {
  return {
    selectedMode: mode,
    deadlineRiskScore: risk,
    reason,
    expectedLatencyMs: expected,
    fallbackNeeded: fallback,
    predictedRemoteArrivalMs: predicted,
    remoteAttemptIntended: attempt,
  };
}

class Policy {
  constructor(cfg, name, backend = "edge") {
    this.cfg = cfg;
    this.name = name;
    this.backend = backend;
    this.isOracle = false;
    this.localEstimateMs = 0.5 * (cfg.local.latency_ms_min + cfg.local.latency_ms_max);
  }

  reset() {}

  /** Nominal predicted remote arrival, from observable quantities only. */
  predictedRemoteMs(obs, model) {
    return (
      model.totalRttMs(obs.rttMs, this.backend) +
      model.nominalComputeMs[this.backend] * (1.0 + model.loadComputeScale * obs.edgeLoad)
    );
  }
}

class LocalOnlyPolicy extends Policy {
  decide() {
    return decision(MODE_LOCAL_FAST, "fixed_baseline:local_only", this.localEstimateMs, Infinity, false);
  }
}

class FixedRemotePolicy extends Policy {
  decide(obs, trace, i, model) {
    const predicted = this.predictedRemoteMs(obs, model);
    if (!obs.edgeAvailable && model.edgeHealthCheck) {
      return decision(
        MODE_EDGE_ACCURATE,
        `fixed_baseline:${this.name}:unreachable`,
        this.localEstimateMs,
        predicted,
        false,
        true,
      );
    }
    return decision(MODE_EDGE_ACCURATE, `fixed_baseline:${this.name}`, predicted, predicted, true);
  }
}

class DapperPolicy extends Policy {
  constructor(cfg, opts = {}) {
    super(cfg, opts.name || "dapper", "edge");
    this.scheduler = opts.scheduler || schedulerFromConfig(cfg, opts);
  }

  decide(obs) {
    return this.scheduler.decide(obs);
  }
}

class DeadlineGreedyPolicy extends Policy {
  constructor(cfg, margin) {
    super(cfg, "deadline_greedy");
    this.margin = margin;
  }

  decide(obs, trace, i, model) {
    const predicted = this.predictedRemoteMs(obs, model);
    if (obs.edgeAvailable && predicted <= this.margin * obs.deadlineMs) {
      return decision(MODE_EDGE_ACCURATE, "deadline_greedy_edge", predicted, predicted, true);
    }
    return decision(MODE_LOCAL_FAST, "deadline_greedy_local", this.localEstimateMs, predicted, false);
  }
}

class ConfidenceDeadlinePolicy extends Policy {
  constructor(cfg, confidenceThreshold, margin) {
    super(cfg, "confidence_deadline");
    this.confidenceThreshold = confidenceThreshold;
    this.margin = margin;
  }

  decide(obs, trace, i, model) {
    const predicted = this.predictedRemoteMs(obs, model);
    if (obs.localConfidence >= this.confidenceThreshold) {
      return decision(MODE_LOCAL_FAST, "confidence_deadline_local_confident", this.localEstimateMs, predicted, false);
    }
    if (obs.edgeAvailable && predicted <= this.margin * obs.deadlineMs) {
      return decision(MODE_EDGE_ACCURATE, "confidence_deadline_edge", predicted, predicted, true);
    }
    return decision(MODE_LOCAL_FAST, "confidence_deadline_local_fallback", this.localEstimateMs, predicted, false);
  }
}

class RttThresholdPolicy extends Policy {
  constructor(cfg, rttThresholdMs) {
    super(cfg, "rtt_threshold");
    this.rttThresholdMs = rttThresholdMs;
  }

  decide(obs, trace, i, model) {
    const predicted = this.predictedRemoteMs(obs, model);
    if (obs.edgeAvailable && obs.rttMs <= this.rttThresholdMs) {
      return decision(MODE_EDGE_ACCURATE, "rtt_threshold_edge", predicted, predicted, true);
    }
    return decision(MODE_LOCAL_FAST, "rtt_threshold_local", this.localEstimateMs, predicted, false);
  }
}

/**
 * OFFLINE ORACLE UPPER BOUND — NOT DEPLOYABLE.
 * Reads the pre-generated frame outcome and offloads exactly when the remote
 * reply would in fact arrive on time, fresh and unlost, and would be better.
 */
class OracleFeasiblePolicy extends Policy {
  constructor(cfg) {
    super(cfg, "oracle_feasible");
    this.isOracle = true;
  }

  decide(obs, trace, i, model) {
    const predicted = this.predictedRemoteMs(obs, model);
    if (!obs.edgeAvailable) {
      return decision(MODE_LOCAL_FAST, "oracle_unreachable_local", this.localEstimateMs, predicted, false);
    }
    const lost = trace.edge_loss_draw[i] < trace.packet_loss[i];
    const arrival = model.trueArrivalMs(trace.rtt_ms[i], trace.edge_compute_ms[i], trace.edge_load[i], this.backend);
    const timeout = model.clientTimeoutMs(trace.rtt_ms[i], trace.edge_load[i], this.backend);
    const feasible = !lost && arrival <= timeout && arrival <= model.deadlineMs;
    const better = trace.edge_confidence[i] > trace.local_confidence[i];
    if (feasible && better) {
      return decision(MODE_EDGE_ACCURATE, "oracle_edge_feasible", arrival, predicted, true);
    }
    return decision(MODE_LOCAL_FAST, "oracle_local", this.localEstimateMs, predicted, false);
  }
}

/** Construct a policy by name, mirroring dapper.policies.build_policy. */
export function buildPolicy(name, cfg, opts = {}) {
  const bp = opts.baselineParams || cfg.baselines || {};
  switch (name) {
    case "local_only":
      return new LocalOnlyPolicy(cfg, "local_only");
    case "edge_only":
      return new FixedRemotePolicy(cfg, "edge_only", "edge");
    case "cloud_only":
      return new FixedRemotePolicy(cfg, "cloud_only", "cloud");
    case "dapper":
      return new DapperPolicy(cfg, opts);
    case "deadline_greedy":
      return new DeadlineGreedyPolicy(cfg, bp.deadline_greedy_margin ?? 0.9);
    case "confidence_deadline":
      return new ConfidenceDeadlinePolicy(
        cfg,
        bp.confidence_deadline_threshold ?? 0.725,
        bp.confidence_deadline_margin ?? 0.9,
      );
    case "rtt_threshold":
      return new RttThresholdPolicy(cfg, bp.rtt_threshold_ms ?? 30.0);
    case "oracle_feasible":
      return new OracleFeasiblePolicy(cfg);
    default:
      throw new Error(`unknown policy: ${name}`);
  }
}
