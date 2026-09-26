// The published-results section: Tables III-VII and the Fig. 2 / Fig. 3 series,
// rendered from site/data/results.json.
//
// Every number here was read out of a CSV under results/ by
// experiments/export_web_results.py. Nothing is typed into this file, and the
// cells marked as best are marked from results.json's `rankings` block, which
// derives the direction from dapper.metrics.LOWER_IS_BETTER -- so "best" is a
// lookup, not an opinion.

import { groupedBars, stackedArea, xyLines } from "./charts.js";
import { MODE_COLOR, POLICY_COLOR, ci, int, num, pct, titleCase } from "./fmt.js";

const TABS = [
  ["main", "Table III · main benchmark"],
  ["placement", "Table IV · placement"],
  ["adaptive", "Table IV · comparators"],
  ["ablation", "Table V · ablation"],
  ["detector", "Table VI · detector"],
  ["replay", "Table VII · replay"],
];

export class Research {
  constructor(results, els) {
    this.r = results;
    this.els = els;
    this.active = "main";
  }

  render() {
    this.headline();
    this.tabs();
    this.table();
    this.honest();
    this.charts();
    this.els.chip.textContent =
      `${int(this.r.protocol.frame_decisions)} frame decisions · ${this.r.protocol.seeds} seeds · ` +
      `${this.r.protocol.profiles.length} profiles · ${this.r.protocol.policies.length} policies`;
  }

  headline() {
    const f = this.r.facts;
    const p = this.r.protocol;
    const cards = [
      [pct(f.dapper_worst_miss_pct), `worst deadline-miss rate over all ${p.profiles.length} profiles at D = ${p.deadline_ms} ms`],
      [pct(f.worst_stale_pct), "stale outputs accepted, over every policy, profile and seed"],
      [`${num(f.dapper_profile_mean_p95_ms, 1)} ms`, `profile-mean p95 control latency, against ${num(f.local_profile_mean_p95_ms, 1)} ms for local-only`],
      [`${num(f.dapper_profile_mean_bandwidth_mb_per_1k, 2)} MB`, "uplink charged per 1,000 frames, including replies that never arrive usable"],
      [`${num(f.scheduler_decision_us?.mean, 2)} µs`, `mean cost of one decision over ${int(f.scheduler_decision_us?.calls)} timed calls`],
      [`${f.calibration?.zero_worst_profile_miss}/${f.calibration?.candidates}`, "calibration candidates reaching zero worst-profile misses"],
    ];
    this.els.headline.innerHTML = cards
      .map(([v, s]) => `<div class="metric"><b>${v}</b><span>${s}</span></div>`)
      .join("");
  }

  tabs() {
    const available = TABS.filter(([id]) => this.r.tables[id]);
    this.els.tabs.innerHTML = available
      .map(([id, label]) => `<button class="tab${id === this.active ? " on" : ""}" data-tab="${id}" type="button" role="tab">${label}</button>`)
      .join("");
    for (const b of this.els.tabs.querySelectorAll("button")) {
      b.addEventListener("click", () => {
        this.active = b.dataset.tab;
        this.tabs();
        this.table();
      });
    }
  }

  table() {
    const t = this.r.tables[this.active];
    if (!t) return;
    this.els.title.textContent = `${t.title} (paper Table ${t.paper_table})`;
    this.els.caption.innerHTML = `${t.caption} <span class="ci">Source: ${t.source}</span>`;
    this.els.table.innerHTML = this[`t_${this.active}`](t);
  }

  // ------------------------------------------------------------- Table III
  t_main(t) {
    const head = `<thead><tr><th class="l">Profile</th><th class="l">Policy</th>
      <th>p95 latency (ms)</th><th>Deadline miss (%)</th><th>Confidence proxy</th>
      <th>Uplink (MB/1k frames)</th></tr></thead>`;
    let last = null;
    const rows = t.rows.map((r) => {
      const rank = this.r.rankings[r.profile] || {};
      const best = (metric) => (rank[metric]?.best_policies || []).includes(r.policy);
      const group = r.profile !== last ? `<tr class="group"><td colspan="6">${titleCase(r.profile)}</td></tr>` : "";
      last = r.profile;
      return `${group}<tr class="${r.policy === "dapper" ? "ours" : ""}">
        <td class="l dim">${titleCase(r.profile)}</td>
        <td class="l">${r.policy_label}</td>
        <td class="${best("p95_latency_ms") ? "best" : ""}">${ci(r.p95_latency_ms, 1)}</td>
        <td class="${best("deadline_miss_rate") ? "best" : ""}">${ci(r.deadline_miss_pct, 2)}</td>
        <td class="${best("usable_confidence_proxy") ? "best" : ""}">${num(r.usable_confidence_proxy.mean, 3)}</td>
        <td class="${best("bandwidth_per_1000_frames_kb") ? "best" : ""}">${num(r.bandwidth_mb_per_1k.mean, 2)}</td>
      </tr>`;
    });
    return head + `<tbody>${rows.join("")}</tbody>`;
  }

  // -------------------------------------------------------- Table IV left
  t_placement(t) {
    const head = `<thead><tr><th class="l">Profile</th><th>local-fast</th><th>edge-accurate</th>
      <th>hybrid</th><th>degraded-safe</th><th>transmitted</th><th>accepted</th>
      <th>reuse</th><th>stale</th></tr></thead>`;
    const rows = t.rows.map((r) => `<tr>
      <td class="l">${titleCase(r.profile)}</td>
      <td>${num(r.pct_local_fast, 1)}</td>
      <td>${num(r.pct_edge_accurate, 1)}</td>
      <td>${num(r.pct_hybrid, 1)}</td>
      <td>${num(r.pct_degraded_safe, 1)}</td>
      <td class="dim">${num(r.transmit_pct, 1)}</td>
      <td>${num(r.accepted_pct, 1)}</td>
      <td>${num(r.reuse_pct, 1)}</td>
      <td class="best">${num(r.stale_pct, 2)}</td></tr>`);
    return head + `<tbody>${rows.join("")}</tbody>`;
  }

  // ------------------------------------------------------- Table IV right
  t_adaptive(t) {
    const head = `<thead><tr><th class="l">Policy</th><th>Transmitted (%)</th><th>Deadline miss (%)</th>
      <th>p95 (ms)</th><th>Confidence proxy</th><th>Uplink (MB/1k)</th></tr></thead>`;
    const rows = t.rows.map((r) => `<tr class="${r.policy === "dapper" ? "ours" : ""}${r.is_oracle ? " oracle" : ""}">
      <td class="l">${r.policy_label}${r.is_oracle ? '<span class="ci"> — not deployable</span>' : ""}</td>
      <td>${num(r.transmit_pct, 2)}</td>
      <td>${num(r.deadline_miss_pct, 2)}</td>
      <td>${num(r.p95_latency_ms, 1)}</td>
      <td>${num(r.usable_confidence_proxy, 4)}</td>
      <td>${num(r.bandwidth_mb_per_1k, 2)}</td></tr>`);
    return head + `<tbody>${rows.join("")}</tbody>`;
  }

  // ----------------------------------------------------------- Table V
  t_ablation(t) {
    const head = `<thead><tr><th class="l">Variant</th><th class="l">Family</th><th>Deadline miss (%)</th>
      <th>p95 (ms)</th><th>Confidence proxy</th><th>Δ conf.</th><th>Uplink (MB/1k)</th>
      <th>Δ uplink</th><th>Reuse (%)</th></tr></thead>`;
    const rows = t.rows.map((r) => `<tr class="${r.variant === "dapper_full" ? "ours" : ""}">
      <td class="l">${r.label}</td>
      <td class="l dim">${r.family}</td>
      <td>${num(r.deadline_miss_pct, 2)}</td>
      <td>${num(r.p95_latency_ms, 2)}</td>
      <td>${num(r.usable_confidence_proxy, 4)}</td>
      <td class="dim">${r.variant === "dapper_full" ? "—" : signedCell(r.delta_confidence, 4)}</td>
      <td>${num(r.bandwidth_mb_per_1k, 2)}</td>
      <td class="dim">${r.variant === "dapper_full" ? "—" : signedCell(r.delta_bandwidth_mb_per_1k, 2)}</td>
      <td>${num(r.reuse_pct, 1)}</td></tr>`);
    return head + `<tbody>${rows.join("")}</tbody>`;
  }

  // ---------------------------------------------------------- Table VI
  t_detector(t) {
    const head = `<thead><tr><th class="l">Model</th><th class="l">Role</th><th>Images</th><th>Recall</th>
      <th>Safety recall</th><th>Precision</th><th>mAP50</th><th>mAP50–95</th>
      <th>GPU (ms)</th><th>CPU (ms)</th></tr></thead>`;
    const rows = t.rows.map((r) => `<tr class="${r.role === "remote" ? "ours" : ""}">
      <td class="l">${r.model}</td><td class="l dim">${r.note}</td>
      <td>${int(r.images)}</td><td>${num(r.recall, 3)}</td><td>${num(r.safety_recall, 3)}</td>
      <td>${num(r.precision, 3)}</td><td>${num(r.map50, 3)}</td><td>${num(r.map50_95, 3)}</td>
      <td>${num(r.gpu_ms, 1)}</td><td>${num(r.cpu_ms, 1)}</td></tr>`);
    const gap = this.r.facts.detector_gap;
    const note = gap
      ? `<tr class="group"><td colspan="10">the remote model leads by ${num(gap.recall_points, 1)} recall points and ${num(gap.safety_recall_points, 1)} safety-recall points</td></tr>`
      : "";
    return head + `<tbody>${rows.join("")}${note}</tbody>`;
  }

  // --------------------------------------------------------- Table VII
  t_replay(t) {
    const head = `<thead><tr><th class="l">Profile</th><th class="l">Policy</th>
      <th>Recall, all frames</th><th>Recall, fresh frames</th><th>Safety recall</th>
      <th>Deadline miss (%)</th><th>Reuse (%)</th></tr></thead>`;
    let last = null;
    const rows = t.rows.map((r) => {
      const group = r.profile !== last ? `<tr class="group"><td colspan="7">${titleCase(r.profile)}${r.in_paper ? "" : " — not printed in the camera-ready"}</td></tr>` : "";
      last = r.profile;
      const localAll = t.rows.find((x) => x.profile === r.profile && x.policy === "local_only")?.recall_all;
      const below = r.policy === "dapper" && localAll !== undefined && r.recall_all < localAll;
      return `${group}<tr class="${r.policy === "dapper" ? "ours" : ""}">
        <td class="l dim">${titleCase(r.profile)}</td><td class="l">${r.policy_label}</td>
        <td class="${below ? "miss" : ""}">${num(r.recall_all, 3)}</td>
        <td>${num(r.recall_fresh, 3)}</td>
        <td>${num(r.safety_recall, 3)}</td>
        <td>${num(r.deadline_miss_pct, 2)}</td>
        <td>${num(r.reuse_pct, 1)}</td></tr>`;
    });
    return head + `<tbody>${rows.join("")}</tbody>`;
  }

  // ------------------------------------------------------------- honest
  honest() {
    this.els.honest.innerHTML = (this.r.honest || [])
      .map((h) => {
        let detail = "";
        if (h.id === "all_frame_recall") {
          detail = `<div class="table-wrap"><table class="data"><thead><tr><th class="l">Profile</th>
            <th>DAPPER, all frames</th><th>local-only</th><th>DAPPER, fresh frames</th><th>Reuse (%)</th></tr></thead><tbody>` +
            h.profiles.map((p) => `<tr><td class="l">${titleCase(p.profile)}</td>
              <td class="miss">${num(p.dapper_recall_all, 3)}</td><td>${num(p.local_recall_all, 3)}</td>
              <td>${num(p.dapper_recall_fresh, 3)}</td><td>${num(p.reuse_pct, 1)}</td></tr>`).join("") +
            `</tbody></table></div>`;
        } else if (h.id === "bandwidth_cost") {
          detail = `<div class="table-wrap"><table class="data"><thead><tr><th class="l">Profile</th>
            <th>Uplink (MB/1k)</th><th>Accepted (%)</th><th>Late (%)</th><th>Lost (%)</th>
            <th>Wasted (MB/1k)</th></tr></thead><tbody>` +
            h.profiles.map((p) => `<tr><td class="l">${titleCase(p.profile)}</td>
              <td>${num(p.bandwidth_mb_per_1k, 2)}</td><td>${num(p.accepted_pct, 1)}</td>
              <td class="miss">${num(p.rejected_deadline_pct, 1)}</td><td class="miss">${num(p.lost_pct, 1)}</td>
              <td class="miss">${num(p.wasted_mb_per_1k, 2)}</td></tr>`).join("") +
            `</tbody></table></div>`;
        } else if (h.id === "larger_deadlines") {
          detail = `<div class="table-wrap"><table class="data"><thead><tr><th class="l">Deadline D</th>
            <th>Deadline miss (%)</th><th>edge-accurate share (%)</th></tr></thead><tbody>` +
            h.deadlines.map((d) => `<tr><td class="l">${num(d.deadline_ms, 0)} ms</td>
              <td class="miss">${num(d.miss_pct, 2)}</td><td>${num(d.pct_edge_accurate, 1)}</td></tr>`).join("") +
            `</tbody></table></div>`;
        }
        return `<div class="honest-item"><h3>${h.headline}</h3><p>${h.explanation}</p>${detail}
          <p class="src">${h.source} · paper ${h.paper_ref}</p></div>`;
      })
      .join("");
  }

  // ------------------------------------------------------------- charts
  charts() {
    const s = this.r.series;
    const profiles = this.r.protocol.profiles;
    const policies = ["local_only", "edge_only", "cloud_only", "dapper"];
    const label = { local_only: "local-only", edge_only: "edge-only", cloud_only: "cloud-only", dapper: "DAPPER" };
    const bars = (key) =>
      policies
        .filter((p) => s[key] && s[key][p] && s[key][p].length)
        .map((p) => ({
          label: label[p],
          color: POLICY_COLOR[p],
          points: profiles.map((prof) => s[key][p].find((q) => q.profile === prof) || null),
        }));

    if (this.els.p95) {
      groupedBars(this.els.p95, { categories: profiles, series: bars("p95_by_profile"), height: 220 });
    }
    if (this.els.confidence) {
      groupedBars(this.els.confidence, {
        categories: profiles,
        series: bars("confidence_by_profile"),
        yMax: 1,
        yFormat: (v) => v.toFixed(1),
        height: 220,
      });
    }
    if (this.els.deadline && s.mode_vs_deadline) {
      const x = s.mode_vs_deadline.map((d) => d.deadline_ms);
      stackedArea(this.els.deadline, {
        x,
        xLabel: "control deadline D (ms)",
        height: 220,
        series: [
          { label: "local-fast", color: MODE_COLOR.local_fast, values: s.mode_vs_deadline.map((d) => d.pct_local_fast) },
          { label: "edge-accurate", color: MODE_COLOR.edge_accurate, values: s.mode_vs_deadline.map((d) => d.pct_edge_accurate) },
          { label: "hybrid", color: MODE_COLOR.hybrid, values: s.mode_vs_deadline.map((d) => d.pct_hybrid) },
          { label: "degraded-safe", color: MODE_COLOR.degraded_safe, values: s.mode_vs_deadline.map((d) => d.pct_degraded_safe) },
        ],
      });
    }
    if (this.els.bias && s.miss_vs_bias) {
      const targets = [["rtt", "RTT estimate", "#22d3ee"], ["compute", "edge-compute estimate", "#f5a524"], ["both", "both estimates", "#ff3d71"]];
      const first = s.miss_vs_bias[targets[0][0]] || [];
      xyLines(this.els.bias, {
        x: first.map((p) => `${Math.round(p.bias * 100)}`),
        xLabel: "estimate bias (%), negative = underestimate",
        height: 220,
        yFormat: (v) => v.toFixed(1),
        series: targets
          .filter(([k]) => s.miss_vs_bias[k])
          .map(([k, lab, color]) => ({ label: lab, color, values: s.miss_vs_bias[k].map((p) => p.miss_pct_mean) })),
      });
    }
  }
}

function signedCell(x, digits) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  if (Math.abs(x) < 0.5 * 10 ** -digits) return (0).toFixed(digits);
  return (x > 0 ? "+" : "−") + Math.abs(x).toFixed(digits);
}
