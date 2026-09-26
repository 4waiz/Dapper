// The paper's Fig. 1 decision path, lit per frame, plus the numeric test behind
// every gate and the five-term breakdown of R_t.
//
// The graph below mirrors the branch structure of dapper/scheduler.py, which is
// what Fig. 1 was drawn from. `light()` takes a decision and the gate tests the
// scheduler computed for that frame, so the path shown is the path taken -- the
// reason string decides which nodes are lit, not a heuristic in this file.

import { MODE_LABEL, TERM_COLOR } from "./fmt.js";

const W = 900;
const H = 320;

// id, x, y, w, h, label, kind
const NODES = [
  ["monitor", 6, 128, 110, 50, "runtime monitor|RTT, loss, load", "in"],
  ["risk", 160, 128, 104, 50, "deadline risk|score Rₜ", "gate"],
  ["conf", 316, 128, 100, 50, "confidence|gate τc", "gate"],
  ["feas", 468, 128, 100, 50, "feasibility|gate", "gate"],
  ["edge", 624, 22, 118, 40, "edge-accurate", "edge"],
  ["hybrid", 624, 84, 118, 40, "hybrid", "hybrid"],
  ["local", 624, 178, 118, 40, "local-fast", "local"],
  ["degraded", 624, 248, 118, 46, "degraded-safe|reuse or local", "degraded"],
  ["accept", 782, 50, 112, 44, "accept?|deadline + fresh", "gate"],
  ["control", 782, 178, 112, 44, "control|output", "out"],
];

// from, to, label, and where the label sits. Label positions are given
// explicitly rather than derived from the curve: three links converge on
// local-fast, and a midpoint rule puts their conditions on top of each other and
// on top of the gate boxes. These coordinates keep every condition in clear
// space, which matters more here than a tidy rule.
const EDGES = [
  ["monitor", "risk", "", 0, 0],
  ["risk", "conf", "Rₜ < θL", 290, 146],
  ["risk", "local", "θL ≤ Rₜ < θD", 300, 206],
  ["risk", "degraded", "Rₜ ≥ θD", 300, 254],
  ["conf", "feas", "cₜ < τc", 442, 146],
  ["conf", "local", "cₜ ≥ τc", 452, 232],
  ["feas", "edge", "T̂ ≤ mD", 596, 96],
  ["feas", "hybrid", "T̂⁻ ≤ W", 596, 142],
  ["feas", "local", "no feasible refresh", 560, 206],
  ["edge", "accept", "", 0, 0],
  ["hybrid", "accept", "", 0, 0],
  ["accept", "control", "on time + fresh", 790, 140],
  ["local", "control", "", 0, 0],
  ["degraded", "control", "", 0, 0],
];

/** Which nodes each reason string lights, in the order the scheduler visited them. */
const REASON_PATH = {
  edge_unavailable_reuse_fresh: ["monitor", "risk", "degraded", "control"],
  edge_unavailable_local_fallback: ["monitor", "risk", "local", "control"],
  risk_high_reuse_fresh: ["monitor", "risk", "degraded", "control"],
  risk_high_local_fallback: ["monitor", "risk", "degraded", "control"],
  risk_moderate_prefer_local: ["monitor", "risk", "local", "control"],
  local_confidence_sufficient: ["monitor", "risk", "conf", "local", "control"],
  low_confidence_edge_margin_ok: ["monitor", "risk", "conf", "feas", "edge", "accept", "control"],
  low_confidence_hybrid_refresh: ["monitor", "risk", "conf", "feas", "hybrid", "accept", "control"],
  low_confidence_refresh_infeasible: ["monitor", "risk", "conf", "feas", "local", "control"],
};

const REASON_EDGES = {
  edge_unavailable_reuse_fresh: [["monitor", "risk"], ["risk", "degraded"], ["degraded", "control"]],
  edge_unavailable_local_fallback: [["monitor", "risk"], ["risk", "local"], ["local", "control"]],
  risk_high_reuse_fresh: [["monitor", "risk"], ["risk", "degraded"], ["degraded", "control"]],
  risk_high_local_fallback: [["monitor", "risk"], ["risk", "degraded"], ["degraded", "control"]],
  risk_moderate_prefer_local: [["monitor", "risk"], ["risk", "local"], ["local", "control"]],
  local_confidence_sufficient: [["monitor", "risk"], ["risk", "conf"], ["conf", "local"], ["local", "control"]],
  low_confidence_edge_margin_ok: [["monitor", "risk"], ["risk", "conf"], ["conf", "feas"], ["feas", "edge"], ["edge", "accept"], ["accept", "control"]],
  low_confidence_hybrid_refresh: [["monitor", "risk"], ["risk", "conf"], ["conf", "feas"], ["feas", "hybrid"], ["hybrid", "accept"], ["accept", "control"]],
  low_confidence_refresh_infeasible: [["monitor", "risk"], ["risk", "conf"], ["conf", "feas"], ["feas", "local"], ["local", "control"]],
};

const byId = Object.fromEntries(NODES.map((n) => [n[0], { id: n[0], x: n[1], y: n[2], w: n[3], h: n[4], label: n[5], kind: n[6] }]));

function anchor(a, b) {
  // Leave from the right edge, arrive at the left edge, unless the target is
  // directly below (the degraded branch), in which case go from the bottom.
  const from = { x: a.x + a.w, y: a.y + a.h / 2 };
  const to = { x: b.x, y: b.y + b.h / 2 };
  return { from, to };
}

function path(a, b) {
  const { from, to } = anchor(a, b);
  const mid = from.x + (to.x - from.x) * 0.45;
  return `M ${from.x} ${from.y} C ${mid} ${from.y}, ${mid} ${to.y}, ${to.x} ${to.y}`;
}

export class DecisionPath {
  constructor(container, testsEl) {
    this.container = container;
    this.testsEl = testsEl;
    this.container.innerHTML = this.svg();
    this.svgEl = this.container.querySelector("svg");
  }

  svg() {
    const edges = EDGES.map(([f, t, label, lx, ly]) => {
      const a = byId[f];
      const b = byId[t];
      // "link", not "edge": a node's kind can be "edge" (edge-accurate), and a
      // shared class name would make the node groups match this selector too.
      // Labels carry a background-coloured halo (paint-order, in the stylesheet)
      // so a link passing behind one never eats a character.
      return `<g class="link" data-link="${f}->${t}">
        <path class="edge-line" d="${path(a, b)}" />
        ${label ? `<text class="edge-label" x="${lx}" y="${ly}" text-anchor="middle">${label}</text>` : ""}
      </g>`;
    }).join("");

    const nodes = NODES.map(([id, x, y, w, h, label]) => {
      const lines = label.split("|");
      const text = lines
        .map((ln, i) => `<tspan x="${x + w / 2}" dy="${i === 0 ? 0 : 12}">${ln}</tspan>`)
        .join("");
      const firstY = y + h / 2 - (lines.length - 1) * 6 + 4;
      return `<g class="node" data-node="${id}">
        <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" />
        <text x="${x + w / 2}" y="${firstY}" text-anchor="middle">${text}</text>
      </g>`;
    }).join("");

    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="DAPPER decision path">
      <g>${edges}</g><g>${nodes}</g>
    </svg>`;
  }

  /** Light the path this frame took and print the numbers behind each gate. */
  light(decision, tests) {
    const reason = decision.reason;
    const mode = decision.selectedMode;
    const litNodes = new Set(REASON_PATH[reason] || []);
    const litEdges = new Set((REASON_EDGES[reason] || []).map(([a, b]) => `${a}->${b}`));

    for (const g of this.svgEl.querySelectorAll(".node")) {
      const id = g.dataset.node;
      g.setAttribute("class", `node${litNodes.has(id) ? " lit" : ""} ${byId[id].kind}`);
    }
    for (const g of this.svgEl.querySelectorAll(".link")) {
      const lit = litEdges.has(g.dataset.link);
      g.querySelector(".edge-line").setAttribute("class", `edge-line${lit ? " lit" : ""}`);
      const label = g.querySelector(".edge-label");
      if (label) label.setAttribute("class", `edge-label${lit ? " lit" : ""}`);
    }

    this.testsEl.innerHTML = this.tests(decision, tests, mode);
  }

  tests(decision, t, mode) {
    const f = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "∞");
    const rows = [];
    const add = (fired, label, expr, verdict) =>
      rows.push(`<li class="${fired ? "fired" : ""}"><span>${label}</span><span>${expr}
        <span class="verdict ${verdict ? "yes" : "no"}">${verdict ? "yes" : "no"}</span></span></li>`);

    add(!t.edgeAvailable, "edge reachable (health check)", "", t.edgeAvailable);
    add(t.risk >= t.thetaD, "Rₜ ≥ θD", `${f(t.risk, 3)} ≥ ${f(t.thetaD, 2)}`, t.risk >= t.thetaD);
    add(t.risk >= t.thetaL && t.risk < t.thetaD, "θL ≤ Rₜ < θD",
      `${f(t.thetaL, 2)} ≤ ${f(t.risk, 3)} < ${f(t.thetaD, 2)}`, t.risk >= t.thetaL && t.risk < t.thetaD);

    const reachedConf = t.risk < t.thetaL && t.edgeAvailable;
    add(reachedConf && t.confidence >= t.tauC, "cₜ ≥ τc",
      `${f(t.confidence, 3)} ≥ ${f(t.tauC, 2)}`, t.confidence >= t.tauC);
    const reachedFeas = reachedConf && t.confidence < t.tauC;
    add(reachedFeas && t.predicted <= t.commitBudget, "T̂ ≤ mD",
      `${f(t.predicted, 1)} ≤ ${f(t.commitBudget, 1)} ms`, t.predicted <= t.commitBudget);
    add(reachedFeas && t.predicted > t.commitBudget && t.optimistic <= t.refreshBudget,
      "T̂⁻ ≤ W", `${f(t.optimistic, 1)} ≤ ${f(t.refreshBudget, 1)} ms`,
      t.optimistic <= t.refreshBudget);
    const reuseOk = t.lastValidAgeMs <= t.lastValidFreshnessMs;
    add(mode === "degraded_safe", "last-valid age ≤ freshness window",
      `${Number.isFinite(t.lastValidAgeMs) ? f(t.lastValidAgeMs, 1) : "—"} ≤ ${f(t.lastValidFreshnessMs, 0)} ms`,
      reuseOk);

    return rows.join("");
  }
}

/** The five weighted, normalised terms of R_t, plus the threshold marks. */
export class RiskPanel {
  constructor({ valueEl, verdictEl, barEl, marksEl, tableEl }) {
    this.valueEl = valueEl;
    this.verdictEl = verdictEl;
    this.barEl = barEl;
    this.marksEl = marksEl;
    this.tableEl = tableEl;
    this.marksDrawn = null;
  }

  update(riskTerms, tests, mode) {
    const { risk, terms } = riskTerms;
    this.valueEl.textContent = risk.toFixed(4);
    this.valueEl.style.color =
      risk >= tests.thetaD ? "var(--degraded)" : risk >= tests.thetaL ? "var(--hybrid)" : "var(--local)";
    const band = risk >= tests.thetaD
      ? `≥ θD (${tests.thetaD.toFixed(2)}) → degraded-safe`
      : risk >= tests.thetaL
        ? `in [θL, θD) → local-fast`
        : `< θL (${tests.thetaL.toFixed(2)}) → reaches the confidence gate`;
    this.verdictEl.textContent = `${band} · selected ${MODE_LABEL[mode]}`;

    this.barEl.innerHTML = terms
      .map((t) => `<i class="t-${t.key}" style="width:${(t.contribution * 100).toFixed(3)}%"></i>`)
      .join("") + `<i style="flex:1"></i>`;

    const marks = `${tests.thetaL}|${tests.thetaD}`;
    if (this.marksDrawn !== marks) {
      this.marksDrawn = marks;
      this.marksEl.innerHTML =
        `<i style="left:${(tests.thetaL * 100).toFixed(2)}%"></i><span style="left:${(tests.thetaL * 100).toFixed(2)}%">θL ${tests.thetaL.toFixed(2)}</span>` +
        `<i style="left:${(tests.thetaD * 100).toFixed(2)}%"></i><span style="left:${(tests.thetaD * 100).toFixed(2)}%">θD ${tests.thetaD.toFixed(2)}</span>`;
    }

    this.tableEl.innerHTML =
      `<thead><tr><th>term</th><th>weight</th><th>normalised</th><th>contribution</th></tr></thead><tbody>` +
      terms
        .map(
          (t) => `<tr class="${t.weight === 0 ? "zero" : ""}">
            <td><span class="sw" style="background:${TERM_COLOR[t.key]}"></span>${t.label}</td>
            <td>${t.weight.toFixed(2)}</td>
            <td>${t.normalised.toFixed(4)}</td>
            <td>${t.contribution.toFixed(4)}</td></tr>`,
        )
        .join("") +
      `<tr><td><b>Rₜ</b></td><td></td><td></td><td><b>${risk.toFixed(4)}</b></td></tr></tbody>`;
  }
}
