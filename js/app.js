// DAPPER Mission Control — orchestration.
//
// Boots the frozen configuration and the published results, loads an immutable
// scenario trace, and drives the real engine one frame at a time: the same
// scheduler.js / executor.js / metrics.js the parity test runs under Node
// against Python. Nothing on this page simulates a decision.
//
// On boot the page re-derives the metrics Python recorded for the loaded trace
// (shipped inside the trace file) and prints the worst disagreement in the
// status bar, so parity is something a visitor watches rather than a claim.

import { fit, lineChart, shareBar } from "./charts.js";
import { DecisionPath, RiskPanel } from "./decision.js";
import { MODE_COLOR, MODE_LABEL, int, mb, ms, num, pct, ratePct, sci } from "./fmt.js";
import { ExecutionModel, RunState, executeRun, stepFrame } from "./executor.js";
import { LiveMetrics, computeMetrics } from "./metrics.js";
import { buildPolicy } from "./policies.js";
import { Research } from "./research.js";
import { REASON_TEXT, schedulerFromConfig } from "./scheduler.js";
import { Scene } from "./scene.js";
import { injectOutage, synthesiseTrace, traceFromExport } from "./trace.js";

const $ = (id) => document.getElementById(id);
const RACE_POLICIES = ["local_only", "edge_only", "cloud_only", "dapper", "oracle_feasible"];
const RACE_LABEL = {
  local_only: "local-only",
  edge_only: "edge-only",
  cloud_only: "cloud-only",
  dapper: "DAPPER",
  oracle_feasible: "oracle",
};
const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
const WINDOW = 240;

const state = {
  cfg: null,
  results: null,
  traceIndex: null,
  trace: null,
  source: "replay",
  profile: "variable",
  seed: 100,
  deadlineMs: 100,
  speedIndex: 2,
  biasRtt: 0,
  biasCompute: 0,
  gates: { confidence: true, freshness: true, degraded: true },
  custom: null,
  playing: true,
  frame: 0,
  runners: new Map(),
  history: { rtt: [], loss: [], load: [], latency: [], confidence: [] },
  parity: null,
  scene: null,
  path: null,
  risk: null,
  research: null,
  lastTs: 0,
};

// --------------------------------------------------------------------- boot
const bootSteps = [];
function boot(text, cls = "run") {
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = text;
  $("boot-log").appendChild(li);
  bootSteps.push(li);
  $("boot-fill").style.width = `${Math.min(100, bootSteps.length * 14)}%`;
  return li;
}
function bootOk(li, text) {
  li.className = "ok";
  if (text) li.textContent = text;
}

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

async function main() {
  try {
    let li = boot("loading frozen configuration (Table I)");
    state.cfg = await getJSON("data/config.json");
    state.deadlineMs = state.cfg.deadline_ms;
    bootOk(li, `frozen configuration loaded · θL ${state.cfg.scheduler.risk_local_threshold}, θD ${state.cfg.scheduler.risk_degraded_threshold}, τc ${state.cfg.scheduler.local_confidence_threshold}`);

    li = boot("loading published results");
    state.results = await getJSON("data/results.json");
    bootOk(li, `published results loaded · ${int(state.results.protocol.frame_decisions)} frame decisions over ${state.results.protocol.seeds} seeds`);

    li = boot("loading scenario index");
    state.traceIndex = await getJSON("data/traces/index.json");
    bootOk(li, `${state.traceIndex.traces.length} immutable scenario traces available`);

    buildControls();

    li = boot(`loading trace ${state.profile}-${state.seed}`);
    await loadTrace();
    bootOk(li, `trace ${state.profile}-${state.seed} loaded · fingerprint ${state.trace.fingerprint}`);

    li = boot("verifying this browser engine against the Python engine");
    const parity = verifyAgainstPython();
    state.parity = parity;
    if (parity.ok) {
      bootOk(li, `engine verified · worst |Python − JS| = ${sci(parity.worst)} over ${int(parity.compared)} values`);
    } else {
      li.className = "bad";
      li.textContent = `engine verification failed: ${parity.message}`;
    }

    li = boot("rendering published results");
    state.research = new Research(state.results, {
      chip: $("research-chip"),
      headline: $("headline"),
      tabs: $("table-tabs"),
      title: $("table-title"),
      table: $("data-table"),
      caption: $("table-caption"),
      honest: $("honest"),
      p95: $("ch-p95"),
      confidence: $("ch-confidence"),
      deadline: $("ch-deadline"),
      bias: $("ch-bias"),
    });
    state.research.render();
    bootOk(li, "published tables and figures rendered");

    li = boot("starting the scheduler");
    state.scene = new Scene($("scene"));
    state.path = new DecisionPath($("pathwrap"), $("gate-tests"));
    state.risk = new RiskPanel({
      valueEl: $("risk-value"),
      verdictEl: $("risk-verdict"),
      barEl: $("risk-bar"),
      marksEl: $("risk-marks"),
      tableEl: $("risk-terms"),
    });
    rebuild();
    bootOk(li, "scheduler running");

    setTimeout(() => $("boot").classList.add("done"), 420);
    requestAnimationFrame(loop);
    window.addEventListener("resize", () => {
      state.research.charts();
      drawCharts();
    });
  } catch (err) {
    const li = boot(String(err && err.message ? err.message : err), "bad");
    li.className = "bad";
    $("status-text").textContent = "FAILED";
    throw err;
  }
}

// ------------------------------------------------------------------ traces
function traceEntry(profile, seed) {
  return state.traceIndex.traces.find((t) => t.profile === profile && t.seed === seed);
}

function seedsFor(profile) {
  return state.traceIndex.traces.filter((t) => t.profile === profile).map((t) => t.seed).sort((a, b) => a - b);
}

async function loadTrace() {
  if (state.source === "live") {
    const params = state.custom || state.cfg.network_profiles[state.profile];
    state.trace = synthesiseTrace({
      profile: state.custom ? "custom" : state.profile,
      params,
      cfg: state.cfg,
      seed: state.seed,
      frames: 4000,
    });
    state.exported = null;
    return;
  }
  const entry = traceEntry(state.profile, state.seed) || state.traceIndex.traces[0];
  state.profile = entry.profile;
  state.seed = entry.seed;
  const payload = await getJSON(`data/traces/${entry.file}`);
  state.exported = payload;
  state.trace = traceFromExport(payload);
}

/**
 * Re-derive the metrics Python recorded for this trace and report the worst
 * disagreement. The reference values ship inside the trace file, so this runs
 * with no network call and no Python.
 */
function verifyAgainstPython() {
  const ref = state.exported && state.exported.reference_metrics;
  if (!ref) return { ok: true, skipped: true, worst: 0, compared: 0, message: "live synthetic trace — nothing to verify against" };
  let worst = 0;
  let compared = 0;
  let where = "";
  try {
    for (const [key, want] of Object.entries(ref)) {
      const at = key.lastIndexOf("@");
      const name = key.slice(0, at);
      const deadline = Number(key.slice(at + 1));
      const { rows, modeSwitches } = executeRun(state.trace, buildPolicy(name, state.cfg), state.cfg, deadline);
      const got = computeMetrics(rows, modeSwitches);
      for (const [metric, value] of Object.entries(want)) {
        if (typeof value !== "number") continue;
        const mine = got[metric];
        if (!Number.isFinite(mine)) continue;
        compared += 1;
        const d = Math.abs(mine - value);
        if (d > worst) {
          worst = d;
          where = `${key} ${metric}`;
        }
      }
    }
  } catch (err) {
    return { ok: false, worst, compared, message: String(err.message || err) };
  }
  const ok = worst < 1e-9;
  return { ok, worst, compared, where, message: ok ? "" : `worst difference ${sci(worst)} at ${where}` };
}

// ---------------------------------------------------------------- runners
function makeRunner(name) {
  const isDapper = name === "dapper";
  const policy = buildPolicy(name, state.cfg, isDapper
    ? {
        scheduler: schedulerFromConfig(state.cfg, {
          rttEstimateBias: state.biasRtt / 100,
          computeEstimateBias: state.biasCompute / 100,
          useConfidenceGate: state.gates.confidence,
          useFreshnessGate: state.gates.freshness,
          useDegradedReuse: state.gates.degraded,
        }),
      }
    : {});
  return {
    name,
    policy,
    model: new ExecutionModel(state.cfg, state.deadlineMs),
    run: new RunState(),
    live: new LiveMetrics(1200),
  };
}

function rebuild() {
  state.frame = 0;
  state.runners = new Map(RACE_POLICIES.map((n) => [n, makeRunner(n)]));
  state.history = { rtt: [], loss: [], load: [], latency: [], confidence: [] };
  state.scene.reset();
  state.scene.deadlineMs = state.deadlineMs;
  $("audit").innerHTML = "";
  $("audit-empty").hidden = false;
  refreshStatus();
  refreshChips();
  drawCharts();
  renderRace();
  // Show the first frame immediately, so a paused page is never blank.
  advance(1);
}

// ------------------------------------------------------------------- loop
function loop(ts) {
  const dt = state.lastTs ? Math.min(120, ts - state.lastTs) : 16;
  state.lastTs = ts;
  if (state.playing) {
    // The trace is a 30 FPS capture, so 1x means 30 scheduler frames a second.
    const want = (dt / 1000) * 30 * SPEEDS[state.speedIndex];
    state.carry = (state.carry || 0) + want;
    const n = Math.floor(state.carry);
    state.carry -= n;
    if (n > 0) advance(n);
  }
  state.scene.tick(dt);
  state.scene.draw();
  requestAnimationFrame(loop);
}

function advance(n) {
  const dapper = state.runners.get("dapper");
  let last = null;
  for (let k = 0; k < n; k++) {
    if (state.frame >= state.trace.frames) {
      if (state.source === "live") {
        // Live mode is endless: extend by synthesising the next block.
        extendLiveTrace();
      } else {
        state.playing = false;
        refreshStatus();
        break;
      }
    }
    const i = state.frame;
    for (const r of state.runners.values()) {
      const step = stepFrame(state.trace, r.policy, r.model, r.run, i);
      r.live.push(step.row);
      if (r.name === "dapper") {
        last = step;
        state.scene.push(step, state.deadlineMs);
        pushHistory(step.row);
        addAudit(step);
      }
    }
    state.frame += 1;
  }
  if (last) {
    const sched = dapper.policy.scheduler;
    const tests = sched.gateTests(last.obs);
    state.path.light(last.decision, tests);
    state.risk.update(sched.riskTerms(last.obs), tests, last.row.selected_mode);
    updateStageBar(last);
  }
  renderKpis();
  renderRace();
  drawCharts();
  refreshStatus();
}

function extendLiveTrace() {
  const params = state.custom || state.cfg.network_profiles[state.profile];
  const next = synthesiseTrace({
    profile: state.custom ? "custom" : state.profile,
    params,
    cfg: state.cfg,
    seed: (state.seed + Math.floor(state.frame / 4000) + 1) >>> 0,
    frames: 4000,
  });
  // Keep the frame index monotonic by restarting the runners' clocks: the scene
  // and metrics carry on, the capture times restart from zero.
  state.trace = next;
  state.frame = 0;
  for (const r of state.runners.values()) r.run = new RunState();
}

function pushHistory(row) {
  const h = state.history;
  h.rtt.push(row.rtt_ms);
  h.loss.push(row.packet_loss * 100);
  h.load.push(row.edge_load * 100);
  h.latency.push(row.control_output_latency_ms);
  h.confidence.push(row.usable_confidence_proxy);
  for (const k of Object.keys(h)) if (h[k].length > WINDOW) h[k].shift();
}

// -------------------------------------------------------------------- UI
function updateStageBar(step) {
  const row = step.row;
  $("sc-frame").textContent = String(row.frame_id);
  $("sc-risk").textContent = num(row.risk_score, 3);
  $("sc-rtt").textContent = ms(row.rtt_ms, 0);
  $("sc-lat").textContent = ms(row.control_output_latency_ms, 1);
  const v = $("sc-verdict");
  if (!row.deadline_met) {
    v.textContent = "DEADLINE MISSED";
    v.style.color = "var(--rose)";
  } else if (row.output_reused) {
    v.textContent = `reused output, ${ms(row.output_age_ms, 0)} old`;
    v.style.color = "var(--degraded)";
  } else if (row.remote_accepted) {
    v.textContent = "refinement accepted in time";
    v.style.color = "var(--green)";
  } else if (row.remote_attempted) {
    v.textContent = row.remote_failed_loss ? "request lost" : "reply rejected at the deadline";
    v.style.color = "var(--rose)";
  } else {
    v.textContent = "on time, local answer";
    v.style.color = "var(--text-2)";
  }
  $("stage-chip").textContent = `${MODE_LABEL[row.selected_mode]} · ${row.decision_reason}`;
  $("path-chip").textContent = `${MODE_LABEL[row.selected_mode]} · frame ${row.frame_id}`;
}

function addAudit(step) {
  const row = step.row;
  const list = $("audit");
  $("audit-empty").hidden = true;
  const li = document.createElement("li");
  li.className = row.selected_mode;
  const outcome = row.remote_accepted
    ? '<span class="acc">reply accepted</span>'
    : row.remote_rejected_deadline
      ? '<span class="miss">reply late</span>'
      : row.remote_rejected_freshness
        ? '<span class="miss">reply outside W</span>'
        : row.remote_failed_loss
          ? '<span class="miss">request lost</span>'
          : row.remote_failed_unavailable
            ? '<span class="miss">edge unreachable</span>'
            : "";
  li.innerHTML =
    `<div class="audit-top"><span class="audit-mode">${MODE_LABEL[row.selected_mode]}</span><span>frame ${row.frame_id}</span></div>` +
    `<div class="audit-why">${REASON_TEXT[row.decision_reason] || row.decision_reason}</div>` +
    `<div class="audit-nums"><span>R=${num(row.risk_score, 3)}</span><span>RTT ${ms(row.rtt_ms, 0)}</span>` +
    `<span class="${row.deadline_met ? "" : "miss"}">latency ${ms(row.control_output_latency_ms, 1)}</span>` +
    `<span>conf ${num(row.usable_confidence_proxy, 3)}</span>${outcome}</div>`;
  // New entries arrive at the top. If the reader has not deliberately scrolled
  // back, hold the view at the newest entry rather than letting it drift.
  const pinned = list.scrollTop < 40;
  list.prepend(li);
  while (list.children.length > 30) list.lastElementChild.remove();
  // Trimming the tail shrinks the content, so a reader who has scrolled back
  // must be clamped or they end up staring at the gap the removed entries left.
  list.scrollTop = pinned ? 0 : Math.min(list.scrollTop, Math.max(0, list.scrollHeight - list.clientHeight));
}

const KPIS = [
  ["miss", "Deadline miss"],
  ["p95", "p95 control latency"],
  ["conf", "Confidence proxy"],
  ["bw", "Uplink / 1k frames"],
  ["remote", "Remote replies"],
  ["reuse", "Reuse share"],
  ["stale", "Stale outputs accepted"],
  ["switch", "Mode switches / 1k"],
];

function renderKpis() {
  const m = state.runners.get("dapper").live.snapshot();
  const el = $("kpis");
  const cards = {
    miss: [pct(m.deadline_miss_rate * 100, 2), `over ${int(m.frames)} frames`, m.deadline_miss_rate > 0 ? "bad" : "good"],
    p95: [ms(m.p95_latency_ms, 1), `deadline ${state.deadlineMs} ms`, m.p95_latency_ms <= state.deadlineMs ? "good" : "bad"],
    conf: [num(m.usable_confidence_proxy, 3), "0 when late or stale", ""],
    bw: [mb(m.bandwidth_per_1000_frames_kb / 1000, 2), "charged for every request sent", ""],
    remote: [
      `${int(m.remote_accepted)}/${int(m.remote_rejected_deadline + m.remote_rejected_freshness)}/${int(m.remote_failed_loss)}`,
      "accepted / late / lost",
      "",
    ],
    reuse: [pct(m.reuse_rate * 100, 1), "served from a still-fresh output", ""],
    stale: [int(m.stale_outputs_accepted), "must stay 0", m.stale_outputs_accepted === 0 ? "good" : "bad"],
    switch: [num(m.mode_switches_per_1000, 0), "placement changes", ""],
  };
  el.innerHTML = KPIS.map(([k, label]) => {
    const [v, s, cls] = cards[k];
    return `<div class="kpi ${cls}"><label>${label}</label><b>${v}</b><small>${s}</small></div>`;
  }).join("");
  $("kpi-scope").textContent = `${MODE_LABEL[state.runners.get("dapper").live.prevMode] || "—"} · frame ${state.frame}`;
}

function renderRace() {
  const snaps = RACE_POLICIES.map((n) => ({ name: n, m: state.runners.get(n).live.snapshot() }));
  const cols = [
    ["deadline_miss_rate", "Deadline miss (%)", true, (m) => m.deadline_miss_rate * 100, 2],
    ["p95_latency_ms", "p95 (ms)", true, (m) => m.p95_latency_ms, 1],
    ["usable_confidence_proxy", "Confidence proxy", false, (m) => m.usable_confidence_proxy, 3],
    ["bandwidth", "Uplink (MB/1k)", true, (m) => m.bandwidth_per_1000_frames_kb / 1000, 2],
    ["reuse_rate", "Reuse (%)", null, (m) => m.reuse_rate * 100, 1],
    ["stale", "Stale accepted", true, (m) => m.stale_acceptance_rate * 100, 2],
  ];
  // "best" is computed from the numbers, and the oracle is excluded because it
  // is not a deployable comparator.
  const deployable = snaps.filter((s) => s.name !== "oracle_feasible");
  const best = {};
  for (const [key, , lowerBetter, get] of cols) {
    if (lowerBetter === null) continue;
    const vals = deployable.map((s) => get(s.m)).filter(Number.isFinite);
    if (!vals.length) continue;
    best[key] = lowerBetter ? Math.min(...vals) : Math.max(...vals);
  }

  $("race-table").innerHTML =
    `<thead><tr><th>Policy</th>${cols.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead><tbody>` +
    snaps
      .map(({ name, m }) => {
        const oracle = name === "oracle_feasible";
        const cells = cols
          .map(([key, , lowerBetter, get, d]) => {
            const v = get(m);
            const isBest = !oracle && lowerBetter !== null && best[key] !== undefined && Math.abs(v - best[key]) < 10 ** -d / 2;
            const miss = key === "deadline_miss_rate" && v > 0;
            return `<td class="${isBest ? "best" : ""}${miss ? " miss" : ""}">${num(v, d)}</td>`;
          })
          .join("");
        return `<tr class="${name === "dapper" ? "ours" : ""}${oracle ? " oracle" : ""}">
          <td>${RACE_LABEL[name]}${oracle ? '<span class="tag">not deployable</span>' : ""}</td>${cells}</tr>`;
      })
      .join("") +
    `</tbody>`;

  const maxConf = Math.max(...snaps.map((s) => s.m.usable_confidence_proxy), 0.001);
  $("race-bars").innerHTML = snaps
    .map(({ name, m }) => `<div class="race-bar"><span>${RACE_LABEL[name]}</span>
      <span class="track"><i style="width:${((m.usable_confidence_proxy / maxConf) * 100).toFixed(1)}%"></i></span>
      <span class="val">${num(m.usable_confidence_proxy, 3)}</span></div>`)
    .join("");

  const d = snaps.find((s) => s.name === "dapper").m;
  const l = snaps.find((s) => s.name === "local_only").m;
  const o = snaps.find((s) => s.name === "oracle_feasible").m;
  const gain = o.usable_confidence_proxy - l.usable_confidence_proxy;
  const got = d.usable_confidence_proxy - l.usable_confidence_proxy;
  const lead =
    `Usable confidence proxy so far: local-only ${num(l.usable_confidence_proxy, 4)}, ` +
    `DAPPER ${num(d.usable_confidence_proxy, 4)}, oracle ${num(o.usable_confidence_proxy, 4)}`;
  const cost =
    `${num(d.bandwidth_per_1000_frames_kb / 1000, 2)} MB per 1,000 frames against the oracle's ` +
    `${num(o.bandwidth_per_1000_frames_kb / 1000, 2)} MB`;

  let tail;
  if (gain <= 1e-9 && got <= 1e-9) {
    tail = ". Neither is ahead of local-only on this stretch, so no share is computed.";
  } else if (got > gain) {
    // This is not a bug. The oracle commits the frame to edge-accurate and only
    // offloads when the reply would beat its own client timeout; DAPPER's hybrid
    // path keeps the local answer and can still accept a refinement the oracle
    // declined. Over a short window that can put DAPPER ahead of the bound.
    const rows = state.results.tables.adaptive.rows;
    const pub = (name) => rows.find((r) => r.policy === name)?.usable_confidence_proxy;
    const pubDapper = pub("dapper");
    const pubOracle = pub("oracle_feasible");
    tail =
      `. DAPPER is ahead of the oracle on this stretch: the oracle commits each frame it ` +
      `offloads and only does so when the reply would beat its own client timeout, whereas ` +
      `DAPPER's hybrid path keeps the local answer and can still accept a refinement the ` +
      `oracle declined. The oracle bounds a perfect-information committing scheduler, not ` +
      `every policy` +
      (pubDapper && pubOracle
        ? `; over the full ${state.results.protocol.seeds}-seed benchmark the oracle leads, ` +
          `${num(pubOracle, 4)} against DAPPER's ${num(pubDapper, 4)}.`
        : ".");
  } else {
    tail = `, so DAPPER recovers ${num((got / gain) * 100, 0)}% of the oracle's available gain here — at ${cost}.`;
  }
  $("race-note").textContent = lead + tail;
}

function drawCharts() {
  const h = state.history;
  lineChart($("ch-net"), {
    series: [
      { values: h.rtt, color: "#22d3ee", label: "RTT ms" },
      { values: h.load, color: "#8b5cf6", label: "load %" },
      { values: h.loss, color: "#ff3d71", label: "loss %" },
    ],
  });
  lineChart($("ch-lat"), {
    series: [{ values: h.latency, color: "#34d399", label: "control latency ms" }],
    rules: [{ value: state.deadlineMs, color: "#ff3d71", label: `D = ${state.deadlineMs} ms` }],
  });
  $("ch-lat-cap").textContent = `deadline ${state.deadlineMs} ms`;
  lineChart($("ch-conf"), {
    series: [{ values: h.confidence, color: "#f5a524", label: "usable confidence" }],
    yMax: 1,
    yFormat: (v) => v.toFixed(2),
  });
  const m = state.runners.get("dapper").live.snapshot();
  shareBar($("ch-mode"), {
    parts: Object.entries(m.mode_share).map(([k, v]) => ({ label: MODE_LABEL[k], color: MODE_COLOR[k], value: v })),
  });
}

function refreshStatus() {
  const dot = $("status-dot");
  dot.className = `dot ${state.playing ? "online" : "paused"}`;
  $("status-text").textContent = state.playing ? "RUNNING" : "PAUSED";
  $("status-mode").textContent = state.source === "replay" ? `REPLAY ${state.profile}-${state.seed}` : "LIVE SYNTHETIC";
  $("status-frame").textContent = `frame ${int(state.frame)}`;
  const p = state.parity;
  const el = $("status-parity");
  if (!p) el.textContent = "parity —";
  else if (p.skipped) el.textContent = "parity n/a (live)";
  else {
    el.textContent = `parity ${sci(p.worst)}`;
    el.style.color = p.ok ? "var(--green)" : "var(--rose)";
  }
  $("play").textContent = state.playing ? "Pause" : "Play";
}

function refreshChips() {
  const np = state.custom || state.cfg.network_profiles[state.profile];
  $("profile-sub").textContent = np
    ? `RTT ${np.rtt_ms_mean}±${np.rtt_ms_std} ms · loss ${(np.packet_loss * 100).toFixed(0)}% · load ${np.edge_load} · outage ${(np.outage_prob * 100).toFixed(0)}%`
    : "";
  const entry = state.source === "replay" ? traceEntry(state.profile, state.seed) : null;
  $("seed-sub").textContent = entry
    ? `published scenario · fingerprint ${entry.fingerprint} · ${int(entry.frames)} frames`
    : "browser-generated, labelled live synthetic";
  $("source-hint").innerHTML =
    state.source === "replay"
      ? `Replaying the immutable scenario <b>${state.profile}-${state.seed}</b> exported from <code>dapper.scenario</code>: the same trace the 30-seed benchmark replayed, fingerprint <b>${state.trace.fingerprint}</b>. Every policy below sees the identical frame, so the comparison is exactly paired.`
      : `<b>Live synthetic.</b> The browser is generating the network process from the Table II distributions with its own PRNG. NumPy's PCG64 cannot be reproduced here, so these frames are <em>not</em> a published result — only the scheduler is the published one.`;
}

// --------------------------------------------------------------- controls
function buildControls() {
  const profileSel = $("profile");
  profileSel.innerHTML = state.cfg.profiles
    .map((p) => `<option value="${p}">${p}</option>`)
    .join("") + `<option value="custom">custom…</option>`;
  profileSel.value = state.profile;

  const fillSeeds = () => {
    const seeds = state.source === "replay" ? seedsFor(state.profile) : [state.seed];
    $("seed").innerHTML = seeds.map((s) => `<option value="${s}">${s}</option>`).join("");
    if (!seeds.includes(state.seed)) state.seed = seeds[0];
    $("seed").value = String(state.seed);
    $("seed-val").textContent = String(state.seed);
    $("seed").disabled = state.source !== "replay";
  };
  fillSeeds();
  state.fillSeeds = fillSeeds;

  profileSel.addEventListener("change", async () => {
    if (profileSel.value === "custom") {
      state.custom = { ...state.cfg.network_profiles[state.profile] };
      $("custom-grid").hidden = false;
      syncCustomInputs();
      setSource("live");
    } else {
      state.custom = null;
      $("custom-grid").hidden = true;
      state.profile = profileSel.value;
      fillSeeds();
      await reload();
    }
  });

  $("seed").addEventListener("change", async () => {
    state.seed = Number($("seed").value);
    $("seed-val").textContent = String(state.seed);
    await reload();
  });

  $("src-replay").addEventListener("click", () => setSource("replay"));
  $("src-live").addEventListener("click", () => setSource("live"));

  $("play").addEventListener("click", () => {
    state.playing = !state.playing;
    if (state.playing && state.frame >= state.trace.frames) rebuild();
    refreshStatus();
  });
  $("reset").addEventListener("click", () => {
    state.playing = true;
    rebuild();
  });
  $("audit-clear").addEventListener("click", () => {
    $("audit").innerHTML = "";
    $("audit-empty").hidden = false;
  });

  const deadline = $("deadline");
  deadline.value = String(state.deadlineMs);
  $("deadline-val").textContent = `${state.deadlineMs} ms`;
  deadline.addEventListener("input", () => {
    state.deadlineMs = Number(deadline.value);
    $("deadline-val").textContent = `${state.deadlineMs} ms`;
    rebuild();
  });

  const speed = $("speed");
  speed.value = String(state.speedIndex);
  const speedLabel = () => {
    $("speed-val").textContent = `${SPEEDS[state.speedIndex]}×`;
    $("speed-sub").textContent = `${(SPEEDS[state.speedIndex] * 30).toFixed(0)} scheduler frames per second (capture is 30 FPS)`;
  };
  speedLabel();
  speed.addEventListener("input", () => {
    state.speedIndex = Number(speed.value);
    speedLabel();
  });

  for (const [id, key, label] of [["bias-rtt", "biasRtt", "RTT"], ["bias-compute", "biasCompute", "compute"]]) {
    const el = $(id);
    el.addEventListener("input", () => {
      state[key] = Number(el.value);
      $(`${id}-val`).textContent = `${state[key] > 0 ? "+" : ""}${state[key]}%`;
      rebuild();
    });
  }

  $("inject").addEventListener("click", () => {
    const n = Number($("inject-len").value);
    const applied = injectOutage(state.trace, state.frame + 1, n);
    $("inject").textContent = `dropped ${applied} frames`;
    setTimeout(() => ($("inject").textContent = "Drop the edge for"), 1600);
  });

  for (const [id, key] of [["g-conf", "confidence"], ["g-fresh", "freshness"], ["g-reuse", "degraded"]]) {
    $(id).addEventListener("click", () => {
      state.gates[key] = !state.gates[key];
      $(id).classList.toggle("on", state.gates[key]);
      rebuild();
    });
  }

  for (const [id, key, scale, fmt] of [
    ["c-rtt", "rtt_ms_mean", 1, (v) => `${v} ms`],
    ["c-jit", "rtt_ms_std", 1, (v) => `±${v} ms`],
    ["c-loss", "packet_loss", 0.01, (v) => `${(v * 100).toFixed(0)}%`],
    ["c-load", "edge_load", 0.01, (v) => v.toFixed(2)],
    ["c-out", "outage_prob", 0.01, (v) => `${(v * 100).toFixed(0)}%`],
  ]) {
    $(id).addEventListener("input", async () => {
      if (!state.custom) state.custom = { ...state.cfg.network_profiles[state.profile] };
      state.custom[key] = Number($(id).value) * scale;
      $(`${id}-val`).textContent = fmt(state.custom[key]);
      await reload();
    });
  }
}

function syncCustomInputs() {
  const c = state.custom;
  const set = (id, value, text) => {
    $(id).value = String(value);
    $(`${id}-val`).textContent = text;
  };
  set("c-rtt", c.rtt_ms_mean, `${c.rtt_ms_mean} ms`);
  set("c-jit", c.rtt_ms_std, `±${c.rtt_ms_std} ms`);
  set("c-loss", Math.round(c.packet_loss * 100), `${(c.packet_loss * 100).toFixed(0)}%`);
  set("c-load", Math.round(c.edge_load * 100), c.edge_load.toFixed(2));
  set("c-out", Math.round(c.outage_prob * 100), `${(c.outage_prob * 100).toFixed(0)}%`);
}

async function setSource(source) {
  state.source = source;
  $("src-replay").classList.toggle("on", source === "replay");
  $("src-live").classList.toggle("on", source === "live");
  if (source === "replay" && state.custom) {
    state.custom = null;
    $("custom-grid").hidden = true;
    $("profile").value = state.profile;
  }
  state.fillSeeds();
  await reload();
}

async function reload() {
  await loadTrace();
  state.parity = verifyAgainstPython();
  refreshChips();
  rebuild();
}

main();
