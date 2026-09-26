// Prove that the browser engine reproduces the Python engine.
//
//   node scripts/js_parity.mjs <referenceDir> [traceDir]
//
// For every reference file written by
//   python experiments/export_web_traces.py --reference-dir <dir>
// this loads the matching shipped trace from site/data/traces, replays it through
// site/js (the exact modules the dashboard loads), and compares every per-frame
// field and every reduced metric against Python's.
//
// Categorical fields (mode, reason, output source) and booleans must match
// exactly -- any difference throws. Floats are reduced to the single worst
// absolute difference, printed on the last line so the caller can threshold it.
// Non-finite values (the NaN refresh latency, the infinite predicted arrival of
// an unreachable edge) are written as null by the exporter, so a null must be
// met by a non-finite double and vice versa.

import { readFileSync, readdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve, join } from "node:path";

const here = resolve(new URL(".", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const root = resolve(here, "..");
const load = (p) => import(pathToFileURL(resolve(root, "site", "js", p)).href);

const { traceFromExport } = await load("trace.js");
const { buildPolicy } = await load("policies.js");
const { executeRun } = await load("executor.js");
const { computeMetrics } = await load("metrics.js");

const refDir = process.argv[2];
if (!refDir) throw new Error("usage: node scripts/js_parity.mjs <referenceDir> [traceDir]");
const traceDir = process.argv[3] || join(root, "site", "data", "traces");
const cfg = JSON.parse(readFileSync(join(root, "site", "data", "config.json"), "utf8"));

const refFiles = readdirSync(refDir).filter((f) => f.endsWith(".reference.json")).sort();
if (!refFiles.length) throw new Error(`no *.reference.json in ${refDir}`);

let worst = 0;
let worstWhere = "";
let compared = 0;
let runs = 0;

/** Metrics whose value is a label, not a number. */
const STRING_METRICS = new Set(["policy", "profile"]);

function note(diff, where) {
  compared += 1;
  if (diff > worst) {
    worst = diff;
    worstWhere = where;
  }
}

function compareFloat(refValue, jsValue, where) {
  const finite = Number.isFinite(jsValue);
  if (refValue === null) {
    if (finite) throw new Error(`${where}: Python had a non-finite value, JS has ${jsValue}`);
    return;
  }
  if (!finite) throw new Error(`${where}: Python had ${refValue}, JS has ${jsValue}`);
  note(Math.abs(refValue - jsValue), where);
}

for (const file of refFiles) {
  const ref = JSON.parse(readFileSync(join(refDir, file), "utf8"));
  const payload = JSON.parse(readFileSync(join(traceDir, ref.trace_file), "utf8"));
  if (payload.fingerprint !== ref.fingerprint) {
    throw new Error(
      `${ref.trace_file}: shipped fingerprint ${payload.fingerprint} != reference ${ref.fingerprint}. ` +
        `Re-run experiments/export_web_traces.py.`,
    );
  }
  const trace = traceFromExport(payload);
  if (trace.frames !== ref.frames) throw new Error(`${file}: frame count differs`);

  for (const [key, run] of Object.entries(ref.runs)) {
    const policy = buildPolicy(run.policy, cfg);
    const { rows, modeSwitches } = executeRun(trace, policy, cfg, run.deadline_ms);
    const tag = `${ref.profile}/${ref.seed} ${key}`;
    runs += 1;

    if (rows.length !== ref.frames) throw new Error(`${tag}: ${rows.length} rows, expected ${ref.frames}`);
    if (modeSwitches !== run.mode_switches) {
      throw new Error(`${tag}: mode switches ${modeSwitches} != ${run.mode_switches}`);
    }

    for (const [field, idx] of Object.entries(run.columns)) {
      const vocab = run.vocab[field];
      for (let i = 0; i < idx.length; i++) {
        const got = rows[i][field];
        if (vocab) {
          const want = vocab[idx[i]];
          if (want !== got) throw new Error(`${tag} frame ${i} ${field}: Python "${want}", JS "${got}"`);
        } else if (typeof got === "boolean") {
          const want = idx[i] === 1;
          if (want !== got) throw new Error(`${tag} frame ${i} ${field}: Python ${want}, JS ${got}`);
        } else if (Number.isInteger(idx[i]) && field === "detections") {
          if (idx[i] !== got) throw new Error(`${tag} frame ${i} ${field}: Python ${idx[i]}, JS ${got}`);
        } else {
          compareFloat(idx[i], got, `${tag} frame ${i} ${field}`);
        }
      }
    }

    const m = computeMetrics(rows, modeSwitches);
    for (const [name, want] of Object.entries(run.metrics)) {
      const got = m[name];
      if (got === undefined) throw new Error(`${tag}: JS metrics lack '${name}'`);
      if (STRING_METRICS.has(name)) {
        if (String(want) !== String(got)) throw new Error(`${tag} ${name}: "${want}" vs "${got}"`);
      } else {
        compareFloat(typeof want === "number" ? want : Number(want), got, `${tag} metric ${name}`);
      }
    }
  }
}

process.stderr.write(
  `[js-parity] ${refFiles.length} traces, ${runs} policy/deadline runs, ` +
    `${compared.toLocaleString("en-US")} float comparisons; ` +
    `worst |Python - JS| = ${worst.toExponential(3)} at ${worstWhere}\n`,
);
console.log(worst);
