// Verify a deployed (or locally served) DAPPER site.
//
//   node scripts/verify_site.mjs https://4waiz.github.io/Dapper/
//   node scripts/verify_site.mjs http://localhost:8765/
//
// Three checks, all of which have to pass:
//   1. every asset the pages reference returns HTTP 200 -- HTML, CSS, JS
//      modules (followed transitively through their imports), images, fonts,
//      the PDF and every JSON under data/;
//   2. every JSON parses, and none of them smuggles a NaN or an Infinity past
//      JSON.parse (a browser rejects both, so a live page would simply break);
//   3. the dashboard and the paper page load in headless Chrome with no console
//      error, no failed request and no horizontal overflow, and the dashboard's
//      own Python-parity self-check agrees to better than 1e-9.
//
// Exits non-zero on the first category that fails, with the offending URLs.

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const base = (process.argv[2] || "http://localhost:8765/").replace(/\/?$/, "/");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fail = [];
const ok = [];

function note(pass, url, detail = "") {
  (pass ? ok : fail).push(detail ? `${url}  ${detail}` : url);
  if (!pass) console.error(`  FAIL ${url} ${detail}`);
}

async function head(url) {
  try {
    // Some static hosts do not answer HEAD; a ranged GET works everywhere.
    const res = await fetch(url, { headers: { Range: "bytes=0-0" } });
    return res.status;
  } catch (e) {
    return `network error: ${e.message}`;
  }
}

async function text(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.text();
}

/** Pull referenced local paths out of an HTML document. */
function hrefsFrom(html, dir) {
  const out = new Set();
  const re = /(?:href|src)\s*=\s*["']([^"'#?]+)["']/g;
  let m;
  while ((m = re.exec(html))) {
    const raw = m[1];
    if (/^(https?:|data:|mailto:|#)/.test(raw)) continue;
    out.add(new URL(raw, dir).href);
  }
  return out;
}

/** Follow ES-module imports so a broken import path is caught, not just the entry point. */
async function moduleGraph(entry) {
  const seen = new Set();
  const queue = [entry];
  while (queue.length) {
    const url = queue.shift();
    if (seen.has(url)) continue;
    seen.add(url);
    const src = await text(url);
    if (src === null) continue;
    const re = /(?:^|[\s;])(?:import|export)[^;'"]*from\s*["']([^"']+)["']|import\s*\(\s*["']([^"']+)["']\s*\)/g;
    let m;
    while ((m = re.exec(src))) {
      const spec = m[1] || m[2];
      if (!spec || /^(https?:|data:)/.test(spec)) continue;
      queue.push(new URL(spec, url).href);
    }
  }
  return seen;
}

console.log(`[verify] ${base}`);

// ---- 1. assets ------------------------------------------------------------
const pages = [base, `${base}paper/`];
const assets = new Set();
for (const page of pages) {
  const html = await text(page);
  if (html === null) {
    note(false, page, "page did not return 200");
    continue;
  }
  note(true, page);
  for (const href of hrefsFrom(html, page)) assets.add(href);
}
for (const mod of await moduleGraph(`${base}js/app.js`)) assets.add(mod);

// The traces are fetched at runtime from the index, not linked in the HTML.
const indexUrl = `${base}data/traces/index.json`;
assets.add(indexUrl);
const idxText = await text(indexUrl);
let traceFiles = [];
if (idxText) {
  try {
    traceFiles = JSON.parse(idxText).traces.map((t) => `${base}data/traces/${t.file}`);
    for (const t of traceFiles) assets.add(t);
  } catch (e) {
    note(false, indexUrl, `does not parse: ${e.message}`);
  }
}

for (const url of [...assets].sort()) {
  const status = await head(url);
  note(status === 200 || status === 206, url, status === 200 || status === 206 ? "" : `HTTP ${status}`);
}
console.log(`  ${ok.length} assets OK, ${fail.length} failed`);

// ---- 2. JSON ---------------------------------------------------------------
const jsonUrls = [...assets].filter((u) => u.endsWith(".json"));
let jsonBad = 0;
for (const url of jsonUrls) {
  const body = await text(url);
  if (body === null) {
    console.error(`  FAIL ${url} unreadable`);
    jsonBad += 1;
    continue;
  }
  try {
    JSON.parse(body);
  } catch (e) {
    console.error(`  FAIL ${url} does not parse: ${e.message}`);
    jsonBad += 1;
    continue;
  }
  // JSON.parse accepts neither, but a hand-edited file might contain them.
  const bare = /(^|[^"\w])(NaN|-?Infinity)([^"\w]|$)/;
  if (bare.test(body)) {
    console.error(`  FAIL ${url} contains a bare NaN or Infinity`);
    jsonBad += 1;
  }
}
console.log(`  ${jsonUrls.length} JSON files checked, ${jsonBad} bad`);

// ---- 3. browser ------------------------------------------------------------
const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find((p) => existsSync(p));

let browserBad = 0;
if (!CHROME) {
  console.log("  Chrome not found; skipped the browser checks");
} else {
  const port = 9336;
  const profile = join(tmpdir(), `dapper-verify-${Date.now()}`);
  const chrome = spawn(CHROME, [
    "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
    "--window-size=1440,900", "--hide-scrollbars", "--disable-gpu", "about:blank",
  ]);
  let wsUrl = null;
  for (let i = 0; i < 60 && !wsUrl; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      wsUrl = list.find((t) => t.type === "page")?.webSocketDebuggerUrl || null;
    } catch {
      /* not up yet */
    }
    if (!wsUrl) await sleep(200);
  }
  const ws = new WebSocket(wsUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let nextId = 1;
  const pending = new Map();
  let errors = [];
  let failedRequests = [];
  ws.addEventListener("message", (e) => {
    const msg = JSON.parse(e.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg);
      pending.delete(msg.id);
      return;
    }
    if (msg.method === "Runtime.exceptionThrown") {
      errors.push(msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
    }
    if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
      errors.push(msg.params.args.map((a) => a.value ?? a.description).join(" "));
    }
    if (msg.method === "Network.responseReceived" && msg.params.response.status >= 400) {
      failedRequests.push(`${msg.params.response.status} ${msg.params.response.url}`);
    }
  });
  const send = (method, params = {}) =>
    new Promise((resolve) => {
      const id = nextId++;
      pending.set(id, resolve);
      ws.send(JSON.stringify({ id, method, params }));
    });
  const evaluate = async (expr) =>
    (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }))
      .result?.result?.value;

  await send("Runtime.enable");
  await send("Page.enable");
  await send("Network.enable");

  for (const page of pages) {
    errors = [];
    failedRequests = [];
    await send("Page.navigate", { url: page });
    await sleep(page === base ? 9000 : 4500);
    const overflow = await evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth");
    const parity = page === base ? await evaluate("document.getElementById('status-parity')?.textContent") : "n/a";
    const booted = page === base ? await evaluate("document.getElementById('boot')?.classList.contains('done')") : true;
    const problems = [];
    if (errors.length) problems.push(`${errors.length} console error(s): ${errors.slice(0, 3).join(" | ")}`);
    if (failedRequests.length) problems.push(`${failedRequests.length} failed request(s): ${failedRequests.slice(0, 3).join(" | ")}`);
    if (overflow > 1) problems.push(`${overflow}px horizontal overflow`);
    if (!booted) problems.push("boot overlay never cleared");
    if (page === base) {
      const worst = Number(String(parity).replace(/[^0-9.eE+-]/g, ""));
      if (!(worst < 1e-9)) problems.push(`parity self-check reads ${parity}`);
    }
    if (problems.length) {
      browserBad += 1;
      console.error(`  FAIL ${page}\n     ${problems.join("\n     ")}`);
    } else {
      console.log(`  OK   ${page}  overflow ${overflow}px${page === base ? `, ${parity}` : ""}`);
    }
  }
  ws.close();
  chrome.kill();
}

const bad = fail.length + jsonBad + browserBad;
console.log(bad ? `[verify] FAILED: ${bad} problem(s)` : "[verify] all checks passed");
process.exit(bad ? 1 : 0);
