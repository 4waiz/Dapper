// Screenshot the running dashboard with headless Chrome (DevTools protocol).
//
//   python -m http.server 8765 --directory site      # in another terminal
//   node scripts/capture_console.mjs [url] [outDir]
//   node scripts/capture_console.mjs [url] [outDir] --mobile
//
// Desktop writes mission-control.png, decision-path.png, telemetry.png,
// policy-race.png and research.png. --mobile drives CDP device emulation
// (Emulation.setDeviceMetricsOverride) at 390x844 with a touch profile, because
// Chrome's --window-size cannot go below roughly 500 px.
//
// It also reports the page's horizontal overflow and any console error, so a
// capture run doubles as a smoke test of the deployed site.

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] || "http://localhost:8765/";
const outDir = process.argv[3] || "assets";
const mobile = process.argv.includes("--mobile");
const VW = mobile ? 390 : 1600;
const VH = mobile ? 844 : 1000;

const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find((p) => existsSync(p));
if (!CHROME) throw new Error("Chrome or Edge not found");

const port = 9334;
const profile = join(tmpdir(), `dapper-capture-${Date.now()}`);
const chrome = spawn(CHROME, [
  "--headless=new",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  `--window-size=${Math.max(VW, 520)},${VH}`,
  "--hide-scrollbars",
  "--disable-gpu",
  "about:blank",
]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch {
      /* not up yet */
    }
    await sleep(200);
  }
  throw new Error("DevTools endpoint did not come up");
}

const ws = new WebSocket(await target());
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let nextId = 1;
const pending = new Map();
const consoleErrors = [];
ws.addEventListener("message", (e) => {
  const msg = JSON.parse(e.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
    return;
  }
  if (msg.method === "Runtime.exceptionThrown") {
    const d = msg.params.exceptionDetails;
    consoleErrors.push(d.exception?.description || d.text);
  }
  if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
    consoleErrors.push(msg.params.args.map((a) => a.value ?? a.description).join(" "));
  }
});
const send = (method, params = {}) =>
  new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) =>
  (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Runtime.enable");
await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", {
  width: VW,
  height: VH,
  deviceScaleFactor: mobile ? 3 : 2,
  mobile,
  screenWidth: VW,
  screenHeight: VH,
});
if (mobile) await send("Emulation.setTouchEmulationEnabled", { enabled: true, maxTouchPoints: 5 });
await send("Page.navigate", { url });
mkdirSync(outDir, { recursive: true });

// Wait for the boot overlay to clear, then let the scheduler build up history.
for (let i = 0; i < 120; i++) {
  if (await evaluate("document.getElementById('boot')?.classList.contains('done')")) break;
  await sleep(250);
}
await sleep(6500);

const parity = await evaluate("document.getElementById('status-parity')?.textContent");
const frames = await evaluate("document.getElementById('status-frame')?.textContent");
const overflow = await evaluate("document.documentElement.scrollWidth - window.innerWidth");
console.log(`viewport ${VW}x${VH} · ${frames} · ${parity} · horizontal overflow ${overflow}px`);

async function shot(name, clip) {
  const res = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    ...(clip ? { clip: { ...clip, scale: 1 } } : {}),
  });
  const path = join(outDir, name);
  writeFileSync(path, Buffer.from(res.result.data, "base64"));
  console.log(`  wrote ${path}`);
}

/** Bounding box of a section, in page coordinates, padded. */
const rect = async (sel, pad = 14) =>
  evaluate(`(() => {
    const el = document.querySelector('${sel}');
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: 0, y: Math.max(0, Math.round(r.top + scrollY - ${pad})), width: ${VW}, height: Math.round(r.height + ${2 * pad}) };
  })()`);

if (mobile) {
  await shot("mobile-mission.png", { x: 0, y: 0, width: VW, height: 1500 });
  const research = await rect("#research");
  if (research) await shot("mobile-research.png", { ...research, height: Math.min(research.height, 1500) });
} else {
  await shot("mission-control.png", { x: 0, y: 0, width: VW, height: 1000 });
  for (const [sel, name, cap] of [
    ["#decision", "decision-path.png", 1200],
    ["#telemetry", "telemetry.png", 900],
    ["#race", "policy-race.png", 1100],
    ["#research", "research.png", 1760],
  ]) {
    const r = await rect(sel);
    if (r) await shot(name, { ...r, height: Math.min(r.height, cap) });
  }
}

if (consoleErrors.length) {
  console.error(`\n${consoleErrors.length} console error(s):`);
  for (const e of consoleErrors.slice(0, 10)) console.error(`  ${e}`);
}
ws.close();
chrome.kill();
if (consoleErrors.length || overflow > 1) process.exit(1);
