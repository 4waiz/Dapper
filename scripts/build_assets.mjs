// Derive every raster asset from the single source SVG.
//
//   node scripts/build_assets.mjs
//
// One mark, site/assets/logo.svg, produces:
//   site/assets/favicon-32.png        browser tab
//   site/assets/favicon-16.png        browser tab, small
//   site/assets/apple-touch-icon.png  iOS home screen (180, opaque)
//   site/assets/logo-256.png          paper card / fallback
//   site/assets/social-preview.png    1280x640 Open Graph card
//   assets/banner.png                 1280x320 README banner
//   assets/logo-256.png               README inline
//
// Rendering goes through headless Chrome so the SVG is rasterised by the same
// engine that will draw it live, rather than by a second renderer that might
// disagree about gradients or stroke caps.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, dirname } from "node:path";

const here = resolve(new URL(".", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const root = resolve(here, "..");
const SVG = readFileSync(join(root, "site", "assets", "logo.svg"), "utf8");

const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find((p) => existsSync(p));
if (!CHROME) throw new Error("Chrome or Edge not found");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9345;
const profile = join(tmpdir(), `dapper-assets-${Date.now()}`);
const chrome = spawn(CHROME, [
  "--headless=new",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  "--window-size=1400,900",
  "--hide-scrollbars",
  "--disable-gpu",
  "about:blank",
]);

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
ws.addEventListener("message", (e) => {
  const msg = JSON.parse(e.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  }
});
const send = (method, params = {}) =>
  new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });

await send("Page.enable");

/** Rasterise an HTML body at an exact pixel size. */
async function shot(outPath, width, height, body, { scale = 1, background = "transparent" } = {}) {
  const html = `<!doctype html><meta charset="utf-8"><style>
    html,body{margin:0;padding:0;width:${width}px;height:${height}px;overflow:hidden;background:${background};}
    *{box-sizing:border-box}
  </style>${body}`;
  await send("Emulation.setDeviceMetricsOverride", {
    width, height, deviceScaleFactor: scale, mobile: false,
  });
  await send("Emulation.setDefaultBackgroundColorOverride",
    background === "transparent" ? { color: { r: 0, g: 0, b: 0, a: 0 } } : {});
  await send("Page.navigate", { url: `data:text/html;charset=utf-8,${encodeURIComponent(html)}` });
  await sleep(320);
  const res = await send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
    clip: { x: 0, y: 0, width, height, scale },
  });
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, Buffer.from(res.result.data, "base64"));
  const kb = (readFileSync(outPath).length / 1024).toFixed(1);
  console.log(`  wrote ${outPath.replace(root + "\\", "").replace(root + "/", "")}  (${width}x${height}, ${kb} KiB)`);
}

const img = (size) => `<img src="data:image/svg+xml;utf8,${encodeURIComponent(SVG)}" width="${size}" height="${size}" style="display:block">`;

// --- icons -----------------------------------------------------------------
await shot(join(root, "site", "assets", "favicon-32.png"), 32, 32, img(32));
await shot(join(root, "site", "assets", "favicon-16.png"), 16, 16, img(16));
await shot(join(root, "site", "assets", "logo-256.png"), 256, 256, img(256));
await shot(join(root, "assets", "logo-256.png"), 256, 256, img(256));
// iOS masks the corners itself and does not honour transparency, so this one
// is drawn opaque, edge to edge.
await shot(join(root, "site", "assets", "apple-touch-icon.png"), 180, 180,
  `<div style="width:180px;height:180px;background:#0b0a09;display:grid;place-items:center">${img(164)}</div>`,
  { background: "#0b0a09" });

// --- social preview and README banner ---------------------------------------
const FONTS = `<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">`;

function card(w, h, { title, sub, tag, logo, pad }) {
  return `${FONTS}<div style="
      width:${w}px;height:${h}px;padding:${pad}px;display:flex;align-items:center;gap:${Math.round(pad * 0.9)}px;
      background:
        radial-gradient(900px 420px at 86% -18%, rgba(217,119,87,.30), transparent 62%),
        radial-gradient(760px 420px at -6% 22%, rgba(154,134,224,.14), transparent 62%),
        linear-gradient(135deg,#151211,#0b0a09);
      font-family:'Inter Tight',Inter,system-ui,sans-serif;color:#f3efec;position:relative;overflow:hidden;">
    <div style="position:absolute;inset:0;background-image:
        linear-gradient(#262220 1px,transparent 1px),linear-gradient(90deg,#262220 1px,transparent 1px);
        background-size:44px 44px;opacity:.28"></div>
    <div style="position:relative;flex:0 0 auto">${img(logo)}</div>
    <div style="position:relative;min-width:0">
      <div style="font-size:${Math.round(h * 0.155)}px;font-weight:800;letter-spacing:.14em;line-height:1">${title}</div>
      <div style="font-size:${Math.round(h * 0.062)}px;color:#a49a93;margin-top:${Math.round(h * 0.038)}px;line-height:1.35;max-width:${Math.round(w * 0.62)}px">${sub}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:${Math.round(h * 0.045)}px;color:#6f6762;margin-top:${Math.round(h * 0.05)}px">${tag}</div>
    </div>
  </div>`;
}

const SUB = "An answer after the deadline isn’t a late answer. It’s no answer.<br>The real per-frame scheduler, running live in your browser.";
const TAG = "IEEE FMEC 2026 &nbsp;·&nbsp; Awaiz Ahmed and Khubaib Amjad Alam &nbsp;·&nbsp; Al Ain University";

await shot(join(root, "site", "assets", "social-preview.png"), 1280, 640,
  card(1280, 640, { title: "DAPPER", sub: SUB, tag: TAG, logo: 300, pad: 84 }), { background: "#0b0a09" });
await shot(join(root, "assets", "social-preview.png"), 1280, 640,
  card(1280, 640, { title: "DAPPER", sub: SUB, tag: TAG, logo: 300, pad: 84 }), { background: "#0b0a09" });
await shot(join(root, "assets", "banner.png"), 1280, 320,
  card(1280, 320, { title: "DAPPER", sub: SUB, tag: TAG, logo: 180, pad: 40 }), { background: "#0b0a09" });

ws.close();
chrome.kill();
try {
  rmSync(profile, { recursive: true, force: true });
} catch {
  /* the profile is in the OS temp dir; leaving it is harmless */
}
console.log("assets built from site/assets/logo.svg");
