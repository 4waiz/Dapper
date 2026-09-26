// A small canvas chart kit: rolling lines, grouped bars with error bars, a
// stacked area and a horizontal mode-share bar.
//
// House rules borrowed from experiments/make_figures.py so the live charts and
// the paper figures cannot disagree about how a number is drawn:
//   * axes start at zero for rates and counts -- no truncated axes;
//   * a mean drawn from the published results is drawn with its 95% interval;
//   * one message per chart.

const FONT = '11px "JetBrains Mono", ui-monospace, Consolas, monospace';
const INK = "#a3b3cc";
const INK_DIM = "#6d809e";
const GRID = "#1a2745";

/** Size the backing store to the CSS box and return a device-pixel context. */
export function fit(canvas, cssHeight = null) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || canvas.parentElement.clientWidth || 320;
  const h = cssHeight || canvas.clientHeight || Number(canvas.getAttribute("height")) || 150;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = FONT;
  return { ctx, w, h };
}

function niceMax(v) {
  if (!Number.isFinite(v) || v <= 0) return 1;
  const exp = Math.floor(Math.log10(v));
  const base = 10 ** exp;
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) if (v <= m * base) return m * base;
  return 10 * base;
}

function axes(ctx, box, { yMax, yMin = 0, yTicks = 4, yFormat = (v) => String(Math.round(v)) }) {
  const { x0, y0, x1, y1 } = box;
  ctx.strokeStyle = GRID;
  ctx.fillStyle = INK_DIM;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= yTicks; i++) {
    const v = yMin + ((yMax - yMin) * i) / yTicks;
    const y = y1 - ((v - yMin) / (yMax - yMin || 1)) * (y1 - y0);
    ctx.beginPath();
    ctx.moveTo(x0, Math.round(y) + 0.5);
    ctx.lineTo(x1, Math.round(y) + 0.5);
    ctx.stroke();
    ctx.fillText(yFormat(v), x0 - 6, y);
  }
}

/**
 * Rolling multi-series line chart.
 * series: [{ values:[...], color, label, dashed?, axis?: 'left'|'right' }]
 * rules:  [{ value, color, label }] horizontal reference lines (e.g. the deadline)
 */
export function lineChart(canvas, { series, rules = [], yMax = null, yMin = 0, pad = 0.08, yFormat, xLabel = null, legend = true }) {
  const { ctx, w, h } = fit(canvas);
  const left = 44;
  const box = { x0: left, y0: 8, x1: w - 8, y1: h - (legend ? 20 : 10) };
  let hi = yMax;
  if (hi === null) {
    hi = 0;
    for (const s of series) for (const v of s.values) if (Number.isFinite(v)) hi = Math.max(hi, v);
    for (const r of rules) hi = Math.max(hi, r.value);
    hi = niceMax(hi * (1 + pad)) || 1;
  }
  axes(ctx, box, { yMax: hi, yMin, yFormat: yFormat || ((v) => v.toFixed(hi <= 2 ? 2 : 0)) });

  const n = Math.max(...series.map((s) => s.values.length), 1);
  const X = (i) => box.x0 + (n <= 1 ? 0 : ((box.x1 - box.x0) * i) / (n - 1));
  const Y = (v) => box.y1 - ((v - yMin) / (hi - yMin || 1)) * (box.y1 - box.y0);

  for (const r of rules) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = r.color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(box.x0, Y(r.value));
    ctx.lineTo(box.x1, Y(r.value));
    ctx.stroke();
    ctx.restore();
    if (r.label) {
      ctx.fillStyle = r.color;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(r.label, box.x0 + 4, Y(r.value) - 2);
    }
  }

  for (const s of series) {
    if (!s.values.length) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.6;
    if (s.dashed) ctx.setLineDash([3, 3]);
    ctx.beginPath();
    let started = false;
    s.values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      const x = X(i);
      const y = Y(Math.min(v, hi));
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    if (s.dots) {
      ctx.fillStyle = s.color;
      s.values.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        ctx.beginPath();
        ctx.arc(X(i), Y(Math.min(v, hi)), 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }

  if (legend) {
    let x = box.x0;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const s of series) {
      if (!s.label) continue;
      ctx.fillStyle = s.color;
      ctx.fillRect(x, h - 12, 9, 3);
      ctx.fillStyle = INK;
      ctx.fillText(s.label, x + 13, h - 10);
      x += 22 + ctx.measureText(s.label).width;
    }
    if (xLabel) {
      ctx.fillStyle = INK_DIM;
      ctx.textAlign = "right";
      ctx.fillText(xLabel, box.x1, h - 10);
    }
  }
}

/**
 * Grouped bars with 95% intervals, the paper Fig. 2 shape.
 * categories: ["stable", ...]
 * series: [{ label, color, points: [{mean, lo, hi}] }]
 */
export function groupedBars(canvas, { categories, series, yMax = null, yFormat, height = null }) {
  const { ctx, w, h } = fit(canvas, height);
  const left = 44;
  const box = { x0: left, y0: 10, x1: w - 8, y1: h - 46 };
  let hi = yMax;
  if (hi === null) {
    hi = 0;
    for (const s of series) for (const p of s.points) if (p && Number.isFinite(p.hi ?? p.mean)) hi = Math.max(hi, p.hi ?? p.mean);
    hi = niceMax(hi * 1.1) || 1;
  }
  axes(ctx, box, { yMax: hi, yFormat: yFormat || ((v) => (hi <= 2 ? v.toFixed(1) : String(Math.round(v)))) });

  const groups = categories.length;
  const gw = (box.x1 - box.x0) / groups;
  const bw = Math.max(3, (gw * 0.74) / series.length);
  const Y = (v) => box.y1 - (Math.min(v, hi) / hi) * (box.y1 - box.y0);

  categories.forEach((cat, gi) => {
    const gx = box.x0 + gi * gw + gw * 0.13;
    series.forEach((s, si) => {
      const p = s.points[gi];
      if (!p || !Number.isFinite(p.mean)) return;
      const x = gx + si * bw;
      ctx.fillStyle = s.color;
      const y = Y(p.mean);
      ctx.fillRect(x, y, bw - 1.5, box.y1 - y);
      if (Number.isFinite(p.lo) && Number.isFinite(p.hi) && p.hi > p.lo) {
        ctx.strokeStyle = "#e8eef8";
        ctx.lineWidth = 1;
        const cx = x + (bw - 1.5) / 2;
        ctx.beginPath();
        ctx.moveTo(cx, Y(p.lo));
        ctx.lineTo(cx, Y(p.hi));
        ctx.moveTo(cx - 2.5, Y(p.lo));
        ctx.lineTo(cx + 2.5, Y(p.lo));
        ctx.moveTo(cx - 2.5, Y(p.hi));
        ctx.lineTo(cx + 2.5, Y(p.hi));
        ctx.stroke();
      }
    });
    ctx.fillStyle = INK_DIM;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(cat, box.x0 + gi * gw + gw / 2, box.y1 + 5);
  });

  let x = box.x0;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(x, h - 11, 9, 7);
    ctx.fillStyle = INK;
    ctx.fillText(s.label, x + 13, h - 8);
    x += 22 + ctx.measureText(s.label).width;
  }
}

/**
 * Stacked area over an x axis, the paper Fig. 3a shape.
 * x: [values], series: [{ label, color, values: [percent] }]
 */
export function stackedArea(canvas, { x, series, xLabel = "", yMax = 100, height = null }) {
  const { ctx, w, h } = fit(canvas, height);
  const box = { x0: 44, y0: 10, x1: w - 8, y1: h - 46 };
  axes(ctx, box, { yMax, yFormat: (v) => `${Math.round(v)}%` });
  const n = x.length;
  const X = (i) => box.x0 + (n <= 1 ? 0 : ((box.x1 - box.x0) * i) / (n - 1));
  const Y = (v) => box.y1 - (v / yMax) * (box.y1 - box.y0);

  const base = new Array(n).fill(0);
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.globalAlpha = 0.82;
    ctx.beginPath();
    for (let i = 0; i < n; i++) ctx.lineTo(X(i), Y(base[i] + (s.values[i] || 0)));
    for (let i = n - 1; i >= 0; i--) ctx.lineTo(X(i), Y(base[i]));
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
    for (let i = 0; i < n; i++) base[i] += s.values[i] || 0;
  }

  ctx.fillStyle = INK_DIM;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i < n; i++) ctx.fillText(String(x[i]), X(i), box.y1 + 5);
  if (xLabel) {
    ctx.textAlign = "right";
    ctx.fillText(xLabel, box.x1, box.y1 + 18);
  }

  let lx = box.x0;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, h - 11, 9, 7);
    ctx.fillStyle = INK;
    ctx.fillText(s.label, lx + 13, h - 8);
    lx += 22 + ctx.measureText(s.label).width;
  }
}

/**
 * Multi-series line over a shared numeric x axis, the paper Fig. 3b shape.
 * x: [values], series: [{ label, color, values }]
 */
export function xyLines(canvas, { x, series, xLabel = "", yFormat, yMax = null, height = null }) {
  const { ctx, w, h } = fit(canvas, height);
  const box = { x0: 44, y0: 10, x1: w - 8, y1: h - 46 };
  let hi = yMax;
  if (hi === null) {
    hi = 0;
    for (const s of series) for (const v of s.values) if (Number.isFinite(v)) hi = Math.max(hi, v);
    hi = niceMax(hi * 1.15) || 1;
  }
  axes(ctx, box, { yMax: hi, yFormat: yFormat || ((v) => v.toFixed(hi <= 2 ? 2 : 1)) });
  const n = x.length;
  const X = (i) => box.x0 + (n <= 1 ? 0 : ((box.x1 - box.x0) * i) / (n - 1));
  const Y = (v) => box.y1 - (Math.min(v, hi) / hi) * (box.y1 - box.y0);

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    s.values.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.stroke();
    ctx.fillStyle = s.color;
    s.values.forEach((v, i) => {
      ctx.beginPath();
      ctx.arc(X(i), Y(v), 2.4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  ctx.fillStyle = INK_DIM;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i < n; i++) ctx.fillText(String(x[i]), X(i), box.y1 + 5);
  if (xLabel) {
    ctx.textAlign = "right";
    ctx.fillText(xLabel, box.x1, box.y1 + 18);
  }

  let lx = box.x0;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, h - 11, 9, 3);
    ctx.fillStyle = INK;
    ctx.fillText(s.label, lx + 13, h - 8);
    lx += 22 + ctx.measureText(s.label).width;
  }
}

/** One horizontal 100%-stacked bar, for the running mode share. */
export function shareBar(canvas, { parts, height = null }) {
  const { ctx, w, h } = fit(canvas, height);
  const barH = 26;
  const y = 12;
  const total = parts.reduce((a, p) => a + (p.value || 0), 0) || 1;
  let x = 0;
  for (const p of parts) {
    const pw = ((p.value || 0) / total) * w;
    ctx.fillStyle = p.color;
    ctx.fillRect(x, y, Math.max(0, pw - 1), barH);
    if (pw > 46) {
      ctx.fillStyle = "#04070e";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace';
      ctx.fillText(`${((p.value / total) * 100).toFixed(0)}%`, x + pw / 2, y + barH / 2);
      ctx.font = FONT;
    }
    x += pw;
  }
  // Legend wraps onto as many rows as the width needs, so a narrow phone
  // viewport does not clip the mode names.
  let lx = 0;
  let ly = y + barH + 16;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  for (const p of parts) {
    const text = `${p.label} ${(((p.value || 0) / total) * 100).toFixed(1)}%`;
    const width = 24 + ctx.measureText(text).width;
    if (lx > 0 && lx + width > w) {
      lx = 0;
      ly += 15;
    }
    ctx.fillStyle = p.color;
    ctx.fillRect(lx, ly - 4, 9, 7);
    ctx.fillStyle = INK;
    ctx.fillText(text, lx + 13, ly);
    lx += width;
  }
}
