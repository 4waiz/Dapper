// Number and label formatting, in one place so the same quantity is never
// rendered two ways on the same page.

export const MODE_LABEL = {
  local_fast: "local-fast",
  edge_accurate: "edge-accurate",
  hybrid: "hybrid",
  degraded_safe: "degraded-safe",
};

export const MODE_COLOR = {
  local_fast: "#22d3ee",
  edge_accurate: "#8b5cf6",
  hybrid: "#f5a524",
  degraded_safe: "#ff3d71",
};

export const TERM_COLOR = {
  rtt: "#22d3ee",
  loss: "#ff3d71",
  load: "#8b5cf6",
  frame_age: "#64748b",
  deadline: "#f5a524",
};

export const POLICY_COLOR = {
  local_only: "#22d3ee",
  edge_only: "#8b5cf6",
  cloud_only: "#f472b6",
  dapper: "#34d399",
  deadline_greedy: "#f5a524",
  confidence_deadline: "#60a5fa",
  rtt_threshold: "#a78bfa",
  oracle_feasible: "#94a3b8",
};

const NBSP = " "; // thin space, for unit separation

/** Fixed-decimal number; em-dash for anything non-finite or missing. */
export function num(x, digits = 2) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

export function pct(x, digits = 2) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return `${x.toFixed(digits)}%`;
}

/** A rate in [0,1] rendered as a percentage. */
export function ratePct(x, digits = 2) {
  return pct(x === null || x === undefined ? NaN : x * 100, digits);
}

export function ms(x, digits = 1) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return `${x.toFixed(digits)}${NBSP}ms`;
}

export function mb(x, digits = 2) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return `${x.toFixed(digits)}${NBSP}MB`;
}

export function signed(x, digits = 4) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  const t = x.toFixed(digits);
  if (Math.abs(x) < 0.5 * 10 ** -digits) return (0).toFixed(digits);
  return (x > 0 ? "+" : "−") + Math.abs(x).toFixed(digits);
}

export function ci(row, digits = 1, scale = 1) {
  if (!row || row.mean === null) return "—";
  const m = row.mean * scale;
  if (row.lo === null || row.hi === null) return m.toFixed(digits);
  return `${m.toFixed(digits)} <span class="ci">[${(row.lo * scale).toFixed(digits)}, ${(row.hi * scale).toFixed(digits)}]</span>`;
}

export function int(x) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return Math.round(x).toLocaleString("en-US");
}

/** 5.7e-14 style, for the parity readout. */
export function sci(x, digits = 1) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  if (x === 0) return "0";
  return x.toExponential(digits);
}

export function titleCase(s) {
  return String(s).charAt(0).toUpperCase() + String(s).slice(1);
}

/** "a", "a and b", "a, b, and c" — IEEE serial comma. */
export function join(items) {
  const xs = [...items];
  if (xs.length === 0) return "";
  if (xs.length <= 2) return xs.join(" and ");
  return `${xs.slice(0, -1).join(", ")}, and ${xs[xs.length - 1]}`;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
