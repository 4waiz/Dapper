// The robot -> edge -> cloud stage.
//
// Nothing here decides anything. Every packet drawn was produced by a real
// decision from executor.js, and its fate on screen is the fate the execution
// model recorded: a local inference that never leaves the robot, a hybrid
// request that returns an immediate local answer while a refinement is in
// flight, a lost packet, a reply rejected at the deadline, or a degraded-safe
// reuse of the last still-fresh output.
//
// Packet flight time is scaled to the frame's *real* latency relative to the
// deadline, so a reply that misses the deadline visibly arrives after the
// deadline ring closes.

import { MODE_COLOR } from "./fmt.js";

const BG = "#02040a";
const GRID = "#101a30";
const INK = "#a3b3cc";
const INK_DIM = "#6d809e";
const OK = "#34d399";
const LATE = "#ff3d71";
const DROP = "#64748b";

const PACKET_LIFETIME_MS = 1500;

export class Scene {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.packets = [];
    this.flashes = [];
    this.now = 0;
    this.latest = null;
    this.deadlineMs = 100;
    this.reuseGlow = 0;
    this.localPulse = 0;
  }

  reset() {
    this.packets = [];
    this.flashes = [];
    this.latest = null;
    this.reuseGlow = 0;
    this.localPulse = 0;
  }

  /**
   * Record one executed frame. `step` is the object stepFrame returned, so the
   * drawing is driven by the engine's own row -- never by a guess.
   */
  push(step, deadlineMs) {
    const { row, arrival, timeout, lost } = step;
    this.deadlineMs = deadlineMs;
    this.latest = row;
    this.localPulse = 1;

    if (row.output_reused) this.reuseGlow = 1;

    if (row.selected_mode === "hybrid" || row.selected_mode === "edge_accurate") {
      const backend = row.output_source === "cloud" || step.backend === "cloud" ? "cloud" : "edge";
      let fate = "accepted";
      if (row.remote_failed_unavailable) fate = "unreachable";
      else if (row.remote_failed_loss) fate = "lost";
      else if (row.remote_rejected_deadline) fate = "late";
      else if (row.remote_rejected_freshness) fate = "stale";
      else if (!row.remote_accepted) fate = "late";
      this.packets.push({
        born: this.now,
        backend,
        fate,
        mode: row.selected_mode,
        // Normalised arrival: 1.0 means "exactly at the deadline".
        arrivalRatio: Number.isFinite(arrival) ? arrival / deadlineMs : 1.4,
        lost,
        timeoutRatio: Number.isFinite(timeout) ? timeout / deadlineMs : 1,
      });
      if (this.packets.length > 26) this.packets.shift();
    }

    this.flashes.push({
      born: this.now,
      missed: !row.deadline_met,
      mode: row.selected_mode,
    });
    if (this.flashes.length > 40) this.flashes.shift();
  }

  /** Advance the animation clock. */
  tick(dtMs) {
    this.now += dtMs;
    this.reuseGlow = Math.max(0, this.reuseGlow - dtMs / 700);
    this.localPulse = Math.max(0, this.localPulse - dtMs / 320);
    this.packets = this.packets.filter((p) => this.now - p.born < PACKET_LIFETIME_MS);
    this.flashes = this.flashes.filter((f) => this.now - f.born < 900);
  }

  draw() {
    const canvas = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 960;
    const h = Math.round((w * 9) / 16);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    this.grid(ctx, w, h);

    // Everything below is sized from the canvas width, so a 390 px phone gets a
    // readable diagram rather than a shrunken desktop one. Below `compact` the
    // secondary captions are dropped instead of being allowed to collide.
    this.s = Math.max(0.52, Math.min(1, w / 1080));
    this.compact = w < 560;

    const robot = { x: w * 0.14, y: h * 0.5 };
    const edge = { x: w * 0.5, y: h * 0.29 };
    const cloud = { x: w * 0.86, y: h * 0.17 };
    // Lifted on a phone: the stage bar wraps to two rows there and would
    // otherwise sit on top of the deadline ring's caption.
    const control = { x: w * 0.53, y: h * (this.compact ? 0.52 : 0.72) };

    this.link(ctx, robot, edge, "access RTT");
    this.link(ctx, edge, cloud, "+80 ms wide area");
    this.controlLink(ctx, robot, control);

    this.packetsLayer(ctx, robot, edge, cloud);
    this.robot(ctx, robot);
    this.node(ctx, edge, "EDGE", "45–90 ms compute", "#8b5cf6");
    this.node(ctx, cloud, "CLOUD", "60–130 ms compute", "#f472b6");
    this.controlLoop(ctx, control);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  grid(ctx, w, h) {
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    const step = Math.max(28, Math.round(w / 34));
    ctx.beginPath();
    for (let x = step; x < w; x += step) {
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, h);
    }
    for (let y = step; y < h; y += step) {
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(w, y + 0.5);
    }
    ctx.stroke();
  }

  link(ctx, a, b, label) {
    ctx.strokeStyle = "#1e2c4d";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
    if (this.compact) return;
    ctx.fillStyle = INK_DIM;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(label, (a.x + b.x) / 2, (a.y + b.y) / 2 - 7);
  }

  controlLink(ctx, robot, control) {
    ctx.strokeStyle = "#1e2c4d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(robot.x, robot.y + 26);
    ctx.quadraticCurveTo(robot.x, control.y, control.x - 60, control.y);
    ctx.stroke();
  }

  robot(ctx, p) {
    const s = this.s;
    const r = 30 * s;
    const glow = 0.25 + 0.5 * this.localPulse;
    ctx.save();
    ctx.shadowColor = `rgba(34, 211, 238, ${glow})`;
    ctx.shadowBlur = 22;
    ctx.fillStyle = "#0b1626";
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    roundRect(ctx, p.x - r, p.y - r * 0.8, r * 2, r * 1.6, 9 * s);
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // camera eye
    ctx.fillStyle = "#22d3ee";
    ctx.beginPath();
    ctx.arc(p.x, p.y - 2, 7 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#02040a";
    ctx.beginPath();
    ctx.arc(p.x, p.y - 2, 3 * s, 0, Math.PI * 2);
    ctx.fill();
    // treads
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 3 * s;
    ctx.beginPath();
    ctx.moveTo(p.x - r + 4 * s, p.y + r * 0.9);
    ctx.lineTo(p.x + r - 4 * s, p.y + r * 0.9);
    ctx.stroke();

    ctx.fillStyle = INK;
    ctx.font = `600 ${(11 * s).toFixed(1)}px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("ROBOT", p.x, p.y + r * 1.15);
    if (!this.compact) {
      ctx.fillStyle = INK_DIM;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText("15–35 ms local model", p.x, p.y + r * 1.15 + 14);
    }

    if (this.reuseGlow > 0) {
      ctx.strokeStyle = `rgba(255, 61, 113, ${0.65 * this.reuseGlow})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r + 12 + 10 * (1 - this.reuseGlow), 0, Math.PI * 2);
      ctx.stroke();
      if (!this.compact) {
        ctx.fillStyle = `rgba(255, 61, 113, ${0.9 * this.reuseGlow})`;
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.textAlign = "left";
        ctx.fillText("REUSING LAST VALID", p.x + r + 16, p.y - 6);
      }
    }
  }

  node(ctx, p, name, sub, color) {
    const s = this.s;
    const wBox = 96 * s;
    const hBox = 48 * s;
    ctx.fillStyle = "#0b1626";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    roundRect(ctx, p.x - wBox / 2, p.y - hBox / 2, wBox, hBox, 9 * s);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = `700 ${(12 * s).toFixed(1)}px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(name, p.x, this.compact ? p.y : p.y - 6 * s);
    if (!this.compact) {
      ctx.fillStyle = INK_DIM;
      ctx.font = `${(9.5 * s).toFixed(1)}px "JetBrains Mono", monospace`;
      ctx.fillText(sub, p.x, p.y + 10 * s);
    }
  }

  /** The deadline ring: the control loop acts when it closes. */
  controlLoop(ctx, p) {
    const s = this.s;
    const r = 34 * s;
    const recent = this.flashes.slice(-14);
    const missed = recent.filter((f) => f.missed).length;
    ctx.strokeStyle = "#243457";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.stroke();

    const last = this.latest;
    if (last) {
      const frac = Math.min(1, last.control_output_latency_ms / this.deadlineMs);
      ctx.strokeStyle = last.deadline_met ? OK : LATE;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = MODE_COLOR[last.selected_mode] || INK;
      ctx.font = `700 ${(13 * s).toFixed(1)}px "JetBrains Mono", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(last.control_output_latency_ms.toFixed(0), p.x, p.y - 4 * s);
      ctx.fillStyle = INK_DIM;
      ctx.font = `${(9 * s).toFixed(1)}px "JetBrains Mono", monospace`;
      ctx.fillText(`of ${this.deadlineMs.toFixed(0)} ms`, p.x, p.y + 10 * s);
    }

    // On a phone the stage bar wraps to three rows and already reports the
    // control latency and the verdict, so the ring speaks for itself there.
    if (this.compact) return;
    ctx.fillStyle = INK;
    ctx.font = `600 ${(11 * s).toFixed(1)}px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText("CONTROL LOOP", p.x, p.y + r + 8 * s);
    {
      ctx.fillStyle = missed ? LATE : INK_DIM;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(missed ? `${missed} of last ${recent.length} frames late` : "every recent frame on time",
        p.x, p.y + r + 22);
    }
  }

  packetsLayer(ctx, robot, edge, cloud) {
    for (const p of this.packets) {
      const age = (this.now - p.born) / PACKET_LIFETIME_MS;
      if (age > 1) continue;
      const target = p.backend === "cloud" ? cloud : edge;
      // Out on the first half, back on the second.
      const out = Math.min(1, age * 2.2);
      const back = Math.max(0, (age - 0.45) * 2.0);
      this.packet(ctx, robot, target, out, "#e8eef8", 0.9 * (1 - age * 0.3));

      if (p.fate === "lost" && back > 0.15) {
        const t = Math.min(0.45, back);
        this.packet(ctx, target, robot, t, DROP, 0.8 * (1 - t / 0.45));
        if (t > 0.4) this.burst(ctx, lerpPoint(target, robot, 0.45), DROP);
        continue;
      }
      if (p.fate === "unreachable") {
        this.burst(ctx, lerpPoint(robot, target, 0.35), DROP);
        continue;
      }
      if (back > 0) {
        const colour = p.fate === "accepted" ? OK : LATE;
        this.packet(ctx, target, robot, Math.min(1, back), colour, 0.95);
        if (back >= 1) {
          this.burst(ctx, robot, colour);
          if (this.compact) continue;
          ctx.fillStyle = colour;
          ctx.font = '600 10px "JetBrains Mono", monospace';
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          const tag = p.fate === "accepted"
            ? "reply accepted"
            : p.fate === "stale" ? "rejected: outside W" : "rejected: after deadline";
          // Below the robot, clear of the access-RTT link label above it.
          ctx.fillText(tag, robot.x + 44, robot.y + 60);
        }
      }
    }
  }

  packet(ctx, a, b, t, colour, alpha) {
    if (t <= 0) return;
    const p = lerpPoint(a, b, Math.min(1, t));
    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
    ctx.fillStyle = colour;
    ctx.shadowColor = colour;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  burst(ctx, p, colour) {
    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 11, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

function lerpPoint(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
