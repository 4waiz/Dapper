"""
Draw the DAPPER decision-path figure used in the paper.

The branch structure is taken from what the scheduler actually implements
(`dapper/scheduler.py`, `decide()`), including the fourth low-risk branch that
returns to local execution when no remote refresh could arrive in time.
Generating the figure from the code rather than drawing it by hand is what keeps
the two from drifting apart.

Box widths are measured from the rendered text rather than guessed, so a label
can never overflow its box.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

C_IN = "#DCE6F1"      # inputs
C_GATE = "#FBE3C4"    # decision gates
C_MODE = "#D6EBE0"    # execution modes
C_OUT = "#E6DEEE"     # control output
EDGE = "#333333"

FIG_W, FIG_H = 7.16, 1.72          # IEEE double column
FS_BOX, FS_EDGE = 6.6, 5.9
PAD_X, PAD_Y = 7.0, 5.0            # points of padding inside each box

#: name -> (column centre x in points, row centre y in points, text, colour)
NODES = {
    "mon":    (36, 48, "runtime monitor\nRTT, loss, load", C_IN),
    "risk":   (110, 48, "deadline risk\nscore $R_t$", C_GATE),
    "conf":   (182, 89, "confidence\ngate", C_GATE),
    "feas":   (256, 89, "feasibility\ngate", C_GATE),
    "edge":   (334, 112, "edge-accurate", C_MODE),
    "hyb":    (334, 80, "hybrid", C_MODE),
    "accept": (416, 96, "accept?\ndeadline + fresh", C_GATE),
    "local":  (256, 46, "local-fast", C_MODE),
    "degr":   (182, 15, "degraded-safe\nreuse or local", C_MODE),
    "out":    (486, 48, "control\noutput", C_OUT),
}

#: (from, side) -> (to, side), label, label dy in points
ARROWS = [
    ("mon", "r", "risk", "l", "", 0),
    ("risk", "r", "conf", "l", r"$R_t<\theta_L$", 7),
    ("risk", "r", "local", "l", r"$\theta_L\!\leq\!R_t\!<\!\theta_D$", 7),
    ("risk", "b", "degr", "l", r"$R_t\geq\theta_D$", -6),
    ("conf", "r", "feas", "l", r"$c_t<\tau_c$", 6),
    ("conf", "b", "local", "t", r"$c_t\geq\tau_c$", 0),
    ("feas", "r", "edge", "l", r"$\hat{T}\leq mD$", 6),
    ("feas", "r", "hyb", "l", r"$\hat{T}^-\!\leq\!W$", -6),
    ("feas", "b", "local", "t", "no feasible\nrefresh", 0),
    ("edge", "r", "accept", "l", "", 0),
    ("hyb", "r", "accept", "l", "", 0),
    ("accept", "r", "out", "t", "on time\n+ fresh", 6),
    ("local", "r", "out", "l", "", 0),
    ("degr", "r", "out", "b", "", 0),
]


def measure(fig, text, fontsize):
    """Rendered width/height of a label, in points."""
    t = fig.text(0, 0, text, fontsize=fontsize, linespacing=1.22)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width * 72.0 / fig.dpi, bb.height * 72.0 / fig.dpi


def draw(out_paths, dpi=300):
    plt.rcParams.update({
        "font.size": FS_BOX, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "cm", "figure.dpi": 100,
    })
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    W, H = FIG_W * 72.0, FIG_H * 72.0
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    boxes = {}
    for key, (cx, cy, text, colour) in NODES.items():
        tw, th = measure(fig, text, FS_BOX)
        w, h = tw + 2 * PAD_X, th + 2 * PAD_Y
        boxes[key] = (cx - w / 2, cy - h / 2, w, h)
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0,rounding_size=3.2",
            linewidth=0.7, edgecolor=EDGE, facecolor=colour, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", zorder=3,
                fontsize=FS_BOX, linespacing=1.22)

    for a in boxes:
        xa, ya, wa, ha = boxes[a]
        assert xa >= -0.5 and xa + wa <= W + 0.5, f"{a} leaves the canvas"
        assert ya >= -0.5 and ya + ha <= H + 0.5, f"{a} leaves the canvas"
        for b in boxes:
            if a >= b:
                continue
            xb, yb, wb, hb = boxes[b]
            if (xa < xb + wb and xb < xa + wa
                    and ya < yb + hb and yb < ya + ha):
                raise AssertionError(f"boxes {a} and {b} overlap")

    def anchor(key, side):
        x, y, w, h = boxes[key]
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
                "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]

    for src, ss, dst, ds, label, dy in ARROWS:
        p0, p1 = anchor(src, ss), anchor(dst, ds)
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=6.0, linewidth=0.6,
            color=EDGE, shrinkA=0.5, shrinkB=1.2, zorder=1))
        if label:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(mx, my + dy, label, ha="center", va="center",
                    fontsize=FS_EDGE, zorder=4, linespacing=1.15,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.95))

    for p in out_paths:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fig.savefig(p, dpi=dpi)
        print(f"  wrote {p}")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", nargs="+", default=[
        os.path.join(ROOT, "results", "paper_figures", "dapper_architecture.pdf"),
        os.path.join(ROOT, "results", "paper_figures", "dapper_architecture.png"),
    ])
    args = p.parse_args(argv)
    draw(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
