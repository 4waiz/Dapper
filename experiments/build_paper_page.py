"""
Build the web edition of the camera-ready, and the posted PDF.

    python experiments/build_paper_page.py
    python experiments/build_paper_page.py --doi 10.1109/FMEC12345.2026.67890

What it does, in order:

1. **Refuses to build a page that disagrees with the paper.** It first runs the
   426 claim checks in experiments/verify_paper_claims.py. If any number in the
   camera-ready no longer matches results/, nothing is written.
2. Fills every {{token}} in paper/paper.template.html from the result CSVs --
   all seven tables and every number quoted in the prose. Nothing is typed.
3. Copies the paper's figures to site/paper/figures/ as PNGs, rasterising
   paper/figures/runtime_bias.pdf, which is the one camera-ready figure the
   repository has no generator for (see the note in write_figures).
4. Builds site/paper/dapper-fmec2026.pdf: the IEEE accepted-manuscript notice on
   a cover page, followed by the untouched pages of the official camera-ready.
   The posted PDF is the certified document, not a re-print of this web page.
5. Rewrites the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> in
   README.md from the same numbers.

The DOI is a placeholder until IEEE assigns one; pass --doi to set it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from dapper.config import load_config  # noqa: E402

from verify_paper_claims import PROFILES, Results, run as verify_run  # noqa: E402

TEMPLATE = os.path.join(ROOT, "paper", "paper.template.html")
OUT_DIR = os.path.join(ROOT, "site", "paper")
SOURCE_PDF = os.path.join(ROOT, "paper", "DAPPER_FMEC2026_CameraReady_IEEE_certified.pdf")
FALLBACK_PDF = os.path.join(ROOT, "paper", "DAPPER_FMEC2026_IEEE_CameraReady.pdf")
POSTED_PDF = os.path.join(OUT_DIR, "dapper-fmec2026.pdf")
DEFAULT_DOI = "10.1109/FMEC.2026.XXXXXXX (to be assigned)"

NUMBER_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
                9: "nine", 10: "ten", 15: "Fifteen"}

POLICY_LABEL = {"local_only": "local-only", "edge_only": "edge-only",
                "cloud_only": "cloud-only", "dapper": "<b>DAPPER</b>",
                "deadline_greedy": "deadline-greedy",
                "confidence_deadline": "conf.+deadline",
                "rtt_threshold": "RTT-threshold",
                "oracle_feasible": "oracle&dagger;"}
ABLATION_LABEL = {
    "dapper_full": "full DAPPER", "dapper_no_rtt": "&minus;&thinsp;RTT",
    "dapper_no_loss": "&minus;&thinsp;packet loss", "dapper_no_load": "&minus;&thinsp;edge load",
    "dapper_no_frame_age": "&minus;&thinsp;frame age",
    "dapper_no_deadline": "&minus;&thinsp;deadline pressure",
    "dapper_no_confidence_gate": "&minus;&thinsp;confidence gate",
    "dapper_no_freshness_gate": "&minus;&thinsp;freshness gate",
    "dapper_no_degraded_reuse": "&minus;&thinsp;degraded reuse",
}
REPLAY_PROFILES = ["stable", "lossy", "variable", "outage"]


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def thousands(x: float) -> str:
    return f"{int(round(x)):,}"


def millions(x: float) -> str:
    """1200000 -> '1.2 million', the form the abstract uses."""
    return f"{x / 1e6:.1f} million"


def ci(mean: float, lo: float, hi: float, d: int) -> str:
    return f"{mean:.{d}f} [{lo:.{d}f},{hi:.{d}f}]"


def row(cells: Sequence[str], cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<tr{attr}>" + "".join(cells) + "</tr>"


def th(text: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<th{attr}>{text}</th>"


def td(text: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<td{attr}>{text}</td>"


# ------------------------------------------------------------------- tables
def table_i(cfg: Dict[str, Any]) -> str:
    import export_web_config

    # The exported labels are plain text because the dashboard renders them in a
    # monospace cell; here they get the subscripts the manuscript sets.
    def pretty(label: str) -> str:
        out = html.escape(label)
        for plain, marked in (
            ("w_r", "w<sub>r</sub>"), ("w_l", "w<sub>l</sub>"), ("w_c", "w<sub>c</sub>"),
            ("w_a", "w<sub>a</sub>"), ("w_d", "w<sub>d</sub>"),
            ("θL", "&theta;<sub>L</sub>"), ("θD", "&theta;<sub>D</sub>"),
            ("τc", "&tau;<sub>c</sub>"),
        ):
            out = out.replace(plain, marked)
        return out

    rows = [row([td(pretty(r["parameter"]), "l"), td(r["value"])])
            for r in export_web_config.build(cfg)["table_i"]]
    return ("<thead>" + row([th("Parameter", "l"), th("Value")]) + "</thead><tbody>"
            + "".join(rows) + "</tbody>")


def table_ii(cfg: Dict[str, Any]) -> str:
    head = row([th("Profile", "l"), th("RTT ms"), th("Loss"), th("Load"), th("Outage")])
    rows = []
    for p in cfg["profiles"]:
        n = cfg["network_profiles"][p]
        rows.append(row([
            td(cap(p), "l"),
            td(f"{n['rtt_ms_mean']:g} &plusmn; {n['rtt_ms_std']:g}"),
            td(f"{n['packet_loss']:.2f}"),
            td(f"{n['edge_load']:g}"),
            td(f"{n['outage_prob']:.2f}"),
        ]))
    return f"<thead>{head}</thead><tbody>" + "".join(rows) + "</tbody>"


def table_iii(res: Results) -> str:
    head = row([th("Profile", "l"), th("Policy", "l"), th("p95 latency ms [95% CI]"),
                th("Miss % [95% CI]"), th("Conf. proxy"), th("BW MB/1k fr.")])
    rows = []
    for profile in PROFILES:
        first = True
        for policy in ("local_only", "edge_only", "cloud_only", "dapper"):
            rows.append(row([
                td(cap(profile) if first else "", "l"),
                td(POLICY_LABEL[policy], "l"),
                td(ci(res.ci(profile, policy, "p95_latency_ms"),
                      res.ci(profile, policy, "p95_latency_ms", "ci_lo"),
                      res.ci(profile, policy, "p95_latency_ms", "ci_hi"), 1)),
                td(ci(100 * res.ci(profile, policy, "deadline_miss_rate"),
                      100 * res.ci(profile, policy, "deadline_miss_rate", "ci_lo"),
                      100 * res.ci(profile, policy, "deadline_miss_rate", "ci_hi"), 2)),
                td(f"{res.ci(profile, policy, 'usable_confidence_proxy'):.3f}"),
                td(f"{1e-3 * res.ci(profile, policy, 'bandwidth_per_1000_frames_kb'):.2f}"),
            ], "b" if policy == "dapper" else ""))
            first = False
    return f"<thead>{head}</thead><tbody>" + "".join(rows) + "</tbody>"


def table_iv(res: Results) -> str:
    left_head = row([th("Profile", "l"), th("local"), th("edge"), th("hybrid"), th("degr."),
                     th("acc. %"), th("%"), th("%")])
    left_rows = []
    for p in PROFILES:
        left_rows.append(row([
            td(cap(p), "l"),
            td(f"{res.mode(p, 'pct_local_fast'):.1f}"),
            td(f"{res.mode(p, 'pct_edge_accurate'):.1f}"),
            td(f"{res.mode(p, 'pct_hybrid'):.1f}"),
            td(f"{res.mode(p, 'pct_degraded_safe'):.1f}"),
            td(f"{100 * res.mode(p, 'remote_accept_rate'):.1f}"),
            td(f"{100 * res.mode(p, 'reuse_rate'):.1f}"),
            td(f"{100 * res.ci(p, 'dapper', 'stale_acceptance_rate'):.2f}"),
        ]))
    left = ("<table><thead>"
            + row([th("", "l"), th("Mode share %", ""), th(""), th(""), th(""),
                   th("Remote"), th("Reuse"), th("Stale")])
            + left_head + "</thead><tbody>" + "".join(left_rows) + "</tbody></table>")

    right_head = row([th("Policy", "l"), th("Tx %"), th("Miss %"), th("p95"),
                      th("Conf."), th("MB/1k")])
    right_rows = []
    for policy in ("local_only", "deadline_greedy", "confidence_deadline",
                   "rtt_threshold", "dapper", "oracle_feasible"):
        right_rows.append(row([
            td(POLICY_LABEL[policy], "l"),
            td(f"{100 * res.ci_over_profiles(policy, 'remote_attempt_rate'):.2f}"),
            td(f"{100 * res.ci_over_profiles(policy, 'deadline_miss_rate'):.2f}"),
            td(f"{res.ci_over_profiles(policy, 'p95_latency_ms'):.1f}"),
            td(f"{res.ci_over_profiles(policy, 'usable_confidence_proxy'):.4f}"),
            td(f"{1e-3 * res.ci_over_profiles(policy, 'bandwidth_per_1000_frames_kb'):.2f}"),
        ], "b" if policy == "dapper" else ""))
    right = f"<table><thead>{right_head}</thead><tbody>" + "".join(right_rows) + "</tbody></table>"
    return left + right


def table_v(res: Results) -> str:
    head = row([th("Variant", "l"), th("Miss %"), th("p95 ms"), th("Conf. proxy"),
                th("MB/1k fr.")])
    rows = []
    for variant, label in ABLATION_LABEL.items():
        rows.append(row([
            td(label, "l"),
            td(f"{100 * res.abl(variant, 'deadline_miss_rate'):.2f}"),
            td(f"{res.abl(variant, 'p95_latency_ms'):.2f}"),
            td(f"{res.abl(variant, 'usable_confidence_proxy'):.4f}"),
            td(f"{1e-3 * res.abl(variant, 'bandwidth_per_1000_frames_kb'):.2f}"),
        ], "b" if variant == "dapper_full" else ""))
    return f"<thead>{head}</thead><tbody>" + "".join(rows) + "</tbody>"


def table_vi(res: Results) -> str:
    head = row([th("Model", "l"), th("Recall"), th("Safety recall"), th("mAP50"),
                th("mAP50&ndash;95")])
    rows = []
    for role, label, model in (("local", "YOLO11n (local)", "yolo11n.pt"),
                               ("remote", "YOLO11m (remote)", "yolo11m.pt")):
        rows.append(row([
            td(label, "l"),
            td(f"{res.detector(role, 'recall'):.3f}"),
            td(f"{res.detector(role, 'safety_recall'):.3f}"),
            td(f"{res.detector_map(model, 'mAP50'):.3f}"),
            td(f"{res.detector_map(model, 'mAP50_95'):.3f}"),
        ]))
    return f"<thead>{head}</thead><tbody>" + "".join(rows) + "</tbody>"


def table_vii(res: Results) -> str:
    head = (row([th("", "l"), th("", "l"), th("Delivered recall"), th(""), th("Miss"), th("Reuse")])
            + row([th("Profile", "l"), th("Policy", "l"), th("all"), th("fresh"), th("%"), th("%")]))
    rows = []
    for profile in REPLAY_PROFILES:
        first = True
        for policy in ("local_only", "edge_only", "dapper"):
            rows.append(row([
                td(cap(profile) if first else "", "l"),
                td(POLICY_LABEL[policy], "l"),
                td(f"{res.replay(profile, policy, 'delivered_recall'):.3f}"),
                td(f"{res.replay(profile, policy, 'delivered_recall_fresh_only'):.3f}"),
                td(f"{100 * res.replay(profile, policy, 'deadline_miss_rate'):.2f}"),
                td(f"{100 * res.replay(profile, policy, 'reuse_rate'):.1f}"),
            ], "b" if policy == "dapper" else ""))
            first = False
    return f"<thead>{head}</thead><tbody>" + "".join(rows) + "</tbody>"


# ------------------------------------------------------------------ values
def count_tests() -> int:
    """Ask pytest how many tests the camera-ready suite collects."""
    out = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+)\s+tests?\s+collected", out.stdout)
    return int(m.group(1)) if m else 0


def values(res: Results, cfg: Dict[str, Any], doi: str) -> Dict[str, str]:
    seeds = int(cfg["evaluation_seeds_count"])
    frames = int(cfg["frames"])
    n_profiles = len(PROFILES)
    n_policies = 8
    v: Dict[str, str] = {}

    # --- protocol scale ---------------------------------------------------
    v["n_seeds"] = str(seeds)
    v["n_frames"] = thousands(frames)
    v["n_profiles_words"] = NUMBER_WORDS[n_profiles]
    v["n_policies_words"] = NUMBER_WORDS[n_policies]
    v["n_policies_words_cap"] = cap(NUMBER_WORDS[n_policies])
    v["n_streams_words"] = NUMBER_WORDS[15]
    v["n_decisions_words"] = millions(seeds * frames * n_profiles * n_policies)
    v["n_runs"] = thousands(seeds * n_profiles * n_policies)
    v["n_tests"] = str(count_tests())
    v["n_boot"] = thousands(10000)
    cal = [int(s) for s in cfg["calibration_seeds"]]
    v["cal_seeds"] = f"{min(cal)}–{max(cal)}"
    start = int(cfg["evaluation_seeds_start"])
    v["eval_seeds"] = f"{start}–{start + seeds - 1}"
    v["n_candidates"] = thousands(len(res.csv("calibration/all_candidates.csv")))
    v["n_zero_miss"] = str(res.zero_miss_candidates())
    v["coco_images"] = thousands(res.detector("local", "n_images"))
    v["replay_seeds"] = str(res.csv("real_detector/dapper_replay.csv")["seed"].nunique())

    # --- model envelopes, straight from config.yaml ------------------------
    for k in ("local", "edge", "cloud"):
        sec = cfg[k]
        v[f"{k}_latency_range"] = f"{sec['latency_ms_min']:g}–{sec['latency_ms_max']:g}"
        v[f"{k}_conf_range"] = f"{sec['confidence_min']:.2f}–{sec['confidence_max']:.2f}"
    v["edge_upload_kb"] = f"{cfg['edge']['bandwidth_kb']:g}"
    v["cloud_upload_kb"] = f"{cfg['cloud']['bandwidth_kb']:g}"
    v["cloud_extra_rtt"] = f"{cfg['cloud']['extra_rtt_ms']:g}"

    # --- headline results --------------------------------------------------
    v["worst_miss"] = f"{100 * res.ci_worst('dapper', 'deadline_miss_rate'):.2f}"
    v["worst_stale"] = f"{100 * max(res.ci(p, q, 'stale_acceptance_rate') for p in PROFILES for q in ('local_only', 'edge_only', 'cloud_only', 'dapper', 'deadline_greedy', 'confidence_deadline', 'rtt_threshold', 'oracle_feasible')):.2f}"

    def p95ci(profile: str, policy: str) -> str:
        return (f"{res.ci(profile, policy, 'p95_latency_ms'):.1f} ms [95% CI: "
                f"{res.ci(profile, policy, 'p95_latency_ms', 'ci_lo'):.1f}–"
                f"{res.ci(profile, policy, 'p95_latency_ms', 'ci_hi'):.1f}]")

    v["p95_var_dapper"] = p95ci("variable", "dapper")
    v["p95_var_edge"] = p95ci("variable", "edge_only")
    v["p95_var_cloud"] = p95ci("variable", "cloud_only")

    for profile in ("stable", "lossy", "variable"):
        d = res.paired("usable_confidence_proxy", "dapper", "local_only", profile)
        v[f"gain_{profile}"] = (f"{d['mean_diff']:.4f} [paired 95% CI: {d['ci_lo']:.4f}–"
                                f"{d['ci_hi']:.4f}]")

    # --- placement and accounting -----------------------------------------
    v["stable_offload"] = f"{res.mode('stable', 'pct_hybrid') + res.mode('stable', 'pct_edge_accurate'):.1f}"
    v["outage_offload"] = f"{res.mode('outage', 'pct_hybrid') + res.mode('outage', 'pct_edge_accurate'):.1f}"
    v["outage_reuse"] = f"{100 * res.mode('outage', 'reuse_rate'):.1f}"
    for profile in ("stable", "lossy"):
        v[f"{profile}_accepted"] = f"{100 * res.ci(profile, 'dapper', 'remote_accept_rate'):.1f}"
        v[f"{profile}_late"] = f"{100 * res.ci(profile, 'dapper', 'remote_rejected_deadline_rate'):.1f}"
        v[f"{profile}_lost"] = f"{100 * res.ci(profile, 'dapper', 'remote_failed_loss_rate'):.1f}"

    # --- comparators and the oracle ---------------------------------------
    # The paper states this one as an upper bound ("transmits on under X% of
    # frames"), so it is rounded UP to the next hundredth rather than to
    # nearest: the exact profile-mean is 0.0147%, and a to-nearest 0.01 would
    # make the sentence false.
    tx = 100 * res.ci_over_profiles("rtt_threshold", "remote_attempt_rate")
    v["rtt_threshold_tx"] = f"{math.ceil(tx * 100) / 100:.2f}"
    loc = res.ci("stable", "local_only", "usable_confidence_proxy")
    dap = res.ci("stable", "dapper", "usable_confidence_proxy")
    ora = res.ci("stable", "oracle_feasible", "usable_confidence_proxy")
    v["stable_local_conf"] = f"{loc:.4f}"
    v["stable_dapper_conf"] = f"{dap:.4f}"
    v["stable_oracle_conf"] = f"{ora:.4f}"
    v["oracle_share"] = f"{100 * (dap - loc) / (ora - loc):.0f}"
    v["stable_dapper_bw"] = f"{1e-3 * res.ci('stable', 'dapper', 'bandwidth_per_1000_frames_kb'):.2f}"
    v["stable_oracle_bw"] = f"{1e-3 * res.ci('stable', 'oracle_feasible', 'bandwidth_per_1000_frames_kb'):.2f}"

    # --- ablation prose -----------------------------------------------------
    v["abl_stable_full"] = f"{res.abl('dapper_full', 'usable_confidence_proxy', 'stable'):.4f}"
    v["abl_stable_noloss"] = f"{res.abl('dapper_no_loss', 'usable_confidence_proxy', 'stable'):.4f}"
    v["abl_lossy_full"] = f"{res.abl('dapper_full', 'usable_confidence_proxy', 'lossy'):.4f}"
    v["abl_lossy_noloss"] = f"{res.abl('dapper_no_loss', 'usable_confidence_proxy', 'lossy'):.4f}"
    full_bw = res.abl("dapper_full", "bandwidth_per_1000_frames_kb", "variable")
    no_fresh_bw = res.abl("dapper_no_freshness_gate", "bandwidth_per_1000_frames_kb", "variable")
    v["abl_var_full_bw"] = f"{1e-3 * full_bw:.2f}"
    v["abl_var_nofresh_bw"] = f"{1e-3 * no_fresh_bw:.2f}"
    v["fresh_saving"] = f"{100 * (no_fresh_bw / full_bw - 1.0):.0f}"
    v["abl_outage_full_p95"] = f"{res.abl('dapper_full', 'p95_latency_ms', 'outage'):.2f}"
    v["abl_outage_noreuse_p95"] = f"{res.abl('dapper_no_degraded_reuse', 'p95_latency_ms', 'outage'):.2f}"

    # --- deadline and margin sweeps ----------------------------------------
    dl = res.csv("sensitivity/deadline_mode_distribution.csv")
    allp = dl[dl.profile == "ALL"].sort_values("deadline_sweep_ms")
    first_edge = allp[allp.pct_edge_accurate > 0].iloc[0]
    largest = allp.iloc[-1]
    v["edge_first_deadline"] = f"{first_edge.deadline_sweep_ms:g}"
    v["edge_share_max"] = f"{largest.pct_edge_accurate:.1f}"
    v["edge_max_deadline"] = f"{largest.deadline_sweep_ms:g}"
    for d in (125, 150, 200):
        v[f"miss_{d}"] = f"{100 * res.deadline(float(d), 'deadline_miss_rate'):.2f}"

    mg = res.csv("sensitivity/commit_margin_sweep_ci.csv")
    swept = mg[(mg.metric == "deadline_miss_rate") & (mg.deadline_sweep_ms >= 100)
               & (mg.deadline_sweep_ms <= 200)]
    worst_by_margin = swept.groupby("remote_deadline_margin")["mean"].max()
    safe = sorted(float(m) for m in worst_by_margin[worst_by_margin <= 0.0].index)
    v["safe_margin"] = f"{safe[-1]:g}" if safe else "—"
    v["margin_sweep_lo"] = f"{swept.deadline_sweep_ms.min():g}"
    v["margin_sweep_hi"] = f"{swept.deadline_sweep_ms.max():g}"
    v["calibrated_margin"] = f"{cfg['scheduler']['remote_deadline_margin']:g}"

    est = res.csv("sensitivity/runtime_estimation_error_ci.csv")
    biases = sorted(float(b) for b in est.bias.unique())
    v["bias_lo"] = f"−{abs(biases[0]) * 100:.0f}%"
    v["bias_hi"] = f"+{biases[-1] * 100:.0f}%"
    v["compute_bias_10"] = f"{100 * res.est('compute', -0.10, 'deadline_miss_rate'):.2f}"
    v["compute_bias_30"] = f"{100 * res.est('compute', -0.30, 'deadline_miss_rate'):.2f}"
    v["both_bias_30"] = f"{100 * res.est('both', -0.30, 'deadline_miss_rate'):.2f}"

    # --- overhead and the detector study ------------------------------------
    v["overhead_calls"] = thousands(res.overhead("n_calls"))
    v["overhead_mean"] = f"{res.overhead('mean_us'):.2f}"
    v["overhead_median"] = f"{res.overhead('median_us'):.2f}"
    v["overhead_p95"] = f"{res.overhead('p95_us'):.2f}"
    v["overhead_p99"] = f"{res.overhead('p99_us'):.2f}"
    v["overhead_added"] = f"{res.wall('dapper') - res.wall('local_only'):.2f}"
    v["switch_min"] = f"{res.switches('dapper', 'min'):.0f}"
    v["switch_max"] = f"{res.switches('dapper', 'max'):.0f}"
    v["recall_gap"] = f"{100 * (res.detector('remote', 'recall') - res.detector('local', 'recall')):.1f}"
    v["safety_gap"] = f"{100 * (res.detector('remote', 'safety_recall') - res.detector('local', 'safety_recall')):.1f}"
    v["yolo_n_cpu"] = f"{res.detector_latency('local_cpu'):.1f}"
    v["yolo_m_cpu"] = f"{res.detector_latency('remote_cpu'):.1f}"
    v["yolo_n_gpu"] = f"{res.detector_latency('local'):.1f}"
    v["yolo_m_gpu"] = f"{res.detector_latency('remote'):.1f}"
    v["rep_stable_dapper"] = f"{res.replay('stable', 'dapper', 'delivered_recall'):.3f}"
    v["rep_stable_local"] = f"{res.replay('stable', 'local_only', 'delivered_recall'):.3f}"
    v["rep_stable_dapper_safety"] = f"{res.replay('stable', 'dapper', 'delivered_safety_recall'):.3f}"
    v["rep_stable_local_safety"] = f"{res.replay('stable', 'local_only', 'delivered_safety_recall'):.3f}"
    v["rep_lossy_edge_miss"] = f"{100 * res.replay('lossy', 'edge_only', 'deadline_miss_rate'):.2f}"
    v["rep_var_reuse"] = f"{100 * res.replay('variable', 'dapper', 'reuse_rate'):.1f}"
    v["rep_outage_reuse"] = f"{100 * res.replay('outage', 'dapper', 'reuse_rate'):.1f}"

    # --- tables and front matter --------------------------------------------
    v["table_i"] = table_i(cfg)
    v["table_ii"] = table_ii(cfg)
    v["table_iii"] = table_iii(res)
    v["table_iv"] = table_iv(res)
    v["table_v"] = table_v(res)
    v["table_vi"] = table_vi(res)
    v["table_vii"] = table_vii(res)
    v["doi"] = html.escape(doi)
    v["ieee_year"] = "2026"
    v["build_date"] = _dt.date.today().isoformat()
    return v


# ------------------------------------------------------------------ outputs
def write_figures() -> None:
    """
    Copy the camera-ready's figures as PNGs.

    Four of the five come from results/paper_figures/, where make_figures.py and
    make_architecture_figure.py write a PDF and a 300-dpi PNG of the same plot.
    The fifth, paper/figures/runtime_bias.pdf, is the camera-ready's Fig. 3b and
    has no generator anywhere in the repository -- make_figures.py writes a
    different, two-panel figure. It is rasterised here so that the web edition
    shows the panel the paper actually prints.
    """
    out = os.path.join(OUT_DIR, "figures")
    os.makedirs(out, exist_ok=True)
    src = os.path.join(ROOT, "results", "paper_figures")
    for name in ("dapper_architecture", "fig2_p95_latency_panel",
                 "fig2b_usable_confidence_panel", "fig3_deadline_mode_distribution"):
        png = os.path.join(src, f"{name}.png")
        if os.path.exists(png):
            shutil.copy(png, os.path.join(out, f"{name}.png"))
            print(f"  figure {name}.png")
        else:
            print(f"  MISSING {png}")

    bias_pdf = os.path.join(ROOT, "paper", "figures", "runtime_bias.pdf")
    target = os.path.join(out, "runtime_bias.png")
    if os.path.exists(bias_pdf):
        try:
            import fitz

            doc = fitz.open(bias_pdf)
            doc[0].get_pixmap(dpi=300).save(target)
            doc.close()
            print("  figure runtime_bias.png (rasterised from the camera-ready PDF)")
        except ImportError:
            print("  PyMuPDF missing; cannot rasterise runtime_bias.pdf")
    else:
        print(f"  MISSING {bias_pdf}")


def notice_text(doi: str) -> List[str]:
    return [
        "© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for",
        "all other uses, in any current or future media, including reprinting/republishing this material for",
        "advertising or promotional purposes, creating new collective works, for resale or redistribution to",
        "servers or lists, or reuse of any copyrighted component of this work in other works.",
        "",
        "This is the ACCEPTED VERSION of the manuscript, posted by the authors. The published version will",
        "appear in the Proceedings of the 2026 IEEE International Conference on Fog and Mobile Edge Computing",
        f"(FMEC).    DOI: {doi}",
        "",
        "Cite the published version. The implementation, the generated results and the reproduction scripts",
        "are available at https://github.com/4waiz/Dapper",
    ]


def write_pdf(doi: str) -> Optional[str]:
    """Prepend the accepted-manuscript notice to the official camera-ready."""
    try:
        import fitz
    except ImportError:
        print("  PyMuPDF missing; posted PDF not built")
        return None

    source = SOURCE_PDF if os.path.exists(SOURCE_PDF) else FALLBACK_PDF
    if not os.path.exists(source):
        print("  no camera-ready PDF found; posted PDF not built")
        return None
    if source == FALLBACK_PDF:
        print("  WARNING: the IEEE-certified PDF is absent; using the local build instead")

    doc = fitz.open(source)
    page0 = doc[0].rect
    cover = doc.new_page(pno=0, width=page0.width, height=page0.height)

    m = 56
    y = 96
    logo = os.path.join(ROOT, "site", "assets", "logo-256.png")
    if os.path.exists(logo):
        cover.insert_image(fitz.Rect(m, y - 34, m + 52, y + 18), filename=logo)
    cover.insert_text((m + 66, y), "DAPPER", fontname="hebo", fontsize=22)
    cover.insert_text((m + 66, y + 18), "Deadline-Aware Perception Placement for Edge Robotics",
                      fontname="helv", fontsize=9.5, color=(0.35, 0.35, 0.4))

    y += 70
    cover.insert_text((m, y), "DAPPER: Deadline-Aware Edge Perception for Mission-Critical",
                      fontname="hebo", fontsize=14)
    cover.insert_text((m, y + 19), "Robots under Dynamic Network Conditions",
                      fontname="hebo", fontsize=14)
    y += 48
    cover.insert_text((m, y), "Awaiz Ahmed and Khubaib Amjad Alam", fontname="helv", fontsize=11)
    cover.insert_text((m, y + 16), "Al Ain University, Abu Dhabi, United Arab Emirates",
                      fontname="helv", fontsize=10, color=(0.3, 0.3, 0.35))
    cover.insert_text((m, y + 32), "2026 IEEE International Conference on Fog and Mobile Edge Computing (FMEC)",
                      fontname="helv", fontsize=10, color=(0.3, 0.3, 0.35))

    y += 76
    cover.draw_line(fitz.Point(m, y), fitz.Point(page0.width - m, y), color=(0.72, 0.74, 0.78), width=0.7)
    y += 24
    for line in notice_text(doi):
        if line:
            cover.insert_text((m, y), line, fontname="helv", fontsize=8.6, color=(0.22, 0.22, 0.26))
        y += 12.4

    y += 10
    cover.draw_line(fitz.Point(m, y), fitz.Point(page0.width - m, y), color=(0.72, 0.74, 0.78), width=0.7)
    cover.insert_text((m, page0.height - 52),
                      "The pages that follow are the camera-ready manuscript, unaltered.",
                      fontname="hebo", fontsize=9, color=(0.22, 0.22, 0.26))

    doc.set_metadata({
        "title": "DAPPER: Deadline-Aware Edge Perception for Mission-Critical Robots under "
                 "Dynamic Network Conditions",
        "author": "Awaiz Ahmed, Khubaib Amjad Alam",
        "subject": f"Accepted manuscript, IEEE FMEC 2026. DOI {doi}. (c) 2026 IEEE.",
        "keywords": "edge computing, fog computing, robotics, real-time perception, "
                    "deadline-aware scheduling, computation offloading",
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    doc.save(POSTED_PDF, garbage=3, deflate=True)
    doc.close()
    size = os.path.getsize(POSTED_PDF) / 1024
    print(f"  wrote {os.path.relpath(POSTED_PDF, ROOT)} "
          f"(cover + {os.path.basename(source)}, {size:.0f} KiB)")
    return POSTED_PDF


def readme_results(res: Results, cfg: Dict[str, Any], v: Dict[str, str]) -> str:
    """The README block, generated from the same numbers as the paper page."""
    deadline = int(float(cfg["deadline_ms"]))
    lines = [
        "At the calibrated **{d} ms** control deadline, over **{n} frame-level decisions** "
        "({s} seeds x {f} frames x {p} profiles x {q} policies):".format(
            d=deadline, n=v["n_decisions_words"], s=v["n_seeds"], f=v["n_frames"],
            p=v["n_profiles_words"], q=v["n_policies_words"]),
        "",
        "| Profile | Policy | p95 latency (ms) | Deadline miss | Confidence proxy | Uplink (MB/1k frames) |",
        "|---|---|---|---|---|---|",
    ]
    for profile in PROFILES:
        for policy in ("local_only", "edge_only", "cloud_only", "dapper"):
            bold = policy == "dapper"
            cells = [
                cap(profile) if policy == "local_only" else "",
                POLICY_LABEL[policy].replace("<b>", "").replace("</b>", ""),
                "{:.1f}".format(res.ci(profile, policy, "p95_latency_ms")),
                "{:.2f}%".format(100 * res.ci(profile, policy, "deadline_miss_rate")),
                "{:.3f}".format(res.ci(profile, policy, "usable_confidence_proxy")),
                "{:.2f}".format(1e-3 * res.ci(profile, policy, "bandwidth_per_1000_frames_kb")),
            ]
            if bold:
                cells = [cells[0]] + ["**{}**".format(c) for c in cells[1:]]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "DAPPER holds a **{miss}% deadline-miss rate on every profile** while keeping tail latency at "
        "local-inference levels, and across all {runs} runs the rate at which an output older than its "
        "freshness window is accepted is **{stale}%**. It is not free: under the stable profile it spends "
        "**{bw} MB per {frames} frames**, of which only {acc}% of requests return a usable reply, and the "
        "frozen {d}-ms calibration produces {m125}%, {m150}% and {m200}% misses at D = 125, 150 and 200 ms. "
        "Full tables, ablations and limitations are in the "
        "[paper](https://4waiz.github.io/Dapper/paper/).".format(
            miss=v["worst_miss"], runs=v["n_runs"], stale=v["worst_stale"],
            bw=v["stable_dapper_bw"], frames=v["n_frames"], acc=v["stable_accepted"],
            d=deadline, m125=v["miss_125"], m150=v["miss_150"], m200=v["miss_200"]),
    ]
    return "\n".join(lines)


def update_readme(text: str) -> None:
    path = os.path.join(ROOT, "README.md")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in content or end not in content:
        print("  README has no RESULTS block; skipped")
        return
    head, rest = content.split(start, 1)
    _, tail = rest.split(end, 1)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{head}{start}\n{text}\n{end}{tail}")
    print("  updated the README results block")


def render(v: Dict[str, str]) -> str:
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html_text = f.read()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", html_text)) - set(v))
    if missing:
        raise SystemExit(f"template tokens without values: {missing}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: v[m.group(1)], html_text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--doi", default=DEFAULT_DOI)
    p.add_argument("--skip-verify", action="store_true",
                   help="build even if a camera-ready claim no longer verifies (not for publishing)")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--no-readme", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    res = Results()

    if not args.skip_verify:
        print("[paper] verifying the camera-ready against results/ before building")
        _, n_fail, _ = verify_run(res, cfg, verbose=False)
        if n_fail:
            raise SystemExit("[paper] refusing to build: the manuscript and the artifact disagree")

    v = values(res, cfg, args.doi)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "index.html")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(v))
    print(f"  wrote {os.path.relpath(out, ROOT)}")
    write_figures()
    if not args.no_pdf:
        write_pdf(args.doi)
    if not args.no_readme:
        update_readme(readme_results(res, cfg, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
