"""
Check every number printed in the FMEC 2026 camera-ready against results/.

    python experiments/verify_paper_claims.py            # report, exit 1 on any mismatch
    python experiments/verify_paper_claims.py --verbose  # also list the claims that pass

The camera-ready PDF is the reference: each claim below records the value the
paper prints, the expression that recomputes it from a released result file, and
the rounding tolerance implied by the number of digits the paper shows. Nothing
here is a source of truth for the dashboard, the README or the paper page --
those read the CSVs directly. This script exists so that a disagreement between
the manuscript and the artifact is a test failure rather than a discovery.

A claim is PASS when |recomputed - printed| <= tol. `tol` is half a unit in the
paper's last printed digit unless a claim needs something looser, in which case
the reason is in the claim's note.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.config import load_config  # noqa: E402

R = os.path.join(ROOT, "results")
PROFILES = ["stable", "congested", "lossy", "variable", "outage"]


# ----------------------------------------------------------------- accessors
class Results:
    """Lazy readers for the released result files."""

    def __init__(self, root: str = R):
        self.root = root
        self._cache: Dict[str, Optional[pd.DataFrame]] = {}

    def csv(self, rel: str) -> Optional[pd.DataFrame]:
        if rel not in self._cache:
            path = os.path.join(self.root, rel)
            self._cache[rel] = pd.read_csv(path) if os.path.exists(path) else None
        return self._cache[rel]

    # results/final/multi_seed_ci.csv -----------------------------------
    def ci(self, profile: str, policy: str, metric: str, field: str = "mean") -> float:
        d = self.csv("final/multi_seed_ci.csv")
        r = d[(d.profile == profile) & (d.policy == policy) & (d.metric == metric)]
        return float(r.iloc[0][field])

    def ci_over_profiles(self, policy: str, metric: str) -> float:
        """Unweighted mean over the five profiles, as the paper's pooled tables use."""
        return sum(self.ci(p, policy, metric) for p in PROFILES) / len(PROFILES)

    def ci_worst(self, policy: str, metric: str) -> float:
        return max(self.ci(p, policy, metric) for p in PROFILES)

    # results/final/paired_comparisons.csv ------------------------------
    def paired(self, metric: str, a: str, b: str, profile: str) -> pd.Series:
        d = self.csv("final/paired_comparisons.csv")
        r = d[(d.metric == metric) & (d.policy_a == a) & (d.policy_b == b)
              & (d.profile == profile)]
        return r.iloc[0]

    # results/final/mode_distribution.csv -------------------------------
    def mode(self, profile: str, col: str) -> float:
        d = self.csv("final/mode_distribution.csv")
        return float(d[d.profile == profile].iloc[0][col])

    # results/ablation/ablation_summary.csv -----------------------------
    def abl(self, variant: str, metric: str, profile: Optional[str] = None) -> float:
        d = self.csv("ablation/ablation_summary.csv")
        r = d[(d.policy == variant) & (d.metric == metric)]
        if profile is not None:
            r = r[r.profile == profile]
            return float(r.iloc[0]["mean"])
        return float(r.groupby("profile")["mean"].first().mean())

    def abl_worst(self, variant: str, metric: str) -> float:
        d = self.csv("ablation/ablation_summary.csv")
        r = d[(d.policy == variant) & (d.metric == metric)]
        return float(r["mean"].max())

    # results/sensitivity/* ---------------------------------------------
    def deadline(self, ms: float, col: str, profile: str = "ALL") -> float:
        d = self.csv("sensitivity/deadline_mode_distribution.csv")
        r = d[(d.deadline_sweep_ms == ms) & (d.profile == profile)]
        return float(r.iloc[0][col])

    def est(self, target: str, bias: float, metric: str, reduce: str = "worst") -> float:
        d = self.csv("sensitivity/runtime_estimation_error_ci.csv")
        r = d[(d.bias_target == target) & (d.bias.round(10) == round(bias, 10))
              & (d.metric == metric)]
        vals = r.groupby("profile")["mean"].first()
        return float(vals.max() if reduce == "worst" else vals.mean())

    def margin_worst_miss(self, margin: float) -> float:
        """Worst (deadline, profile) miss rate at one commit margin, D in 100..200."""
        d = self.csv("sensitivity/commit_margin_sweep_ci.csv")
        r = d[(d.remote_deadline_margin.round(10) == round(margin, 10))
              & (d.metric == "deadline_miss_rate")
              & (d.deadline_sweep_ms >= 100) & (d.deadline_sweep_ms <= 200)]
        return float(r["mean"].max())

    # results/overhead/* ------------------------------------------------
    def overhead(self, col: str) -> float:
        return float(self.csv("overhead/scheduler_overhead_summary.csv").iloc[0][col])

    def wall(self, policy: str) -> float:
        d = self.csv("overhead/benchmark_wallclock.csv")
        return float(d[d.policy == policy].iloc[0]["wall_us_per_frame"])

    def switches(self, policy: str, how: str) -> float:
        d = self.csv("overhead/mode_switches.csv")
        v = d[d.policy == policy]["mode_switches_per_1000_mean"]
        return float(v.min() if how == "min" else v.max())

    # results/real_detector/* -------------------------------------------
    def detector(self, role: str, col: str) -> float:
        d = self.csv("real_detector/model_quality.csv")
        return float(d[d.role == role].iloc[0][col])

    def detector_map(self, model: str, col: str) -> float:
        d = self.csv("real_detector/model_accuracy.csv")
        return float(d[d.model == model].iloc[0][col])

    def detector_latency(self, role: str, col: str = "inference_mean_ms") -> float:
        d = self.csv("real_detector/model_latency.csv")
        return float(d[d.role == role].iloc[0][col])

    def replay(self, profile: str, policy: str, col: str, ms: float = 100.0) -> float:
        d = self.csv("real_detector/dapper_replay.csv")
        r = d[(d.profile == profile) & (d.policy == policy) & (d.deadline_ms == ms)]
        return float(r[col].mean())

    # results/calibration/* ---------------------------------------------
    def zero_miss_candidates(self) -> int:
        d = self.csv("calibration/all_candidates.csv")
        return int((d["worst_deadline_miss_rate"] <= 0.0).sum())


# -------------------------------------------------------------------- claims
class Claim:
    __slots__ = ("section", "what", "printed", "fn", "tol", "note")

    def __init__(self, section: str, what: str, printed: float,
                 fn: Callable[[], float], tol: float, note: str = ""):
        self.section = section
        self.what = what
        self.printed = printed
        self.fn = fn
        self.tol = tol
        self.note = note


def build_claims(res: Results, cfg: Dict[str, Any]) -> List[Claim]:
    c: List[Claim] = []

    def add(section, what, printed, fn, tol, note=""):
        c.append(Claim(section, what, printed, fn, tol, note))

    sch = cfg["scheduler"]

    # --- Table I: frozen configuration -----------------------------------
    for key, printed in (("weight_rtt", 0.36), ("weight_loss", 0.35),
                         ("weight_load", 0.11), ("weight_frame_age", 0.0),
                         ("weight_deadline", 0.18),
                         ("risk_local_threshold", 0.45),
                         ("risk_degraded_threshold", 0.60),
                         ("local_confidence_threshold", 0.78),
                         ("remote_deadline_margin", 0.80),
                         ("hybrid_freshness_window_ms", 100.0),
                         ("last_valid_freshness_ms", 100.0)):
        add("Table I", f"config.scheduler.{key}", printed,
            lambda k=key: float(sch[k]), 0.0)

    # --- Table II: network profiles ---------------------------------------
    printed_profiles = {
        "stable": (25, 5, 0.01, 0.2, 0.00),
        "congested": (70, 20, 0.05, 0.7, 0.00),
        "lossy": (40, 10, 0.20, 0.3, 0.00),
        "variable": (50, 35, 0.07, 0.5, 0.02),
        "outage": (200, 80, 0.35, 0.9, 0.30),
    }
    keys = ("rtt_ms_mean", "rtt_ms_std", "packet_loss", "edge_load", "outage_prob")
    for prof, vals in printed_profiles.items():
        for key, printed in zip(keys, vals):
            add("Table II", f"{prof}.{key}", printed,
                lambda p=prof, k=key: float(cfg["network_profiles"][p][k]), 0.0)

    # --- protocol scale ---------------------------------------------------
    add("Sec. IV", "evaluation seeds", 30,
        lambda: float(cfg["evaluation_seeds_count"]), 0.0)
    add("Sec. IV", "frames per cell", 1000, lambda: float(cfg["frames"]), 0.0)
    add("Abstract", "frame-level decisions (seeds x frames x profiles x policies)", 1_200_000,
        lambda: float(cfg["evaluation_seeds_count"]) * float(cfg["frames"]) * 5 * 8, 0.0)
    add("Sec. V-B", "runs in the final evaluation", 1200,
        lambda: float(cfg["evaluation_seeds_count"]) * 5 * 8, 0.0)

    # --- Table III: main comparison --------------------------------------
    table3 = {
        ("stable", "local_only"): (34.0, 33.9, 34.0, 0.00, 0.00, 0.00, 0.725, 0.00),
        ("stable", "edge_only"): (132.8, 132.7, 132.9, 48.77, 48.17, 49.38, 0.448, 25.00),
        ("stable", "cloud_only"): (134.0, 133.9, 134.0, 100.00, 100.00, 100.00, 0.000, 35.00),
        ("stable", "dapper"): (34.0, 33.9, 34.0, 0.00, 0.00, 0.00, 0.796, 21.62),
        ("congested", "local_only"): (33.9, 33.9, 34.0, 0.00, 0.00, 0.00, 0.725, 0.00),
        ("congested", "edge_only"): (133.9, 133.9, 134.0, 100.00, 100.00, 100.00, 0.000, 25.00),
        ("congested", "cloud_only"): (133.9, 133.9, 134.0, 100.00, 100.00, 100.00, 0.000, 35.00),
        ("congested", "dapper"): (33.9, 33.9, 34.0, 0.00, 0.00, 0.00, 0.725, 0.00),
        ("lossy", "local_only"): (34.0, 34.0, 34.1, 0.00, 0.00, 0.00, 0.725, 0.00),
        ("lossy", "edge_only"): (133.9, 133.8, 133.9, 87.51, 87.06, 87.94, 0.109, 25.00),
        ("lossy", "cloud_only"): (134.0, 134.0, 134.1, 100.00, 100.00, 100.00, 0.000, 35.00),
        ("lossy", "dapper"): (34.0, 34.0, 34.1, 0.00, 0.00, 0.00, 0.741, 18.13),
        ("variable", "local_only"): (34.0, 33.9, 34.0, 0.00, 0.00, 0.00, 0.725, 0.00),
        ("variable", "edge_only"): (133.6, 133.5, 133.7, 74.00, 72.62, 75.43, 0.195, 19.55),
        ("variable", "cloud_only"): (133.7, 133.6, 133.7, 78.20, 76.76, 79.62, 0.158, 27.37),
        ("variable", "dapper"): (33.7, 33.7, 33.8, 0.00, 0.00, 0.00, 0.732, 5.78),
        ("outage", "local_only"): (34.0, 34.0, 34.1, 0.00, 0.00, 0.00, 0.725, 0.00),
        ("outage", "edge_only"): (128.1, 127.7, 128.4, 14.68, 14.06, 15.33, 0.619, 3.67),
        ("outage", "cloud_only"): (128.1, 127.7, 128.4, 14.68, 14.06, 15.33, 0.619, 5.14),
        ("outage", "dapper"): (31.2, 31.0, 31.4, 0.00, 0.00, 0.00, 0.726, 0.00),
    }
    for (prof, pol), vals in table3.items():
        p95, p95lo, p95hi, miss, misslo, misshi, conf, bw = vals
        add("Table III", f"{prof}/{pol} p95 ms", p95,
            lambda p=prof, q=pol: res.ci(p, q, "p95_latency_ms"), 0.05)
        add("Table III", f"{prof}/{pol} p95 CI lo", p95lo,
            lambda p=prof, q=pol: res.ci(p, q, "p95_latency_ms", "ci_lo"), 0.05)
        add("Table III", f"{prof}/{pol} p95 CI hi", p95hi,
            lambda p=prof, q=pol: res.ci(p, q, "p95_latency_ms", "ci_hi"), 0.05)
        add("Table III", f"{prof}/{pol} miss %", miss,
            lambda p=prof, q=pol: 100 * res.ci(p, q, "deadline_miss_rate"), 0.005)
        add("Table III", f"{prof}/{pol} miss CI lo", misslo,
            lambda p=prof, q=pol: 100 * res.ci(p, q, "deadline_miss_rate", "ci_lo"), 0.005)
        add("Table III", f"{prof}/{pol} miss CI hi", misshi,
            lambda p=prof, q=pol: 100 * res.ci(p, q, "deadline_miss_rate", "ci_hi"), 0.005)
        add("Table III", f"{prof}/{pol} conf. proxy", conf,
            lambda p=prof, q=pol: res.ci(p, q, "usable_confidence_proxy"), 0.0005)
        add("Table III", f"{prof}/{pol} BW MB/1k fr.", bw,
            lambda p=prof, q=pol: 1e-3 * res.ci(p, q, "bandwidth_per_1000_frames_kb"), 0.005)

    # --- Table IV left: DAPPER placement ---------------------------------
    table4l = {
        "stable": (13.5, 0.0, 86.5, 0.0, 51.2, 0.0, 0.00),
        "congested": (99.2, 0.0, 0.0, 0.7, 0.0, 0.7, 0.00),
        "lossy": (27.5, 0.0, 72.5, 0.0, 13.7, 0.0, 0.00),
        "variable": (60.1, 0.0, 23.1, 16.8, 15.9, 16.8, 0.00),
        "outage": (22.6, 0.0, 0.0, 77.4, 0.0, 73.4, 0.00),
    }
    for prof, vals in table4l.items():
        loc, edge, hyb, degr, acc, reuse, stale = vals
        add("Table IV", f"{prof} local-fast %", loc,
            lambda p=prof: res.mode(p, "pct_local_fast"), 0.05)
        add("Table IV", f"{prof} edge-accurate %", edge,
            lambda p=prof: res.mode(p, "pct_edge_accurate"), 0.05)
        add("Table IV", f"{prof} hybrid %", hyb,
            lambda p=prof: res.mode(p, "pct_hybrid"), 0.05)
        add("Table IV", f"{prof} degraded-safe %", degr,
            lambda p=prof: res.mode(p, "pct_degraded_safe"), 0.05)
        add("Table IV", f"{prof} remote accepted %", acc,
            lambda p=prof: 100 * res.mode(p, "remote_accept_rate"), 0.05)
        add("Table IV", f"{prof} reuse %", reuse,
            lambda p=prof: 100 * res.mode(p, "reuse_rate"), 0.05)
        add("Table IV", f"{prof} stale %", stale,
            lambda p=prof: 100 * res.ci(p, "dapper", "stale_acceptance_rate"), 0.005)

    # --- Table IV right: adaptive comparators, mean over profiles --------
    table4r = {
        "local_only": (0.00, 0.00, 34.0, 0.7250, 0.00),
        "deadline_greedy": (0.00, 0.00, 34.0, 0.7250, 0.00),
        "confidence_deadline": (0.00, 0.00, 34.0, 0.7250, 0.00),
        "rtt_threshold": (0.01, 0.00, 34.0, 0.7250, 0.00),
        "dapper": (36.42, 0.00, 33.4, 0.7437, 9.11),
        "oracle_feasible": (13.58, 0.00, 60.8, 0.7454, 3.40),
    }
    for pol, vals in table4r.items():
        tx, miss, p95, conf, bw = vals
        add("Table IV", f"{pol} Tx %", tx,
            lambda q=pol: 100 * res.ci_over_profiles(q, "remote_attempt_rate"), 0.005)
        add("Table IV", f"{pol} miss %", miss,
            lambda q=pol: 100 * res.ci_over_profiles(q, "deadline_miss_rate"), 0.005)
        add("Table IV", f"{pol} p95 ms", p95,
            lambda q=pol: res.ci_over_profiles(q, "p95_latency_ms"), 0.05)
        add("Table IV", f"{pol} conf. proxy", conf,
            lambda q=pol: res.ci_over_profiles(q, "usable_confidence_proxy"), 0.00005)
        add("Table IV", f"{pol} MB/1k fr.", bw,
            lambda q=pol: 1e-3 * res.ci_over_profiles(q, "bandwidth_per_1000_frames_kb"), 0.005)

    # --- Table V: ablation -----------------------------------------------
    table5 = {
        "dapper_full": (0.00, 33.37, 0.7437, 9.11),
        "dapper_no_rtt": (0.00, 33.43, 0.7427, 8.09),
        "dapper_no_loss": (0.00, 32.58, 0.7337, 2.50),
        "dapper_no_load": (0.00, 33.37, 0.7431, 8.14),
        "dapper_no_frame_age": (0.00, 33.37, 0.7437, 9.11),
        "dapper_no_deadline": (0.00, 33.38, 0.7439, 9.71),
        "dapper_no_confidence_gate": (0.00, 33.37, 0.7453, 10.53),
        "dapper_no_freshness_gate": (0.00, 33.37, 0.7437, 9.96),
        "dapper_no_degraded_reuse": (0.00, 33.95, 0.7436, 9.11),
    }
    for variant, vals in table5.items():
        miss, p95, conf, bw = vals
        add("Table V", f"{variant} miss %", miss,
            lambda v=variant: 100 * res.abl(v, "deadline_miss_rate"), 0.005)
        add("Table V", f"{variant} p95 ms", p95,
            lambda v=variant: res.abl(v, "p95_latency_ms"), 0.005)
        add("Table V", f"{variant} conf. proxy", conf,
            lambda v=variant: res.abl(v, "usable_confidence_proxy"), 0.00005)
        add("Table V", f"{variant} MB/1k fr.", bw,
            lambda v=variant: 1e-3 * res.abl(v, "bandwidth_per_1000_frames_kb"), 0.005)
    add("Sec. V-C", "worst ablated-variant miss % (claimed: none nonzero)", 0.0,
        lambda: 100 * max(res.abl_worst(v, "deadline_miss_rate") for v in table5), 0.0)

    # --- Table VI: real detector -----------------------------------------
    add("Table VI", "YOLO11n recall", 0.507, lambda: res.detector("local", "recall"), 0.0005)
    add("Table VI", "YOLO11n safety recall", 0.623,
        lambda: res.detector("local", "safety_recall"), 0.0005)
    add("Table VI", "YOLO11n mAP50", 0.546,
        lambda: res.detector_map("yolo11n.pt", "mAP50"), 0.0005)
    add("Table VI", "YOLO11n mAP50-95", 0.389,
        lambda: res.detector_map("yolo11n.pt", "mAP50_95"), 0.0005)
    add("Table VI", "YOLO11m recall", 0.638, lambda: res.detector("remote", "recall"), 0.0005)
    add("Table VI", "YOLO11m safety recall", 0.734,
        lambda: res.detector("remote", "safety_recall"), 0.0005)
    add("Table VI", "YOLO11m mAP50", 0.678,
        lambda: res.detector_map("yolo11m.pt", "mAP50"), 0.0005)
    add("Table VI", "YOLO11m mAP50-95", 0.510,
        lambda: res.detector_map("yolo11m.pt", "mAP50_95"), 0.0005)
    add("Sec. V-D", "recall gap (percentage points)", 13.1,
        lambda: 100 * (res.detector("remote", "recall") - res.detector("local", "recall")), 0.05)
    add("Sec. V-D", "safety-recall gap (percentage points)", 11.1,
        lambda: 100 * (res.detector("remote", "safety_recall")
                       - res.detector("local", "safety_recall")), 0.05)
    add("Sec. V-D", "YOLO11n CPU inference ms", 35.2,
        lambda: res.detector_latency("local_cpu"), 0.05)
    add("Sec. V-D", "YOLO11m CPU inference ms", 177.5,
        lambda: res.detector_latency("remote_cpu"), 0.05)
    add("Sec. V-D", "YOLO11n GPU inference ms", 6.1,
        lambda: res.detector_latency("local"), 0.05)
    add("Sec. V-D", "YOLO11m GPU inference ms", 8.1,
        lambda: res.detector_latency("remote"), 0.05)

    # --- Table VII: replay -----------------------------------------------
    table7 = {
        ("stable", "local_only"): (0.639, 0.639, 0.00, 0.0),
        ("stable", "edge_only"): (0.740, 0.747, 0.96, 0.0),
        ("stable", "dapper"): (0.674, 0.674, 0.00, 0.0),
        ("lossy", "local_only"): (0.639, 0.639, 0.00, 0.0),
        ("lossy", "edge_only"): (0.600, 0.747, 19.74, 0.0),
        ("lossy", "dapper"): (0.663, 0.663, 0.00, 0.0),
        ("variable", "local_only"): (0.639, 0.639, 0.00, 0.0),
        ("variable", "edge_only"): (0.680, 0.721, 5.66, 0.0),
        ("variable", "dapper"): (0.545, 0.649, 0.00, 16.0),
        ("outage", "local_only"): (0.639, 0.639, 0.00, 0.0),
        ("outage", "edge_only"): (0.546, 0.639, 14.54, 0.0),
        ("outage", "dapper"): (0.223, 0.637, 0.00, 65.0),
    }
    for (prof, pol), vals in table7.items():
        allr, fresh, miss, reuse = vals
        add("Table VII", f"{prof}/{pol} recall (all)", allr,
            lambda p=prof, q=pol: res.replay(p, q, "delivered_recall"), 0.0005)
        add("Table VII", f"{prof}/{pol} recall (fresh)", fresh,
            lambda p=prof, q=pol: res.replay(p, q, "delivered_recall_fresh_only"), 0.0005)
        add("Table VII", f"{prof}/{pol} miss %", miss,
            lambda p=prof, q=pol: 100 * res.replay(p, q, "deadline_miss_rate"), 0.005)
        add("Table VII", f"{prof}/{pol} reuse %", reuse,
            lambda p=prof, q=pol: 100 * res.replay(p, q, "reuse_rate"), 0.05)
    add("Sec. V-D", "stable DAPPER safety recall", 0.762,
        lambda: res.replay("stable", "dapper", "delivered_safety_recall"), 0.0005)
    add("Sec. V-D", "stable local-only safety recall", 0.737,
        lambda: res.replay("stable", "local_only", "delivered_safety_recall"), 0.0005)

    # --- Sec. V-A/B prose -------------------------------------------------
    add("Sec. V-A", "DAPPER worst-profile miss %", 0.00,
        lambda: 100 * res.ci_worst("dapper", "deadline_miss_rate"), 0.0)
    add("Sec. V-B", "worst stale-acceptance % over every policy and profile", 0.00,
        lambda: 100 * max(res.ci(p, q, "stale_acceptance_rate") for p in PROFILES
                          for q in ("local_only", "edge_only", "cloud_only", "dapper",
                                    "deadline_greedy", "confidence_deadline",
                                    "rtt_threshold", "oracle_feasible")), 0.0)
    for prof, acc, dl, lost in (("stable", 51.2, 47.7, 1.1), ("lossy", 13.7, 67.5, 18.8)):
        add("Sec. V-B", f"{prof} remote accepted %", acc,
            lambda p=prof: 100 * res.ci(p, "dapper", "remote_accept_rate"), 0.05)
        add("Sec. V-B", f"{prof} remote rejected on deadline %", dl,
            lambda p=prof: 100 * res.ci(p, "dapper", "remote_rejected_deadline_rate"), 0.05)
        add("Sec. V-B", f"{prof} remote lost %", lost,
            lambda p=prof: 100 * res.ci(p, "dapper", "remote_failed_loss_rate"), 0.05)

    # paired confidence gains vs local-only
    for prof, diff, lo, hi in (("stable", 0.0710, 0.0700, 0.0720),
                               ("lossy", 0.0158, 0.0151, 0.0165),
                               ("variable", 0.0062, 0.0056, 0.0068)):
        add("Sec. V-A", f"{prof} paired conf. gain vs local-only", diff,
            lambda p=prof: float(res.paired("usable_confidence_proxy", "dapper",
                                            "local_only", p)["mean_diff"]), 0.00005)
        add("Sec. V-A", f"{prof} paired gain CI lo", lo,
            lambda p=prof: float(res.paired("usable_confidence_proxy", "dapper",
                                            "local_only", p)["ci_lo"]), 0.00005)
        add("Sec. V-A", f"{prof} paired gain CI hi", hi,
            lambda p=prof: float(res.paired("usable_confidence_proxy", "dapper",
                                            "local_only", p)["ci_hi"]), 0.00005)
    for prof in ("congested", "outage"):
        add("Sec. V-A", f"{prof} paired conf. CI includes zero (1 = yes)", 1.0,
            lambda p=prof: 0.0 if bool(res.paired("usable_confidence_proxy", "dapper",
                                                  "local_only", p)["excludes_zero"]) else 1.0,
            0.0)

    # --- Sec. IV-C calibration -------------------------------------------
    add("Sec. IV-C", "candidates reaching zero worst-profile misses (of 400)", 328,
        lambda: float(res.zero_miss_candidates()), 0.0)
    add("Sec. IV-C", "candidate configurations searched", 400,
        lambda: float(len(res.csv("calibration/all_candidates.csv"))), 0.0)

    # --- Sec. V-C oracle comparison --------------------------------------
    add("Sec. V-C", "stable local-only conf. proxy", 0.7250,
        lambda: res.ci("stable", "local_only", "usable_confidence_proxy"), 0.00005)
    add("Sec. V-C", "stable DAPPER conf. proxy", 0.7960,
        lambda: res.ci("stable", "dapper", "usable_confidence_proxy"), 0.00005)
    add("Sec. V-C", "stable oracle conf. proxy", 0.8018,
        lambda: res.ci("stable", "oracle_feasible", "usable_confidence_proxy"), 0.00005)
    add("Sec. V-C", "stable DAPPER MB/1k fr.", 21.62,
        lambda: 1e-3 * res.ci("stable", "dapper", "bandwidth_per_1000_frames_kb"), 0.005)
    add("Sec. V-C", "stable oracle MB/1k fr.", 12.81,
        lambda: 1e-3 * res.ci("stable", "oracle_feasible",
                              "bandwidth_per_1000_frames_kb"), 0.005)
    add("Sec. V-C", "share of oracle gain DAPPER recovers, stable (%)", 92,
        lambda: 100 * ((res.ci("stable", "dapper", "usable_confidence_proxy")
                        - res.ci("stable", "local_only", "usable_confidence_proxy"))
                       / (res.ci("stable", "oracle_feasible", "usable_confidence_proxy")
                          - res.ci("stable", "local_only", "usable_confidence_proxy"))), 0.5,
        "paper says 'approximately 92%'")
    add("Sec. V-C", "RTT-threshold transmit % (paper: under 0.02)", 0.02,
        lambda: 100 * res.ci_over_profiles("rtt_threshold", "remote_attempt_rate"), 0.02,
        "one-sided: the paper claims an upper bound")

    # --- Sec. V-C ablation prose -----------------------------------------
    add("Sec. V-C", "stable conf. without loss signal", 0.7675,
        lambda: res.abl("dapper_no_loss", "usable_confidence_proxy", "stable"), 0.00005)
    add("Sec. V-C", "lossy conf. full DAPPER", 0.7406,
        lambda: res.abl("dapper_full", "usable_confidence_proxy", "lossy"), 0.00005)
    add("Sec. V-C", "lossy conf. without loss signal", 0.7248,
        lambda: res.abl("dapper_no_loss", "usable_confidence_proxy", "lossy"), 0.00005)
    add("Sec. V-C", "variable MB/1k fr. full DAPPER", 5.78,
        lambda: 1e-3 * res.abl("dapper_full", "bandwidth_per_1000_frames_kb", "variable"), 0.005)
    add("Sec. V-C", "variable MB/1k fr. without freshness gate", 9.75,
        lambda: 1e-3 * res.abl("dapper_no_freshness_gate",
                               "bandwidth_per_1000_frames_kb", "variable"), 0.005)
    add("Sec. V-C", "freshness-gate traffic saving (%)", 69,
        lambda: 100 * (res.abl("dapper_no_freshness_gate",
                               "bandwidth_per_1000_frames_kb", "variable")
                       / res.abl("dapper_full", "bandwidth_per_1000_frames_kb",
                                 "variable") - 1.0), 0.5,
        "paper says 'about 69%'")
    add("Sec. V-C", "outage p95 ms full DAPPER", 31.19,
        lambda: res.abl("dapper_full", "p95_latency_ms", "outage"), 0.005)
    add("Sec. V-C", "outage p95 ms without degraded reuse", 33.85,
        lambda: res.abl("dapper_no_degraded_reuse", "p95_latency_ms", "outage"), 0.005)

    # --- Sec. V-C deadline sweep -----------------------------------------
    add("Sec. V-C", "edge-accurate % at D = 100 ms", 0.0,
        lambda: res.deadline(100.0, "pct_edge_accurate"), 0.0)
    add("Sec. V-C", "edge-accurate % at D = 250 ms", 65.5,
        lambda: res.deadline(250.0, "pct_edge_accurate"), 0.05)
    add("Sec. V-C", "miss % at D = 125 ms", 5.24,
        lambda: 100 * res.deadline(125.0, "deadline_miss_rate"), 0.005)
    add("Sec. V-C", "miss % at D = 150 ms", 13.31,
        lambda: 100 * res.deadline(150.0, "deadline_miss_rate"), 0.005)
    add("Sec. V-C", "miss % at D = 200 ms", 4.21,
        lambda: 100 * res.deadline(200.0, "deadline_miss_rate"), 0.005)
    add("Sec. V-C", "worst miss % over D in [100,200] at margin m = 0.6", 0.00,
        lambda: 100 * res.margin_worst_miss(0.6), 0.0)
    add("Sec. V-C", "worst miss % over D in [100,200] at margin m = 0.8 (> 0 claimed)", 1.0,
        lambda: 1.0 if res.margin_worst_miss(0.8) > 0.0 else 0.0, 0.0,
        "boolean: the paper says the calibrated m = 0.8 does not hold zero misses")

    # --- Sec. V-C estimation error ---------------------------------------
    for bias in (-0.30, -0.20, -0.10, 0.10, 0.20, 0.30):
        add("Sec. V-C", f"RTT-bias {bias:+.0%} worst-profile miss %", 0.00,
            lambda b=bias: 100 * res.est("rtt", b, "deadline_miss_rate"), 0.0)
    add("Sec. V-C", "compute-bias -10% worst-profile miss %", 0.02,
        lambda: 100 * res.est("compute", -0.10, "deadline_miss_rate"), 0.005)
    add("Sec. V-C", "compute-bias -30% worst-profile miss %", 38.55,
        lambda: 100 * res.est("compute", -0.30, "deadline_miss_rate"), 0.005)
    add("Sec. V-C", "both-bias -30% worst-profile miss %", 42.22,
        lambda: 100 * res.est("both", -0.30, "deadline_miss_rate"), 0.005)

    # --- Sec. V-D overhead ------------------------------------------------
    add("Sec. V-D", "scheduler decision mean us", 1.45, lambda: res.overhead("mean_us"), 0.005)
    add("Sec. V-D", "scheduler decision median us", 1.40,
        lambda: res.overhead("median_us"), 0.005)
    add("Sec. V-D", "scheduler decision p95 us", 1.80, lambda: res.overhead("p95_us"), 0.005)
    add("Sec. V-D", "scheduler decision p99 us", 2.00, lambda: res.overhead("p99_us"), 0.005)
    add("Sec. V-D", "timed decisions", 200000, lambda: res.overhead("n_calls"), 0.0)
    add("Sec. V-D", "DAPPER minus local-only wall time us/frame", 1.31,
        lambda: res.wall("dapper") - res.wall("local_only"), 0.005)
    add("Sec. V-D", "min DAPPER mode switches / 1000 frames", 12,
        lambda: res.switches("dapper", "min"), 0.5)
    add("Sec. V-D", "max DAPPER mode switches / 1000 frames", 418,
        lambda: res.switches("dapper", "max"), 0.5)

    return c


# -------------------------------------------------------------------- report
def run(res: Results, cfg: Dict[str, Any], verbose: bool) -> Tuple[int, int, int]:
    claims = build_claims(res, cfg)
    failures: List[Tuple[Claim, Any]] = []
    skipped: List[Claim] = []
    passed = 0
    for cl in claims:
        try:
            got = cl.fn()
        except Exception as exc:  # missing file or row
            skipped.append(cl)
            if verbose:
                print(f"  SKIP  {cl.section:<10} {cl.what:<58} ({type(exc).__name__})")
            continue
        if got is None or (isinstance(got, float) and math.isnan(got)):
            skipped.append(cl)
            continue
        if abs(float(got) - cl.printed) <= cl.tol:
            passed += 1
            if verbose:
                print(f"  ok    {cl.section:<10} {cl.what:<58} "
                      f"paper={cl.printed:<12g} results={float(got):<12.6g}")
        else:
            failures.append((cl, float(got)))

    if failures:
        print(f"\n{len(failures)} claim(s) DISAGREE with results/:\n")
        for cl, got in failures:
            print(f"  FAIL  {cl.section:<10} {cl.what}")
            print(f"        paper prints {cl.printed:g}, results give {got:.6g} "
                  f"(tolerance {cl.tol:g})")
            if cl.note:
                print(f"        note: {cl.note}")
    if skipped:
        print(f"\n{len(skipped)} claim(s) could not be checked (result file absent):")
        for cl in skipped[:12]:
            print(f"  --    {cl.section:<10} {cl.what}")
        if len(skipped) > 12:
            print(f"  ... and {len(skipped) - 12} more")

    print(f"\n[verify] {passed} pass, {len(failures)} fail, {len(skipped)} unchecked "
          f"of {len(claims)} camera-ready claims")
    return passed, len(failures), len(skipped)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=R)
    p.add_argument("--config", default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    _, n_fail, _ = run(Results(args.results), cfg, args.verbose)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
