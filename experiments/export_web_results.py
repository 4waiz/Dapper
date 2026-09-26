"""
Export the published results to site/data/results.json.

    python experiments/export_web_results.py

One file feeds three consumers -- the dashboard's research section, the README
results block and the paper page -- so a number can only be wrong in one place.
Every value is read from a CSV under results/. Nothing is typed.

Two rules the paper's honesty depends on are enforced here rather than left to
whoever writes the prose:

**Rankings are computed, never asserted.** `rankings` orders the policies on
each metric using dapper.metrics.LOWER_IS_BETTER for the direction, so a word
like "best" or "lower" in any consumer is a lookup, not a claim.

**The losses are exported next to the wins.** `honest` carries the three places
the paper shows DAPPER behind or paying a price: all-frame delivered recall
under variable and outage replay, the bandwidth charged for replies that never
arrive usable, and the deadline misses the frozen 100-ms calibration produces at
larger deadlines. A consumer that renders `tables` and `series` but skips
`honest` is visibly incomplete.

NaN and Infinity are illegal in JSON, so every non-finite value becomes null and
the dump uses allow_nan=False.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.config import load_config  # noqa: E402
from dapper.metrics import LOWER_IS_BETTER  # noqa: E402
from dapper.policies import ALL_POLICY_NAMES  # noqa: E402
from dapper.stats import BOOTSTRAP_SEED, DEFAULT_N_BOOT  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
DEFAULT_OUT = os.path.join(ROOT, "site", "data", "results.json")

PROFILE_ORDER = ["stable", "congested", "lossy", "variable", "outage"]
MAIN_POLICY_ORDER = ["local_only", "edge_only", "cloud_only", "dapper"]
ADAPTIVE_ORDER = ["local_only", "deadline_greedy", "confidence_deadline",
                  "rtt_threshold", "dapper", "oracle_feasible"]
POLICY_LABEL = {
    "local_only": "local-only", "edge_only": "edge-only", "cloud_only": "cloud-only",
    "dapper": "DAPPER", "deadline_greedy": "deadline-greedy",
    "confidence_deadline": "conf.+deadline", "rtt_threshold": "RTT-threshold",
    "oracle_feasible": "oracle",
}
ABLATION_LABEL = {
    "dapper_full": "full DAPPER", "dapper_no_rtt": "− RTT",
    "dapper_no_loss": "− packet loss", "dapper_no_load": "− edge load",
    "dapper_no_frame_age": "− frame age",
    "dapper_no_deadline": "− deadline pressure",
    "dapper_no_confidence_gate": "− confidence gate",
    "dapper_no_freshness_gate": "− freshness gate",
    "dapper_no_degraded_reuse": "− degraded reuse",
}
ABLATION_ORDER = list(ABLATION_LABEL)

#: Metrics the ranking block covers, with the unit and the digits the paper uses.
RANKED_METRICS = [
    ("deadline_miss_rate", "%", 100.0, 2),
    ("p95_latency_ms", " ms", 1.0, 1),
    ("usable_confidence_proxy", "", 1.0, 3),
    ("bandwidth_per_1000_frames_kb", " MB/1k fr.", 1e-3, 2),
]


def _f(x: Any) -> Optional[float]:
    """JSON-safe float: NaN and +/-Infinity become null."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def read(rel: str) -> Optional[pd.DataFrame]:
    path = os.path.join(RESULTS, rel)
    return pd.read_csv(path) if os.path.exists(path) else None


class Long:
    """Accessor for the long-format (profile, policy, metric) CI frame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._idx = df.set_index(["profile", "policy", "metric"]).sort_index()

    def has(self, profile: str, policy: str, metric: str) -> bool:
        return (profile, policy, metric) in self._idx.index

    def row(self, profile: str, policy: str, metric: str) -> Dict[str, Optional[float]]:
        r = self._idx.loc[(profile, policy, metric)]
        return {"mean": _f(r["mean"]), "lo": _f(r["ci_lo"]), "hi": _f(r["ci_hi"]),
                "std": _f(r["std"]), "n_seeds": int(r["n_seeds"])}

    def mean(self, profile: str, policy: str, metric: str) -> float:
        return float(self._idx.loc[(profile, policy, metric)]["mean"])

    def profile_mean(self, policy: str, metric: str,
                     profiles: Sequence[str] = tuple(PROFILE_ORDER)) -> float:
        return sum(self.mean(p, policy, metric) for p in profiles) / len(profiles)

    def profiles(self) -> List[str]:
        present = set(self.df["profile"])
        return [p for p in PROFILE_ORDER if p in present]

    def policies(self) -> List[str]:
        present = set(self.df["policy"])
        return [p for p in ALL_POLICY_NAMES if p in present]


# ------------------------------------------------------------------- tables
def table_main(long: Long) -> Dict[str, Any]:
    rows = []
    for profile in long.profiles():
        for policy in MAIN_POLICY_ORDER:
            if not long.has(profile, policy, "p95_latency_ms"):
                continue
            rows.append({
                "profile": profile,
                "policy": policy,
                "policy_label": POLICY_LABEL[policy],
                "p95_latency_ms": long.row(profile, policy, "p95_latency_ms"),
                "deadline_miss_pct": _scaled(long.row(profile, policy, "deadline_miss_rate"), 100.0),
                "usable_confidence_proxy": long.row(profile, policy, "usable_confidence_proxy"),
                "bandwidth_mb_per_1k": _scaled(
                    long.row(profile, policy, "bandwidth_per_1000_frames_kb"), 1e-3),
            })
    return {
        "id": "main",
        "paper_table": "III",
        "title": "Repeated evaluation over 30 independent seeds",
        "caption": ("1,000 frames per seed, profile and policy at D = 100 ms. Intervals "
                    "are 95% percentile bootstrap over seeds. The confidence proxy is "
                    "the synthetic value the control loop holds at the deadline, not "
                    "detector accuracy."),
        "source": "results/final/multi_seed_ci.csv",
        "rows": rows,
    }


def _scaled(row: Dict[str, Optional[float]], scale: float) -> Dict[str, Optional[float]]:
    return {k: (None if v is None else (v * scale if k != "n_seeds" else v))
            for k, v in row.items()}


def table_placement(modes: pd.DataFrame, long: Long) -> Dict[str, Any]:
    rows = []
    for profile in PROFILE_ORDER:
        sub = modes[modes["profile"] == profile]
        if not len(sub):
            continue
        r = sub.iloc[0]
        rows.append({
            "profile": profile,
            "pct_local_fast": _f(r["pct_local_fast"]),
            "pct_edge_accurate": _f(r["pct_edge_accurate"]),
            "pct_hybrid": _f(r["pct_hybrid"]),
            "pct_degraded_safe": _f(r["pct_degraded_safe"]),
            "transmit_pct": _f(100.0 * r["remote_attempt_rate"]),
            "accepted_pct": _f(100.0 * r["remote_accept_rate"]),
            "reuse_pct": _f(100.0 * r["reuse_rate"]),
            "stale_pct": _f(100.0 * long.mean(profile, "dapper", "stale_acceptance_rate")),
            "mode_switches_per_1000": _f(r["mode_switches_per_1000"]),
        })
    return {
        "id": "placement",
        "paper_table": "IV (left)",
        "title": "DAPPER placement and remote accounting",
        "caption": ("At D = 100 ms over 30 seeds. Accepted is a percentage of "
                    "transmitted requests; reuse and stale are percentages of all "
                    "frames. Stale acceptance is zero by construction and measured "
                    "so the claim is checked rather than asserted."),
        "source": "results/final/mode_distribution.csv",
        "rows": rows,
    }


def table_adaptive(long: Long) -> Dict[str, Any]:
    profiles = long.profiles()
    rows = []
    for policy in ADAPTIVE_ORDER:
        if not long.has(profiles[0], policy, "p95_latency_ms"):
            continue
        rows.append({
            "policy": policy,
            "policy_label": POLICY_LABEL[policy],
            "is_oracle": policy == "oracle_feasible",
            "transmit_pct": _f(100.0 * long.profile_mean(policy, "remote_attempt_rate", profiles)),
            "deadline_miss_pct": _f(100.0 * long.profile_mean(policy, "deadline_miss_rate", profiles)),
            "p95_latency_ms": _f(long.profile_mean(policy, "p95_latency_ms", profiles)),
            "usable_confidence_proxy": _f(long.profile_mean(policy, "usable_confidence_proxy", profiles)),
            "bandwidth_mb_per_1k": _f(1e-3 * long.profile_mean(
                policy, "bandwidth_per_1000_frames_kb", profiles)),
        })
    return {
        "id": "adaptive",
        "paper_table": "IV (right)",
        "title": "DAPPER against the calibrated adaptive comparators and the offline oracle",
        "caption": ("Averaged over the five profiles at D = 100 ms. Each comparator's "
                    "hyperparameters were fitted on the calibration seeds with the same "
                    "objective used for DAPPER. The oracle reads the realised frame "
                    "outcome and is not deployable."),
        "source": "results/final/multi_seed_ci.csv",
        "rows": rows,
    }


def table_ablation(abl: pd.DataFrame) -> Dict[str, Any]:
    idx = abl.set_index(["policy", "metric", "profile"]).sort_index()

    def value(variant: str, metric: str) -> Optional[float]:
        try:
            sub = idx.loc[(variant, metric)]
        except KeyError:
            return None
        return _f(sub["mean"].mean())

    full_conf = value("dapper_full", "usable_confidence_proxy")
    full_bw = value("dapper_full", "bandwidth_per_1000_frames_kb")
    rows = []
    for variant in ABLATION_ORDER:
        conf = value(variant, "usable_confidence_proxy")
        bw = value(variant, "bandwidth_per_1000_frames_kb")
        rows.append({
            "variant": variant,
            "label": ABLATION_LABEL[variant],
            "family": ("risk signal" if variant in (
                "dapper_full", "dapper_no_rtt", "dapper_no_loss", "dapper_no_load",
                "dapper_no_frame_age", "dapper_no_deadline") else "mechanism"),
            "deadline_miss_pct": _mul(value(variant, "deadline_miss_rate"), 100.0),
            "p95_latency_ms": value(variant, "p95_latency_ms"),
            "usable_confidence_proxy": conf,
            "bandwidth_mb_per_1k": _mul(bw, 1e-3),
            "reuse_pct": _mul(value(variant, "reuse_rate"), 100.0),
            "delta_confidence": None if (conf is None or full_conf is None) else conf - full_conf,
            "delta_bandwidth_mb_per_1k": None if (bw is None or full_bw is None)
            else 1e-3 * (bw - full_bw),
        })
    return {
        "id": "ablation",
        "paper_table": "V",
        "title": "One-at-a-time ablation",
        "caption": ("Averaged over the five profiles and 30 seeds. Removing a risk "
                    "signal sets its weight to zero and renormalises the remainder. "
                    "No ablated variant produced a nonzero deadline-miss rate."),
        "source": "results/ablation/ablation_summary.csv",
        "rows": rows,
    }


def _mul(x: Optional[float], k: float) -> Optional[float]:
    return None if x is None else x * k


def table_detector(quality: pd.DataFrame, accuracy: Optional[pd.DataFrame],
                   latency: Optional[pd.DataFrame]) -> Dict[str, Any]:
    rows = []
    for role, label, note in (("local", "YOLO11n", "local model"),
                              ("remote", "YOLO11m", "remote model")):
        q = quality[quality["role"] == role]
        if not len(q):
            continue
        q = q.iloc[0]
        row = {"role": role, "model": label, "note": note,
               "images": int(q["n_images"]),
               "recall": _f(q["recall"]), "precision": _f(q["precision"]),
               "safety_recall": _f(q["safety_recall"])}
        if accuracy is not None:
            a = accuracy[accuracy["model"] == q["model"]]
            if len(a):
                row["map50"] = _f(a.iloc[0]["mAP50"])
                row["map50_95"] = _f(a.iloc[0]["mAP50_95"])
        if latency is not None:
            for key, dev in (("gpu_ms", role), ("cpu_ms", f"{role}_cpu")):
                l = latency[latency["role"] == dev]
                row[key] = _f(l.iloc[0]["inference_mean_ms"]) if len(l) else None
        rows.append(row)
    return {
        "id": "detector",
        "paper_table": "VI",
        "title": "Measured detector quality on all 5,000 COCO val2017 images",
        "caption": ("Confidence threshold 0.25, IoU matching at 0.50. Safety-relevant "
                    "classes are person, bicycle, car, motorcycle, bus and truck. "
                    "Measured on one workstation; not embedded or robot timing."),
        "source": "results/real_detector/model_quality.csv, model_accuracy.csv, model_latency.csv",
        "rows": rows,
    }


#: Profiles the camera-ready prints in Table VII. The replay CSV also holds
#: congested, which the manuscript omits; the dashboard shows the full table and
#: the paper page renders only these rows, so the paper edition stays faithful.
REPLAY_PAPER_PROFILES = ["stable", "lossy", "variable", "outage"]


def table_replay(replay: pd.DataFrame, deadline_ms: float = 100.0) -> Dict[str, Any]:
    r = replay[replay["deadline_ms"] == deadline_ms]
    rows = []
    for profile in PROFILE_ORDER:
        for policy in ("local_only", "edge_only", "dapper"):
            sub = r[(r["profile"] == profile) & (r["policy"] == policy)]
            if not len(sub):
                continue
            rows.append({
                "profile": profile,
                "policy": policy,
                "policy_label": POLICY_LABEL[policy],
                "recall_all": _f(sub["delivered_recall"].mean()),
                "recall_fresh": _f(sub["delivered_recall_fresh_only"].mean()),
                "safety_recall": _f(sub["delivered_safety_recall"].mean()),
                "deadline_miss_pct": _f(100.0 * sub["deadline_miss_rate"].mean()),
                "reuse_pct": _f(100.0 * sub["reuse_rate"].mean()),
                "in_paper": profile in REPLAY_PAPER_PROFILES,
            })
    return {
        "id": "replay",
        "paper_table": "VII",
        "paper_profiles": list(REPLAY_PAPER_PROFILES),
        "title": "Detection quality delivered when the measured detector outputs are replayed",
        "caption": ("D = 100 ms, 10 seeds, synthetic network profiles. “all” "
                    "scores a reused output as detecting nothing, which is a lower "
                    "bound because consecutive COCO images are unrelated; "
                    "“fresh” averages only freshly computed outputs. Both are "
                    "reported rather than choosing whichever is more favourable."),
        "source": "results/real_detector/dapper_replay.csv",
        "rows": rows,
    }


# ------------------------------------------------------------------- series
def series_by_profile(long: Long, metric: str, scale: float = 1.0) -> Dict[str, Any]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for policy in MAIN_POLICY_ORDER:
        points = []
        for profile in long.profiles():
            if not long.has(profile, policy, metric):
                continue
            r = long.row(profile, policy, metric)
            points.append({"profile": profile, "mean": _mul(r["mean"], scale),
                           "lo": _mul(r["lo"], scale), "hi": _mul(r["hi"], scale)})
        out[policy] = points
    return out


def series_mode_vs_deadline(dl: pd.DataFrame) -> List[Dict[str, Any]]:
    a = dl[dl["profile"] == "ALL"].sort_values("deadline_sweep_ms")
    return [{
        "deadline_ms": _f(r["deadline_sweep_ms"]),
        "pct_local_fast": _f(r["pct_local_fast"]),
        "pct_edge_accurate": _f(r["pct_edge_accurate"]),
        "pct_hybrid": _f(r["pct_hybrid"]),
        "pct_degraded_safe": _f(r["pct_degraded_safe"]),
        "deadline_miss_pct": _f(100.0 * r["deadline_miss_rate"]),
        "p95_latency_ms": _f(r["p95_latency_ms"]),
        "usable_confidence_proxy": _f(r["usable_confidence_proxy"]),
        "bandwidth_mb_per_1k": _f(1e-3 * r["bandwidth_per_1000_frames_kb"]),
    } for _, r in a.iterrows()]


def series_miss_vs_bias(est: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Profile-mean deadline misses versus scheduler estimate bias (paper Fig. 3b)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    miss = est[est["metric"] == "deadline_miss_rate"]
    for target, sub in miss.groupby("bias_target"):
        pts = []
        for bias, g in sub.groupby("bias"):
            per_profile = g.groupby("profile")["mean"].first()
            pts.append({"bias": _f(bias),
                        "miss_pct_mean": _f(100.0 * per_profile.mean()),
                        "miss_pct_worst": _f(100.0 * per_profile.max())})
        out[str(target)] = sorted(pts, key=lambda p: p["bias"])
    return out


# ----------------------------------------------------------------- rankings
def rankings(long: Long) -> Dict[str, Any]:
    """
    Order the policies on each metric, with the direction taken from
    dapper.metrics.LOWER_IS_BETTER. A consumer that wants to write "best" or
    "lower" looks it up here instead of deciding.
    """
    out: Dict[str, Any] = {}
    for profile in long.profiles():
        per_metric = {}
        for metric, unit, scale, digits in RANKED_METRICS:
            lower_better = metric in LOWER_IS_BETTER
            entries = [{"policy": p, "value": _mul(long.mean(profile, p, metric), scale)}
                       for p in MAIN_POLICY_ORDER if long.has(profile, p, metric)]
            entries.sort(key=lambda e: e["value"], reverse=not lower_better)
            best = entries[0]["value"] if entries else None
            for rank, e in enumerate(entries, start=1):
                e["rank"] = rank
                e["is_best"] = best is not None and abs(e["value"] - best) < 10.0 ** (-digits) / 2
            per_metric[metric] = {
                "lower_is_better": lower_better,
                "unit": unit,
                "digits": digits,
                "order": entries,
                "best_policies": [e["policy"] for e in entries if e["is_best"]],
            }
        out[profile] = per_metric
    return out


# -------------------------------------------------------------------- honest
def honest(long: Long, replay: Optional[pd.DataFrame], dl: Optional[pd.DataFrame],
           abl: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """The three places the paper shows DAPPER behind or paying a price."""
    items: List[Dict[str, Any]] = []

    if replay is not None:
        r = replay[replay["deadline_ms"] == 100.0]
        worse = []
        for profile in PROFILE_ORDER:
            d = r[(r["profile"] == profile) & (r["policy"] == "dapper")]
            l = r[(r["profile"] == profile) & (r["policy"] == "local_only")]
            if not len(d) or not len(l):
                continue
            da, la = d["delivered_recall"].mean(), l["delivered_recall"].mean()
            if da < la:
                worse.append({
                    "profile": profile,
                    "dapper_recall_all": _f(da),
                    "local_recall_all": _f(la),
                    "dapper_recall_fresh": _f(d["delivered_recall_fresh_only"].mean()),
                    "reuse_pct": _f(100.0 * d["reuse_rate"].mean()),
                })
        if worse:
            items.append({
                "id": "all_frame_recall",
                "headline": "All-frame delivered recall falls below local-only where DAPPER reuses",
                "explanation": ("A reused output is scored as detecting nothing in the "
                                "current image. Consecutive COCO validation images are "
                                "unrelated, so this is a lower bound a real video stream "
                                "would not incur; the fresh-frame column shows no such "
                                "gap. Both views are reported."),
                "profiles": worse,
                "source": "results/real_detector/dapper_replay.csv",
                "paper_ref": "Table VII, Sec. V-D",
            })

    rows = []
    for profile in long.profiles():
        if not long.has(profile, "dapper", "remote_attempt_rate"):
            continue
        acc = long.mean(profile, "dapper", "remote_accept_rate")
        bw = 1e-3 * long.mean(profile, "dapper", "bandwidth_per_1000_frames_kb")
        rows.append({
            "profile": profile,
            "bandwidth_mb_per_1k": _f(bw),
            "transmit_pct": _f(100.0 * long.mean(profile, "dapper", "remote_attempt_rate")),
            "accepted_pct": _f(100.0 * acc),
            "rejected_deadline_pct": _f(100.0 * long.mean(
                profile, "dapper", "remote_rejected_deadline_rate")),
            "lost_pct": _f(100.0 * long.mean(profile, "dapper", "remote_failed_loss_rate")),
            "wasted_mb_per_1k": _f(bw * (1.0 - acc)),
        })
    if rows:
        items.append({
            "id": "bandwidth_cost",
            "headline": "Bandwidth is the price, and a large share of it buys nothing",
            "explanation": ("Every transmitted request is charged its upload bandwidth "
                            "even when the packet is lost or the reply arrives too late "
                            "to accept. Under stable conditions roughly half the traffic, "
                            "and under lossy conditions the large majority, returns no "
                            "usable result."),
            "profiles": rows,
            "source": "results/final/multi_seed_ci.csv",
            "paper_ref": "Table III, Table IV, Sec. V-B",
        })

    if dl is not None:
        a = dl[dl["profile"] == "ALL"].sort_values("deadline_sweep_ms")
        misses = [{"deadline_ms": _f(r["deadline_sweep_ms"]),
                   "miss_pct": _f(100.0 * r["deadline_miss_rate"]),
                   "pct_edge_accurate": _f(r["pct_edge_accurate"])}
                  for _, r in a.iterrows() if r["deadline_miss_rate"] > 0.0]
        if misses:
            items.append({
                "id": "larger_deadlines",
                "headline": "The frozen 100-ms calibration is not reliable at every larger deadline",
                "explanation": ("Parameters were selected at D = 100 ms, where no frame "
                                "enters edge-accurate, so the commit margin m is "
                                "unidentifiable there. At larger deadlines frames do "
                                "commit, and the calibrated m = 0.80 produces misses. "
                                "The zero-miss result is scoped to D = 100 ms; "
                                "multi-deadline calibration is the remedy."),
                "deadlines": misses,
                "source": "results/sensitivity/deadline_mode_distribution.csv",
                "paper_ref": "Fig. 3a, Sec. V-C, Sec. VI",
            })

    return items


# --------------------------------------------------------------------- facts
def facts(long: Long, modes: Optional[pd.DataFrame], abl: Optional[pd.DataFrame],
          paired: Optional[pd.DataFrame], overhead: Optional[pd.DataFrame],
          wall: Optional[pd.DataFrame], switches: Optional[pd.DataFrame],
          candidates: Optional[pd.DataFrame], margin: Optional[pd.DataFrame],
          quality: Optional[pd.DataFrame]) -> Dict[str, Any]:
    profiles = long.profiles()
    out: Dict[str, Any] = {
        "dapper_worst_miss_pct": _f(100.0 * max(
            long.mean(p, "dapper", "deadline_miss_rate") for p in profiles)),
        "worst_stale_pct": _f(100.0 * max(
            long.mean(p, q, "stale_acceptance_rate")
            for p in profiles for q in long.policies())),
        "dapper_profile_mean_p95_ms": _f(long.profile_mean("dapper", "p95_latency_ms", profiles)),
        "local_profile_mean_p95_ms": _f(long.profile_mean("local_only", "p95_latency_ms", profiles)),
        "dapper_profile_mean_bandwidth_mb_per_1k": _f(
            1e-3 * long.profile_mean("dapper", "bandwidth_per_1000_frames_kb", profiles)),
    }
    if paired is not None:
        gains = []
        for _, r in paired[(paired["metric"] == "usable_confidence_proxy")
                           & (paired["policy_a"] == "dapper")
                           & (paired["policy_b"] == "local_only")
                           & (paired["profile"] != "ALL")].iterrows():
            gains.append({"profile": r["profile"], "mean_diff": _f(r["mean_diff"]),
                          "lo": _f(r["ci_lo"]), "hi": _f(r["ci_hi"]),
                          "excludes_zero": bool(r["excludes_zero"])})
        order = {p: i for i, p in enumerate(PROFILE_ORDER)}
        out["confidence_gain_vs_local"] = sorted(gains, key=lambda g: order.get(g["profile"], 99))
    if overhead is not None and len(overhead):
        o = overhead.iloc[0]
        out["scheduler_decision_us"] = {
            "mean": _f(o["mean_us"]), "median": _f(o["median_us"]),
            "p95": _f(o["p95_us"]), "p99": _f(o["p99_us"]),
            "calls": int(o["n_calls"]),
        }
    if wall is not None and len(wall):
        w = wall.set_index("policy")["wall_us_per_frame"]
        if "dapper" in w and "local_only" in w:
            out["dapper_added_wall_us_per_frame"] = _f(float(w["dapper"]) - float(w["local_only"]))
    if switches is not None and len(switches):
        d = switches[switches["policy"] == "dapper"]["mode_switches_per_1000_mean"]
        out["dapper_mode_switches_per_1000"] = {"min": _f(d.min()), "max": _f(d.max())}
    if candidates is not None and len(candidates):
        out["calibration"] = {
            "candidates": int(len(candidates)),
            "zero_worst_profile_miss": int((candidates["worst_deadline_miss_rate"] <= 0.0).sum()),
        }
    if margin is not None and len(margin):
        m = margin[(margin["metric"] == "deadline_miss_rate")
                   & (margin["deadline_sweep_ms"] >= 100)
                   & (margin["deadline_sweep_ms"] <= 200)]
        safe = sorted(float(v) for v in m.groupby("remote_deadline_margin")["mean"].max()
                      .pipe(lambda s: s[s <= 0.0]).index)
        out["commit_margin"] = {
            "swept_deadlines_ms": sorted(float(v) for v in m["deadline_sweep_ms"].unique()),
            "margins_with_zero_miss": safe,
            "largest_safe_margin": safe[-1] if safe else None,
            "calibrated_margin_worst_miss_pct": _f(100.0 * float(
                m[m["remote_deadline_margin"] == 0.8]["mean"].max())) if len(
                    m[m["remote_deadline_margin"] == 0.8]) else None,
        }
    if quality is not None and len(quality):
        loc = quality[quality["role"] == "local"]
        rem = quality[quality["role"] == "remote"]
        if len(loc) and len(rem):
            out["detector_gap"] = {
                "recall_points": _f(100.0 * (rem.iloc[0]["recall"] - loc.iloc[0]["recall"])),
                "safety_recall_points": _f(100.0 * (
                    rem.iloc[0]["safety_recall"] - loc.iloc[0]["safety_recall"])),
            }
    return out


# ---------------------------------------------------------------------- main
def build(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ci = read("final/multi_seed_ci.csv")
    if ci is None:
        raise SystemExit("results/final/multi_seed_ci.csv is missing; run experiments/final_eval.py")
    long = Long(ci)
    modes = read("final/mode_distribution.csv")
    paired = read("final/paired_comparisons.csv")
    abl = read("ablation/ablation_summary.csv")
    dl = read("sensitivity/deadline_mode_distribution.csv")
    est = read("sensitivity/runtime_estimation_error_ci.csv")
    margin = read("sensitivity/commit_margin_sweep_ci.csv")
    overhead = read("overhead/scheduler_overhead_summary.csv")
    wall = read("overhead/benchmark_wallclock.csv")
    switches = read("overhead/mode_switches.csv")
    candidates = read("calibration/all_candidates.csv")
    quality = read("real_detector/model_quality.csv")
    accuracy = read("real_detector/model_accuracy.csv")
    latency = read("real_detector/model_latency.csv")
    replay = read("real_detector/dapper_replay.csv")

    n_profiles = len(long.profiles())
    n_policies = len(long.policies())
    seeds = int(cfg["evaluation_seeds_count"])
    frames = int(cfg["frames"])

    tables: Dict[str, Any] = {"main": table_main(long)}
    if modes is not None:
        tables["placement"] = table_placement(modes, long)
    tables["adaptive"] = table_adaptive(long)
    if abl is not None:
        tables["ablation"] = table_ablation(abl)
    if quality is not None:
        tables["detector"] = table_detector(quality, accuracy, latency)
    if replay is not None:
        tables["replay"] = table_replay(replay)

    series: Dict[str, Any] = {
        "p95_by_profile": series_by_profile(long, "p95_latency_ms"),
        "confidence_by_profile": series_by_profile(long, "usable_confidence_proxy"),
        "miss_by_profile": series_by_profile(long, "deadline_miss_rate", 100.0),
        "bandwidth_by_profile": series_by_profile(long, "bandwidth_per_1000_frames_kb", 1e-3),
    }
    if dl is not None:
        series["mode_vs_deadline"] = series_mode_vs_deadline(dl)
    if est is not None:
        series["miss_vs_bias"] = series_miss_vs_bias(est)

    return {
        "generated_by": "experiments/export_web_results.py",
        "note": ("Every value is read from a CSV under results/. The main benchmark is "
                 "a synthetic network study; confidence is a simulated proxy, not "
                 "detector accuracy. The secondary detector study reports measured "
                 "COCO recall and is labelled separately. Nothing here is a "
                 "physical-robot or safety result."),
        "paper": {
            "title": ("DAPPER: Deadline-Aware Edge Perception for Mission-Critical "
                      "Robots under Dynamic Network Conditions"),
            "venue": "IEEE FMEC 2026",
            "authors": ["Awaiz Ahmed", "Khubaib Amjad Alam"],
            "affiliation": "Al Ain University, Abu Dhabi, United Arab Emirates",
        },
        "protocol": {
            "deadline_ms": float(cfg["deadline_ms"]),
            "seeds": seeds,
            "seed_start": int(cfg["evaluation_seeds_start"]),
            "calibration_seeds": [int(s) for s in cfg["calibration_seeds"]],
            "frames_per_cell": frames,
            "profiles": long.profiles(),
            "policies": long.policies(),
            "runs": seeds * n_profiles * n_policies,
            "frame_decisions": seeds * frames * n_profiles * n_policies,
            "bootstrap": {"n_boot": DEFAULT_N_BOOT, "seed": BOOTSTRAP_SEED, "alpha": 0.05},
        },
        "tables": tables,
        "series": series,
        "rankings": rankings(long),
        "facts": facts(long, modes, abl, paired, overhead, wall, switches,
                       candidates, margin, quality),
        "honest": honest(long, replay, dl, abl),
    }


def write_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(text) / 1024:.0f} KiB)")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args(argv)
    payload = build(load_config(args.config))
    write_json(payload, args.out)
    print(f"  tables: {', '.join(payload['tables'])}")
    print(f"  honest items: {', '.join(h['id'] for h in payload['honest'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
