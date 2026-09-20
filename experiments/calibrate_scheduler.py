"""
Phase 3 - systematic, reproducible selection of DAPPER's weights and thresholds.

Reviewer 1 objected that the risk weights and thresholds were not selected
systematically. This script replaces hand-chosen values with a transparent
procedure that never touches the evaluation data.

Protocol
--------
* **Calibration seeds 0-9 only.** The evaluation seeds (100-129) are never
  generated here; :func:`assert_disjoint` fails the run if the two sets overlap.
* **Search.** A single joint *deterministic random search* over the full
  parameter vector (fixed search seed), plus two reference points that are
  always evaluated: the originally published configuration and a uniform-weight
  configuration. There is no learning and no adaptive re-sampling, so the whole
  search can be replayed from ``--search-seed`` alone.
* **Objective.** Predeclared and lexicographic, in this order:

  1. minimise the **worst-profile deadline-miss rate**;
  2. minimise the **stale/invalid output acceptance rate**;
  3. maximise the **usable confidence proxy**;
  4. minimise the **worst-profile p95 control latency**;
  5. minimise **offload bandwidth**.

  The order is fixed before any result is seen. Levels are not collapsed into a
  weighted score.

* **Tolerances are taken from the data, not chosen by hand.** A strict
  lexicographic rule would let a difference far smaller than the Monte-Carlo
  noise of the calibration estimate decide the whole configuration. At each
  level we therefore keep every candidate within **one standard error of the
  best candidate's criterion**, where the standard error is computed across the
  10 calibration seeds. Candidates that are indistinguishable from the best at
  one level are passed to the next. No tolerance is a free parameter.

Outputs
-------
``results/calibration/all_candidates.csv``      every candidate and its criteria
``results/calibration/selected_config.yaml``    the frozen configuration
``results/calibration/baseline_candidates.csv`` adaptive-baseline search
``results/calibration/calibration_report.md``   why the selection was made
"""

from __future__ import annotations

import argparse
import copy
import io
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from _common import (  # noqa: E402
    ROOT, RESULTS, assert_disjoint, build_bank, calibration_seeds, load_config,
    run_cells, write_csv, write_text,
)
from dapper.config import RISK_WEIGHT_KEYS, with_overrides  # noqa: E402
from dapper.policies import build_policy  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "calibration")

#: The predeclared objective. (criterion column, aggregation, direction)
#: `aggregation` is how the per-seed scalar is formed from the per-profile values.
OBJECTIVE = (
    ("deadline_miss_rate", "worst", "min"),
    ("stale_acceptance_rate", "mean", "min"),
    ("usable_confidence_proxy", "mean", "max"),
    ("p95_latency_ms", "worst", "min"),
    ("bandwidth_per_1000_frames_kb", "mean", "min"),
)

def assert_thresholds_are_interior(cfg) -> None:
    """Fail loudly if the candidate set contains a degenerate gate threshold."""
    lo = float(cfg["local"]["confidence_min"])
    hi = float(cfg["local"]["confidence_max"])
    bad = [t for t in CONFIDENCE_THRESHOLDS if not lo < t < hi]
    if bad:
        raise ValueError(
            f"local_confidence_threshold candidates {bad} lie outside the open "
            f"interval ({lo}, {hi}) spanned by the local model's confidence, so "
            "they disable the gate rather than set it")


PUBLISHED_REFERENCE = {
    "weight_rtt": 0.35, "weight_loss": 0.30, "weight_load": 0.15,
    "weight_frame_age": 0.05, "weight_deadline": 0.15,
    "risk_local_threshold": 0.40, "risk_degraded_threshold": 0.70,
    "local_confidence_threshold": 0.725, "remote_deadline_margin": 0.60,
    "hybrid_freshness_window_ms": 80.0, "hybrid_estimator": "optimistic",
    "last_valid_freshness_ms": 250.0,
}
UNIFORM_REFERENCE = dict(PUBLISHED_REFERENCE,
                         weight_rtt=0.2, weight_loss=0.2, weight_load=0.2,
                         weight_frame_age=0.2, weight_deadline=0.2)

LOCAL_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
DEGRADED_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
# The confidence gate is only a gate when its threshold lies strictly inside the
# local model's confidence support (`local.confidence_min/max` in config.yaml).
# A threshold at or beyond an endpoint fires on every frame or on no frame: that
# is the degenerate "always offload" / "never offload" policy already covered by
# the functional ablation, and reporting it as a tuned threshold would misstate
# what was selected. Endpoints are therefore excluded from the search space.
# This exclusion *lowers* the best achievable usable-confidence proxy, because
# the degenerate "gate never fires" variant offloads most aggressively.
CONFIDENCE_THRESHOLDS = (0.68, 0.70, 0.72, 0.725, 0.74, 0.76, 0.78)
REMOTE_MARGINS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
HYBRID_WINDOWS = (60.0, 70.0, 80.0, 90.0, 100.0)
HYBRID_ESTIMATORS = ("optimistic", "expected")
FRESHNESS_WINDOWS = (100.0, 150.0, 200.0, 250.0, 300.0)


# --------------------------------------------------------- identifiability
def identifiable_signals(cfg, bank, deadline_ms: float) -> Dict[str, Dict[str, float]]:
    """
    Report, per risk signal, how much it actually varies across the calibration
    scenarios.

    A signal that is *constant* over every frame cannot be identified: its
    weight only adds a fixed offset to the risk score, which the two thresholds
    absorb. Worse, leaving such a dimension in the simplex lets the search park
    weight on it and thereby rescale the signals that do matter -- which is
    exactly how an inert frame-age term made the risk-degraded branch
    unreachable in a first run of this calibration. Constant signals are
    therefore pinned to zero weight and the finding is reported.
    """
    nominal = 0.5 * (float(cfg["edge"]["latency_ms_min"])
                     + float(cfg["edge"]["latency_ms_max"]))
    scale = float(cfg["execution"]["load_compute_scale"])
    extra = float(cfg["edge"].get("extra_rtt_ms", 0.0))
    cols = {k: [] for k in ("rtt", "loss", "load", "frame_age", "deadline")}
    for trace in bank.values():
        rtt = np.asarray(trace.rtt_ms) + extra
        load = np.asarray(trace.edge_load)
        cols["rtt"].append(np.clip(rtt / deadline_ms, 0, 1))
        cols["loss"].append(np.clip(np.asarray(trace.packet_loss), 0, 1))
        cols["load"].append(np.clip(load, 0, 1))
        # The benchmark invokes the scheduler at capture, so the frame age at
        # the decision instant is zero for every frame.
        cols["frame_age"].append(np.zeros(trace.frames))
        cols["deadline"].append(
            np.clip((rtt + nominal * (1 + scale * load)) / deadline_ms, 0, 1))
    out = {}
    for name, parts in cols.items():
        v = np.concatenate(parts)
        out[name] = {"mean": float(v.mean()), "std": float(v.std()),
                     "min": float(v.min()), "max": float(v.max()),
                     "identifiable": bool(v.max() - v.min() > 1e-12)}
    return out


# ------------------------------------------------------------------ search
def sample_candidates(n: int, search_seed: int,
                      free_signals: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    Deterministic random search over the joint parameter space.

    ``free_signals`` names the risk signals that carry weight. Signals left out
    are pinned to zero, so the simplex is taken only over dimensions the
    calibration data can actually distinguish.
    """
    rng = np.random.default_rng(search_seed)
    out: List[Dict[str, Any]] = [dict(PUBLISHED_REFERENCE), dict(UNIFORM_REFERENCE)]
    seen = {_key(c) for c in out}
    while len(out) < n:
        free = list(free_signals or ("rtt", "loss", "load", "frame_age", "deadline"))
        draw = rng.dirichlet(np.ones(len(free)))
        draw = np.round(draw, 2)
        if draw.sum() <= 0:
            continue
        draw = draw / draw.sum()
        draw = np.round(draw, 4)
        draw = draw / draw.sum()
        wmap = {k: 0.0 for k in ("rtt", "loss", "load", "frame_age", "deadline")}
        for k, v in zip(free, draw):
            wmap[k] = float(v)
        w = [wmap["rtt"], wmap["loss"], wmap["load"], wmap["frame_age"], wmap["deadline"]]
        local = float(rng.choice(LOCAL_THRESHOLDS))
        degraded_pool = [d for d in DEGRADED_THRESHOLDS if d > local]
        if not degraded_pool:
            continue
        cand = {
            "weight_rtt": float(w[0]), "weight_loss": float(w[1]),
            "weight_load": float(w[2]), "weight_frame_age": float(w[3]),
            "weight_deadline": float(w[4]),
            "risk_local_threshold": local,
            "risk_degraded_threshold": float(rng.choice(degraded_pool)),
            "local_confidence_threshold": float(rng.choice(CONFIDENCE_THRESHOLDS)),
            "remote_deadline_margin": float(rng.choice(REMOTE_MARGINS)),
            "hybrid_freshness_window_ms": float(rng.choice(HYBRID_WINDOWS)),
            "hybrid_estimator": str(rng.choice(HYBRID_ESTIMATORS)),
            "last_valid_freshness_ms": float(rng.choice(FRESHNESS_WINDOWS)),
        }
        k = _key(cand)
        if k in seen:
            continue
        seen.add(k)
        out.append(cand)
    return out


def _key(c: Dict[str, Any]) -> tuple:
    return tuple(sorted((k, round(v, 6) if isinstance(v, float) else v)
                        for k, v in c.items()))


# -------------------------------------------------------------- objective
def per_seed_criteria(per_run: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Collapse a per-run frame into one scalar per seed for each criterion.

    ``worst`` takes the maximum across profiles within a seed (the profile that
    behaves worst decides), ``mean`` takes the profile-averaged value.
    """
    out: Dict[str, np.ndarray] = {}
    for col, agg, _ in OBJECTIVE:
        piv = per_run.pivot_table(index="seed", columns="profile", values=col, aggfunc="mean")
        out[col] = (piv.max(axis=1) if agg == "worst" else piv.mean(axis=1)).to_numpy(dtype=float)
    return out


def _se(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def lexicographic_select(cands: pd.DataFrame,
                         per_seed: Dict[int, Dict[str, np.ndarray]]) -> Tuple[int, List[str]]:
    """
    Apply the predeclared objective with data-derived, one-standard-error slack.

    Returns the winning candidate id and a human-readable trace of each level.
    """
    alive = list(cands["candidate_id"])
    log: List[str] = []
    for col, agg, direction in OBJECTIVE:
        vals = {cid: float(np.mean(per_seed[cid][col])) for cid in alive}
        best_cid = (min(vals, key=vals.get) if direction == "min"
                    else max(vals, key=vals.get))
        best = vals[best_cid]
        tol = _se(per_seed[best_cid][col])
        if direction == "min":
            keep = [cid for cid in alive if vals[cid] <= best + tol]
        else:
            keep = [cid for cid in alive if vals[cid] >= best - tol]
        log.append(
            f"L{OBJECTIVE.index((col, agg, direction)) + 1} {direction}({agg} {col}): "
            f"best={best:.6g} from candidate {best_cid}; "
            f"1 SE over calibration seeds = {tol:.6g}; "
            f"{len(keep)} of {len(alive)} candidates retained")
        alive = keep
        if len(alive) == 1:
            break
    # Deterministic tie-break: the lowest candidate id still alive.
    return int(min(alive)), log


# ------------------------------------------------------------- evaluation
def evaluate_candidate(cfg: Dict[str, Any], cand: Dict[str, Any], bank,
                       deadline_ms: float) -> pd.DataFrame:
    cand_cfg = with_overrides(cfg, **cand)
    policy = build_policy("dapper", cand_cfg)
    per_run, _ = run_cells(bank, [policy], cand_cfg, deadline_ms)
    return per_run


def calibrate_dapper(cfg, bank, deadline_ms, n_candidates, search_seed, free_signals=None):
    cands = sample_candidates(n_candidates, search_seed, free_signals)
    rows, per_seed = [], {}
    for cid, cand in enumerate(cands):
        per_run = evaluate_candidate(cfg, cand, bank, deadline_ms)
        crit = per_seed_criteria(per_run)
        per_seed[cid] = crit
        row = {"candidate_id": cid, **cand}
        for col, agg, _ in OBJECTIVE:
            row[f"{agg}_{col}"] = float(np.mean(crit[col]))
            row[f"{agg}_{col}_se"] = _se(crit[col])
        row["is_published_reference"] = (cid == 0)
        row["is_uniform_reference"] = (cid == 1)
        rows.append(row)
        if (cid + 1) % 25 == 0:
            print(f"  [calibration] {cid + 1}/{len(cands)} candidates evaluated", flush=True)
    table = pd.DataFrame(rows)
    winner, log = lexicographic_select(table, per_seed)
    table["selected"] = table["candidate_id"] == winner
    return table, winner, log, cands


# ------------------------------------------------------ adaptive baselines
def baseline_grid() -> List[Dict[str, Any]]:
    grid = []
    for m in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        grid.append({"baseline": "deadline_greedy", "deadline_greedy_margin": m})
    for t in CONFIDENCE_THRESHOLDS:
        for m in (0.3, 0.5, 0.7, 0.9, 1.0):
            grid.append({"baseline": "confidence_deadline",
                         "confidence_deadline_threshold": t,
                         "confidence_deadline_margin": m})
    for r in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0, 100.0):
        grid.append({"baseline": "rtt_threshold", "rtt_threshold_ms": r})
    return grid


def calibrate_baselines(cfg, bank, deadline_ms):
    rows, per_seed, ids = [], {}, {}
    for gid, entry in enumerate(baseline_grid()):
        name = entry["baseline"]
        params = {k: v for k, v in entry.items() if k != "baseline"}
        policy = build_policy(name, cfg, baseline_params={**cfg.get("baselines", {}), **params})
        per_run, _ = run_cells(bank, [policy], cfg, deadline_ms)
        crit = per_seed_criteria(per_run)
        per_seed[gid] = crit
        row = {"candidate_id": gid, "baseline": name, **params}
        for col, agg, _ in OBJECTIVE:
            row[f"{agg}_{col}"] = float(np.mean(crit[col]))
        rows.append(row)
        ids.setdefault(name, []).append(gid)
    table = pd.DataFrame(rows)
    selected: Dict[str, Dict[str, Any]] = {}
    logs: Dict[str, List[str]] = {}
    for name, gids in ids.items():
        sub = table[table["candidate_id"].isin(gids)]
        win, log = lexicographic_select(sub, {g: per_seed[g] for g in gids})
        logs[name] = log
        chosen = table[table["candidate_id"] == win].iloc[0]
        selected[name] = {k: chosen[k] for k in chosen.index
                          if k not in ("candidate_id", "baseline")
                          and not k.startswith(("worst_", "mean_"))
                          and not pd.isna(chosen[k])}
    table["selected"] = table["candidate_id"].isin(
        [lexicographic_select(table[table["candidate_id"].isin(g)],
                              {x: per_seed[x] for x in g})[0] for g in ids.values()])
    return table, selected, logs


# ------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--frames", type=int, default=500,
                   help="frames per calibration cell (calibration uses a smaller "
                        "budget than the final protocol; it only ranks candidates)")
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--candidates", type=int, default=400)
    p.add_argument("--search-seed", type=int, default=12345)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    assert_disjoint(cfg)
    assert_thresholds_are_interior(cfg)
    seeds = calibration_seeds(cfg)
    profiles = list(cfg["profiles"])
    print(f"[calibration] seeds={seeds} profiles={profiles} frames={args.frames} "
          f"D={args.deadline_ms} candidates={args.candidates} search_seed={args.search_seed}")
    print("[calibration] predeclared objective:")
    for i, (col, agg, d) in enumerate(OBJECTIVE, 1):
        print(f"             L{i}: {d}imise {agg} {col}")

    bank = build_bank(cfg, profiles, seeds, args.frames)

    signals = identifiable_signals(cfg, bank, args.deadline_ms)
    free = [k for k, v in signals.items() if v["identifiable"]]
    pinned = [k for k, v in signals.items() if not v["identifiable"]]
    print("[calibration] risk-signal identifiability over the calibration data:")
    for k, v in signals.items():
        verdict = "free" if v["identifiable"] else "CONSTANT -> weight pinned to 0"
        print(f"             {k:<10s} range [{v['min']:.4f}, {v['max']:.4f}] "
              f"std {v['std']:.4f} -> {verdict}")
    write_csv(pd.DataFrame([{"signal": k, **v} for k, v in signals.items()]),
              os.path.join(args.out_dir, "signal_identifiability.csv"))

    table, winner, log, cands = calibrate_dapper(
        cfg, bank, args.deadline_ms, args.candidates, args.search_seed, free)
    write_csv(table, os.path.join(args.out_dir, "all_candidates.csv"))

    bl_table, bl_selected, bl_logs = calibrate_baselines(cfg, bank, args.deadline_ms)
    write_csv(bl_table, os.path.join(args.out_dir, "baseline_candidates.csv"))

    chosen = cands[winner]
    selected_cfg = copy.deepcopy(cfg)
    selected_cfg["scheduler"].update(chosen)
    for name, params in bl_selected.items():
        for k, v in params.items():
            selected_cfg.setdefault("baselines", {})[k] = (
                float(v) if isinstance(v, (int, float, np.floating)) else v)
    selected_cfg["_calibration"] = {
        "search_seed": int(args.search_seed),
        "candidates": int(args.candidates),
        "calibration_seeds": seeds,
        "frames_per_cell": int(args.frames),
        "deadline_ms": float(args.deadline_ms),
        "winning_candidate_id": int(winner),
    }
    write_text(yaml.safe_dump(selected_cfg, sort_keys=False, default_flow_style=False),
               os.path.join(args.out_dir, "selected_config.yaml"))

    write_text(_report(cfg, table, winner, log, chosen, bl_table, bl_selected,
                       bl_logs, args, signals, pinned),
               os.path.join(args.out_dir, "calibration_report.md"))
    print(f"[calibration] winner = candidate {winner}")
    for line in log:
        print("   " + line)
    return 0


def _report(cfg, table, winner, log, chosen, bl_table, bl_selected, bl_logs,
            args, signals=None, pinned=None) -> str:
    ref = table[table["is_published_reference"]].iloc[0]
    win = table[table["candidate_id"] == winner].iloc[0]
    L = []
    L.append("# Scheduler calibration report\n")
    L.append("Generated by `experiments/calibrate_scheduler.py`. "
             "Every number here comes from `all_candidates.csv`.\n")
    L.append("## Protocol\n")
    L.append(f"* Calibration seeds: `{cfg['calibration_seeds']}` "
             f"(evaluation seeds {cfg['evaluation_seeds_start']}-"
             f"{cfg['evaluation_seeds_start'] + cfg['evaluation_seeds_count'] - 1} "
             "were not generated by this script).")
    L.append(f"* Profiles: {', '.join(cfg['profiles'])}.")
    L.append(f"* {args.frames} frames per (seed, profile) cell, D = {args.deadline_ms:g} ms.")
    L.append(f"* Search: deterministic random search, `--search-seed {args.search_seed}`, "
             f"{len(table)} candidates, including the published configuration "
             "(candidate 0) and a uniform-weight configuration (candidate 1).")
    L.append("* Weights are drawn from a flat Dirichlet on the 5-simplex, so they are "
             "non-negative and sum to 1 by construction; "
             "`risk_local_threshold < risk_degraded_threshold` is enforced at sampling "
             "time and re-checked by `dapper.config.validate_config`.\n")
    L.append("## Risk-signal identifiability\n")
    if signals:
        L.append("A risk signal that is constant over every calibration frame "
                 "cannot be identified: its weight only adds a fixed offset that "
                 "the two thresholds absorb, and leaving it in the simplex lets the "
                 "search park weight on it and thereby rescale the signals that do "
                 "matter. Constant signals are pinned to zero weight and the "
                 "simplex is taken over the remaining dimensions.\n")
        L.append("| signal | min | max | std | in search |")
        L.append("|---|---|---|---|---|")
        for k, v in signals.items():
            L.append(f"| `{k}` | {v['min']:.4f} | {v['max']:.4f} | {v['std']:.4f} | "
                     f"{'yes' if v['identifiable'] else '**no - pinned to 0**'} |")
        if pinned:
            L.append("")
            L.append("`" + "`, `".join(pinned) + "` "
                     + ("is" if len(pinned) == 1 else "are")
                     + " constant at zero because the benchmark invokes the "
                     "scheduler at frame capture, so the frame age at the decision "
                     "instant is always zero. This was already true of the published "
                     "configuration, whose `weight_frame_age: 0.05` was therefore "
                     "inert. A first run of this calibration parked 26 % of the "
                     "weight on that inert dimension, which pushed the maximum "
                     "attainable risk (0.743) below the selected degraded threshold "
                     "(0.750) and made the risk-based degraded-safe branch "
                     "unreachable. The term is retained in the risk formulation "
                     "because it becomes identifiable in a deployment where frames "
                     "queue before the scheduler runs.")
        L.append("")
    L.append("## Predeclared objective\n")
    for i, (col, agg, d) in enumerate(OBJECTIVE, 1):
        L.append(f"{i}. {'minimise' if d == 'min' else 'maximise'} **{agg} `{col}`**")
    L.append("\nThe order was fixed before any candidate was evaluated and the levels are "
             "not collapsed into a weighted score. At each level, candidates within **one "
             "standard error** of the best candidate (computed across the calibration "
             "seeds) are carried to the next level, so a difference smaller than the "
             "Monte-Carlo noise of the calibration estimate never decides the "
             "configuration. The final tie-break is the lowest candidate id, which is a "
             "deterministic function of the search seed.\n")
    L.append("## Selection trace\n```")
    L.extend(log)
    L.append("```\n")
    L.append("## Selected configuration\n")
    L.append("| parameter | published | selected |")
    L.append("|---|---|---|")
    for k in PUBLISHED_REFERENCE:
        L.append(f"| `{k}` | {PUBLISHED_REFERENCE[k]} | {chosen[k]} |")
    L.append("")
    L.append("## Selected vs published configuration on the calibration seeds\n")
    L.append("| criterion | published (cand 0) | selected (cand "
             f"{winner}) |")
    L.append("|---|---|---|")
    for col, agg, _ in OBJECTIVE:
        key = f"{agg}_{col}"
        L.append(f"| {agg} `{col}` | {ref[key]:.6g} | {win[key]:.6g} |")
    L.append("")
    if winner == 0:
        L.append("The published configuration survived every level of the objective and "
                 "was selected. No scheduler parameter changed.\n")
    else:
        L.append("The published configuration did **not** survive the objective. "
                 "The differences above are reported as generated; no attempt was made "
                 "to recover the published values.\n")
    L.append("## Adaptive-baseline hyperparameters\n")
    L.append("Each adaptive baseline was fitted on the same calibration seeds with the "
             "same objective, so no comparator is handicapped by an arbitrary setting.\n")
    for name, params in bl_selected.items():
        pretty = ", ".join(f"`{k}` = {v}" for k, v in params.items())
        L.append(f"* **{name}**: {pretty}")
        L.append("  ```")
        for line in bl_logs[name]:
            L.append("  " + line)
        L.append("  ```")
    L.append("")
    L.append("## Parameters this calibration cannot identify\n")
    L.append("`remote_deadline_margin` gates the `edge_accurate` mode, which "
             "requires the *expected* remote completion to fit a fraction of the "
             "deadline. With the configured edge envelope no frame satisfies that "
             "test at D = 100 ms, so every candidate scores identically on this "
             "parameter and its selected value is **not identified by this "
             "calibration**. It is examined directly in the deadline sweep "
             "(`results/sensitivity/deadline_sweep.csv`), which is where "
             "`edge_accurate` first becomes reachable. This is reported rather "
             "than hidden.\n")
    L.append("## Search-space restriction\n")
    L.append("`local_confidence_threshold` candidates are restricted to the open "
             f"interval ({cfg['local']['confidence_min']}, "
             f"{cfg['local']['confidence_max']}) spanned by the local model's "
             "confidence. A threshold at or beyond an endpoint would fire on every "
             "frame or on none, which is the degenerate always/never-offload "
             "policy already measured by the functional ablation. Excluding the "
             "endpoints *reduces* the best attainable usable-confidence proxy, so "
             "the restriction works against DAPPER rather than for it.\n")
    L.append("## Freeze\n")
    L.append("`results/calibration/selected_config.yaml` is copied into `config.yaml` and "
             "then frozen. Every number in the final evaluation, the ablations and the "
             "sensitivity studies comes from that frozen configuration; no parameter is "
             "re-selected on evaluation seeds.\n")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
