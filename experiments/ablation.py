"""
Phase 7 - DAPPER ablation study.

Two one-factor-at-a-time families; no Cartesian product.

Component ablation
    Remove one risk signal at a time by setting its weight to zero and
    **renormalising the remaining weights to sum to 1**, so the risk score stays
    on the same [0, 1] scale and thresholds keep their meaning. Variants:
    ``DAPPER-full``, ``-no-rtt``, ``-no-loss``, ``-no-load``, ``-no-frame-age``,
    ``-no-deadline-pressure``.

Functional ablation
    Disable one mechanism at a time: the local-confidence gate (always offload
    in the low-risk region), the freshness gate (offload for a refresh even when
    it cannot land in time) and degraded-safe reuse (never reuse a previous
    output; always recompute locally).

Everything runs on evaluation seeds, over the same paired scenario traces, with
the frozen calibrated configuration. Observations are emitted only where the
generated data supports them.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from _common import (  # noqa: E402
    RESULTS, assert_disjoint, build_bank, evaluation_seeds, load_config,
    run_cells, write_csv, write_text,
)
from dapper.config import RISK_COMPONENTS, renormalised_weights  # noqa: E402
from dapper.policies import DapperPolicy  # noqa: E402
from dapper.scheduler import build_scheduler_from_config  # noqa: E402
from dapper.stats import aggregate_over_seeds, paired_difference  # noqa: E402

OUT_DIR = os.path.join(RESULTS, "ablation")

ABLATION_METRICS = (
    "deadline_miss_rate", "p95_latency_ms", "mean_latency_ms",
    "usable_confidence_proxy", "bandwidth_per_1000_frames_kb",
    "stale_acceptance_rate", "fallback_rate", "reuse_rate",
    "remote_attempt_rate", "remote_accept_rate",
    "pct_local_fast", "pct_edge_accurate", "pct_hybrid", "pct_degraded_safe",
)

FUNCTIONAL_VARIANTS = (
    ("dapper_full", {}),
    ("dapper_no_confidence_gate", {"use_confidence_gate": False}),
    ("dapper_no_freshness_gate", {"use_freshness_gate": False}),
    ("dapper_no_degraded_reuse", {"use_degraded_reuse": False}),
)


def component_variants(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, float]]]:
    base = {k: float(cfg["scheduler"][k]) for k in
            ("weight_rtt", "weight_loss", "weight_load",
             "weight_frame_age", "weight_deadline")}
    out = [("dapper_full", renormalised_weights(base, None))]
    for comp in RISK_COMPONENTS:
        if base[f"weight_{comp}"] == 0.0:
            # Removing a signal that already carries no weight is a no-op; say so
            # rather than silently reporting an identical row.
            out.append((f"dapper_no_{comp}", renormalised_weights(base, comp)))
        else:
            out.append((f"dapper_no_{comp}", renormalised_weights(base, comp)))
    return out


def _run_variants(cfg, bank, deadline_ms, variants, build) -> pd.DataFrame:
    frames = []
    for name, spec in variants:
        policy = build(name, spec)
        per_run, _ = run_cells(bank, [policy], cfg, deadline_ms)
        per_run["variant"] = name
        frames.append(per_run)
        print(f"  [ablation] {name} done", flush=True)
    return pd.concat(frames, ignore_index=True)


def _observations(per_run: pd.DataFrame, family: str, n_boot: int) -> List[str]:
    """
    Emit only factual, generated statements of the form
    "Removing X changed Y from A to B under profile Z".

    A statement is emitted for a (variant, profile, metric) cell only when the
    paired 95 % bootstrap interval of the difference from ``dapper_full``
    excludes zero, so no difference is described that the data cannot support.
    """
    lines: List[str] = []
    tidy = per_run.rename(columns={"variant": "policy"})
    for variant in sorted(set(tidy["policy"])):
        if variant == "dapper_full":
            continue
        for profile in sorted(set(tidy["profile"])):
            for metric, unit, digits in (
                ("deadline_miss_rate", " (fraction)", 4),
                ("p95_latency_ms", " ms", 2),
                ("usable_confidence_proxy", "", 4),
                ("bandwidth_per_1000_frames_kb", " KB/1000 frames", 1),
                ("stale_acceptance_rate", " (fraction)", 4),
            ):
                try:
                    d = paired_difference(tidy, metric, variant, "dapper_full",
                                          profile=profile, n_boot=n_boot)
                except ValueError:
                    continue
                if not d["excludes_zero"]:
                    continue
                verb = "increased" if d["mean_diff"] > 0 else "decreased"
                lines.append(
                    f"* Under **{profile}**, `{variant}` {verb} `{metric}` from "
                    f"{d['b_mean']:.{digits}f}{unit} to {d['a_mean']:.{digits}f}{unit} "
                    f"(paired difference {d['mean_diff']:+.{digits}f}{unit}, "
                    f"95% CI [{d['ci_lo']:+.{digits}f}, {d['ci_hi']:+.{digits}f}], "
                    f"{d['n_seeds']} seeds).")
    if not lines:
        lines.append(f"* No {family} variant produced a difference from `dapper_full` "
                     "whose paired 95 % bootstrap interval excluded zero on any "
                     "reported metric.")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--deadline-ms", type=float, default=100.0)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--profiles", nargs="*", default=None,
                   help="default: variable, congested, lossy, outage, stable")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    assert_disjoint(cfg)
    seeds = evaluation_seeds(cfg, args.seeds)
    profiles = args.profiles or ["variable", "congested", "lossy", "outage", "stable"]
    print(f"[ablation] seeds={seeds[0]}-{seeds[-1]} profiles={profiles} "
          f"frames={args.frames} D={args.deadline_ms}")
    bank = build_bank(cfg, profiles, seeds, args.frames)

    comp = _run_variants(
        cfg, bank, args.deadline_ms, component_variants(cfg),
        lambda name, w: DapperPolicy(
            cfg, scheduler=build_scheduler_from_config(cfg, weights=w), name=name))
    write_csv(comp, os.path.join(args.out_dir, "component_ablation.csv"))

    func = _run_variants(
        cfg, bank, args.deadline_ms, list(FUNCTIONAL_VARIANTS),
        lambda name, kw: DapperPolicy(
            cfg, scheduler=build_scheduler_from_config(cfg, **kw), name=name))
    write_csv(func, os.path.join(args.out_dir, "functional_ablation.csv"))

    summary_rows = []
    for family, df in (("component", comp), ("functional", func)):
        agg = aggregate_over_seeds(df.rename(columns={"variant": "policy"}),
                                   ("profile", "policy"),
                                   [m for m in ABLATION_METRICS if m in df.columns],
                                   n_boot=args.n_boot)
        agg["family"] = family
        summary_rows.append(agg)
    summary = pd.concat(summary_rows, ignore_index=True)
    write_csv(summary, os.path.join(args.out_dir, "ablation_summary.csv"))

    obs = ["# Ablation observations\n",
           "Generated by `experiments/ablation.py` from "
           "`component_ablation.csv` and `functional_ablation.csv`. Each line is "
           "a paired, seed-level comparison against `dapper_full` on the same "
           "scenario traces; only differences whose 95 % bootstrap interval "
           "excludes zero are stated.\n",
           "## Risk-component ablation\n"]
    obs += _observations(comp, "component", args.n_boot)
    obs += ["\n## Functional ablation\n"]
    obs += _observations(func, "functional", args.n_boot)
    write_text("\n".join(obs) + "\n", os.path.join(args.out_dir, "ablation_observations.md"))

    pd.set_option("display.width", 220)
    print("\n[ablation] component means:")
    print(comp.groupby(["variant", "profile"], as_index=False)[
        ["deadline_miss_rate", "p95_latency_ms", "usable_confidence_proxy",
         "bandwidth_per_1000_frames_kb"]].mean().to_string(index=False))
    print("\n[ablation] functional means:")
    print(func.groupby(["variant", "profile"], as_index=False)[
        ["deadline_miss_rate", "p95_latency_ms", "usable_confidence_proxy",
         "bandwidth_per_1000_frames_kb", "reuse_rate"]].mean().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
