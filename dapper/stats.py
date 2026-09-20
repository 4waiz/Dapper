"""
Seed-level statistics.

The evaluation produces 30 seeds x 1000 frames per cell. Those 30,000 frame
observations are **not** 30,000 independent samples: the network process is
AR(1)-smoothed, outages last 5-25 frames, and the last-valid store couples
neighbouring frames. Treating frames as independent would understate the
variance by roughly the autocorrelation length and manufacture false precision.

The independent unit of replication is therefore the **seed**. Each seed
contributes one scalar per metric (its own run-level value), and all dispersion
is computed across those 30 values.

Confidence intervals
--------------------
Percentile bootstrap over seeds: resample the 30 per-seed values with
replacement ``n_boot`` times, take the statistic of each resample, and report
the 2.5th and 97.5th percentiles. The bootstrap generator is seeded from a fixed
constant so the intervals are reproducible.

Paired comparisons
------------------
Because every policy replays the *same* scenario trace for a given seed, a
policy pair can be compared seed-by-seed. :func:`paired_difference` bootstraps
the per-seed difference, which is strictly more powerful than comparing two
independent intervals and is the correct test for this design. No difference is
described as significant unless its interval excludes zero, and the language
used in the generated reports is fixed by :func:`describe_difference`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Fixed bootstrap seed. Changing it changes every reported interval.
BOOTSTRAP_SEED = 20260920
DEFAULT_N_BOOT = 10000


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Percentile bootstrap of ``statistic`` over the supplied per-seed values."""
    x = np.asarray(list(values), dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "mean": float("nan"), "std": float("nan"), "n": 0}
    point = float(statistic(x))
    if n == 1:
        return {"point": point, "lo": point, "hi": point,
                "mean": point, "std": 0.0, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.apply_along_axis(statistic, 1, x[idx]) if statistic is not np.mean \
        else x[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "n": int(n),
    }


def aggregate_over_seeds(
    per_run: pd.DataFrame,
    group_cols: Sequence[str] = ("profile", "policy"),
    metrics: Optional[Sequence[str]] = None,
    n_boot: int = DEFAULT_N_BOOT,
) -> pd.DataFrame:
    """
    Collapse a per-run metrics frame to one row per group, with bootstrap CIs.

    Returns a long-format frame: one row per (group, metric) with
    ``mean``, ``std``, ``ci_lo``, ``ci_hi`` and ``n_seeds``.
    """
    group_cols = list(group_cols)
    skip = set(group_cols) | {"seed", "frames"}
    if metrics is None:
        metrics = [c for c in per_run.columns
                   if c not in skip and pd.api.types.is_numeric_dtype(per_run[c])]
    rows: List[Dict[str, Any]] = []
    for key, sub in per_run.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key))
        for m in metrics:
            ci = bootstrap_ci(sub[m].to_numpy(dtype=float), n_boot=n_boot)
            rows.append({**base, "metric": m, "mean": ci["mean"], "std": ci["std"],
                         "ci_lo": ci["lo"], "ci_hi": ci["hi"], "n_seeds": ci["n"]})
    return pd.DataFrame(rows)


def wide_aggregate(long_df: pd.DataFrame,
                   group_cols: Sequence[str] = ("profile", "policy")) -> pd.DataFrame:
    """Pivot :func:`aggregate_over_seeds` output into one row per group."""
    group_cols = list(group_cols)
    out = long_df.pivot_table(index=group_cols, columns="metric",
                              values=["mean", "std", "ci_lo", "ci_hi"], aggfunc="first")
    out.columns = [f"{metric}__{stat}" for stat, metric in out.columns]
    return out.reset_index()


def paired_difference(
    per_run: pd.DataFrame,
    metric: str,
    policy_a: str,
    policy_b: str,
    profile: Optional[str] = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """
    Bootstrap the per-seed difference ``policy_a - policy_b`` for one metric.

    Both policies replayed identical scenarios for each seed, so the difference
    is paired. Returns the point estimate, the 95 % interval, the number of
    seeds, and whether the interval excludes zero.
    """
    sub = per_run if profile is None else per_run[per_run["profile"] == profile]
    a = sub[sub["policy"] == policy_a].set_index("seed")[metric]
    b = sub[sub["policy"] == policy_b].set_index("seed")[metric]
    common = sorted(set(a.index) & set(b.index))
    if not common:
        raise ValueError(f"no shared seeds between {policy_a} and {policy_b}")
    d = (a.loc[common].to_numpy(dtype=float) - b.loc[common].to_numpy(dtype=float))
    ci = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    excludes_zero = bool((ci["lo"] > 0.0) or (ci["hi"] < 0.0))
    return {
        "metric": metric, "policy_a": policy_a, "policy_b": policy_b,
        "profile": profile or "ALL", "n_seeds": len(common),
        "mean_diff": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
        "std": ci["std"], "excludes_zero": excludes_zero,
        "a_mean": float(a.loc[common].mean()), "b_mean": float(b.loc[common].mean()),
    }


def describe_difference(d: Dict[str, Any], unit: str = "", lower_is_better: bool = True) -> str:
    """
    Render a paired difference in fixed, non-inflationary language.

    The word "significant" is used only when the bootstrap interval excludes
    zero, and never as a synonym for "large".
    """
    direction = "lower" if d["mean_diff"] < 0 else "higher"
    verdict = ("the 95% interval excludes zero" if d["excludes_zero"]
               else "the 95% interval includes zero")
    return (f"{d['policy_a']} produced {abs(d['mean_diff']):.4g}{unit} {direction} "
            f"{d['metric']} than {d['policy_b']} under {d['profile']} "
            f"(paired over {d['n_seeds']} seeds, 95% CI "
            f"[{d['ci_lo']:.4g}, {d['ci_hi']:.4g}]{unit}; {verdict})")


def fmt_ci(mean: float, lo: float, hi: float, digits: int = 1, unit: str = "") -> str:
    """Paper-friendly ``12.3 [95% CI: 11.8-12.9]`` rendering."""
    return f"{mean:.{digits}f}{unit} [95% CI: {lo:.{digits}f}-{hi:.{digits}f}]"
