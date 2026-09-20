"""
Plot generation for DAPPER runs.

Conventions:
    * Pure matplotlib, no seaborn.
    * Each chart is its own figure.
    * No explicit colour choices - we rely on matplotlib defaults so that
    the output looks consistent with the rest of the toolchain.

The plot functions accept tidy DataFrames produced by benchmark.py / metrics.py
and write PNG files to a results directory.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")  # Headless: do not require an X display.
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_latency_over_time(
    per_frame: pd.DataFrame,
    out_path: str,
    title: str = "Latency over time",
) -> str:
    """One line per mode. X axis is frame index, Y axis is latency."""
    fig = plt.figure()
    for mode, sub in per_frame.groupby("run_mode"):
        sub = sub.sort_values("frame_id")
        plt.plot(sub["frame_id"], sub["latency_ms"], label=mode, linewidth=1.0)
    plt.xlabel("Frame")
    plt.ylabel("Latency (ms)")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_p95_latency_by_mode(summary: pd.DataFrame, out_path: str) -> str:
    fig = plt.figure()
    labels = summary["mode"].astype(str).tolist()
    values = summary["p95_latency_ms"].astype(float).tolist()
    plt.bar(labels, values)
    plt.xlabel("Mode")
    plt.ylabel("p95 latency (ms)")
    plt.title("p95 Latency by Mode")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_deadline_miss_by_mode(summary: pd.DataFrame, out_path: str) -> str:
    fig = plt.figure()
    labels = summary["mode"].astype(str).tolist()
    values = (summary["deadline_miss_rate"].astype(float) * 100.0).tolist()
    plt.bar(labels, values)
    plt.xlabel("Mode")
    plt.ylabel("Deadline miss rate (%)")
    plt.title("Deadline Miss Rate by Mode")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_confidence_by_mode(summary: pd.DataFrame, out_path: str) -> str:
    fig = plt.figure()
    labels = summary["mode"].astype(str).tolist()
    values = summary["mean_confidence"].astype(float).tolist()
    plt.bar(labels, values)
    plt.xlabel("Mode")
    plt.ylabel("Mean confidence")
    plt.title("Mean Confidence by Mode")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_bandwidth_by_mode(summary: pd.DataFrame, out_path: str) -> str:
    fig = plt.figure()
    labels = summary["mode"].astype(str).tolist()
    values = summary["total_bandwidth_kb"].astype(float).tolist()
    plt.bar(labels, values)
    plt.xlabel("Mode")
    plt.ylabel("Total bandwidth (KB)")
    plt.title("Total Bandwidth by Mode")
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_dapper_mode_distribution(
    per_frame: pd.DataFrame,
    out_path: str,
    run_mode_filter: str = "dapper",
) -> str:
    """Show how DAPPER distributes its decisions across the four modes."""
    sub = per_frame[per_frame["run_mode"] == run_mode_filter]
    fig = plt.figure()
    if len(sub) == 0:
        plt.text(0.5, 0.5, "No DAPPER frames in input", ha="center", va="center")
        plt.axis("off")
    else:
        counts = sub["selected_mode"].value_counts()
        plt.bar(counts.index.astype(str).tolist(), counts.values.tolist())
        plt.xlabel("Selected mode")
        plt.ylabel("Frame count")
        plt.title("DAPPER mode distribution")
        plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_plots(
    per_frame: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: str,
    profile_tag: Optional[str] = None,
) -> Iterable[str]:
    """Convenience wrapper that produces every plot for a profile."""
    _ensure_dir(out_dir)
    suffix = f"_{profile_tag}" if profile_tag else ""

    paths = [
        plot_latency_over_time(
            per_frame,
            os.path.join(out_dir, f"latency_over_time{suffix}.png"),
            title=f"Latency over time ({profile_tag or 'all'})",
        ),
        plot_p95_latency_by_mode(
            summary, os.path.join(out_dir, f"p95_by_mode{suffix}.png")
        ),
        plot_deadline_miss_by_mode(
            summary, os.path.join(out_dir, f"deadline_miss_by_mode{suffix}.png")
        ),
        plot_confidence_by_mode(
            summary, os.path.join(out_dir, f"confidence_by_mode{suffix}.png")
        ),
        plot_bandwidth_by_mode(
            summary, os.path.join(out_dir, f"bandwidth_by_mode{suffix}.png")
        ),
        plot_dapper_mode_distribution(
            per_frame, os.path.join(out_dir, f"dapper_mode_distribution{suffix}.png")
        ),
    ]
    return paths
