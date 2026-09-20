"""
Record a real network trace in the format the benchmark replays.

This utility exists so that a future measurement campaign - a robot on a real
Wi-Fi or 5G link, probing a real edge node - can drive the *same* scheduler and
execution model without touching either. Nothing in the camera-ready results
was produced by this tool; the released results use seeded synthetic profiles,
and no synthetic profile may be described as a measured trace.

It probes an endpoint at a fixed interval and writes::

    timestamp_ms,rtt_ms,packet_loss,edge_load,edge_available

* ``rtt_ms`` is the measured round trip of a TCP connect (default) or of an
  HTTP request to ``--health-url``.
* ``packet_loss`` is the failure fraction over a trailing window of
  ``--loss-window`` probes, which is the only loss estimate available without
  raw sockets or elevated privileges.
* ``edge_load`` is read from the ``--load-url`` endpoint if one is given
  (expected to return a JSON number or ``{"load": x}`` in [0, 1]); otherwise it
  is left at ``--default-load`` and that fact is recorded in the sidecar
  metadata.
* ``edge_available`` is 0 when a probe fails outright.

Example::

    python tools/capture_network_trace.py --host 192.168.1.50 --port 8000 \\
        --duration-s 600 --interval-ms 100 \\
        --out results/traces/lab_wifi_2026-09-20.csv
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import socket
import sys
import time
from typing import Deque, Optional, Sequence

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dapper.trace import TRACE_COLUMNS, write_network_trace  # noqa: E402


def probe_tcp(host: str, port: int, timeout_s: float) -> Optional[float]:
    """Round-trip time of a TCP connect, in milliseconds, or None on failure."""
    t0 = time.perf_counter_ns()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except Exception:
        return None
    return (time.perf_counter_ns() - t0) / 1e6


def probe_http(url: str, timeout_s: float) -> Optional[float]:
    import urllib.request
    t0 = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            r.read(1)
    except Exception:
        return None
    return (time.perf_counter_ns() - t0) / 1e6


def read_load(url: Optional[str], timeout_s: float, default: float) -> float:
    if not url:
        return default
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            payload = json.loads(r.read().decode("utf-8"))
        value = payload["load"] if isinstance(payload, dict) else payload
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=None, help="edge host for a TCP-connect probe")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--health-url", default=None, help="probe this URL instead of TCP")
    p.add_argument("--load-url", default=None, help="endpoint returning edge load in [0,1]")
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument("--interval-ms", type=float, default=100.0)
    p.add_argument("--timeout-s", type=float, default=1.0)
    p.add_argument("--loss-window", type=int, default=50)
    p.add_argument("--default-load", type=float, default=0.5)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if not args.host and not args.health_url:
        p.error("give --host (TCP probe) or --health-url (HTTP probe)")

    rows = []
    recent: Deque[int] = collections.deque(maxlen=max(1, args.loss_window))
    t_start = time.perf_counter()
    last_rtt = float("nan")
    n = 0
    while (time.perf_counter() - t_start) < args.duration_s:
        tick = time.perf_counter()
        rtt = (probe_http(args.health_url, args.timeout_s) if args.health_url
               else probe_tcp(args.host, args.port, args.timeout_s))
        ok = rtt is not None
        recent.append(0 if ok else 1)
        if ok:
            last_rtt = rtt
        rows.append({
            "timestamp_ms": (tick - t_start) * 1000.0,
            "rtt_ms": rtt if ok else (last_rtt if last_rtt == last_rtt
                                      else args.timeout_s * 1000.0),
            "packet_loss": sum(recent) / len(recent),
            "edge_load": read_load(args.load_url, args.timeout_s, args.default_load),
            "edge_available": int(ok),
        })
        n += 1
        if n % 100 == 0:
            print(f"  [capture] {n} probes, {rows[-1]['timestamp_ms'] / 1000:.0f}s, "
                  f"rtt={rows[-1]['rtt_ms']:.1f}ms loss={rows[-1]['packet_loss']:.3f}",
                  flush=True)
        sleep_s = args.interval_ms / 1000.0 - (time.perf_counter() - tick)
        if sleep_s > 0:
            time.sleep(sleep_s)

    df = pd.DataFrame(rows)
    write_network_trace(args.out, df)
    meta = {
        "target": args.health_url or f"tcp://{args.host}:{args.port}",
        "probe": "http" if args.health_url else "tcp_connect",
        "n_samples": len(df), "interval_ms": args.interval_ms,
        "duration_s": args.duration_s, "loss_window_probes": args.loss_window,
        "edge_load_source": args.load_url or f"constant {args.default_load}",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "columns": list(TRACE_COLUMNS),
        "note": "measured capture; provenance must be stated wherever it is used",
    }
    with io.open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[capture] wrote {args.out} ({len(df)} samples) and its .meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
