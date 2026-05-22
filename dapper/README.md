# DAPPER: Deadline-Aware Perception Placement for Edge Robotics

A minimal but serious research prototype for evaluating **latency-aware edge AI
scheduling** for real-time robotic perception in mission-critical
environments.

DAPPER chooses, on a frame-by-frame basis, between four execution modes:

| Mode             | What it does                                                                  |
|------------------|-------------------------------------------------------------------------------|
| `local_fast`     | Lightweight local inference. Lowest latency, lowest accuracy proxy.           |
| `edge_accurate`  | Remote inference over an HTTP endpoint. Higher accuracy, depends on network.  |
| `hybrid`         | Returns local immediately, refreshes with the edge result if it arrives in time. |
| `degraded_safe`  | Fallback for high-risk frames: reuse last valid result or local fallback.     |

The benchmark compares DAPPER against three fixed baselines:
`local_only`, `edge_only`, and `cloud_only`.

> **Honest scope.** This is a *controlled synthetic benchmark*, not a physical
> robot deployment. Network and inference timings are sampled from explicit,
> configurable distributions, and we never claim to have run a real YOLO
> model. Every simulated number is documented in [`config.yaml`](config.yaml)
> and can be overridden.

## Project layout

```
dapper/
  README.md            <- you are here
  requirements.txt     <- pip dependencies
  config.yaml          <- all simulated parameters
  app.py               <- optional FastAPI dashboard
  scheduler.py         <- DAPPER risk-driven scheduler
  monitor.py           <- network condition simulator
  perception.py        <- synthetic / OpenCV perception backend
  edge_server.py       <- simulated FastAPI edge inference server
  benchmark.py         <- main CLI runner
  metrics.py           <- summary metrics
  plots.py             <- matplotlib charts
  data/                <- optional video files
  results/             <- output CSVs and PNGs (gitignored content)
  tests/test_scheduler.py
```

## Setup

```bash
cd dapper
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the simulated edge server (optional)

The benchmark works entirely offline; the edge server only matters if you want
to exercise the HTTP path (or replace it with a real model later).

```bash
python edge_server.py
# or:
uvicorn edge_server:app --port 8001
```

Then `GET http://127.0.0.1:8001/health` should return `{"status": "ok"}`.

## Run a single benchmark

```bash
python benchmark.py --frames 1000 --deadline-ms 100 \
    --profile variable --mode dapper \
    --output results/run_variable.csv
```

Available profiles: `stable`, `congested`, `lossy`, `variable`, `outage`.
Available modes: `local_only`, `edge_only`, `cloud_only`, `dapper`.

## Run all modes against all profiles

```bash
python benchmark.py run-all --frames 1000 --deadline-ms 100 --out-dir results
```

This writes:

- `results/all_runs.csv` – every per-frame row from every (profile, mode) run
- `results/summary.csv`  – one summary row per (profile, mode)
- `results/*.png`        – matplotlib plots, per profile

## Optional video mode

```bash
python benchmark.py --frames 500 --mode dapper \
    --profile variable --video data/sample.mp4 \
    --output results/run_video.csv
```

This reads frames with OpenCV but still simulates the inference call – the
prototype never claims to run a real detector.

## Interpreting the output

`results/summary.csv` has one row per (profile, mode) combination with these
columns:

| column                    | meaning                                              |
|---------------------------|------------------------------------------------------|
| `mean_latency_ms`         | Average end-to-end perception latency                |
| `p95_latency_ms` / `p99`  | Tail latency                                         |
| `deadline_miss_rate`      | Fraction of frames whose latency exceeded the deadline |
| `mean_confidence`         | Average detector confidence proxy                    |
| `stale_output_rate`       | Fraction of frames served from last-valid cache      |
| `fallback_rate`           | Fraction of frames where a fallback path was used    |
| `mode_switch_count`       | Times DAPPER changed mode between adjacent frames    |
| `total_bandwidth_kb`      | Total simulated bytes sent for inference offload     |
| `bandwidth_per_frame_kb`  | Mean bandwidth per frame                             |
| `reliability_score`       | Composite: `(1 - miss) * conf * (1 - 0.5*stale)`     |

Expected qualitative findings the prototype is designed to demonstrate:

- DAPPER **reduces deadline misses** compared with fixed `edge_only` and
  `cloud_only` perception under `congested`, `lossy`, and `variable`
  profiles.
- DAPPER **preserves higher confidence than `local_only`** when network
  conditions allow it to use `edge_accurate` or `hybrid`.
- DAPPER **uses `degraded_safe` during the `outage` profile** instead of
  returning stale remote results.
- DAPPER **reduces bandwidth** versus `edge_only` and `cloud_only` because
  it only offloads frames when the network budget actually permits it.

## Per-frame CSV schema

| column            | type    | meaning                                         |
|-------------------|---------|-------------------------------------------------|
| `frame_id`        | int     | Sequential frame number                         |
| `timestamp`       | float   | Virtual clock in ms (30 FPS)                    |
| `run_mode`        | str     | Policy under test (e.g. `dapper`)               |
| `profile`         | str     | Network profile name                            |
| `selected_mode`   | str     | Perception mode actually used for this frame    |
| `rtt_ms`          | float   | Sampled round trip time                         |
| `packet_loss`     | float   | Sampled packet loss probability                 |
| `edge_load`       | float   | Simulated edge server load 0..1                 |
| `edge_available`  | bool    | False during simulated outage windows           |
| `deadline_ms`     | float   | Per-frame perception deadline                   |
| `latency_ms`      | float   | Simulated end-to-end perception latency         |
| `deadline_met`    | bool    | `latency_ms <= deadline_ms`                     |
| `confidence`      | float   | Detector confidence proxy                       |
| `stale_output`    | bool    | Reused a previous last-valid result             |
| `fallback_used`   | bool    | Any fallback path triggered                     |
| `bandwidth_kb`    | float   | Bytes sent for this frame                       |
| `risk_score`      | float   | Scheduler's risk in [0, 1]                      |
| `decision_reason` | str     | Why this mode was chosen                        |
| `source`          | str     | `local`, `edge`, `cloud`, `last_valid`, ...     |
| `detections`      | int     | Number of synthetic detections                  |

## Tests

```bash
pytest tests
```

The test suite exercises the scheduler's decision boundaries (stable network
prefers edge/hybrid, high RTT/loss steers to local/degraded, outage forces
degraded-safe, deadline pressure increases risk, etc.).

## Dashboard API (optional)

```bash
uvicorn app:app --port 8000
```

Endpoints:

- `GET  /status` – service status and available modes/profiles
- `GET  /last-run` – head of the most recent per-frame CSV
- `GET  /metrics` – summary CSV as JSON
- `POST /run-benchmark` – run a benchmark synchronously

## Mapping to the DAPPER research paper

This prototype mirrors the paper's contributions one-to-one:

| Paper concept                                | Code location                              |
|----------------------------------------------|--------------------------------------------|
| Mission-critical deadline model              | `deadline_ms` everywhere                   |
| Runtime monitor                              | `monitor.py` (5 profiles, AR(1) smoothing) |
| Risk-driven placement decision               | `scheduler.py` (`compute_risk` + `decide`) |
| Local / edge / hybrid / degraded-safe        | `perception.py` (`*_infer`, `degraded_safe`) |
| Fixed baselines (local/edge/cloud only)      | `benchmark.py` (`_choose_fixed_mode`)      |
| Metrics: tail latency, miss rate, reliability | `metrics.py`                              |
| Reproducible experiment harness              | `benchmark.py run-all` + seeded RNG        |

## Limitations

- **Synthetic only.** The numbers come from distributions in `config.yaml`,
  not measured robot hardware. A real evaluation would replace
  `perception.py` and `edge_server.py` with measured timings from a robot +
  Jetson + WAN link, and feed the same scheduler.
- **No real detector.** We never load YOLO or any model. Confidence values
  are sampled from configurable bands.
- **Single robot.** The scheduler treats edge load as an exogenous signal;
  multi-robot contention would need an extension.
- **No actuation loop.** We measure perception latency, not control-loop
  stability. A future extension could close the loop.

## License

This is research code. Use freely for academic comparison; ground-truth your
claims with measured numbers before deploying to a robot.
