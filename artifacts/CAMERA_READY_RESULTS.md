# DAPPER camera-ready results

Every number in this document was produced by the scripts in `experiments/` and can be traced to a CSV under `results/`. Nothing here was typed by hand. Statements that the data does not support are marked as such rather than omitted.

## 1. Experimental protocol

* **Repetition.** 30 independent evaluation seeds (100-129), 1000 frames per (seed, profile, policy) cell.
* **Coverage.** 5 network profiles x 8 policies x 30 seeds = 1200 runs = 1,200,000 frame-level decisions at D = 100 ms.
* **Paired scenarios.** For every (seed, profile) an immutable scenario trace is generated in advance from 15 independent random streams: the network state, and the *potential* local, edge and cloud outcomes (latency, confidence, detections, loss draw) of every frame. All policies replay the identical trace, so a policy's decision selects which potential outcome is realised but cannot change the random future. Comparisons are exactly paired at the frame level. `results/final/scenario_fingerprints.csv` records a content hash of every trace.
* **Parameter selection.** Weights and thresholds were selected on calibration seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] only, which are disjoint from the evaluation seeds; `experiments/_common.assert_disjoint` fails the run otherwise. The search was a deterministic random search over 400 joint candidates scored by a predeclared lexicographic objective. The selection is frozen in `config.yaml` and no parameter was re-selected on evaluation data.
* **Statistics.** The independent unit of replication is the **seed**, not the frame: frames within a run are autocorrelated (AR(1) RTT, 5-25 frame outage bursts, carried-over last-valid state), so treating 30,000 frames as independent samples would manufacture precision. Each seed contributes one scalar per metric and all intervals are percentile bootstrap over seeds (10,000 resamples, fixed bootstrap seed). Policy pairs are compared seed-by-seed because they replay identical scenarios.
* **Timing model.** A per-frame independent timing model with carried-over last-valid state, not an event-driven simulation: frames do not queue, but an output produced for frame *t* becomes available only at capture(*t*) + latency and ages from its own capture instant. Every transmitted offload is charged its upload bandwidth, including requests that are lost and replies that arrive too late to accept; a lost or overdue reply costs a deadline-bounded client timeout before the local fallback runs.
* **Environment.** See `artifacts/ENVIRONMENT.md`.

## 2. Main repeated-measures table

Source: `results/final/multi_seed_ci.csv`. Confidence proxy is the mean per-frame confidence the control loop holds at the deadline under the **synthetic** perception model; it is not detector accuracy.

| Profile | Policy | p95 control latency (ms) | Deadline miss (%) | Confidence proxy | Bandwidth (MB / 1000 frames) |
|---|---|---|---|---|---|
| stable | local_only | 34.0 [95% CI: 33.9-34.0] | 0.00 [95% CI: 0.00-0.00] | 0.7250 [95% CI: 0.7244-0.7255] | 0.00 |
| stable | edge_only | 132.8 [95% CI: 132.7-132.9] | 48.77 [95% CI: 48.17-49.38] | 0.4483 [95% CI: 0.4429-0.4535] | 25.00 |
| stable | cloud_only | 134.0 [95% CI: 133.9-134.0] | 100.00 [95% CI: 100.00-100.00] | 0.0000 [95% CI: 0.0000-0.0000] | 35.00 |
| stable | dapper | 34.0 [95% CI: 33.9-34.0] | 0.00 [95% CI: 0.00-0.00] | 0.7960 [95% CI: 0.7949-0.7970] | 21.62 |
| congested | local_only | 33.9 [95% CI: 33.9-34.0] | 0.00 [95% CI: 0.00-0.00] | 0.7249 [95% CI: 0.7244-0.7255] | 0.00 |
| congested | edge_only | 133.9 [95% CI: 133.9-134.0] | 100.00 [95% CI: 100.00-100.00] | 0.0000 [95% CI: 0.0000-0.0000] | 25.00 |
| congested | cloud_only | 133.9 [95% CI: 133.9-134.0] | 100.00 [95% CI: 100.00-100.00] | 0.0000 [95% CI: 0.0000-0.0000] | 35.00 |
| congested | dapper | 33.9 [95% CI: 33.9-34.0] | 0.00 [95% CI: 0.00-0.00] | 0.7250 [95% CI: 0.7244-0.7255] | 0.00 |
| lossy | local_only | 34.0 [95% CI: 34.0-34.1] | 0.00 [95% CI: 0.00-0.00] | 0.7248 [95% CI: 0.7243-0.7253] | 0.00 |
| lossy | edge_only | 133.9 [95% CI: 133.8-133.9] | 87.51 [95% CI: 87.06-87.94] | 0.1093 [95% CI: 0.1055-0.1133] | 25.00 |
| lossy | cloud_only | 134.0 [95% CI: 134.0-134.1] | 100.00 [95% CI: 100.00-100.00] | 0.0000 [95% CI: 0.0000-0.0000] | 35.00 |
| lossy | dapper | 34.0 [95% CI: 34.0-34.1] | 0.00 [95% CI: 0.00-0.00] | 0.7406 [95% CI: 0.7398-0.7414] | 18.13 |
| variable | local_only | 34.0 [95% CI: 33.9-34.0] | 0.00 [95% CI: 0.00-0.00] | 0.7253 [95% CI: 0.7248-0.7259] | 0.00 |
| variable | edge_only | 133.6 [95% CI: 133.5-133.7] | 74.00 [95% CI: 72.62-75.43] | 0.1948 [95% CI: 0.1843-0.2048] | 19.55 |
| variable | cloud_only | 133.7 [95% CI: 133.6-133.7] | 78.20 [95% CI: 76.76-79.62] | 0.1580 [95% CI: 0.1477-0.1684] | 27.37 |
| variable | dapper | 33.7 [95% CI: 33.7-33.8] | 0.00 [95% CI: 0.00-0.00] | 0.7315 [95% CI: 0.7306-0.7324] | 5.78 |
| outage | local_only | 34.0 [95% CI: 34.0-34.1] | 0.00 [95% CI: 0.00-0.00] | 0.7249 [95% CI: 0.7244-0.7254] | 0.00 |
| outage | edge_only | 128.1 [95% CI: 127.7-128.4] | 14.68 [95% CI: 14.06-15.33] | 0.6187 [95% CI: 0.6140-0.6232] | 3.67 |
| outage | cloud_only | 128.1 [95% CI: 127.7-128.4] | 14.68 [95% CI: 14.06-15.33] | 0.6187 [95% CI: 0.6140-0.6232] | 5.14 |
| outage | dapper | 31.2 [95% CI: 31.0-31.4] | 0.00 [95% CI: 0.00-0.00] | 0.7255 [95% CI: 0.7245-0.7265] | 0.00 |

**DAPPER produced a 0.00 % deadline-miss rate in all 5 profiles across all 30 seeds** (150,000 DAPPER frames). Every per-seed miss rate was exactly zero, so the bootstrap interval is degenerate at [0.00, 0.00].

### DAPPER mode distribution

| Profile | local-fast % | edge-accurate % | hybrid % | degraded-safe % | remote attempts / frame | remote accept % | reuse % |
|---|---|---|---|---|---|---|---|
| congested | 99.2 | 0.0 | 0.0 | 0.7 | 0.000 | 0.0 | 0.7 |
| lossy | 27.5 | 0.0 | 72.5 | 0.0 | 0.725 | 13.7 | 0.0 |
| outage | 22.6 | 0.0 | 0.0 | 77.4 | 0.000 | 0.0 | 73.4 |
| stable | 13.5 | 0.0 | 86.5 | 0.0 | 0.865 | 51.2 | 0.0 |
| variable | 60.1 | 0.0 | 23.1 | 16.8 | 0.231 | 15.9 | 16.8 |

`edge_accurate` receives **zero frames at D = 100 ms**, exactly as in the published prototype. Section 5 shows this is a consequence of the configured timing envelope and not dead code: the mode is selected once the control budget is large enough to admit a committed remote round trip.

## 3. Comparison with local-only and with adaptive baselines

Source: `results/final/paired_comparisons.csv` (paired over seeds; each policy replayed the identical scenario traces).

### DAPPER versus local-only, confidence proxy

| Profile | local-only | DAPPER | paired difference [95% CI] | interval excludes zero |
|---|---|---|---|---|
| stable | 0.7250 | 0.7960 | +0.0710 [+0.0700, +0.0720] | yes |
| congested | 0.7249 | 0.7250 | +0.0000 [-0.0000, +0.0001] | no |
| lossy | 0.7248 | 0.7406 | +0.0158 [+0.0151, +0.0165] | yes |
| variable | 0.7253 | 0.7315 | +0.0062 [+0.0056, +0.0068] | yes |
| outage | 0.7249 | 0.7255 | +0.0006 [-0.0002, +0.0014] | no |
| ALL | 0.7250 | 0.7437 | +0.0187 [+0.0184, +0.0191] | yes |

DAPPER produced a higher confidence proxy than local-only under **stable, lossy, variable**, with paired 95 % intervals that exclude zero. Under **congested, outage** the interval includes zero, i.e. the data does not show a difference. This is the direct answer to the reviewer observation that local-only already achieves zero deadline misses: the two policies are equally deadline-safe, and the question is what perception quality each delivers at that reliability.

### Bandwidth cost of that quality

| Profile | local-only | DAPPER | edge-only | DAPPER as % of edge-only |
|---|---|---|---|---|
| stable | 0.00 | 21.62 | 25.00 | 86 % |
| congested | 0.00 | 0.00 | 25.00 | 0 % |
| lossy | 0.00 | 18.13 | 25.00 | 73 % |
| variable | 0.00 | 5.78 | 19.55 | 30 % |
| outage | 0.00 | 0.00 | 3.67 | 0 % |

This is the headline change from the published artifact. In the published prototype an offload attempt whose reply was lost or arrived too late was billed **no** bandwidth, which is what produced the reported 25 KB under the variable profile. With every transmitted request charged - which is what the manuscript already claims the system does - DAPPER's offload traffic is a substantial fraction of edge-only's, not a rounding error. The 25 KB figure cannot be carried into the camera-ready paper.

### Adaptive baselines

| Profile | Policy | Miss % | p95 ms | Confidence proxy | MB / 1000 frames |
|---|---|---|---|---|---|
| stable | local_only | 0.00 | 34.0 | 0.7250 | 0.00 |
| stable | deadline_greedy | 0.00 | 34.0 | 0.7250 | 0.00 |
| stable | confidence_deadline | 0.00 | 34.0 | 0.7250 | 0.00 |
| stable | rtt_threshold | 0.00 | 34.0 | 0.7250 | 0.00 |
| stable | dapper | 0.00 | 34.0 | 0.7960 | 21.62 |
| stable | oracle_feasible | 0.00 | 97.5 | 0.8018 | 12.81 |
| congested | local_only | 0.00 | 33.9 | 0.7249 | 0.00 |
| congested | deadline_greedy | 0.00 | 33.9 | 0.7249 | 0.00 |
| congested | confidence_deadline | 0.00 | 33.9 | 0.7249 | 0.00 |
| congested | rtt_threshold | 0.00 | 33.9 | 0.7249 | 0.00 |
| congested | dapper | 0.00 | 33.9 | 0.7250 | 0.00 |
| congested | oracle_feasible | 0.00 | 33.9 | 0.7249 | 0.00 |
| lossy | local_only | 0.00 | 34.0 | 0.7248 | 0.00 |
| lossy | deadline_greedy | 0.00 | 34.0 | 0.7248 | 0.00 |
| lossy | confidence_deadline | 0.00 | 34.0 | 0.7248 | 0.00 |
| lossy | rtt_threshold | 0.00 | 34.0 | 0.7248 | 0.00 |
| lossy | dapper | 0.00 | 34.0 | 0.7406 | 18.13 |
| lossy | oracle_feasible | 0.00 | 96.3 | 0.7435 | 3.12 |
| variable | local_only | 0.00 | 34.0 | 0.7253 | 0.00 |
| variable | deadline_greedy | 0.00 | 34.0 | 0.7253 | 0.00 |
| variable | confidence_deadline | 0.00 | 34.0 | 0.7253 | 0.00 |
| variable | rtt_threshold | 0.02 | 34.0 | 0.7252 | 0.02 |
| variable | dapper | 0.00 | 33.7 | 0.7315 | 5.78 |
| variable | oracle_feasible | 0.00 | 42.0 | 0.7317 | 1.05 |
| outage | local_only | 0.00 | 34.0 | 0.7249 | 0.00 |
| outage | deadline_greedy | 0.00 | 34.0 | 0.7249 | 0.00 |
| outage | confidence_deadline | 0.00 | 34.0 | 0.7249 | 0.00 |
| outage | rtt_threshold | 0.00 | 34.0 | 0.7249 | 0.00 |
| outage | dapper | 0.00 | 31.2 | 0.7255 | 0.00 |
| outage | oracle_feasible | 0.00 | 34.0 | 0.7249 | 0.00 |

All three adaptive baselines were fitted on the calibration seeds with the same predeclared objective used for DAPPER, so none is a strawman. At D = 100 ms each selected a setting under which it barely offloads: deadline-greedy and confidence-and-deadline transmit on 0.000 % of frames (identical to local-only) and RTT-threshold on 0.015 %. `results/calibration/baseline_candidates.csv` shows why: the least conservative setting that still reaches a zero worst-profile miss rate offloads on essentially no frames, and the next setting up jumps to a double-digit miss rate. At this deadline any policy that *commits* a frame to the edge misses deadlines, because the configured edge round trip does not fit the budget. DAPPER is the only evaluated policy that still extracts remote quality there, because its hybrid mode never commits the frame: a local answer is always in hand and the remote reply is accepted only if it genuinely arrives in time.

### Distance from the offline oracle

* **stable**: local-only 0.7250, DAPPER 0.7960, oracle 0.8018. DAPPER captured 92 % of the gain available to a scheduler with perfect knowledge of each frame's outcome, using 21.62 MB / 1000 frames against the oracle's 12.81 MB.
* **congested**: the oracle achieves no measurable gain over local-only (0.7249 vs 0.7249), so there is nothing for any scheduler to capture under this profile.
* **lossy**: local-only 0.7248, DAPPER 0.7406, oracle 0.7435. DAPPER captured 85 % of the gain available to a scheduler with perfect knowledge of each frame's outcome, using 18.13 MB / 1000 frames against the oracle's 3.12 MB.
* **variable**: local-only 0.7253, DAPPER 0.7315, oracle 0.7317. DAPPER captured 98 % of the gain available to a scheduler with perfect knowledge of each frame's outcome, using 5.78 MB / 1000 frames against the oracle's 1.05 MB.
* **outage**: the oracle achieves no measurable gain over local-only (0.7249 vs 0.7249), so there is nothing for any scheduler to capture under this profile.

The oracle is an **offline upper bound**: it reads the realised outcome of the frame it is deciding about. It is not deployable and must never be presented as a baseline.

## 4. Ablation

Source: `results/ablation/`. Risk-signal variants set one weight to zero and renormalise the rest, so the score stays on [0, 1] and the thresholds keep their meaning. Each statement below is a paired seed-level comparison against the full scheduler on identical traces, and is emitted only when the 95 % bootstrap interval of the difference excludes zero.

## Risk-component ablation

* Under **congested**, `dapper_no_deadline` increased `p95_latency_ms` from 33.94 ms to 33.94 ms (paired difference +0.01 ms, 95% CI [+0.00, +0.01], 30 seeds).
* Under **lossy**, `dapper_no_deadline` increased `usable_confidence_proxy` from 0.7406 to 0.7419 (paired difference +0.0014, 95% CI [+0.0012, +0.0015], 30 seeds).
* Under **lossy**, `dapper_no_deadline` increased `bandwidth_per_1000_frames_kb` from 18126.7 KB/1000 frames to 21150.0 KB/1000 frames (paired difference +3023.3 KB/1000 frames, 95% CI [+2915.8, +3128.3], 30 seeds).
* Under **variable**, `dapper_no_deadline` increased `p95_latency_ms` from 33.74 ms to 33.75 ms (paired difference +0.00 ms, 95% CI [+0.00, +0.01], 30 seeds).
* Under **congested**, `dapper_no_load` increased `p95_latency_ms` from 33.94 ms to 33.94 ms (paired difference +0.00 ms, 95% CI [+0.00, +0.01], 30 seeds).
* Under **lossy**, `dapper_no_load` decreased `usable_confidence_proxy` from 0.7406 to 0.7376 (paired difference -0.0030, 95% CI [-0.0032, -0.0027], 30 seeds).
* Under **lossy**, `dapper_no_load` decreased `bandwidth_per_1000_frames_kb` from 18126.7 KB/1000 frames to 13292.5 KB/1000 frames (paired difference -4834.2 KB/1000 frames, 95% CI [-4980.0, -4685.0], 30 seeds).
* Under **congested**, `dapper_no_loss` decreased `p95_latency_ms` from 33.94 ms to 31.16 ms (paired difference -2.77 ms, 95% CI [-2.92, -2.63], 30 seeds).
* Under **lossy**, `dapper_no_loss` decreased `p95_latency_ms` from 34.01 ms to 33.99 ms (paired difference -0.02 ms, 95% CI [-0.03, -0.01], 30 seeds).
* Under **lossy**, `dapper_no_loss` decreased `usable_confidence_proxy` from 0.7406 to 0.7248 (paired difference -0.0158, 95% CI [-0.0165, -0.0152], 30 seeds).
* Under **lossy**, `dapper_no_loss` decreased `bandwidth_per_1000_frames_kb` from 18126.7 KB/1000 frames to 0.0 KB/1000 frames (paired difference -18126.7 KB/1000 frames, 95% CI [-18280.8, -17975.0], 30 seeds).
* Under **stable**, `dapper_no_loss` decreased `usable_confidence_proxy` from 0.7960 to 0.7675 (paired difference -0.0285, 95% CI [-0.0295, -0.0275], 30 seeds).
* Under **stable**, `dapper_no_loss` decreased `bandwidth_per_1000_frames_kb` from 21621.7 KB/1000 frames to 12323.3 KB/1000 frames (paired difference -9298.3 KB/1000 frames, 95% CI [-9530.8, -9055.0], 30 seeds).
* Under **variable**, `dapper_no_loss` decreased `p95_latency_ms` from 33.74 ms to 32.56 ms (paired difference -1.18 ms, 95% CI [-1.29, -1.07], 30 seeds).
* Under **variable**, `dapper_no_loss` decreased `usable_confidence_proxy` from 0.7315 to 0.7260 (paired difference -0.0056, 95% CI [-0.0064, -0.0046], 30 seeds).
* Under **variable**, `dapper_no_loss` decreased `bandwidth_per_1000_frames_kb` from 5777.5 KB/1000 frames to 172.5 KB/1000 frames (paired difference -5605.0 KB/1000 frames, 95% CI [-5815.0, -5395.0], 30 seeds).
* Under **congested**, `dapper_no_rtt` increased `p95_latency_ms` from 33.94 ms to 33.94 ms (paired difference +0.01 ms, 95% CI [+0.00, +0.01], 30 seeds).
* Under **lossy**, `dapper_no_rtt` decreased `usable_confidence_proxy` from 0.7406 to 0.7358 (paired difference -0.0048, 95% CI [-0.0051, -0.0045], 30 seeds).
* Under **lossy**, `dapper_no_rtt` decreased `bandwidth_per_1000_frames_kb` from 18126.7 KB/1000 frames to 13035.8 KB/1000 frames (paired difference -5090.8 KB/1000 frames, 95% CI [-5220.9, -4960.8], 30 seeds).
* Under **outage**, `dapper_no_rtt` increased `p95_latency_ms` from 31.19 ms to 31.46 ms (paired difference +0.27 ms, 95% CI [+0.09, +0.44], 30 seeds).
* Under **variable**, `dapper_no_rtt` increased `p95_latency_ms` from 33.74 ms to 33.75 ms (paired difference +0.00 ms, 95% CI [+0.00, +0.01], 30 seeds).

**Null results (reported because absence of an effect is itself a finding).** On every profile and every reported metric, the paired 95 % bootstrap interval of the difference from `dapper_full` included zero for: `dapper_no_frame_age`.

## Functional ablation

* Under **lossy**, `dapper_no_confidence_gate` increased `usable_confidence_proxy` from 0.7406 to 0.7420 (paired difference +0.0014, 95% CI [+0.0013, +0.0016], 30 seeds).
* Under **lossy**, `dapper_no_confidence_gate` increased `bandwidth_per_1000_frames_kb` from 18126.7 KB/1000 frames to 20925.8 KB/1000 frames (paired difference +2799.2 KB/1000 frames, 95% CI [+2698.3, +2901.7], 30 seeds).
* Under **stable**, `dapper_no_confidence_gate` increased `usable_confidence_proxy` from 0.7960 to 0.8018 (paired difference +0.0058, 95% CI [+0.0056, +0.0061], 30 seeds).
* Under **stable**, `dapper_no_confidence_gate` increased `bandwidth_per_1000_frames_kb` from 21621.7 KB/1000 frames to 25000.0 KB/1000 frames (paired difference +3378.3 KB/1000 frames, 95% CI [+3264.2, +3500.8], 30 seeds).
* Under **variable**, `dapper_no_confidence_gate` increased `usable_confidence_proxy` from 0.7315 to 0.7320 (paired difference +0.0004, 95% CI [+0.0004, +0.0005], 30 seeds).
* Under **variable**, `dapper_no_confidence_gate` increased `bandwidth_per_1000_frames_kb` from 5777.5 KB/1000 frames to 6705.8 KB/1000 frames (paired difference +928.3 KB/1000 frames, 95% CI [+875.0, +985.0], 30 seeds).
* Under **outage**, `dapper_no_degraded_reuse` increased `p95_latency_ms` from 31.19 ms to 33.85 ms (paired difference +2.67 ms, 95% CI [+2.51, +2.82], 30 seeds).
* Under **variable**, `dapper_no_degraded_reuse` increased `p95_latency_ms` from 33.74 ms to 33.97 ms (paired difference +0.22 ms, 95% CI [+0.18, +0.27], 30 seeds).
* Under **congested**, `dapper_no_freshness_gate` increased `bandwidth_per_1000_frames_kb` from 2.5 KB/1000 frames to 150.0 KB/1000 frames (paired difference +147.5 KB/1000 frames, 95% CI [+120.0, +179.2], 30 seeds).
* Under **lossy**, `dapper_no_freshness_gate` increased `bandwidth_per_1000_frames_kb` from 18126.7 KB/1000 frames to 18267.5 KB/1000 frames (paired difference +140.8 KB/1000 frames, 95% CI [+120.8, +161.7], 30 seeds).
* Under **variable**, `dapper_no_freshness_gate` increased `bandwidth_per_1000_frames_kb` from 5777.5 KB/1000 frames to 9745.8 KB/1000 frames (paired difference +3968.3 KB/1000 frames, 95% CI [+3840.0, +4102.5], 30 seeds).


## 5. Deadline sensitivity

Source: `results/sensitivity/deadline_mode_distribution.csv` (pooled over the five profiles and all evaluation seeds).

| D (ms) | local-fast % | edge-accurate % | hybrid % | degraded-safe % | Miss % | p95 (ms) | Confidence proxy | MB / 1000 frames |
|---|---|---|---|---|---|---|---|---|
| 50 | 48.4 | 0.0 | 0.0 | 51.6 | 0.00 | 32.6 | 0.7251 | 0.00 |
| 75 | 57.4 | 0.0 | 10.6 | 32.1 | 0.00 | 33.2 | 0.7258 | 2.64 |
| 100 | 44.6 | 0.0 | 36.4 | 19.0 | 0.00 | 33.4 | 0.7437 | 9.11 |
| 125 | 42.4 | 11.3 | 27.5 | 18.8 | 5.24 | 56.4 | 0.7066 | 9.70 |
| 150 | 42.4 | 32.0 | 6.9 | 18.8 | 13.31 | 108.6 | 0.6488 | 9.71 |
| 200 | 26.1 | 55.3 | 0.0 | 18.6 | 4.21 | 152.6 | 0.7140 | 13.83 |
| 250 | 16.0 | 65.5 | 0.0 | 18.4 | 0.00 | 156.3 | 0.7441 | 16.38 |

`edge_accurate` first receives frames at **D = 125 ms** (11.3 % of frames) and reaches 65.5 % at D = 250 ms. Its absence at D = 100 ms is therefore a consequence of the configured timing envelope - the expected edge round trip does not fit the budget - and not unreachable code.

**Unfavourable result, reported in full.** DAPPER is *not* deadline-safe at every deadline: with the frozen configuration its miss rate is 5.24 % at D = 125 ms, 13.31 % at D = 150 ms, 4.21 % at D = 200 ms. The cause is identified in `results/calibration/calibration_report.md`: `remote_deadline_margin`, the gate on the *committing* `edge_accurate` mode, is **not identified** by a calibration performed at D = 100 ms, because at that deadline no frame passes the commit test under any margin, so every candidate scores identically on it. The value the search happened to select is permissive enough to admit commitments whose realised completion can overrun an intermediate deadline, and a committed frame has no local answer to fall back on.

### Commit-margin diagnostic

Source: `results/sensitivity/commit_margin_sweep.csv`. This is a sensitivity analysis on evaluation seeds, **not** a re-selection: its result is deliberately not fed back into `config.yaml`, because choosing a parameter on evaluation data is exactly what the calibration/evaluation split exists to prevent.

| D (ms) | largest margin with 0.00 % miss | miss at the frozen margin (0.8) |
|---|---|---|
| 100 | 0.8 | 0.00 % |
| 125 | 0.6 | 5.24 % |
| 150 | 0.6 | 13.31 % |
| 200 | 0.7 | 4.21 % |

A commit margin of at most **0.6** keeps the deadline-miss rate at 0.00 % at every deadline in this sweep. Reporting this is the honest response to the finding; acting on it would require a fresh calibration on the calibration seeds across several deadlines, which is named as future work rather than done post hoc.

## 6. Robustness to inaccurate runtime estimates

Source: `results/sensitivity/runtime_estimation_error_ci.csv`. The scheduler's RTT and edge-compute estimates are multiplied by (1 + bias); the executed scenario is unchanged, so only the scheduler's beliefs are wrong. A **negative** bias means the scheduler underestimates the true cost, which is the dangerous direction. Profiles: stable, congested, variable.

Values below are averaged over the three profiles; the summary table that follows uses the **worst** profile, because a policy is only as reliable as its worst case.

| Biased quantity | Bias | Miss % (profile mean) | p95 (ms) | Confidence proxy | MB / 1000 frames |
|---|---|---|---|---|---|
| rtt | -30 % | 0.00 | 33.9 | 0.7508 | 12.11 |
| rtt | -20 % | 0.00 | 33.9 | 0.7508 | 10.78 |
| rtt | -10 % | 0.00 | 33.9 | 0.7508 | 9.83 |
| rtt | +0 % | 0.00 | 33.9 | 0.7508 | 9.13 |
| rtt | +10 % | 0.00 | 33.9 | 0.7507 | 8.60 |
| rtt | +20 % | 0.00 | 33.8 | 0.7504 | 8.21 |
| rtt | +30 % | 0.00 | 33.6 | 0.7502 | 7.94 |
| compute | -30 % | 13.13 | 66.8 | 0.6570 | 10.51 |
| compute | -20 % | 0.28 | 34.0 | 0.7488 | 10.49 |
| compute | -10 % | 0.01 | 33.9 | 0.7508 | 9.95 |
| compute | +0 % | 0.00 | 33.9 | 0.7508 | 9.13 |
| compute | +10 % | 0.00 | 33.9 | 0.7506 | 8.39 |
| compute | +20 % | 0.00 | 33.9 | 0.7501 | 7.85 |
| compute | +30 % | 0.00 | 33.9 | 0.7496 | 7.52 |
| both | -30 % | 15.29 | 80.2 | 0.6415 | 16.07 |
| both | -20 % | 8.73 | 66.2 | 0.6884 | 13.09 |
| both | -10 % | 0.01 | 33.9 | 0.7508 | 10.83 |
| both | +0 % | 0.00 | 33.9 | 0.7508 | 9.13 |
| both | +10 % | 0.00 | 33.9 | 0.7503 | 8.03 |
| both | +20 % | 0.00 | 33.8 | 0.7495 | 7.53 |
| both | +30 % | 0.00 | 33.6 | 0.7467 | 6.45 |

**The response is asymmetric, and the asymmetry is the finding.**

| Biased quantity | Bias range with 0.00 % worst-profile misses | Worst-profile miss rate at -10 % | Worst-profile miss rate over the whole sweep | at bias |
|---|---|---|---|---|
| rtt | [-30 %, +30 %] | 0.000 % | 0.00 % | -30 % |
| compute | [+0 %, +30 %] | 0.020 % | 38.55 % | -30 % |
| both | [+0 %, +30 %] | 0.020 % | 42.22 % | -30 % |

* **Over**estimating the remote cost is harmless at every bias swept: the scheduler becomes more conservative and simply offloads less.
* **Under**estimating the round-trip time is also harmless across the whole swept range.
* **Under**estimating the *edge compute time* is the dangerous direction. The mechanism is the one the deadline sweep also exposed: an underestimated remote completion passes the commit test, so the scheduler selects the committing `edge_accurate` mode, and a committed frame has no local answer to fall back on when the reply turns out to be slow. Hybrid is never implicated, because a hybrid frame always holds a local result.

The degradation is graded rather than a cliff: a 10 % under-estimate of edge compute costs a worst-profile miss rate of well under a tenth of a percent, a 20 % under-estimate costs single-digit percent, and a 30 % under-estimate is severe.

The supportable statement is therefore bounded and directional: biasing the round-trip-time estimate left the deadline-miss rate at exactly 0.00 % for every bias in [-30 %, +30 %]; biasing the edge-compute estimate left the deadline-miss rate at exactly 0.00 % for every bias in [+0 %, +30 %]; biasing both estimates jointly left the deadline-miss rate at exactly 0.00 % for every bias in [+0 %, +30 %]. No robustness is claimed outside the range actually swept, and no single symmetric tolerance is claimed, because the data does not show one.

## 7. Scheduling and switching overhead

Source: `results/overhead/`. Over 200,000 timed calls after warm-up, one `DapperScheduler.decide()` call took **mean 1.45 us, median 1.40 us, p95 1.80 us, p99 2.00 us** (691,118 decisions/s) on the machine in `artifacts/ENVIRONMENT.md`.

This is the **software** cost of the policy in CPython on a desktop CPU. It is not a robot-hardware measurement, and it is not the simulated application latency reported everywhere else in this document. The two must not be mixed.

End-to-end benchmark wall time per frame (same distinction applies):

| Policy | wall time per frame |
|---|---|
| local_only | 4.90 us |
| edge_only | 5.98 us |
| dapper | 6.20 us |

Running DAPPER instead of the fixed local policy added 1.31 us per frame, consistent with the per-decision measurement above.

Mode-switch rate (DAPPER, per 1000 frames, mean +/- sd over seeds):

| Profile | switches / 1000 frames |
|---|---|
| congested | 12.1 +/- 4.0 |
| lossy | 363.3 +/- 20.1 |
| outage | 418.4 +/- 10.5 |
| stable | 233.3 +/- 23.6 |
| variable | 315.4 +/- 16.7 |

Switching is a logical change of execution site between consecutive frames. No physical reconfiguration cost is modelled, so these counts must not be converted into a latency or energy figure.

## 8. Secondary real-detector validation

**Scope.** Two real detectors were run over the full 5,000-image COCO val2017 split on one workstation. This is a secondary validation of the *quality gap* between a lightweight and a large detector, and of what each policy delivers when that gap is real. It is **not** a physical robot deployment, not embedded (Jetson-class) timing and not a measured wireless link.

Source: `results/real_detector/`.

| Model | role | recall | precision | safety-class recall | mAP50 | mAP50-95 | GPU inference (ms) | CPU inference (ms) |
|---|---|---|---|---|---|---|---|---|
| yolo11n.pt | local | 0.507 | 0.704 | 0.623 | 0.546 | 0.389 | 6.1 | 35.2 |
| yolo11m.pt | remote | 0.638 | 0.730 | 0.734 | 0.678 | 0.510 | 8.1 | 177.5 |

The large detector detected 13.1 percentage points more ground-truth objects overall and 11.1 points more safety-relevant objects (person, bicycle, car, motorcycle, bus, truck) than the lightweight one, at IoU 0.50 and confidence 0.25. This is the quality the scheduler is trading against timeliness, measured rather than assumed.

### Replay of every policy over the measured detector outputs

The scheduler sees the confidence YOLO11n actually produced for each image, and the compute times both detectors actually took; the network remains the seeded synthetic profile. `delivered recall` is the fraction of the current image's ground-truth objects detected by the output the control loop holds at the deadline, counting a missed deadline as zero. Two views are given because consecutive COCO validation images are unrelated, so a *reused* output contains no detections of the current image: `all frames` scores reuse as zero (a lower bound that a real video stream would not incur), and `fresh frames` averages only over frames that computed a new output.

| Profile | Policy | recall (all fr.) | recall (fresh fr.) | safety recall (all fr.) | Miss % | reuse % | MB / 1000 frames |
|---|---|---|---|---|---|---|---|
| stable | local_only | 0.639 | 0.639 | 0.737 | 0.00 | 0.0 | 0.00 |
| stable | edge_only | 0.740 | 0.747 | 0.815 | 0.96 | 0.0 | 25.00 |
| stable | dapper | 0.674 | 0.674 | 0.762 | 0.00 | 0.0 | 5.59 |
| stable | oracle_feasible | 0.734 | 0.734 | 0.814 | 0.00 | 0.0 | 21.07 |
| congested | local_only | 0.639 | 0.639 | 0.737 | 0.00 | 0.0 | 0.00 |
| congested | edge_only | 0.698 | 0.747 | 0.769 | 6.54 | 0.0 | 25.00 |
| congested | dapper | 0.634 | 0.639 | 0.730 | 0.00 | 0.9 | 0.00 |
| congested | oracle_feasible | 0.729 | 0.729 | 0.810 | 0.00 | 0.0 | 19.89 |
| lossy | local_only | 0.639 | 0.639 | 0.737 | 0.00 | 0.0 | 0.00 |
| lossy | edge_only | 0.600 | 0.747 | 0.661 | 19.74 | 0.0 | 25.00 |
| lossy | dapper | 0.663 | 0.663 | 0.755 | 0.00 | 0.0 | 4.68 |
| lossy | oracle_feasible | 0.717 | 0.717 | 0.800 | 0.00 | 0.0 | 17.08 |
| variable | local_only | 0.639 | 0.639 | 0.737 | 0.00 | 0.0 | 0.00 |
| variable | edge_only | 0.680 | 0.721 | 0.757 | 5.66 | 0.0 | 19.18 |
| variable | dapper | 0.545 | 0.649 | 0.626 | 0.00 | 16.0 | 1.47 |
| variable | oracle_feasible | 0.707 | 0.707 | 0.792 | 0.00 | 0.0 | 15.14 |
| outage | local_only | 0.639 | 0.639 | 0.737 | 0.00 | 0.0 | 0.00 |
| outage | edge_only | 0.546 | 0.639 | 0.631 | 14.54 | 0.0 | 3.63 |
| outage | dapper | 0.223 | 0.637 | 0.262 | 0.00 | 65.0 | 0.00 |
| outage | oracle_feasible | 0.639 | 0.639 | 0.737 | 0.00 | 0.0 | 0.00 |

* **stable** (DAPPER never reused an output here, so both views coincide): DAPPER delivered 0.674 recall against local-only's 0.639 (+3.5 points) and 0.762 vs 0.737 safety recall, both at 0.00 % deadline misses, while edge-only reached 0.740 recall but missed 0.96 % of deadlines.
* **lossy** (DAPPER never reused an output here, so both views coincide): DAPPER delivered 0.663 recall against local-only's 0.639 (+2.4 points) and 0.755 vs 0.737 safety recall, both at 0.00 % deadline misses, while edge-only reached 0.600 recall but missed 19.74 % of deadlines.

Under `variable` and `outage`, DAPPER's all-frames recall falls below local-only's because it reuses outputs on 65 % and 16 % of frames respectively and every reuse is scored as zero. On the fresh-frame view DAPPER matches or exceeds local-only on those profiles. Which view is right depends on how much a slightly stale detection is worth in the target application, which this dataset cannot answer; both are therefore reported.

## 9. Limitations that remain

* **No physical robot.** Nothing in this package was run on a robot, a Jetson-class device or any embedded platform. No claim about on-robot timing, actuation or safety is supported.
* **No measured wireless trace.** The five network profiles are seeded, parameterised distributions. Trace-driven replay is *implemented* (`dapper/trace.py`, `tools/capture_network_trace.py`) and unit tested, but no Wi-Fi, 5G or private-edge capture was collected, so no result here rests on measured network data.
* **No energy measurement.** Energy is not instrumented anywhere in this package. Any statement about energy would be unsupported.
* **No safety validation.** `degraded-safe` is the name of a conservative fallback mode. It is not a certified safety property, there is no hazard analysis, and no formal guarantee is established.
* **Synthetic perception in the main benchmark.** The headline tables use a simulated confidence proxy. The measured detector study is secondary and covers perception quality only - its network side is still synthetic.
* **Single-machine detector timing.** Running a small and a large model on one workstation is not an edge deployment; it measures the model quality gap and a plausible CPU-versus-GPU compute gap, nothing more.
* **One deadline calibrated.** Parameters were selected at D = 100 ms. The deadline sweep shows the committing mode's margin is unsafe at some larger deadlines, and that parameter is not identifiable from a single-deadline calibration.
* **Bandwidth model.** Upload is charged per frame at a fixed size and treated as the dominant cost; downlink, headers and codec effects are not modelled.
