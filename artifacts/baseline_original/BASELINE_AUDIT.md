# Baseline audit — reproduction of the published DAPPER results

**Repository state audited:** git commit `e0a728e` (working tree clean).
**Date of audit:** 2026-09-20.
**Environment:** see [environment.txt](environment.txt).
**Commands:** see [commands.txt](commands.txt).

## 1. Test status of the frozen implementation

```
python -m pytest -q
..........                                                               [100%]
10 passed in 0.05s
```

## 2. Reproduction of the paper benchmark

Protocol re-run exactly as documented in the manuscript supplementary material
(S1): **500 frames, D = 100 ms, seed 42, 5 profiles x 4 policies = 10,000
frame-level decisions.**

```
python benchmark.py run-all --frames 500 --deadline-ms 100 --seed 42 \
    --out-dir artifacts/baseline_original
```

Artifacts: [summary.csv](summary.csv), [all_runs.csv](all_runs.csv).

### Verdict: the published numbers reproduce exactly.

The regenerated `summary.csv` is identical to the previously committed
`results/summary.csv`, and every headline value in Table III / Table S1 of the
manuscript is recovered:

| Paper claim | Paper value | Regenerated value | Match |
|---|---|---|---|
| DAPPER p95, stable | 78.2 ms | 78.1972 ms | yes |
| DAPPER miss, all profiles | 0.0 % | 0.0 % (all 5) | yes |
| edge-only miss, congested | 94.6 % | 94.6 % | yes |
| cloud-only miss, congested | 94.6 % | 94.6 % | yes |
| DAPPER p95, variable | 33.6 ms | 33.613 ms | yes |
| edge-only p95, variable | 166.3 ms | 166.270 ms | yes |
| cloud-only p95, variable | 290.2 ms | 290.179 ms | yes |
| DAPPER bandwidth, variable | 25 KB | 25.0 KB | yes |
| edge-only bandwidth, variable | 9,150 KB | 9,150.0 KB | yes |
| cloud-only bandwidth, variable | 12,810 KB | 12,810.0 KB | yes |
| DAPPER degraded-safe frames, outage | 452 / 500 | 452 / 500 | yes |

The published artifact is therefore **reproducible**. Everything that follows is
a correctness critique of *what* was computed, not of whether it can be
re-computed.

## 3. DAPPER mode distribution in the frozen implementation

Counts of `selected_mode` over 500 frames per profile (D = 100 ms, seed 42):

| Profile | local_fast | edge_accurate | hybrid | degraded_safe |
|---|---|---|---|---|
| stable | 0 | **0** | 500 | 0 |
| congested | 499 | **0** | 1 | 0 |
| lossy | 204 | **0** | 296 | 0 |
| variable | 241 | **0** | 165 | 94 |
| outage | 48 | **0** | 0 | 452 |

`edge_accurate` receives **zero frames in every profile** — confirmed by direct
count over `all_runs.csv`. One of the four advertised modes is never exercised
at the paper's operating point.

## 4. Confirmed implementation defects

Each item below was verified by reading the frozen source and by direct
inspection of `all_runs.csv`. Line references are to commit `e0a728e`.

### D1. `local_confidence` is declared but never used (CONFIRMED)

`SchedulerInputs.local_confidence` exists (`scheduler.py:40`) but appears
nowhere in `DapperScheduler.compute_risk()` or `DapperScheduler.decide()`
(`scheduler.py:124-225`). The manuscript (Sec. III and supplementary S5)
states the low-risk branch "keeps the frame local when the local model is
confident enough, and otherwise offloads". **The implementation does not do
this.** The scheduler is not confidence-aware.

### D2. `benchmark.py` feeds a constant confidence (CONFIRMED)

`benchmark.py:200-201` passes `0.5 * (confidence_min + confidence_max)` =
**0.725 for every frame**, so even if the scheduler consumed the field it would
carry no per-frame information.

### D3. `edge.base_rtt_ms` is ignored (CONFIRMED)

`benchmark.py:114`:
`extra_rtt = float(cfg[backend_section].get("base_rtt_ms", 0.0)) if use_cloud_latency else 0.0`.
The static access RTT is added **only** for `cloud_only`. `edge.base_rtt_ms: 20`
in `config.yaml` is dead configuration for every edge path — including DAPPER's
own offloads and the scheduler's prediction. The manuscript (Sec. IV-B) claims
"Each remote path is charged a static access RTT on top of profile-driven
jitter". **The implementation charges it for cloud only.**

### D4. Failed offloads are free (CONFIRMED)

`perception.edge_infer()` returns `None` on packet loss or unavailability
(`perception.py:107-110`) *before* any bandwidth is recorded, and
`benchmark._execute_mode()` then substitutes a local result whose
`bandwidth_kb` is 0.0 (`benchmark.py:123-127`). The manuscript (Sec. III, IV-B
and S6) claims "Every remote round trip is charged its configured access RTT
and upload bandwidth, including attempts that are lost or arrive too late to be
accepted." **The implementation charges none of them.**

The same applies to hybrid: `hybrid_infer()` discards the edge result when it
overshoots the freshness window and returns the local result with
`bandwidth_kb = 0` (`perception.py:158-167`). The effect is large and it
inflates DAPPER's headline bandwidth advantage:

* stable / DAPPER: 500 hybrid frames were executed, but only 1,550 KB was
  recorded = 62 charged offloads. **438 offload attempts were billed nothing.**
* variable / DAPPER: 165 hybrid frames, 25 KB recorded = **1 charged offload**.
  The paper's "25 KB vs 9,150 KB" comparison rests on this.

### D5. Failed offloads are also instantaneous (CONFIRMED)

Because `edge_infer()` returns `None` with no elapsed time, a lost request costs
only the subsequent local inference (15-35 ms). A real client cannot know a
packet was lost until a timer fires. No wait / timeout / RTT penalty is modelled.

### D6. Reused-but-fresh outputs are labelled stale (CONFIRMED)

`perception.degraded_safe()` returns `stale=True` on the reuse branch
(`perception.py:182-190`) — the branch that is only taken when
`last_valid_age_ms <= last_valid_freshness_ms`, i.e. when the output **is**
fresh by the system's own definition. `stale_output_rate` therefore measures
"reused" and not "expired": the reported 87.0 % stale rate under outage and
18.8 % under variable are reuse rates of *valid* outputs, and they are
propagated into `reliability_score` as a penalty (`metrics.py:92`).

### D7. Policies are not perfectly paired (CONFIRMED)

`build_monitor(..., seed=seed)` and `build_perception(..., seed=seed)` use two
separate generators, so the **network** trace is identical across policies for a
given seed. The **perception** stream is not: `local_infer()` consumes 3 draws,
`edge_infer()` consumes 4 (loss draw + 3), `hybrid_infer()` consumes 7, and
`degraded_safe()` consumes 0 or 3. Any policy that takes a different branch
desynchronises the perception RNG for all subsequent frames, so the "potential"
local/edge outcomes are **not** the same across policies. Comparisons are not
paired at the frame level.

### D8. Result availability time is not modelled (CONFIRMED)

`benchmark.py:258-259` sets `last_valid_time = virtual_clock_ms`, the frame
*capture* time, regardless of how long the result took to produce. An 80 ms edge
result is therefore treated as having existed at t=0 of its own frame. Ages fed
to the freshness gate are optimistic by up to one full inference latency.

### D9. Hybrid conflates two different latencies (CONFIRMED)

`hybrid_infer()` returns a single `PerceptionResult`. When the refresh is
accepted, `latency_ms` is the **edge** arrival time, even though a local answer
was in hand much earlier; when it is rejected, `latency_ms` is the local time
and the offload disappears from the record entirely. A single `latency_ms`
column cannot express "first usable output" and "final refined output", and the
deadline metric silently switches meaning between frames.

### D10. Single seed, no repetition (CONFIRMED)

Every published number comes from `seed = 42`. There are no repeated trials,
no dispersion estimate and no confidence intervals.

### D11. README / paper protocol mismatch (CONFIRMED)

`README.md` documents `--frames 1000`; `config.yaml` defaults to `frames: 1000`;
the published results and the manuscript use 500. The exact published protocol
was not pinned in the repository.

### D12. Weights and thresholds are unjustified (CONFIRMED)

`config.yaml` sets `weight_rtt: 0.35, weight_loss: 0.30, weight_load: 0.15,
weight_frame_age: 0.05, weight_deadline: 0.15` and thresholds `0.40 / 0.70`
with no recorded selection procedure, no validation that weights sum to 1
(they do, but nothing enforces it) and no sensitivity analysis.

## 5. Net effect on the published claims

| Published claim | Status after audit |
|---|---|
| "DAPPER achieved 0.0 % deadline misses in every profile" | Reproduces, but D9 means the latency being compared is not consistently defined, and D5 makes failed offloads unrealistically cheap. |
| "DAPPER used 25 KB vs 9,150 KB / 12,810 KB (variable)" | **Artefact of D4.** 164 of 165 offload attempts were billed nothing. |
| "Every remote round trip is charged access RTT and upload bandwidth" | **False in the code** (D3, D4). |
| "confidence- and margin-aware" low-risk branch | **Not implemented** (D1, D2). |
| "edge-accurate ... chosen only when the round trip fits the deadline with margin" | Mode exists but is selected on **zero** frames at D = 100 ms (Sec. 3). |
| stale-output rate | Measures reuse, not expiry (D6). |
| "every policy is evaluated under identical seeded conditions" | True for the network, **false for perception** (D7). |

The baseline snapshot in this directory is retained unmodified as the record of
the published artifact. All subsequent work is reported against the corrected
implementation and the new camera-ready protocol.
