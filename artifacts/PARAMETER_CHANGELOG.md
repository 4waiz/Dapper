# Parameter changelog

Every parameter that differs from the published configuration (`legacy/config_original.yaml`), with the reason. Permitted reasons are exactly three: **(A)** fixing an implementation inconsistency, **(B)** selected by the declared calibration protocol, **(C)** varied only inside a sensitivity study. No value was changed because it improved DAPPER's numbers.

## Structural changes (reason A)

| Change | Was | Now | Why |
|---|---|---|---|
| `edge.base_rtt_ms` | 20 (never applied) | removed | Dead configuration: the published executor added the static access RTT for the cloud path only. Rather than start charging it - which would have made the fixed edge baseline worse and DAPPER look better by comparison - we adopt the reading that the network-profile RTT *is* the edge access round trip, which leaves the edge timing exactly as published. |
| `cloud.base_rtt_ms` | 80 | `cloud.extra_rtt_ms: 80` | Same value, renamed for the semantics above, and now applied identically in the scheduler's prediction and in execution. A unit test asserts the two expressions agree. |
| `execution.*` | absent | new section | The timing constants that were previously implicit in code: frame period, load-compute scaling, the client response-timer slack, the cost of serving a cached output, and whether edge availability is observable before transmission. |
| `scheduler.local_confidence_threshold` | absent | selected | The published scheduler declared a confidence-aware branch and did not implement it; the threshold is the parameter that branch needs. |
| `scheduler.remote_deadline_margin` | absent | selected | Makes explicit the manuscript's "consumes at most a configurable fraction of the deadline" rule for committing to the remote path. |
| `scheduler.hybrid_estimator` | absent | selected | Which edge compute estimate gates a hybrid refresh. Left to the calibration rather than fixed by hand. |

## Values selected by the calibration protocol (reason B)

Selected on calibration seeds 0-9 only, by the predeclared objective in `experiments/calibrate_scheduler.py`; see `results/calibration/calibration_report.md`.

| Parameter | Published | Selected |
|---|---|---|
| `weight_rtt` | 0.35 | 0.36 |
| `weight_loss` | 0.3 | 0.35 |
| `weight_load` | 0.15 | 0.11 |
| `weight_frame_age` | 0.05 | 0 |
| `weight_deadline` | 0.15 | 0.18 |
| `risk_local_threshold` | 0.4 | 0.45 |
| `risk_degraded_threshold` | 0.7 | 0.6 |
| `hybrid_freshness_window_ms` | 80.0 | 100 |
| `last_valid_freshness_ms` | 250.0 | 100 |
| `local_confidence_threshold` | n/a (unused) | 0.78 |
| `remote_deadline_margin` | n/a | 0.8 |
| `hybrid_estimator` | n/a | optimistic |

`weight_frame_age` is pinned to zero rather than searched. The frame-age signal is constant at zero in this benchmark because the scheduler is invoked at frame capture, so its weight is unidentifiable: it only adds an offset the thresholds absorb while occupying a simplex dimension and rescaling the signals that do vary. A first calibration run parked 26 % of the weight there, which pushed the maximum attainable risk below the degraded threshold and made the risk-based degraded-safe branch unreachable. The term stays in the risk formulation because it is identifiable wherever frames queue before scheduling. Source: `results/calibration/signal_identifiability.csv`.

`local_confidence_threshold` candidates were restricted to the open interval spanned by the local model's confidence (0.65, 0.8). A threshold at an endpoint would fire on every frame or on none, i.e. disable the gate rather than set it. This restriction *lowers* the best attainable confidence proxy, so it works against DAPPER.

## Values varied only inside sensitivity studies (reason C)

* the control deadline D (50-250 ms);
* `risk_local_threshold`, `risk_degraded_threshold` and `local_confidence_threshold` around their selected values;
* `remote_deadline_margin` (commit-margin diagnostic);
* the scheduler's RTT and compute estimate biases (-30 % to +30 %);
* the risk weights, one at a time, in the ablation.

None of these sweeps feeds back into `config.yaml`. In particular, the commit-margin diagnostic identifies a value that would remove the deadline misses observed at D = 125-200 ms, and that value is **deliberately not adopted**, because it was found on evaluation seeds.

## Unchanged

Every network-profile parameter (RTT mean and standard deviation, packet loss, edge load, outage probability), every perception latency and confidence range, both bandwidth figures, the AR(1) smoothing coefficients and the outage burst length are exactly as published. The distributions that generate the results were not touched.

