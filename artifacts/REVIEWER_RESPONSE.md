# Response to reviewers

Every response below points at a file in the artifact. Where a request could not be satisfied, the response says so plainly instead of restating it as satisfied.

## Reviewer 1

### R1.1 Risk weights and thresholds are not selected systematically enough

**Comment.** The risk weights and thresholds appear arbitrary.

**Response.** We agree, and replaced the hand-chosen values with an explicit selection procedure run on data that is never used for evaluation. Ten calibration seeds are held disjoint from the 30 evaluation seeds and the split is enforced in code. A deterministic random search evaluates 400 joint candidates, drawing weights from a flat Dirichlet distribution so that they are non-negative and sum to one by construction, and always including the previously published configuration and a uniform-weight configuration as reference points. Ranking uses a lexicographic objective fixed before any result was seen, with a per-level tolerance of one standard error taken from the calibration data rather than chosen by hand. We also added an identifiability check that pins to zero the weight of any risk signal that is constant over the calibration data, after finding that an inert frame-age term let the search rescale the signals that do vary.

**Changes made.** New `experiments/calibrate_scheduler.py`; the selected configuration is frozen in `config.yaml`; `dapper/config.py` now enforces that the weights sum to one and that the local threshold is below the degraded threshold.

**Evidence.** `results/calibration/calibration_report.md`, `all_candidates.csv`, `signal_identifiability.csv`, `baseline_candidates.csv`. 328 of 400 candidates reach a zero worst-profile miss rate, so the reliability result is not an artefact of the chosen weights.

### R1.2 Only one fixed seed; no confidence intervals or repeated trials

**Comment.** A single seed (42) is used and no dispersion is reported.

**Response.** Addressed. Every reported quantity now comes from 30 independent seeds with 1000 frames per cell. Because frames within a run are autocorrelated, we treat the seed - not the frame - as the unit of replication, and report 95 % percentile bootstrap intervals over seeds with a fixed bootstrap seed. Since all policies replay identical pre-generated scenarios, policies are compared seed by seed, and no difference is described as distinguishable unless its paired interval excludes zero.

**Changes made.** New `experiments/final_eval.py` and `dapper/stats.py`; every table and figure now carries intervals.

**Evidence.** `results/final/multi_seed_ci.csv`, `multi_seed_per_run.csv`, `paired_comparisons.csv`.

### R1.3 Sensitivity to inaccurate runtime estimates is missing

**Comment.** No analysis of what happens when the scheduler's runtime estimates are wrong.

**Response.** Added. The scheduler's round-trip-time and edge-compute estimates are biased from -30 % to +30 %, separately and jointly, while the executed scenario is left unchanged, so only the scheduler's beliefs are wrong. Negative bias - underestimating the true cost - is the dangerous direction and is included. The measured response is asymmetric and we report it as such rather than as one tolerance figure: biasing the round-trip-time estimate left the deadline-miss rate at exactly 0.00 % for every bias in [-30 %, +30 %]; biasing the edge-compute estimate left the deadline-miss rate at exactly 0.00 % for every bias in [+0 %, +30 %]; biasing both estimates jointly left the deadline-miss rate at exactly 0.00 % for every bias in [+0 %, +30 %]. Beyond those ranges, under-estimating the edge compute time admits commitments to the remote path that cannot be recovered, which is the same vulnerability the deadline sweep exposes. We claim no robustness outside the range actually swept.

**Changes made.** New estimation-error study in `experiments/sensitivity.py`; `rtt_estimate_bias` and `compute_estimate_bias` were added to the scheduler and are covered by unit tests.

**Evidence.** `results/sensitivity/runtime_estimation_error.csv` and its CI file; figure `fig4_runtime_estimation_error`.

### R1.4 Local-only already achieves zero misses, so DAPPER's quality advantage is unclear

**Comment.** If local-only never misses a deadline, what does DAPPER add?

**Response.** This was the most important criticism and it exposed a genuine gap: the published scheduler declared a confidence-aware branch that the code did not implement, so DAPPER could not in fact trade quality against timeliness. We implemented the branch as described, giving the scheduler a per-frame local-confidence estimate instead of a constant. With that fixed, both policies remain deadline-safe and the comparison becomes the quality delivered at that reliability: DAPPER raises the usable confidence proxy above local-only on the profiles where a remote result can arrive in time (stable 0.7250 to 0.7960, paired interval excluding zero), and is indistinguishable from it where it cannot. The secondary detector experiment measures the same effect in real detection recall rather than a proxy.

**Changes made.** `dapper/scheduler.py` implements the confidence gate; `dapper/scenario.py` supplies a per-frame local confidence; the real-detector replay reports delivered recall.

**Evidence.** `results/final/paired_comparisons.csv` (metric `usable_confidence_proxy`, `policy_b=local_only`); `results/real_detector/dapper_replay.csv`.

### R1.5 Stronger adaptive baselines are missing

**Comment.** Comparison is only against fixed policies.

**Response.** Added three adaptive comparators - deadline-greedy, confidence-and-deadline, and RTT-threshold - and an offline oracle upper bound. Each baseline's hyperparameters were fitted on the calibration seeds with the same objective used for DAPPER, so none is a strawman. We report the outcome as measured: at the primary deadline all three baselines select their most conservative setting and coincide with local-only, because with the configured edge envelope any setting that commits a frame to the edge at 100 ms produces a double-digit miss rate. We also report the oracle gap, which shows how far DAPPER is from perfect information.

**Changes made.** New `dapper/policies.py` with all baselines and the oracle; baselines are included in the deadline sweep so they are also evaluated where they are active.

**Evidence.** `results/final/multi_seed_ci.csv`, `results/calibration/baseline_candidates.csv`, figure `fig8_adaptive_baselines`, table `TABLE_ADAPTIVE_BASELINES.csv`.

### R1.6 Ablations for risk components, thresholds and operating modes are missing

**Comment.** No ablation study.

**Response.** Added two one-at-a-time ablation families and three sensitivity sweeps. Each risk signal is removed by zeroing its weight and renormalising the rest, so the score keeps its scale. Each functional mechanism - the confidence gate, the freshness gate and degraded-safe reuse - is disabled in turn. Thresholds are swept around their selected values. Every statement generated from the ablation is a paired comparison on identical scenarios and is emitted only where the bootstrap interval excludes zero; variants with no detectable effect are reported as null results rather than omitted.

**Changes made.** New `experiments/ablation.py` and `experiments/sensitivity.py`.

**Evidence.** `results/ablation/ablation_observations.md`, `component_ablation.csv`, `functional_ablation.csv`, `results/sensitivity/threshold_sweep.csv`; figure `fig5_component_ablation`; table `TABLE_ABLATION.csv`.

### R1.7 Stale-result rejection should be quantified

**Comment.** The stale-rejection mechanism is described but not measured.

**Response.** Addressed, and a definitional error was corrected first. The published prototype flagged every re-used output as stale, including re-use its own freshness rule had just declared valid, so the reported stale-output rate measured re-use rather than expiry. We now record re-use, output age, freshness validity, and remote rejections separated into deadline rejections, freshness rejections, loss and unreachability. An output older than the freshness window is never accepted; the stale-acceptance rate is identically zero across every profile and policy, and a unit test asserts this rather than leaving it as a claim.

**Changes made.** `dapper/executor.py` records the separated counters; `tests/test_execution.py` asserts zero stale acceptance end to end.

**Evidence.** `results/final/multi_seed_ci.csv` (metrics `stale_acceptance_rate`, `reuse_rate`, `remote_rejected_deadline_rate`, `remote_rejected_freshness_rate`, `remote_failed_loss_rate`).

### R1.8 Switching overhead is missing

**Comment.** The cost of running the scheduler and of switching modes is not reported.

**Response.** Measured. One placement decision takes a mean of 1.45 us (median 1.40, p95 1.80, p99 2.00) over 200,000 timed calls, and running DAPPER rather than a fixed policy adds a comparable amount per frame end to end. We report the measured microsecond values rather than calling the overhead negligible, and we state explicitly that these are software costs in CPython on a desktop CPU, not robot-hardware measurements. Mode-switch rates per 1000 frames are reported per profile; no physical reconfiguration cost is modelled, so we do not convert them into latency or energy.

**Changes made.** New `experiments/overhead.py`.

**Evidence.** `results/overhead/scheduler_overhead_summary.csv`, `mode_switches.csv`, `benchmark_wallclock.csv`; figure `fig6_scheduler_overhead`; table `TABLE_OVERHEAD.csv`.

### R1.9 Actual detection accuracy and safety-critical recall are missing

**Comment.** Confidence values are simulated; no real detector accuracy or safety-class recall.

**Response.** Added as a clearly delimited secondary experiment. YOLO11n and YOLO11m were run over the full 5,000-image COCO 2017 validation split. Overall recall was 0.507 versus 0.638, and recall restricted to safety-relevant classes (person, bicycle, car, motorcycle, bus, truck) was 0.623 versus 0.734; per-class figures are also reported. Every policy was then replayed over the cached per-image outputs, so the quality each policy delivers per frame is measured rather than proxied. We state explicitly that running both models on one workstation is a secondary validation and not a physical edge-robot experiment, and that the network side of the replay remains synthetic.

**Changes made.** New `real_validation/` package: dataset preparation, cached inference, accuracy evaluation and DAPPER replay.

**Evidence.** `results/real_detector/model_accuracy.csv`, `model_quality.csv`, `model_latency.csv`, `dapper_replay.csv`; figure `fig7_real_detector`; tables `TABLE_REAL_DETECTOR*.csv`.

### R1.10 Energy consumption is missing

**Comment.** No energy analysis.

**Response.** **Not satisfied, and we do not claim otherwise.** No energy measurement was performed: this package has no power instrumentation and no hardware platform on which a meaningful measurement could be taken. Rather than substitute a model-based estimate that would not be a measurement, we state in the limitations that energy is not evaluated, and we name energy-aware placement as future work.

**Changes made.** Limitations and conclusion updated to say so explicitly.

**Evidence.** None. No energy number appears anywhere in the artifact.

### R1.11 The benchmark is synthetic; confidence, network and latency are simulated

**Comment.** The evaluation is entirely simulated.

**Response.** Partly addressed, and the remainder is stated as a limitation. Perception is no longer entirely simulated: the secondary experiment uses two real detectors on a real labelled dataset, and the replay drives the scheduler with measured per-image confidences and compute times. The network side remains synthetic. We implemented and unit tested trace-driven replay and shipped a capture tool so that a measured trace can be dropped in without touching the scheduler, but **no measured Wi-Fi, 5G or edge trace was collected**, and no result in this paper rests on one. We do not describe any generated profile as a measured trace.

**Changes made.** New `dapper/trace.py` and `tools/capture_network_trace.py`, both tested; `real_validation/` for the detector side.

**Evidence.** `tests/test_scenario_and_policies.py::test_trace_round_trip_drives_the_benchmark`; `results/real_detector/`.

## Reviewer 2

### R2.1 Original contribution versus the state of the art needs to be clearer

**Comment.** The delta over existing work is not sharp enough.

**Response.** The revised text states the contribution as the combination that the comparators lack: a per-frame placement decision that is auditable (every decision carries a numeric risk score and a reason string from a closed set), deadline-aware, freshness-gated on accepting remote results, and equipped with a conservative fallback - together with a package in which all policies are compared on identical pre-generated scenarios. The adaptive-baseline section now makes the delta empirical rather than rhetorical: the intuitive confidence-and-deadline rule is implemented, fitted and measured, and the text explains exactly which structural feature of DAPPER accounts for the difference at the primary deadline.

**Changes made.** Sections A, D, E and H of `artifacts/PAPER_PATCH_TEXT.md`.

**Evidence.** `results/final/multi_seed_ci.csv`, `results/calibration/baseline_candidates.csv`.

### R2.2 Relations among DAPPER components need stronger technical explanation

**Comment.** How the components interact is under-explained.

**Response.** The methodology subsection now describes one shared execution model and states how the monitor, the scheduler and the execution path are coupled: the scheduler's predicted remote arrival and the executor's realised arrival are the same expression with the estimate substituted for the unobservable compute time, which is asserted by a unit test so the two cannot drift apart. The decision procedure is given as an ordered rule with a named reason for each branch, and the ablation quantifies what each component contributes.

**Changes made.** `dapper/scheduler.py` and `dapper/executor.py` docstrings; `tests/test_execution.py::test_prediction_uses_the_same_form_as_execution`; Sections A and F of the patch text.

**Evidence.** `results/ablation/ablation_observations.md`.

### R2.3 More concrete examples are needed

**Comment.** The paper is abstract about what the scheduler actually does.

**Response.** The mode-distribution table gives the concrete per-profile behaviour, the deadline sweep shows the policy moving between modes as the control budget changes, and the reason-string counts show which branch fired and how often. The scenario traces are saveable and replayable as CSV, so a specific frame's decision can be reproduced and inspected.

**Changes made.** `results/final/mode_distribution.csv`, `results/sensitivity/deadline_mode_distribution.csv`, `dapper.scenario.save_scenarios`.

**Evidence.** Figure `fig3_deadline_mode_distribution`; `results/final/scenario_fingerprints.csv`.

### R2.4 Key observations should be summarised

**Comment.** The main takeaways are not collected anywhere.

**Response.** Added a generated summary of the findings, including the unfavourable ones, and a claim-evidence matrix in which every camera-ready claim is paired with the experiment and artifact that supports it and marked supported or not supported.

**Changes made.** `artifacts/CAMERA_READY_RESULTS.md` and `artifacts/CLAIM_EVIDENCE_MATRIX.md`.

**Evidence.** Both documents are generated from the result CSVs.

### R2.5 2025-2026 references should be added

**Comment.** The related-work section needs newer citations.

**Response.** **Not addressed by this artifact.** The work reported here is experimental; no literature search was performed as part of it and we do not supply citations we have not verified. The related-work update is left to the authors.

**Changes made.** None.

**Evidence.** None.

## Requests we did not satisfy

* **Energy consumption** (R1.10) - not measured; no instrumentation, no platform, no estimate substituted.
* **Physical-robot validation** - not performed. The detector study runs on one workstation and is described as such throughout.
* **Measured wireless traces** - trace-driven replay is implemented and tested, but no trace was captured, so no result depends on one.
* **2025-2026 references** (R2.5) - outside the scope of this artifact.

We also report one result that is unfavourable to DAPPER and was not asked for: in the deadline sweep, the frozen configuration is **not** deadline-safe at some deadlines above the calibrated one, because the gate on the committing mode is unidentifiable from a single-deadline calibration. The mechanism is diagnosed, the fix is identified, and it is deliberately left to a future calibration rather than applied to the evaluation data.
