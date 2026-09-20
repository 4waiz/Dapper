# DAPPER — Deadline-Aware Perception Placement for Edge Robotics

Reproducible research artifact for *DAPPER: Deadline-Aware Edge Perception for
Mission-Critical Robots under Dynamic Network Conditions* (FMEC 2026).

DAPPER decides, **per frame**, whether a robot should run perception locally,
offload it to an edge server, do both, or fall back to a previous result. The
decision is deterministic and auditable: every frame carries a numeric
deadline-risk score and a reason string drawn from a closed set.

---

## Quick reproduction

```bash
pip install -r requirements.txt
python -m pytest -q
python experiments/run_camera_ready.py --full
```

`--full` runs the exact camera-ready protocol (30 seeds × 1000 frames × 5
profiles × 8 policies, plus ablations, sweeps and overhead) and takes roughly
30 minutes on a desktop CPU. `--quick` runs a 4-seed version in about a minute
for CI. The optional real-detector stage needs a CUDA GPU and is skipped if its
cached predictions are absent.

Everything is deterministic: identical commands on the same commit produce
identical CSVs, because every stochastic quantity comes from a seeded
pre-generated scenario and the bootstrap has a fixed seed.

## What to read first

| File | What it is |
|---|---|
| [artifacts/CAMERA_READY_RESULTS.md](artifacts/CAMERA_READY_RESULTS.md) | The full result narrative, generated from the CSVs |
| [artifacts/CLAIM_EVIDENCE_MATRIX.md](artifacts/CLAIM_EVIDENCE_MATRIX.md) | Every claim, its evidence, and the claims the data does **not** support |
| [artifacts/PAPER_PATCH_TEXT.md](artifacts/PAPER_PATCH_TEXT.md) | Ready-to-paste IEEE prose with a source comment on every number |
| [artifacts/REVIEWER_RESPONSE.md](artifacts/REVIEWER_RESPONSE.md) | Point-by-point response, including what was **not** addressed |
| [artifacts/baseline_original/BASELINE_AUDIT.md](artifacts/baseline_original/BASELINE_AUDIT.md) | Audit of the originally published implementation |
| [artifacts/PARAMETER_CHANGELOG.md](artifacts/PARAMETER_CHANGELOG.md) | Every parameter change and its justification |
| [results/calibration/calibration_report.md](results/calibration/calibration_report.md) | How the weights and thresholds were selected |

## Layout

```
dapper/              corrected implementation
  config.py            configuration loading + invariant checks
  scenario.py          immutable pre-generated per-frame scenarios
  scheduler.py         the DAPPER decision rule
  policies.py          all policies, adaptive baselines, offline oracle
  executor.py          shared timing / bandwidth / freshness accounting
  metrics.py           per-run metric reduction
  stats.py             seed-level bootstrap confidence intervals
  trace.py             trace-driven network input
experiments/         one driver per phase, plus run_camera_ready.py
real_validation/     optional YOLO11n / YOLO11m study on COCO val2017
tools/               network-trace capture utility
tests/               69 unit tests
legacy/              the published prototype, frozen and still reproducible
artifacts/           generated reports and the frozen baseline snapshot
results/             every generated CSV, figure and table
```

## Individual experiments

```bash
python experiments/calibrate_scheduler.py     # parameter selection (seeds 0-9)
python experiments/final_eval.py              # 30-seed evaluation (seeds 100-129)
python experiments/ablation.py                # risk-signal + mechanism ablations
python experiments/sensitivity.py             # deadline, threshold, estimation, margin
python experiments/overhead.py                # scheduler decision cost
python experiments/make_figures.py            # results/paper_figures/
python experiments/make_tables.py             # results/paper_tables/
python experiments/make_reports.py            # artifacts/*.md
```

Optional real-detector study (CUDA GPU, ~1 GB download):

```bash
python real_validation/prepare_dataset.py          # COCO val2017 + YOLO labels
python real_validation/collect_detector_outputs.py --cpu-subset-local -1
python real_validation/evaluate_detection.py
python real_validation/replay_dapper.py
```

## Experimental design

* **Paired scenarios.** For each (seed, profile), the *potential* outcome of
  every execution site on every frame — latency, confidence, detections and the
  loss draw — is generated in advance from 15 independent random streams. All
  policies replay the identical trace, so a decision selects which outcome is
  realised but cannot change the random future. Comparisons are paired at frame
  granularity; `results/final/scenario_fingerprints.csv` lets a reviewer verify
  that a replay used the same scenarios.
* **Disjoint seeds.** Parameters are selected on seeds 0–9 and never on the
  evaluation seeds 100–129. The split is enforced in code.
* **Seed-level statistics.** Frames within a run are autocorrelated, so the
  seed is the unit of replication. All intervals are percentile bootstrap over
  seeds with a fixed bootstrap seed, and policy pairs are compared seed by seed.
* **One shared execution model.** No policy has a private cost model. Every
  transmitted offload is charged its upload bandwidth, including requests that
  are lost or whose reply arrives too late to accept, and a failed request costs
  a deadline-bounded client timeout before the local fallback runs.

## Reproducing the originally published results

The published prototype is frozen under `legacy/` and still reproduces the
FMEC 2026 submission numbers exactly:

```bash
python -m legacy.benchmark --config legacy/config_original.yaml run-all \
    --frames 500 --deadline-ms 100 --seed 42 --out-dir artifacts/baseline_original
```

`artifacts/baseline_original/BASELINE_AUDIT.md` documents twelve confirmed
implementation defects in that version and what each one did to the reported
numbers.

## Scope and limitations

This is a **controlled synthetic benchmark** with a secondary real-detector
study. It is **not**:

* a physical robot deployment — nothing here ran on a robot or embedded hardware;
* a measured wireless evaluation — trace replay is implemented and tested, but
  no Wi-Fi, 5G or edge trace was captured, so no result depends on one;
* an energy study — energy is not instrumented anywhere;
* a safety validation — `degraded-safe` names a conservative fallback mode, not
  a certified safety property.

Confidence values in the main tables are **simulated proxies**, never detector
accuracies. The real-detector study reports measured recall on COCO val2017 and
is labelled as such throughout.
