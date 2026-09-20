"""
One command that reproduces every camera-ready result.

    python experiments/run_camera_ready.py --full     # exact protocol
    python experiments/run_camera_ready.py --quick    # small, for CI / dev

The full protocol is deterministic: the same command on the same commit
produces the same CSVs, because every stochastic quantity is drawn from a
seeded, pre-generated scenario trace and the bootstrap has a fixed seed.

Stages, in dependency order:

    0  tests                 the suite must pass before any result is produced
    1  calibration           parameter selection on seeds 0-9      (optional)
    2  final evaluation      seeds 100-129, 8 policies, 5 profiles
    3  ablation              risk-signal and mechanism ablations
    4  sensitivity           deadline, threshold and estimation-error sweeps
    5  overhead              scheduler decision cost and switch rates
    6  real detector         secondary YOLO11n/YOLO11m validation   (optional)
    7  figures and tables    every paper artifact
    8  reports               narrative, patch text, reviewer response, matrix

Stage 1 is **not** run by default: `config.yaml` already holds the frozen
calibrated values, and re-running calibration would overwrite them. Pass
``--calibrate`` to redo it from scratch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run(cmd: List[str], label: str, cwd: str = ROOT) -> float:
    print(f"\n{'=' * 78}\n[{label}] {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"[{label}] FAILED with exit code {r.returncode}")
    print(f"[{label}] done in {dt:.1f}s", flush=True)
    return dt


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="exact camera-ready protocol")
    mode.add_argument("--quick", action="store_true", help="few seeds, for CI")
    p.add_argument("--calibrate", action="store_true",
                   help="re-run parameter selection and overwrite config.yaml")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-real-detector", action="store_true",
                   help="skip the optional YOLO validation (needs torch + the dataset)")
    p.add_argument("--stages", nargs="*", default=None,
                   choices=["final", "ablation", "sensitivity", "overhead",
                            "real", "figures", "tables", "reports"])
    args = p.parse_args(argv)
    quick = args.quick and not args.full

    py = [sys.executable]
    E = lambda s: [os.path.join(HERE, s)]  # noqa: E731
    seeds = ["--seeds", "4"] if quick else []
    frames = ["--frames", "200"] if quick else ["--frames", "1000"]
    nboot = ["--n-boot", "500"] if quick else []
    want = set(args.stages or ["final", "ablation", "sensitivity", "overhead",
                               "real", "figures", "tables", "reports"])

    t0 = time.time()
    if not args.skip_tests:
        run(py + ["-m", "pytest", "-q"], "0 tests")

    if args.calibrate:
        cal = ["--candidates", "40", "--frames", "100"] if quick else \
              ["--candidates", "400", "--frames", "500"]
        run(py + E("calibrate_scheduler.py") + cal, "1 calibration")
        print("\n*** config.yaml is NOT updated automatically. Copy the selected "
              "values from results/calibration/selected_config.yaml and re-run. ***")

    if "final" in want:
        run(py + E("final_eval.py") + frames + seeds + nboot, "2 final evaluation")
    if "ablation" in want:
        run(py + E("ablation.py") + frames + seeds + nboot, "3 ablation")
    if "sensitivity" in want:
        run(py + E("sensitivity.py") + frames + seeds
            + (["--n-boot", "500"] if quick else ["--n-boot", "5000"]), "4 sensitivity")
    if "overhead" in want:
        calls = ["--calls", "20000", "--warmup", "1000"] if quick else \
                ["--calls", "200000", "--warmup", "5000"]
        run(py + E("overhead.py") + calls + frames + seeds, "5 overhead")

    if "real" in want and not args.skip_real_detector:
        rv = os.path.join(ROOT, "real_validation")
        parquet = os.path.join(rv, "detector_results.parquet")
        if not os.path.exists(parquet):
            print("\n[6 real detector] detector_results.parquet is missing. Run:\n"
                  "    python real_validation/prepare_dataset.py\n"
                  "    python real_validation/collect_detector_outputs.py\n"
                  "Skipping; see artifacts/REAL_DETECTOR_NOT_RUN.md if it was "
                  "never executed.")
        else:
            run(py + [os.path.join(rv, "evaluate_detection.py")], "6a detector metrics")
            replay = ["--seeds", "3"] if quick else ["--seeds", "10"]
            run(py + [os.path.join(rv, "replay_dapper.py")] + replay, "6b DAPPER replay")

    if "figures" in want:
        run(py + E("make_figures.py"), "7a figures")
    if "tables" in want:
        run(py + E("make_tables.py"), "7b tables")
    if "reports" in want:
        run(py + E("make_reports.py"), "8 reports")

    print(f"\n{'=' * 78}\nAll stages complete in {time.time() - t0:.0f}s.\n"
          f"Results:   results/\n"
          f"Figures:   results/paper_figures/\n"
          f"Tables:    results/paper_tables/\n"
          f"Narrative: artifacts/CAMERA_READY_RESULTS.md\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
