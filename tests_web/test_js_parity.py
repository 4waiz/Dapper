"""
The browser engine must reproduce the Python engine, decision for decision.

This suite lives outside `tests/` on purpose. The paper states that the prototype
ships 69 automated tests, and `python -m pytest -q` must keep reporting exactly
that (pytest.ini pins `testpaths = tests`). The web engine is a post-camera-ready
addition, so its tests are a separate suite:

    python -m pytest tests_web -q

What is checked, in the order a failure would matter:

1. `site/data/config.json` is what `export_web_config.py` produces from the live
   `config.yaml`, so the dashboard cannot drift from Table I.
2. Every shipped trace in `site/data/traces/` is bit-for-bit the scenario
   `dapper.scenario.generate_scenarios` draws for that (profile, seed) -- same
   fingerprint, same arrays. Replay mode therefore plays a published scenario.
3. The JavaScript engine, run under Node over those traces, reproduces Python's
   per-frame mode, reason string, risk score, control and final latency, remote
   outcome, freshness, bandwidth and fallback flag exactly, and every reduced
   metric to within 1e-9. Five policies at three deadlines, because
   edge-accurate receives no frames at D <= 100 ms and would otherwise never be
   exercised.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from dapper.config import load_config  # noqa: E402
from dapper.scenario import generate_scenarios  # noqa: E402

import export_web_config  # noqa: E402
import export_web_traces  # noqa: E402

SITE_DATA = ROOT / "site" / "data"
TRACE_DIR = SITE_DATA / "traces"
PARITY_TOLERANCE = 1e-9

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def trace_index():
    index = json.loads((TRACE_DIR / "index.json").read_text(encoding="utf-8"))
    assert index["traces"], "no traces exported; run experiments/export_web_traces.py"
    return index


def test_exported_config_matches_config_yaml(cfg):
    """site/data/config.json is generated, not maintained."""
    on_disk = json.loads((SITE_DATA / "config.json").read_text(encoding="utf-8"))
    assert on_disk == export_web_config.build(cfg), (
        "site/data/config.json is stale; re-run experiments/export_web_config.py")


def test_exported_config_is_json_safe():
    """A browser rejects NaN and Infinity, so the export must contain neither."""
    raw = (SITE_DATA / "config.json").read_text(encoding="utf-8")
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in raw
    assert "\r" not in raw, "config.json must use LF endings"


def test_shipped_traces_cover_every_profile(cfg, trace_index):
    shipped = {(t["profile"], t["seed"]) for t in trace_index["traces"]}
    for profile in cfg["profiles"]:
        seeds = sorted(s for p, s in shipped if p == profile)
        assert seeds, f"no trace shipped for profile '{profile}'"
        assert len(seeds) >= 2, f"profile '{profile}' ships only one seed"
    assert all(t["frames"] == 1000 for t in trace_index["traces"])


def test_shipped_traces_are_the_python_scenarios(cfg, trace_index):
    """Every array in every shipped trace equals the Python scenario exactly."""
    for entry in trace_index["traces"]:
        trace = generate_scenarios(entry["profile"], cfg, entry["seed"], entry["frames"])
        assert trace.fingerprint() == entry["fingerprint"], (
            f"{entry['file']}: index fingerprint is stale")
        payload = json.loads((TRACE_DIR / entry["file"]).read_text(encoding="utf-8"))
        assert payload["fingerprint"] == trace.fingerprint()
        assert payload["frames"] == trace.frames
        assert payload["frame_period_ms"] == float(cfg["execution"]["frame_period_ms"])
        for key in export_web_traces.TRACE_FLOAT_ARRAYS:
            got = np.asarray(payload["arrays"][key], dtype=float)
            want = np.asarray(getattr(trace, key), dtype=float)
            # Exported with Python's shortest round-trip repr, so this is equality,
            # not approximate equality.
            assert np.array_equal(got, want), f"{entry['file']}: array '{key}' differs"
        for key in export_web_traces.TRACE_INT_ARRAYS:
            assert np.array_equal(np.asarray(payload["arrays"][key], dtype=np.int64),
                                  np.asarray(getattr(trace, key), dtype=np.int64)), key
        avail = np.asarray(payload["arrays"]["edge_available"], dtype=bool)
        assert np.array_equal(avail, np.asarray(trace.edge_available, dtype=bool))


def test_shipped_traces_carry_no_non_finite_values(trace_index):
    for entry in trace_index["traces"]:
        raw = (TRACE_DIR / entry["file"]).read_text(encoding="utf-8")
        for token in ("NaN", "Infinity"):
            assert token not in raw, f"{entry['file']} contains {token}"


@needs_node
def test_browser_engine_matches_python(cfg, tmp_path):
    """Replay the exported traces through site/js under Node and compare."""
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    export_web_traces.write_reference(
        cfg,
        str(ref_dir),
        list(cfg["profiles"]),
        list(export_web_traces.DEFAULT_SEEDS),
        export_web_traces.DEFAULT_FRAMES,
        export_web_traces.REFERENCE_POLICIES,
        export_web_traces.REFERENCE_DEADLINES_MS,
    )
    written = sorted(p.name for p in ref_dir.glob("*.reference.json"))
    assert len(written) == len(cfg["profiles"]) * len(export_web_traces.DEFAULT_SEEDS)

    out = subprocess.run(
        [shutil.which("node"), str(ROOT / "scripts" / "js_parity.mjs"), str(ref_dir)],
        capture_output=True, text=True, timeout=900, cwd=str(ROOT),
    )
    assert out.returncode == 0, f"js_parity.mjs failed:\n{out.stderr}\n{out.stdout}"
    print(out.stderr.strip())
    worst = float(out.stdout.strip().splitlines()[-1])
    assert worst < PARITY_TOLERANCE, (
        f"worst absolute Python/JS difference {worst:g} exceeds {PARITY_TOLERANCE:g}")


@needs_node
def test_every_reason_and_mode_is_reachable_in_the_browser_engine(cfg, trace_index):
    """
    The parity check is only as good as its coverage: assert that the shipped
    traces actually drive the engine through every mode and every reason string
    the scheduler can emit, across the deadlines the reference covers.
    """
    script = r"""
import { readFileSync, readdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { resolve, join } from "node:path";
const root = process.argv[2];
const load = (p) => import(pathToFileURL(resolve(root, "site", "js", p)).href);
const { traceFromExport } = await load("trace.js");
const { buildPolicy } = await load("policies.js");
const { executeRun } = await load("executor.js");
const cfg = JSON.parse(readFileSync(join(root, "site", "data", "config.json"), "utf8"));
const dir = join(root, "site", "data", "traces");
const modes = new Set(), reasons = new Set();
for (const f of readdirSync(dir).filter((f) => f !== "index.json")) {
  const trace = traceFromExport(JSON.parse(readFileSync(join(dir, f), "utf8")));
  for (const D of [100, 150, 250]) {
    for (const row of executeRun(trace, buildPolicy("dapper", cfg), cfg, D).rows) {
      modes.add(row.selected_mode);
      reasons.add(row.decision_reason);
    }
  }
}
console.log(JSON.stringify({ modes: [...modes].sort(), reasons: [...reasons].sort() }));
"""
    path = ROOT / "scripts" / ".coverage_probe.mjs"
    path.write_text(script, encoding="utf-8", newline="\n")
    try:
        out = subprocess.run([shutil.which("node"), str(path), str(ROOT)],
                             capture_output=True, text=True, timeout=600, cwd=str(ROOT))
    finally:
        path.unlink(missing_ok=True)
    assert out.returncode == 0, out.stderr
    seen = json.loads(out.stdout)
    from dapper.scheduler import ALL_MODES, ALL_REASONS
    assert set(seen["modes"]) == set(ALL_MODES), (
        f"modes not exercised: {sorted(set(ALL_MODES) - set(seen['modes']))}")
    assert set(seen["reasons"]) == set(ALL_REASONS), (
        f"reasons not exercised: {sorted(set(ALL_REASONS) - set(seen['reasons']))}")
