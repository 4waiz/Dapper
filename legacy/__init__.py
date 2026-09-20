"""Frozen copy of the originally published DAPPER prototype (commit e0a728e).

These modules are retained unmodified so that the published FMEC 2026 results
in artifacts/baseline_original/ remain reproducible verbatim:

    python -m legacy.benchmark run-all --config legacy/config_original.yaml \
        --frames 500 --deadline-ms 100 --seed 42 --out-dir <dir>

They are NOT used by any camera-ready experiment. The corrected implementation
lives in the dapper/ package. See artifacts/baseline_original/BASELINE_AUDIT.md
for the defects that motivated the rewrite.
"""

import os as _os
import sys as _sys

# The frozen modules use flat imports (``from monitor import ...``) exactly as
# published. Make that resolve without editing them.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
