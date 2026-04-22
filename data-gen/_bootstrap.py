"""sys.path shim — lets `common`, `stage_a_curator`, `stage_c_datadesigner`
import cleanly even though the containing directory is `data-gen/` (hyphens
aren't valid Python package identifiers). Every runnable entrypoint under
this tree does `import _bootstrap  # noqa: F401` before its local imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

_DATA_GEN = Path(__file__).resolve().parent
if str(_DATA_GEN) not in sys.path:
    sys.path.insert(0, str(_DATA_GEN))
