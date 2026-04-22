"""Put the project root on sys.path so tests can `import demo.env.*`.

Also default pytest-asyncio to strict/auto mode so the bare `@pytest.mark.asyncio`
decorators Just Work without requiring repo-wide config.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_configure(config):
    config.inicfg.setdefault("asyncio_mode", "auto")
