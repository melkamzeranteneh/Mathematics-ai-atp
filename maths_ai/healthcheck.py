"""Lightweight Docker healthcheck for the core service.

Verifies that core modules are importable and the data directory is accessible.
Intentionally avoids heavy imports (torch, pantograph) to stay fast.
"""
import sys
from pathlib import Path


def _check() -> bool:
    from maths_ai.core.config import DATA_ROOT, LEAN_PROJECT_PATH

    if not DATA_ROOT.exists():
        print(f"FAIL: DATA_ROOT does not exist: {DATA_ROOT}", file=sys.stderr)
        return False

    if not LEAN_PROJECT_PATH.exists():
        print(f"FAIL: LEAN_PROJECT_PATH does not exist: {LEAN_PROJECT_PATH}", file=sys.stderr)
        return False

    return True


if __name__ == "__main__":
    sys.exit(0 if _check() else 1)
