#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import platform
import sys


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main() -> int:
    print("== Environment Check ==")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    modules = ["yaml", "ultralytics", "torch", "cv2", "duckdb"]
    for m in modules:
        print(f"module[{m}]: {'FOUND' if _has(m) else 'MISSING'}")

    if _has("torch"):
        import torch  # type: ignore

        print(f"torch.mps.is_available: {torch.backends.mps.is_available()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
