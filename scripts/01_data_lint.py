#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.data_lint import lint_yolo_labels


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", required=True)
    args = p.parse_args()

    result = lint_yolo_labels(args.labels_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
