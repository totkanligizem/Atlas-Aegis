#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CANDIDATE_REPORT="reports/detector_eval_candidate_highrecall_poisson_focus_1600_report.json"

echo "[loop-1600] train start"
make train-det-highrecall-poisson-focus-1600-ep1

echo "[loop-1600] candidate eval start"
make eval-det-candidate-highrecall-poisson-focus-1600

echo "[loop-1600] candidate gate check"
export CANDIDATE_REPORT
PASS="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["CANDIDATE_REPORT"])
if not p.exists():
    raise FileNotFoundError(f"Candidate report not found: {p}")
obj = json.loads(p.read_text(encoding="utf-8"))
print("1" if bool(obj["gates"]["pass"]) else "0")
PY
)"

if [[ "$PASS" == "1" ]]; then
  echo "[loop-1600] candidate PASS; full41 eval start"
  make eval-det-full41-highrecall-poisson-focus-1600
  echo "[loop-1600] done"
else
  echo "[loop-1600] candidate FAIL; full41 skipped"
fi
