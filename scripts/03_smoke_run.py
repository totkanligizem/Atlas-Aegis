#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerial_stack.config import load_yaml
from aerial_stack.risk import BandThresholds, RiskWeights
from aerial_stack.smoke import run_smoke


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=120)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    weights = cfg.get("weights", {})
    bands = cfg.get("bands", {})

    result = run_smoke(
        num_frames=args.frames,
        weights=RiskWeights(
            roi_dwell=float(weights.get("roi_dwell", 1.4)),
            approach=float(weights.get("approach", 1.0)),
            duration=float(weights.get("duration", 0.8)),
            conf_mean=float(weights.get("conf_mean", 0.6)),
            conf_stability=float(weights.get("conf_stability", 0.5)),
            occlusion=float(weights.get("occlusion", 0.3)),
            bias=float(cfg.get("bias", -1.8)),
        ),
        bands=BandThresholds(
            green_max=float(bands.get("green_max", 39.999)),
            yellow_max=float(bands.get("yellow_max", 69.999)),
        ),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
