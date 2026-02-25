from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .risk import BandThresholds, RiskFeatures, RiskWeights, band_from_score, score_risk


@dataclass
class FrameSummary:
    frame_id: int
    detections: int
    conf_mean: float
    risk_score: float
    band: str


def run_smoke(num_frames: int, weights: RiskWeights, bands: BandThresholds) -> dict:
    random.seed(42)
    summaries: list[FrameSummary] = []

    for frame_id in range(num_frames):
        detections = random.randint(0, 6)
        conf_mean = 0.2 if detections == 0 else round(random.uniform(0.35, 0.95), 4)

        features = RiskFeatures(
            duration_frames=min(frame_id + 1, 120),
            conf_mean=conf_mean,
            conf_std=round(random.uniform(0.02, 0.22), 4),
            bbox_area_slope=round(random.uniform(-0.03, 0.03), 4),
            roi_dwell=random.randint(0, 60),
            occlusion_count=random.randint(0, 6),
        )

        risk = score_risk(features, weights)
        band = band_from_score(risk, bands)

        summaries.append(
            FrameSummary(
                frame_id=frame_id,
                detections=detections,
                conf_mean=conf_mean,
                risk_score=round(risk, 3),
                band=band,
            )
        )

    red_frames = sum(1 for s in summaries if s.band == "RED")
    yellow_frames = sum(1 for s in summaries if s.band == "YELLOW")
    green_frames = sum(1 for s in summaries if s.band == "GREEN")

    return {
        "num_frames": num_frames,
        "band_histogram": {
            "GREEN": green_frames,
            "YELLOW": yellow_frames,
            "RED": red_frames,
        },
        "samples": [asdict(s) for s in summaries[:10]],
    }
