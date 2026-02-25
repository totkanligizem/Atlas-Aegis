from __future__ import annotations

import math
from dataclasses import dataclass


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class RiskFeatures:
    duration_frames: int
    conf_mean: float
    conf_std: float
    bbox_area_slope: float
    roi_dwell: int
    occlusion_count: int


@dataclass
class RiskWeights:
    roi_dwell: float = 1.4
    approach: float = 1.0
    duration: float = 0.8
    conf_mean: float = 0.6
    conf_stability: float = 0.5
    occlusion: float = 0.3
    bias: float = -1.8


@dataclass
class BandThresholds:
    green_max: float = 39.999
    yellow_max: float = 69.999


def score_risk(features: RiskFeatures, weights: RiskWeights) -> float:
    p = _clip(features.duration_frames / 30.0)
    c = _clip(features.conf_mean)
    cs = 1.0 - _clip(features.conf_std / 0.25)
    a = _sigmoid(features.bbox_area_slope / 0.02)
    r = _clip(features.roi_dwell / 30.0)
    o = _clip(features.occlusion_count / 10.0)

    z = (
        weights.roi_dwell * r
        + weights.approach * a
        + weights.duration * p
        + weights.conf_mean * c
        + weights.conf_stability * cs
        + weights.occlusion * o
        + weights.bias
    )
    return 100.0 * _sigmoid(z)


def band_from_score(score: float, bands: BandThresholds) -> str:
    if score <= bands.green_max:
        return "GREEN"
    if score <= bands.yellow_max:
        return "YELLOW"
    return "RED"
