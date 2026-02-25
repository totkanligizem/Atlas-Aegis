from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class CorruptionSpec:
    name: str
    severity: int


def parse_condition(condition: str) -> CorruptionSpec:
    # Supported forms:
    # - clean
    # - s3_blur, s3_low_light, ...
    # - blur_s1, low_light_s3, ...
    if condition == "clean":
        return CorruptionSpec(name="clean", severity=0)

    if condition.startswith("s3_"):
        return CorruptionSpec(name=condition[3:], severity=3)

    if "_s" in condition:
        left, right = condition.rsplit("_s", 1)
        try:
            sev = int(right)
        except ValueError as exc:
            raise ValueError(f"Invalid condition severity: {condition}") from exc
        if sev < 1 or sev > 5:
            raise ValueError(f"Severity must be in 1..5: {condition}")
        return CorruptionSpec(name=left, severity=sev)

    raise ValueError(f"Unsupported condition format: {condition}")


def _rng_for(key: str) -> np.random.Generator:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    seed = int(digest, 16) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def _motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    import cv2

    ks = [3, 5, 9, 15, 21][severity - 1]
    kernel = np.zeros((ks, ks), dtype=np.float32)
    kernel[ks // 2, :] = 1.0 / ks
    return cv2.filter2D(img, -1, kernel)


def _low_light(img: np.ndarray, severity: int) -> np.ndarray:
    factors = [0.85, 0.70, 0.55, 0.40, 0.25]
    f = factors[severity - 1]
    out = np.clip(img.astype(np.float32) * f, 0, 255)
    return out.astype(np.uint8)


def _jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    import cv2

    q = [70, 50, 35, 20, 10][severity - 1]
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def _gauss(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    sigma = [3.0, 6.0, 12.0, 18.0, 25.0][severity - 1]
    noise = rng.normal(0.0, sigma, img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def _poisson(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    scales = [0.02, 0.05, 0.10, 0.15, 0.20]
    scale = scales[severity - 1]
    # Normalize then apply poisson noise with controlled amplitude.
    norm = img.astype(np.float32) / 255.0
    noisy = rng.poisson(norm / max(scale, 1e-6)) * scale
    out = np.clip(noisy * 255.0, 0, 255)
    return out.astype(np.uint8)


def _dropout(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    holes = [1, 2, 3, 4, 5][severity - 1]
    h, w = out.shape[:2]
    for _ in range(holes):
        hh = int(rng.integers(max(8, h // 20), max(9, h // 8)))
        ww = int(rng.integers(max(8, w // 20), max(9, w // 8)))
        y = int(rng.integers(0, max(1, h - hh)))
        x = int(rng.integers(0, max(1, w - ww)))
        out[y : y + hh, x : x + ww] = 0
    return out


def _fog(img: np.ndarray, severity: int) -> np.ndarray:
    import cv2

    coefs = [0.05, 0.12, 0.25, 0.40, 0.60]
    alpha = coefs[severity - 1]
    fog = np.full_like(img, 255, dtype=np.uint8)
    blend = cv2.addWeighted(img, 1.0 - alpha, fog, alpha, 0)
    return blend


def _downsample(img: np.ndarray, severity: int) -> np.ndarray:
    import cv2

    scales = [0.85, 0.70, 0.55, 0.40, 0.25]
    s = scales[severity - 1]
    h, w = img.shape[:2]
    nw = max(2, int(w * s))
    nh = max(2, int(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return up


def apply_condition(image_bgr: np.ndarray, condition: str, seed_key: str) -> np.ndarray:
    spec = parse_condition(condition)
    if spec.name == "clean":
        return image_bgr

    rng = _rng_for(f"{seed_key}|{condition}")
    name = spec.name
    sev = spec.severity

    if name == "blur":
        return _motion_blur(image_bgr, sev)
    if name == "low_light":
        return _low_light(image_bgr, sev)
    if name == "jpeg":
        return _jpeg(image_bgr, sev)
    if name == "gauss":
        return _gauss(image_bgr, sev, rng)
    if name == "poisson":
        return _poisson(image_bgr, sev, rng)
    if name == "dropout":
        return _dropout(image_bgr, sev, rng)
    if name == "fog":
        return _fog(image_bgr, sev)
    if name == "downsample":
        return _downsample(image_bgr, sev)

    raise ValueError(f"Unsupported corruption name: {name}")
