#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def exists_any(path: Path, patterns: list[str]) -> bool:
    for pat in patterns:
        if any(path.glob(pat)):
            return True
    return False


def is_uavdt_extracted(base_dir: Path) -> bool:
    candidate_roots = [
        base_dir / "UAVDT",
        base_dir / "UAV-benchmark-M",
    ]
    for root in candidate_roots:
        if root.exists() and (root / "train").exists() and (root / "val").exists():
            return True
    return False


def main() -> int:
    checks = {
        "uavdt_zip": ROOT / "data/raw/uavdt/UAVDT.zip",
        "uavdt_extracted_base": ROOT / "data/raw/uavdt",
        "visdrone_manual_archives": ROOT / "data/manual/visdrone",
        "visdrone_extracted_dir": ROOT / "data/raw/visdrone",
        "dota_manual_archives": ROOT / "data/manual/dota",
        "dota_extracted_dir": ROOT / "data/raw/dota",
    }

    vis_manual = checks["visdrone_manual_archives"].exists()
    vis_manual_archives = vis_manual and exists_any(
        checks["visdrone_manual_archives"], ["*.zip", "*.tar", "*.tar.gz"]
    )
    vis_manual_extracted = vis_manual and (
        (checks["visdrone_manual_archives"] / "VisDrone2019-DET-train").exists()
        and (checks["visdrone_manual_archives"] / "VisDrone2019-MOT-train").exists()
    )

    dota_manual = checks["dota_manual_archives"].exists()
    dota_manual_archives = dota_manual and exists_any(
        checks["dota_manual_archives"], ["*.zip", "*.tar", "*.tar.gz"]
    )
    dota_manual_extracted = dota_manual and (
        (checks["dota_manual_archives"] / "images").exists()
        and (checks["dota_manual_archives"] / "labelTxt").exists()
    )

    uavdt_zip_present = checks["uavdt_zip"].exists()
    uavdt_extracted_present = is_uavdt_extracted(checks["uavdt_extracted_base"])
    vis_extracted_present = checks["visdrone_extracted_dir"].exists() and any(
        checks["visdrone_extracted_dir"].iterdir()
    )
    dota_extracted_present = checks["dota_extracted_dir"].exists() and any(
        checks["dota_extracted_dir"].iterdir()
    )

    next_steps: list[str] = []
    if not uavdt_extracted_present:
        next_steps.append("UAVDT extraction: bash scripts/01_download_data.sh uavdt")
    if not vis_extracted_present:
        next_steps.append(
            "Manual VisDrone files -> data/manual/visdrone/, then: bash scripts/01_prepare_manual_data.sh"
        )
    if not dota_extracted_present:
        next_steps.append(
            "Manual DOTA files -> data/manual/dota/, then: bash scripts/01_prepare_manual_data.sh"
        )
    if not next_steps:
        next_steps.append("All required datasets look ready.")

    status = {
        "uavdt": {
            "zip_present": uavdt_zip_present,
            "extracted_present": uavdt_extracted_present,
            "zip_size_mb": round(checks["uavdt_zip"].stat().st_size / (1024 * 1024), 2)
            if uavdt_zip_present
            else 0.0,
        },
        "visdrone": {
            "manual_archives_present": vis_manual_archives,
            "manual_extracted_present": vis_manual_extracted,
            "extracted_present": vis_extracted_present,
        },
        "dota": {
            "manual_archives_present": dota_manual_archives,
            "manual_extracted_present": dota_manual_extracted,
            "extracted_present": dota_extracted_present,
        },
        "next_steps": next_steps,
    }

    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
