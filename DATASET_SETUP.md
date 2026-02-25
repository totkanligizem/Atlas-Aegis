# Dataset Setup (What I can do vs what you must do)

## 1) Automated by me
- `UAVDT` direct download from Zenodo (no login).
- Target path:
  - ZIP: `data/raw/uavdt/UAVDT.zip`
  - Extracted: `data/raw/uavdt/UAVDT/` (or legacy `data/raw/uavdt/UAV-benchmark-M/`)

Command (already started):
```bash
bash scripts/01_download_data.sh uavdt
```

## 2) Manual required by you (to avoid login/terms blockers)
Some datasets require registration/login or manual terms acceptance.

### VisDrone (manual)
1. Open: https://github.com/VisDrone/VisDrone-Dataset
2. Download required archives (DET/MOT as needed)
3. Place files under:
   - `data/manual/visdrone/`

### DOTA (manual)
1. Open: https://captain-whu.github.io/DOTA/dataset.html
2. Download required archives
3. Place files under:
   - `data/manual/dota/`

## 3) After manual download (I handle extraction)
Run:
```bash
bash scripts/01_prepare_manual_data.sh
```

This extracts archives to:
- `data/raw/visdrone/`
- `data/raw/dota/`

## 4) Quick verification
```bash
ls -lah data/raw/uavdt
ls -lah data/raw/visdrone
ls -lah data/raw/dota
```

## 5) Important legal reminder
- Free download does **not** always mean commercial-use allowed.
- Keep R&D and production data pipelines separate.
- Track dataset usage policy in `governance/legal_status.yaml`.
