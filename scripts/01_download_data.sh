#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/raw"

mkdir -p "$DATA_DIR/uavdt" "$DATA_DIR/visdrone" "$DATA_DIR/dota" "$DATA_DIR/manual"

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Data dir: $DATA_DIR"

download_uavdt() {
  local out_zip="$DATA_DIR/uavdt/UAVDT.zip"
  if [[ -f "$out_zip" ]]; then
    echo "[INFO] UAVDT.zip exists, resuming download if incomplete: $out_zip"
    curl -L -C - "https://zenodo.org/records/14575517/files/UAVDT.zip?download=1" -o "$out_zip"
  else
    echo "[INFO] Downloading UAVDT (Zenodo direct)..."
    curl -L -C - "https://zenodo.org/records/14575517/files/UAVDT.zip?download=1" -o "$out_zip"
  fi

  if [[ ! -d "$DATA_DIR/uavdt/UAVDT" && ! -d "$DATA_DIR/uavdt/UAV-benchmark-M" ]]; then
    echo "[INFO] Extracting UAVDT..."
    unzip -q "$out_zip" -d "$DATA_DIR/uavdt"
  else
    echo "[SKIP] UAVDT already extracted."
  fi
}

manual_required_notice() {
  cat <<'TXT'

[MANUAL REQUIRED DATASETS]
1) VisDrone (registration/drive mirrors can require manual steps)
   - Official: https://github.com/VisDrone/VisDrone-Dataset
   - Put archives under: data/manual/visdrone/

2) DOTA (license/terms check and source mirrors)
   - Official: https://captain-whu.github.io/DOTA/dataset.html
   - Put archives under: data/manual/dota/

3) xView / AI-TOD / SODA-A / TinyBenchmark
   - Often include registration, NC terms or external drive steps.
   - Put archives under: data/manual/<dataset_name>/

After manual download, run:
  bash scripts/01_prepare_manual_data.sh
TXT
}

case "${1:-all}" in
  uavdt)
    download_uavdt
    ;;
  all)
    download_uavdt
    manual_required_notice
    ;;
  *)
    echo "Usage: $0 [uavdt|all]"
    exit 1
    ;;
esac

echo "[DONE]"
