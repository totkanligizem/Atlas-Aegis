#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANUAL_DIR="$ROOT_DIR/data/manual"
RAW_DIR="$ROOT_DIR/data/raw"

mkdir -p "$RAW_DIR/visdrone" "$RAW_DIR/dota"

echo "[INFO] Preparing manually downloaded archives..."

if compgen -G "$MANUAL_DIR/visdrone/*.zip" > /dev/null; then
  for z in "$MANUAL_DIR"/visdrone/*.zip; do
    echo "[INFO] Extracting VisDrone archive: $z"
    unzip -q "$z" -d "$RAW_DIR/visdrone"
  done
else
  if [[ -d "$MANUAL_DIR/visdrone/VisDrone2019-DET-train" ]] || [[ -d "$MANUAL_DIR/visdrone/VisDrone2019-MOT-train" ]]; then
    echo "[INFO] No VisDrone zip found; using existing extracted folders in $MANUAL_DIR/visdrone"
  else
    echo "[WARN] No VisDrone zip found in $MANUAL_DIR/visdrone"
  fi
fi

if compgen -G "$MANUAL_DIR/dota/*.zip" > /dev/null; then
  for z in "$MANUAL_DIR"/dota/*.zip; do
    echo "[INFO] Extracting DOTA archive: $z"
    unzip -q "$z" -d "$RAW_DIR/dota"
  done
else
  if [[ -d "$MANUAL_DIR/dota/images" ]] || compgen -G "$MANUAL_DIR/dota/images-*" > /dev/null; then
    echo "[INFO] No DOTA zip found; using existing extracted folders in $MANUAL_DIR/dota"
  else
    echo "[WARN] No DOTA zip found in $MANUAL_DIR/dota"
  fi
fi

# If user already extracted datasets manually under data/manual,
# create symlinks in data/raw to avoid duplicated storage.
if [[ -d "$MANUAL_DIR/visdrone/VisDrone2019-DET-train" ]] && [[ -d "$MANUAL_DIR/visdrone/VisDrone2019-MOT-train" ]]; then
  if [[ -z "$(ls -A "$RAW_DIR/visdrone" 2>/dev/null || true)" ]]; then
    echo "[INFO] Linking manual VisDrone extraction into data/raw/visdrone"
    rm -rf "$RAW_DIR/visdrone"
    ln -s "$MANUAL_DIR/visdrone" "$RAW_DIR/visdrone"
  fi
fi

if [[ -d "$MANUAL_DIR/dota/images" ]] && [[ -d "$MANUAL_DIR/dota/labelTxt" ]]; then
  if [[ -z "$(ls -A "$RAW_DIR/dota" 2>/dev/null || true)" ]]; then
    echo "[INFO] Linking manual DOTA extraction into data/raw/dota"
    rm -rf "$RAW_DIR/dota"
    ln -s "$MANUAL_DIR/dota" "$RAW_DIR/dota"
  fi
fi

echo "[DONE] Manual dataset prep complete."
