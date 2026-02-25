#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/weights"
mkdir -p "$WEIGHTS_DIR"

download() {
  local name="$1"
  local url="$2"
  local out="$WEIGHTS_DIR/$name"
  echo "[INFO] Downloading $name ..."
  curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
  ls -lah "$out"
}

case "${1:-all}" in
  yolov8n)
    download "yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
    ;;
  yolov8s)
    download "yolov8s.pt" "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt"
    ;;
  all)
    download "yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
    download "yolov8s.pt" "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt"
    ;;
  *)
    echo "Usage: $0 [yolov8n|yolov8s|all]"
    exit 1
    ;;
esac

echo "[DONE] Weights ready under $WEIGHTS_DIR"
