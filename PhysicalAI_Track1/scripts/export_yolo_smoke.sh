#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/PhysicalAI-SmartSpaces}"
OUT_DIR="${OUT_DIR:-runs/yolo_smoke}"

python -m physicalai_track1 export-yolo \
  --data-root "$DATA_ROOT" \
  --year 2026 \
  --split val \
  --scenes Warehouse_020 \
  --frame-stride 60 \
  --max-frames-per-scene 180 \
  --output-dir "$OUT_DIR"

