#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/PhysicalAI-SmartSpaces}"
SCENE="${SCENE:-Warehouse_020}"
MAX_FRAMES="${MAX_FRAMES:-120}"
RUN_DIR="${RUN_DIR:-runs/geometric_smoke}"

mkdir -p "$RUN_DIR"

python -m physicalai_track1 build-priors \
  --data-root "$DATA_ROOT" \
  --year 2026 \
  --split train \
  --max-frames-per-scene 300 \
  --out "$RUN_DIR/priors_train_sample.json"

python -m physicalai_track1 export-gt-2d \
  --data-root "$DATA_ROOT" \
  --year 2026 \
  --split val \
  --scenes "$SCENE" \
  --max-frames-per-scene "$MAX_FRAMES" \
  --out "$RUN_DIR/oracle_2d.tsv"

python -m physicalai_track1 lift-2d \
  --data-root "$DATA_ROOT" \
  --year 2026 \
  --split val \
  --detections "$RUN_DIR/oracle_2d.tsv" \
  --priors "$RUN_DIR/priors_train_sample.json" \
  --out "$RUN_DIR/lifted.tsv"

python -m physicalai_track1 fuse-3d \
  --lifted "$RUN_DIR/lifted.tsv" \
  --distance-m 1.5 \
  --min-sources 1 \
  --out "$RUN_DIR/fused.tsv"

python -m physicalai_track1 track-online \
  --fused "$RUN_DIR/fused.tsv" \
  --max-distance-m 2.5 \
  --max-age 45 \
  --decimals 6 \
  --out "$RUN_DIR/track1_geometric_baseline.txt"

python -m physicalai_track1 validate \
  --submission "$RUN_DIR/track1_geometric_baseline.txt"

python -m physicalai_track1 eval \
  --data-root "$DATA_ROOT" \
  --year 2026 \
  --split val \
  --scenes "$SCENE" \
  --max-frames-per-scene "$MAX_FRAMES" \
  --pred "$RUN_DIR/track1_geometric_baseline.txt"

