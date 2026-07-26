#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/PhysicalAI-SmartSpaces}"
YEAR="${YEAR:-2026}"
SPLIT="${SPLIT:-val}"
SCENE="${SCENE:-Warehouse_020}"
MAX_FRAMES="${MAX_FRAMES:-120}"
OUT="${OUT:-runs/oracle_${SPLIT}_track1.txt}"

python -m physicalai_track1 gt-to-submission \
  --data-root "$DATA_ROOT" \
  --year "$YEAR" \
  --split "$SPLIT" \
  --scenes "$SCENE" \
  --max-frames-per-scene "$MAX_FRAMES" \
  --decimals 6 \
  --out "$OUT"

python -m physicalai_track1 validate --submission "$OUT"

python -m physicalai_track1 eval \
  --data-root "$DATA_ROOT" \
  --year "$YEAR" \
  --split "$SPLIT" \
  --scenes "$SCENE" \
  --max-frames-per-scene "$MAX_FRAMES" \
  --pred "$OUT"
