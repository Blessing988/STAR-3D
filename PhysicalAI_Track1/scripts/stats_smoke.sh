#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/PhysicalAI-SmartSpaces}"

python -m physicalai_track1 stats \
  --data-root "$DATA_ROOT" \
  --year 2026 \
  --split val \
  --scenes Warehouse_020 \
  --max-frames-per-scene 180

