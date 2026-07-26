#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${MANIFEST:-runs/yolo_smoke/val_frames.tsv}"

python -m physicalai_track1 extract-frames --manifest "$MANIFEST"

