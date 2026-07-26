#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/PhysicalAI-SmartSpaces}"
YEAR="${YEAR:-2026}"
SPLIT="${SPLIT:-val}"
SCENES="${SCENES:-Warehouse_020 Warehouse_021 Warehouse_022}"
MAX_FRAMES="${MAX_FRAMES:-600}"
RUN_DIR="${RUN_DIR:-/path/to/scratch/PhysicalAI_Track1/runs/star_geometry_val}"
DETECTIONS="${DETECTIONS:-}"
DO_EVAL="${DO_EVAL:-auto}"
RESIDUAL_MODEL="${RESIDUAL_MODEL:-}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-1.0}"
RESIDUAL_MAX_UNCERTAINTY="${RESIDUAL_MAX_UNCERTAINTY:-}"
LIFT_USE_DEPTH="${LIFT_USE_DEPTH:-0}"
LIFT_DEPTH_SCALE="${LIFT_DEPTH_SCALE:-0.001}"
LIFT_DEPTH_ROOT="${LIFT_DEPTH_ROOT:-}"
LIFT_DEPTH_MODE="${LIFT_DEPTH_MODE:-none}"
LIFT_DEPTH_BLEND_ALPHA="${LIFT_DEPTH_BLEND_ALPHA:-0.65}"
LIFT_DEPTH_PERCENTILE="${LIFT_DEPTH_PERCENTILE:-35.0}"
LIFT_DEPTH_MAX_REPROJECTION_PX="${LIFT_DEPTH_MAX_REPROJECTION_PX:-80.0}"
STATIC_DEPTH_ROOT="${STATIC_DEPTH_ROOT:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
SUBMISSION_DECIMALS="${SUBMISSION_DECIMALS:-6}"

FUSE_DISTANCE_M="${FUSE_DISTANCE_M:-1.35}"
FUSE_CLASS_DISTANCE_M="${FUSE_CLASS_DISTANCE_M:-}"
FUSE_MIN_SOURCES="${FUSE_MIN_SOURCES:-1}"
FUSE_MERGE_IOU="${FUSE_MERGE_IOU:-0.08}"
FUSE_NMS_IOU="${FUSE_NMS_IOU:-0.35}"
FUSE_NMS_DISTANCE_M="${FUSE_NMS_DISTANCE_M:-0.25}"
FUSE_SINGLE_CAMERA_SCORE_FACTOR="${FUSE_SINGLE_CAMERA_SCORE_FACTOR:-0.92}"
FUSE_STREAMING_SORTED="${FUSE_STREAMING_SORTED:-0}"
FUSE_SORT_TMP_DIR="${FUSE_SORT_TMP_DIR:-$RUN_DIR/sort_tmp}"

TRACK_MAX_DISTANCE_M="${TRACK_MAX_DISTANCE_M:-2.2}"
TRACK_MAX_AGE="${TRACK_MAX_AGE:-45}"
TRACK_MIN_SCORE="${TRACK_MIN_SCORE:-0.0}"
TRACK_CLASS_MIN_SCORES="${TRACK_CLASS_MIN_SCORES:-}"
TRACK_CLASS_MAX_DISTANCES_M="${TRACK_CLASS_MAX_DISTANCES_M:-}"
TRACK_CLASS_MAX_COSTS="${TRACK_CLASS_MAX_COSTS:-}"
TRACK_CLASS_MAX_AGES="${TRACK_CLASS_MAX_AGES:-}"
TRACK_MAX_COST="${TRACK_MAX_COST:-1.35}"
TRACK_DISTANCE_WEIGHT="${TRACK_DISTANCE_WEIGHT:-1.0}"
TRACK_IOU_WEIGHT="${TRACK_IOU_WEIGHT:-0.35}"
TRACK_YAW_WEIGHT="${TRACK_YAW_WEIGHT:-0.08}"
TRACK_SCORE_WEIGHT="${TRACK_SCORE_WEIGHT:-0.18}"
TRACK_SOURCE_WEIGHT="${TRACK_SOURCE_WEIGHT:-0.12}"
TRACK_POSITION_ALPHA="${TRACK_POSITION_ALPHA:-0.85}"
TRACK_VELOCITY_ALPHA="${TRACK_VELOCITY_ALPHA:-0.70}"
TRACK_CONFIRMATION_HITS="${TRACK_CONFIRMATION_HITS:-1}"
TRACK_CONFIRMATION_MODE="${TRACK_CONFIRMATION_MODE:-immediate}"
TRACK_CLASS_CONFIRMATION_HITS="${TRACK_CLASS_CONFIRMATION_HITS:-}"
TRACK_DUPLICATE_BIRTH_DISTANCE_M="${TRACK_DUPLICATE_BIRTH_DISTANCE_M:-0.0}"
TRACK_CLASS_DUPLICATE_BIRTH_DISTANCES_M="${TRACK_CLASS_DUPLICATE_BIRTH_DISTANCES_M:-}"
TRACK_DUPLICATE_BIRTH_IOU="${TRACK_DUPLICATE_BIRTH_IOU:-0.0}"
TRACK_IMMEDIATE_BIRTH_SCORE="${TRACK_IMMEDIATE_BIRTH_SCORE:-}"
TRACK_CLASS_IMMEDIATE_BIRTH_SCORES="${TRACK_CLASS_IMMEDIATE_BIRTH_SCORES:-}"
TRACK_IMMEDIATE_BIRTH_MIN_SOURCES="${TRACK_IMMEDIATE_BIRTH_MIN_SOURCES:-}"

mkdir -p "$RUN_DIR"

if [[ "$DO_EVAL" == "auto" ]]; then
  if [[ "$SPLIT" == "test" ]]; then
    DO_EVAL=0
  else
    DO_EVAL=1
  fi
fi

echo "run_dir=$RUN_DIR"
echo "data_root=$DATA_ROOT"
echo "year=$YEAR split=$SPLIT scenes=$SCENES max_frames=$MAX_FRAMES"
echo "detections=${DETECTIONS:-oracle_2d}"
echo "do_eval=$DO_EVAL submission_decimals=$SUBMISSION_DECIMALS"

python -m physicalai_track1 build-priors \
  --data-root "$DATA_ROOT" \
  --year "$YEAR" \
  --split train \
  --max-frames-per-scene 900 \
  --out "$RUN_DIR/priors_train_sample.json"

if [[ -z "$DETECTIONS" && "$SPLIT" == "test" ]]; then
  echo "SPLIT=test requires DETECTIONS because no test ground truth is available." >&2
  exit 2
fi

if [[ -z "$DETECTIONS" ]]; then
  python -m physicalai_track1 export-gt-2d \
    --data-root "$DATA_ROOT" \
    --year "$YEAR" \
    --split "$SPLIT" \
    --scenes $SCENES \
    --frame-stride "$FRAME_STRIDE" \
    --max-frames-per-scene "$MAX_FRAMES" \
    --out "$RUN_DIR/oracle_2d.tsv"
  DETECTIONS="$RUN_DIR/oracle_2d.tsv"
fi

LIFT_ARGS=(python -m physicalai_track1 lift-2d \
  --data-root "$DATA_ROOT" \
  --year "$YEAR" \
  --split "$SPLIT" \
  --detections "$DETECTIONS" \
  --priors "$RUN_DIR/priors_train_sample.json" \
  --out "$RUN_DIR/lifted.tsv")
if [[ -n "$RESIDUAL_MODEL" ]]; then
  LIFT_ARGS+=(--residual-model "$RESIDUAL_MODEL" --residual-scale "$RESIDUAL_SCALE")
fi
if [[ -n "$RESIDUAL_MAX_UNCERTAINTY" ]]; then
  LIFT_ARGS+=(--max-residual-uncertainty "$RESIDUAL_MAX_UNCERTAINTY")
fi
if [[ "$LIFT_USE_DEPTH" == "1" ]]; then
  LIFT_ARGS+=(--use-depth --depth-scale "$LIFT_DEPTH_SCALE")
fi
if [[ -n "$LIFT_DEPTH_ROOT" ]]; then
  LIFT_ARGS+=(--depth-root "$LIFT_DEPTH_ROOT")
fi
if [[ "$LIFT_DEPTH_MODE" != "none" ]]; then
  LIFT_ARGS+=(--depth-lift-mode "$LIFT_DEPTH_MODE" --depth-scale "$LIFT_DEPTH_SCALE" --depth-blend-alpha "$LIFT_DEPTH_BLEND_ALPHA" --depth-percentile "$LIFT_DEPTH_PERCENTILE" --depth-max-reprojection-px "$LIFT_DEPTH_MAX_REPROJECTION_PX")
fi
if [[ -n "$STATIC_DEPTH_ROOT" ]]; then
  LIFT_ARGS+=(--static-depth-root "$STATIC_DEPTH_ROOT")
fi
"${LIFT_ARGS[@]}"

LIFTED_FOR_FUSION="$RUN_DIR/lifted.tsv"
if [[ "$FUSE_STREAMING_SORTED" == "1" ]]; then
  mkdir -p "$FUSE_SORT_TMP_DIR"
  LIFTED_FOR_FUSION="$RUN_DIR/lifted.sorted.tsv"
  if [[ ! -s "$LIFTED_FOR_FUSION" ]]; then
    {
      head -1 "$RUN_DIR/lifted.tsv"
      tail -n +2 "$RUN_DIR/lifted.tsv" | sort -T "$FUSE_SORT_TMP_DIR" -S 50% -t $'\t' -k1,1n -k2,2n -k3,3n
    } > "$LIFTED_FOR_FUSION.tmp"
    mv "$LIFTED_FOR_FUSION.tmp" "$LIFTED_FOR_FUSION"
  fi
fi

FUSE_ARGS=(python -m physicalai_track1 fuse-3d \
  --lifted "$LIFTED_FOR_FUSION" \
  --distance-m "$FUSE_DISTANCE_M" \
  --min-sources "$FUSE_MIN_SOURCES" \
  --merge-iou "$FUSE_MERGE_IOU" \
  --nms-iou "$FUSE_NMS_IOU" \
  --nms-distance-m "$FUSE_NMS_DISTANCE_M" \
  --single-camera-score-factor "$FUSE_SINGLE_CAMERA_SCORE_FACTOR" \
  --out "$RUN_DIR/fused.tsv")
if [[ -n "$FUSE_CLASS_DISTANCE_M" ]]; then
  FUSE_ARGS+=(--class-distance-m "$FUSE_CLASS_DISTANCE_M")
fi
if [[ "$FUSE_STREAMING_SORTED" == "1" ]]; then
  FUSE_ARGS+=(--streaming-sorted)
fi
"${FUSE_ARGS[@]}"

TRACK_ARGS=(python -m physicalai_track1 track-online \
  --fused "$RUN_DIR/fused.tsv" \
  --max-distance-m "$TRACK_MAX_DISTANCE_M" \
  --max-age "$TRACK_MAX_AGE" \
  --min-score "$TRACK_MIN_SCORE" \
  --max-cost "$TRACK_MAX_COST" \
  --distance-weight "$TRACK_DISTANCE_WEIGHT" \
  --iou-weight "$TRACK_IOU_WEIGHT" \
  --yaw-weight "$TRACK_YAW_WEIGHT" \
  --score-weight "$TRACK_SCORE_WEIGHT" \
  --source-weight "$TRACK_SOURCE_WEIGHT" \
  --position-alpha "$TRACK_POSITION_ALPHA" \
  --velocity-alpha "$TRACK_VELOCITY_ALPHA" \
  --confirmation-hits "$TRACK_CONFIRMATION_HITS" \
  --confirmation-mode "$TRACK_CONFIRMATION_MODE" \
  --duplicate-birth-distance-m "$TRACK_DUPLICATE_BIRTH_DISTANCE_M" \
  --duplicate-birth-iou "$TRACK_DUPLICATE_BIRTH_IOU" \
  --decimals "$SUBMISSION_DECIMALS" \
  --out "$RUN_DIR/track1_star_online.txt")
if [[ -n "$TRACK_CLASS_MIN_SCORES" ]]; then
  TRACK_ARGS+=(--class-min-scores "$TRACK_CLASS_MIN_SCORES")
fi
if [[ -n "$TRACK_CLASS_MAX_DISTANCES_M" ]]; then
  TRACK_ARGS+=(--class-max-distances-m "$TRACK_CLASS_MAX_DISTANCES_M")
fi
if [[ -n "$TRACK_CLASS_MAX_COSTS" ]]; then
  TRACK_ARGS+=(--class-max-costs "$TRACK_CLASS_MAX_COSTS")
fi
if [[ -n "$TRACK_CLASS_MAX_AGES" ]]; then
  TRACK_ARGS+=(--class-max-ages "$TRACK_CLASS_MAX_AGES")
fi
if [[ -n "$TRACK_CLASS_CONFIRMATION_HITS" ]]; then
  TRACK_ARGS+=(--class-confirmation-hits "$TRACK_CLASS_CONFIRMATION_HITS")
fi
if [[ -n "$TRACK_CLASS_DUPLICATE_BIRTH_DISTANCES_M" ]]; then
  TRACK_ARGS+=(--class-duplicate-birth-distances-m "$TRACK_CLASS_DUPLICATE_BIRTH_DISTANCES_M")
fi
if [[ -n "$TRACK_IMMEDIATE_BIRTH_SCORE" ]]; then
  TRACK_ARGS+=(--immediate-birth-score "$TRACK_IMMEDIATE_BIRTH_SCORE")
fi
if [[ -n "$TRACK_CLASS_IMMEDIATE_BIRTH_SCORES" ]]; then
  TRACK_ARGS+=(--class-immediate-birth-scores "$TRACK_CLASS_IMMEDIATE_BIRTH_SCORES")
fi
if [[ -n "$TRACK_IMMEDIATE_BIRTH_MIN_SOURCES" ]]; then
  TRACK_ARGS+=(--immediate-birth-min-sources "$TRACK_IMMEDIATE_BIRTH_MIN_SOURCES")
fi
"${TRACK_ARGS[@]}"

python -m physicalai_track1 validate \
  --submission "$RUN_DIR/track1_star_online.txt" \
  > "$RUN_DIR/validate.json"

if [[ "$DO_EVAL" == "1" ]]; then
  python -m physicalai_track1 eval \
    --data-root "$DATA_ROOT" \
    --year "$YEAR" \
    --split "$SPLIT" \
    --scenes $SCENES \
    --max-frames-per-scene "$MAX_FRAMES" \
    --frame-stride "$FRAME_STRIDE" \
    --pred "$RUN_DIR/track1_star_online.txt" \
    > "$RUN_DIR/eval.json"
else
  printf '{"eval_skipped": true, "reason": "split has no local ground truth or DO_EVAL=0"}\n' > "$RUN_DIR/eval.json"
fi

cat "$RUN_DIR/validate.json"
cat "$RUN_DIR/eval.json"
