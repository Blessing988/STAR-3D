# Reproducing STAR-3D

This guide describes the intended path for reproducing the STAR-3D pipeline on a
PBS-based HPC cluster.

## 1. Environment

Install the lightweight package first:

```bash
cd PhysicalAI_Track1
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

On CCAST/PBS, the project scripts create separate conda environments for the
core geometry pipeline, detector training, D-FINE, depth experiments, and
V-DETR. These environments are separated because their CUDA and compiler
requirements conflict.

```bash
qsub scripts/pbs/setup_env.pbs
qsub scripts/pbs/setup_detector_env_h100.pbs
qsub scripts/pbs/setup_rfdetr_env.pbs
qsub scripts/pbs/setup_depth_pro_env.pbs
qsub scripts/pbs/setup_vdetr_env.pbs
```

## 2. Dataset

Download AI City Challenge 2026 Track 1 data from the NVIDIA PhysicalAI Smart
Spaces Hugging Face repository:

- Dataset card:
  <https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/blob/main/README.md>
- Train:
  <https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/tree/main/MTMC_Tracking_2026/train>
- Validation:
  <https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/tree/main/MTMC_Tracking_2026/val>

Recommended RGB-only download:

```bash
pip install -U huggingface_hub

export PHYSICALAI_DATA_ROOT=/path/to/PhysicalAI-SmartSpaces

hf download nvidia/PhysicalAI-SmartSpaces \
  --repo-type dataset \
  --include "MTMC_Tracking_2026/**" \
  --exclude "MTMC_Tracking_2026/**/depth_maps/**" \
  --local-dir "$PHYSICALAI_DATA_ROOT"
```

Set paths:

```bash
export STAR3D_SCRATCH=/path/to/scratch/PhysicalAI_Track1
```

Expected layout:

```text
$PHYSICALAI_DATA_ROOT/MTMC_Tracking_2026/
  train/Warehouse_000/ ... Warehouse_019/
  val/Warehouse_020/ ... Warehouse_022/
  test/Warehouse_023/ ... Warehouse_027/
```

Train and validation scenes must include `videos/`, `ground_truth.json`,
`calibration.json`, and `map.png`; `depth_maps/` are optional for the RGB-only
pipeline. Test scenes need videos and calibration for inference.

## 3. Smoke Tests

Run these before launching detector jobs:

```bash
python -m physicalai_track1 inspect \
  --data-root "$PHYSICALAI_DATA_ROOT" \
  --year 2026

python -m physicalai_track1 gt-to-submission \
  --data-root "$PHYSICALAI_DATA_ROOT" \
  --year 2026 \
  --split val \
  --frame-stride 30 \
  --out "$STAR3D_SCRATCH/oracle_val_stride30.txt"

python -m physicalai_track1 eval \
  --data-root "$PHYSICALAI_DATA_ROOT" \
  --year 2026 \
  --split val \
  --frame-stride 30 \
  --pred "$STAR3D_SCRATCH/oracle_val_stride30.txt" \
  --by-class
```

The oracle run is a format and coordinate-system sanity check.

## 4. Prepare Detector Data

```bash
qsub scripts/pbs/export_yolo_trainval.pbs

qsub -v MANIFEST=$STAR3D_SCRATCH/datasets/yolo_2026_stride30/train_frames.tsv \
  scripts/pbs/extract_yolo_frames.pbs

qsub -v MANIFEST=$STAR3D_SCRATCH/datasets/yolo_2026_stride30/val_frames.tsv \
  scripts/pbs/extract_yolo_frames.pbs
```

The default stride-30 set is for development and validation. Use stride 1 for
full test submissions.

## 5. Detector Training

YOLO training:

```bash
qsub scripts/pbs/train_yolo_h100_preemptible.pbs
```

D-FINE training/inference:

```bash
qsub scripts/pbs/train_dfine_l_h100.pbs
qsub scripts/pbs/infer_dfine_val_to_star.pbs
```

RF-DETR/YOLOR/YOLOv5-CBAM experiments are represented by their own scripts in
`scripts/pbs/`.

## 6. Validation Pipeline

A normal STAR-3D validation run converts per-camera detections into the common
TSV format, lifts them into 3D, fuses multi-view candidates, tracks online, and
evaluates against validation GT.

```bash
qsub scripts/pbs/ensemble_yolo_dfine_val_to_star.pbs
```

Cached sweeps can tune association and fusion without rerunning GPU inference:

```bash
qsub -v FUSED=/path/to/star/fused.tsv scripts/pbs/sweep_star_tracker.pbs
qsub -v LIFTED=/path/to/star/lifted.tsv scripts/pbs/sweep_star_class_fusion.pbs
```

## 7. Test Submission

Prepare full test frames:

```bash
qsub scripts/pbs/prepare_yolo_test_frames.pbs
```

Create a submission:

```bash
qsub scripts/pbs/ensemble_yolo_dfine_test_submission.pbs
```

Validate the output before zipping or submitting:

```bash
python -m physicalai_track1 validate \
  --submission /path/to/track1.txt
```

The final file must be named `track1.txt` inside the submitted zip.

## 8. Qualitative Figures

Generate BEV and RGB projection visualizations:

```bash
python scripts/visualize_submission_qualitative.py \
  --submission /path/to/track1.txt \
  --out-dir /path/to/figures

python scripts/visualize_multiview_projection.py \
  --submission /path/to/track1.txt \
  --dataset-root "$PHYSICALAI_DATA_ROOT/MTMC_Tracking_2026" \
  --frames-root "$STAR3D_SCRATCH/datasets/yolo_2026_test_stride1/images/test" \
  --scene-id 25 \
  --frame-id 3000 \
  --out /path/to/qual_multiview_scene25_frame3000.png
```

## 9. Notes

- Use `freenodes -cg` before submitting GPU jobs.
- Prefer the `gpus` queue when an appropriate node is available.
- Use H100/preemptible jobs for high-resolution detector training or V-DETR.
- Keep generated data on scratch, not in the Git repository.
