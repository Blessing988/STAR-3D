# PBS Job Guide

Submit jobs from the project root:

```bash
cd /path/to/PhysicalAI_Track1
qsub scripts/pbs/setup_env.pbs
qsub scripts/pbs/smoke_oracle_eval.pbs
qsub scripts/pbs/smoke_geometric_baseline.pbs
```

The project conda environment path is:

```text
/path/to/conda_envs/physicalai_track1
```

The default `setup_env.pbs` creates a lean baseline environment for dataset
inspection, geometric lifting, fusion, tracking, and evaluation. It does not
install PyTorch/Ultralytics, because solving a full GPU stack with the cluster's
old `conda 4.12.0` can hang for a long time.

Use this separate environment for detector training:

```text
/path/to/conda_envs/physicalai_track1_detector
```

```bash
qsub scripts/pbs/setup_detector_env_h100.pbs
```

Use `freenodes -cg` before heavy jobs:

```bash
freenodes -cg
bash scripts/hpc/recommend_pbs_node.sh train-large
bash scripts/hpc/gpu_memory_snapshot.sh
```

Observed on 2026-06-07:

- `gpu0063` has NVIDIA H100 NVL GPUs with about 95,830 MiB VRAM each.
- `gpu0063` belongs to `condo08,preemptible`, so H100 jobs should normally use
  `#PBS -q preemptible` with `host=gpu0063`.
- `gpu0036` has NVIDIA A100-PCIE-40GB GPUs and belongs to `gpus,preemptible`.
- For ordinary detector jobs, use `gpus` first. For very high-memory training,
  use H100 through `preemptible` when enough GPUs are free.
- For CPU/high-memory preprocessing, use `bigmem` or `preemptible` without
  requesting GPUs.

The templates are intentionally editable. If `freenodes -cg` shows a better
node, change the `#PBS -q` and `#PBS -l select=...:host=...` lines before
submitting.

## Detector Dataset Preparation

Prepare labels and manifests first:

```bash
qsub scripts/pbs/export_yolo_trainval.pbs
```

Default output is on scratch:

```text
/path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/
```

A project symlink is also created at:

```text
/path/to/PhysicalAI_Track1/runs/yolo_2026_stride30
```

After the label job finishes, inspect manifest sizes:

```bash
wc -l /path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/*_frames.tsv
du -sh /path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30
```

Then extract frames one manifest at a time:

```bash
qsub -v MANIFEST=/path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/train_frames.tsv scripts/pbs/extract_yolo_frames.pbs
qsub -v MANIFEST=/path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/val_frames.tsv scripts/pbs/extract_yolo_frames.pbs
```

Use `FRAME_STRIDE=15` or `FRAME_STRIDE=10` only after the stride-30 detector
baseline is working.

## Detector-To-STAR Validation

Run YOLO inference and the full 3D pipeline with:

```bash
qsub scripts/pbs/infer_yolo_val_to_star.pbs
```

Important environment overrides:

```text
MODEL       trained Ultralytics checkpoint
IMGSZ       inference size, default 1536
CONF        low detector threshold, default 0.01
IOU         detector/class-wise NMS threshold, default 0.70
SCENES      validation scene names
MAX_FRAMES  first N source frames per scene
FRAME_STRIDE sampled validation stride, default 30
BATCH       inference chunk/batch size, default 8
FUSE_CLASS_DISTANCE_M  class-specific 3D fusion radii
TRACK_CLASS_MIN_SCORES class-specific confidence gates
```

The default output layout is:

```text
runs/inference/<run_name>/detections.tsv       common per-camera detections
runs/inference/<run_name>/star/lifted.tsv      per-view 3D candidates
runs/inference/<run_name>/star/fused.tsv       fused frame detections
runs/inference/<run_name>/star/track1_star_online.txt
runs/inference/<run_name>/star/eval.json
```

For a promotion gate on any new checkpoint, use the fixed best-known YOLO +
STAR settings:

```bash
qsub -v MODEL=/path/to/best.pt,RUN_NAME=yolo_candidate_01 scripts/pbs/eval_yolo_checkpoint_to_star.pbs
```

This runs detector inference on the sampled validation manifest, applies the
current selected post-processing defaults
`FUSE_CLASS_DISTANCE_M=0:1.2,1:2.0,6:2.5` and
`TRACK_CLASS_MIN_SCORES=0:0.6,1:0.8,6:0.4`, then writes
`star/eval.json`.

## Test Submission

The 2026 test split has videos and calibration but no `ground_truth.json`.
Prepare a no-label manifest and extract frames to scratch:

```bash
qsub scripts/pbs/prepare_yolo_test_frames.pbs
```

Default output:

```text
/path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_test_stride1/
```

`FRAME_STRIDE=1` is the correct default for a real leaderboard submission. For
a fast smoke test only, run:

```bash
qsub -v FRAME_STRIDE=30,MAX_FRAMES_PER_SCENE=600 scripts/pbs/prepare_yolo_test_frames.pbs
```

Create the test submission from a detector checkpoint:

```bash
qsub -v MODEL=/path/to/best.pt,RUN_NAME=yolo_test_01 scripts/pbs/infer_yolo_test_to_submission.pbs
```

The submission job runs `infer-yolo-manifest`, lifts detections with test
calibration, fuses and tracks in STAR with `DO_EVAL=0`, validates the text
format, and writes:

```text
runs/submissions/<run_name>/track1_test_submission.txt
runs/submissions/<run_name>/track1_test_submission.txt.zip
```

Use `DATASET_DIR`, `MANIFEST`, and `IMAGE_DIR` overrides if you prepared a
different test stride or a dry-run subset.

For D-FINE, export standard COCO result JSON and convert it with:

```bash
python -m physicalai_track1 import-coco-predictions \
  --predictions path/to/results.json \
  --annotations /path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/annotations/instances_val_dfine.json \
  --manifest /path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/val_frames.tsv \
  --out path/to/detections.tsv
```

When predictions were generated from stride-30 frames, evaluation must also use
`--frame-stride 30`; otherwise unsampled GT frames are incorrectly counted as
false negatives.

Direct D-FINE checkpoint inference plus STAR evaluation is available with:

```bash
qsub scripts/pbs/infer_dfine_val_to_star.pbs
```

It defaults to `best_stg1.pth` from the active 960-pixel D-FINE-L run and can
be overridden with `CHECKPOINT`, `CONFIG`, `BATCH`, and `CONF`.

After a detector run has produced `star/fused.tsv`, tune confidence and
association without rerunning GPU inference:

```bash
qsub -v FUSED=/path/to/star/fused.tsv scripts/pbs/sweep_star_tracker.pbs
```

The sweep writes ranked metrics and the best submission to
`runs/sweeps/<run_name>/`.

For detectors whose score distributions differ strongly by class, run:

```bash
qsub -v FUSED=/path/to/star/fused.tsv scripts/pbs/sweep_star_class_thresholds.pbs
```

Tune class-specific multi-view clustering on cached lifted candidates:

```bash
qsub -v LIFTED=/path/to/star/lifted.tsv scripts/pbs/sweep_star_class_fusion.pbs
```

This writes `class_fusion_sweep_results.json`, `fused_best.tsv`, and
`track1_best_class_fusion.txt` under the selected scratch run directory.

## Learned Geometry Residual

Build supervised residual targets on scratch:

```bash
qsub scripts/pbs/build_geometry_residual_dataset.pbs
```

Then train the uncertainty-aware MLP:

```bash
qsub -W depend=afterok:<dataset_job_id> scripts/pbs/train_geometry_residual.pbs
```

The exported `.npz` model runs with NumPy in the lean geometry environment.
Evaluate it by setting:

```bash
RESIDUAL_MODEL=/path/to/scratch/PhysicalAI_Track1/runs/geometry_residual/geometry_mlp_stride15.npz \
bash scripts/run_star_geometry_val.sh
```
