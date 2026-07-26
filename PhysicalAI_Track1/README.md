# PhysicalAI Track 1: Multi-Camera 3D Perception

This project is for AI City Challenge 2026 Track 1: Multi-Camera 3D
Perception (Sim2Real).

The challenge output is a single `track1.txt` file. Each line is:

```text
scene_id class_id object_id frame_id x y z width length height yaw
```

Important local paths on the HPC cluster:

```text
Dataset root:
/path/to/PhysicalAI-SmartSpaces

Project root:
/path/to/PhysicalAI_Track1

Scratch output root for prepared datasets/runs:
/path/to/scratch/PhysicalAI_Track1
```

The directory `/path/to/aicty_tar_dataset` is the
Track 3 Traffic Anomaly Reasoning dataset. It is not the Track 1 3D perception
dataset, although it matches the BCQ/MCQ leaderboard screenshot.

## Quick Start

From the HPC project root:

```bash
python -m physicalai_track1 inspect \
  --data-root /path/to/PhysicalAI-SmartSpaces \
  --year 2026

python -m physicalai_track1 gt-to-submission \
  --data-root /path/to/PhysicalAI-SmartSpaces \
  --year 2026 \
  --split val \
  --decimals 6 \
  --out runs/oracle_val_track1.txt

python -m physicalai_track1 eval \
  --data-root /path/to/PhysicalAI-SmartSpaces \
  --year 2026 \
  --split val \
  --pred runs/oracle_val_track1.txt
```

The oracle validation conversion should score near 1.0. That smoke test proves
that parsing, submission formatting, and local metric code agree on the basic
box convention.

For official submissions, do not use `--decimals 6`; the challenge page requires
all floating-point values to be rounded to two decimals, which is the default.

## PBS On The HPC

This cluster uses PBS. Submit project jobs from:

```bash
cd /path/to/PhysicalAI_Track1
```

Check free nodes and pick a template:

```bash
freenodes -cg
bash scripts/hpc/recommend_pbs_node.sh train-large
bash scripts/hpc/gpu_memory_snapshot.sh
```

Create or update the project conda environment:

```bash
qsub scripts/pbs/setup_env.pbs
```

Run the parser/evaluator smoke test through PBS:

```bash
qsub scripts/pbs/smoke_oracle_eval.pbs
qsub scripts/pbs/smoke_geometric_baseline.pbs
```

Export detector labels/frames and train a detector:

```bash
qsub scripts/pbs/export_yolo_trainval.pbs
# inspect scratch manifests, then extract frames:
qsub -v MANIFEST=/path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/train_frames.tsv scripts/pbs/extract_yolo_frames.pbs
qsub -v MANIFEST=/path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30/val_frames.tsv scripts/pbs/extract_yolo_frames.pbs
qsub scripts/pbs/train_yolo_gpus.pbs
# or, for high-VRAM H100 training:
qsub scripts/pbs/train_yolo_h100_preemptible.pbs
```

Run a trained YOLO checkpoint through the calibrated STAR validation pipeline:

```bash
qsub scripts/pbs/infer_yolo_val_to_star.pbs
```

The job performs YOLO inference directly into the common per-camera detection
TSV, lifts and fuses detections in global 3D, runs the online tracker, and
evaluates at the same sampled frame stride. Inference is manifest-filtered and
chunked so pilot runs do not process unrelated frames. Outputs are written
under:

```text
/path/to/scratch/PhysicalAI_Track1/runs/inference/
```

Evaluate any candidate checkpoint with the current best fixed STAR settings:

```bash
qsub -v MODEL=/path/to/best.pt,RUN_NAME=my_checkpoint_gate scripts/pbs/eval_yolo_checkpoint_to_star.pbs
```

Prepare the real 2026 test frames and create a leaderboard-format submission:

```bash
qsub scripts/pbs/prepare_yolo_test_frames.pbs
qsub -v MODEL=/path/to/best.pt,RUN_NAME=my_test_submission scripts/pbs/infer_yolo_test_to_submission.pbs
```

The test prep job defaults to `FRAME_STRIDE=1`, because the leaderboard
submission should cover the full test videos. Use `FRAME_STRIDE=30` only for a
quick dry run. The submission job writes both a text file and a best-effort zip
under:

```text
/path/to/scratch/PhysicalAI_Track1/runs/submissions/
```

Detector adapters can also be run independently:

```bash
python -m physicalai_track1 import-yolo-predictions --help
python -m physicalai_track1 import-coco-predictions --help
```

The COCO adapter is the standard entry point for D-FINE result JSON files.
For direct checkpoint inference, use:

```bash
qsub scripts/pbs/infer_dfine_val_to_star.pbs
```

Build and train the first learned geometry residual model:

```bash
qsub scripts/pbs/build_geometry_residual_dataset.pbs
qsub -W depend=afterok:<dataset_job_id> scripts/pbs/train_geometry_residual.pbs
```

This model learns center, height, dimensions, yaw, and uncertainty from
synthetic 3D GT and calibration. It exports a NumPy-compatible `.npz` model so
the normal CPU geometry pipeline does not require PyTorch.

Sweep conservative residual blending and uncertainty gates:

```bash
qsub scripts/pbs/sweep_geometry_residual_scale.pbs
```

The ranked result is written to
`runs/sweeps/geometry_residual_scale/residual_scale_sweep_results.json`.

Tune multi-view fusion per class on cached lifted candidates:

```bash
qsub -v LIFTED=/path/to/lifted.tsv scripts/pbs/sweep_star_class_fusion.pbs
```

The sweep learns separate BEV clustering radii and source-count constraints
for each validation class, then writes the best fused TSV and submission.
Apply selected radii in normal STAR runs with, for example,
`FUSE_CLASS_DISTANCE_M=0:1.2,1:2.0,6:2.5`.

See `scripts/pbs/README.md` for queue selection notes. Current observations:
`gpu0063` is an H100 NVL node with about 95 GB VRAM per GPU and is accessed via
`preemptible`/`host=gpu0063`; ordinary detector jobs should try `gpus` first.

## Why LLaMA Factory Is Not The Core Tool Here

Track 1 is a geometric 3D detection and multi-object tracking task scored by
3D HOTA. LLaMA Factory is useful for language/VLM fine-tuning, especially the
Track 3 TAR dataset, but it is not the correct core training stack for this
Track 1 detector/tracker. We can still use VLMs as an auxiliary component for
scene/camera metadata reasoning or paper narration, but the leaderboard score
will come from calibrated RGB detection, 3D lifting, cross-view fusion, and
identity association.

## Proposed System

The working name is `STAR-3D`: Sim2Real Temporal Association and Ray-guided 3D
tracking.

1. Train a strong RGB detector on synthetic 2D visible boxes.
2. Use calibration to lift detections into global 3D with learned center,
   height, size, and yaw priors.
3. Fuse multi-view hypotheses with camera deduplication, confidence and
   reprojection weighting, adaptive BEV/3D-IoU grouping, and 3D duplicate
   suppression.
4. Track online with global assignment using motion-predicted BEV distance,
   3D IoU, yaw, confidence, and distinct-camera support. Kalman/IMM and
   appearance association remain the next tracker upgrades.
5. Add an optional offline linker for leaderboard submissions, but keep a
   fully online configuration for the official +10 percent bonus.
6. Use synthetic-to-real domain adaptation: detector augmentations,
   style/lighting randomization, CT2.5 vs IsaacSim balancing, test-time
   enhancement, and self-training on high-confidence real test detections only
   if competition rules make that strategically worthwhile.

The 2026-specific strategy is RGB/calibration-first:

- The current downloaded `depth_maps` directories are empty, so do not depend on
  depth maps.
- Use RGB, calibration, ray geometry, class priors, and online multi-view fusion
  for real test scenes and the official online bonus.

Key documents:

- `docs/WINNING_PLAN.md`: complete method and competition strategy.
- `docs/RESEARCH_INNOVATION.md`: research contribution and why it can win.
- `docs/TOP_TEAM_REPO_NOTES.md`: what to learn from the public 2025 repos.
- `docs/IMPLEMENTATION_BACKLOG.md`: ordered engineering tasks and deliverables.
- `docs/EXPERIMENT_LOG.md`: comparable runs, settings, metrics, and decisions.
- `docs/PAPER_OUTLINE.md`: ECCV workshop paper structure.
