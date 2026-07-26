# Implementation Backlog

This backlog is ordered by leaderboard impact. Each item should produce a
measurable validation result before moving on.

## 0. Validation And Data Plumbing

Status: baseline implemented, plus an upgraded geometric end-to-end baseline,
detector prediction adapters, and stride-correct sampled evaluation.

- Run `scripts/smoke_oracle_eval.sh` on the HPC after every parser/evaluator
  change.
- Run `scripts/smoke_geometric_baseline.sh` after every calibration, lifting,
  fusion, or tracking change.
- Keep local evaluator results and official evaluator results side by side.
- Add official evaluation wrapper when NVIDIA releases or updates the exact
  2026 scorer.
- Track all experiments with: git hash, data split, stride, detector weights,
  postprocess thresholds, and HOTA/DetA/AssA/LocA.

## 1. Detector Training

Goal: high-recall 2D detections with low class confusion.

Steps:

- Export YOLO labels from all train scenes at stride 5 to 15.
- Use dense export for rare classes and sparse export for common person frames.
- Train a fast baseline first: YOLO11/YOLO12/RT-DETR at 1280.
- Train a larger teacher model at 1536 or full 1920 if GPU memory permits.
- Tune class-specific thresholds against validation 3D HOTA, not only 2D AP.
- Add Sim2Real augmentation sweeps:
  - brightness/contrast/gamma/color temperature
  - blur/compression/noise
  - synthetic shadows and low-light enhancement
  - crop/truncation and small-box filtering

Deliverables:

- `runs/detector/yolo_baseline/`
- validation per-class AP report
- inference TSV/JSON with per-camera boxes, confidences, and embeddings

Implemented plumbing:

- Ultralytics `save_txt` predictions to the common per-camera TSV.
- COCO result JSON to the same TSV for D-FINE.
- confidence filtering and class-wise 2D NMS.
- manifest-based scene, camera, and frame recovery.
- YOLO inference-to-STAR PBS runner with scratch outputs.

## 2. RGB-To-3D Lifting

Goal: convert each 2D detection into a calibrated global 3D box.

Implemented first-stage steps:

- class-conditioned width, length, height, z, and yaw priors;
- calibrated bottom-center lifting from image coordinates to the world plane;
- reprojection-error measurement for each single-view candidate;
- oracle-2D and detector-prediction inputs through one interface.
- supervised residual dataset export for center, z, dimensions, and circular
  yaw;
- scene-held-out heteroscedastic MLP training;
- NumPy-only residual inference with uncertainty-weighted fusion.

Next steps:

- Replace sampled priors with robust all-scene priors, stratified by simulator
  domain where useful.
- Add visible-center and truncation-aware ray lifting alongside bottom-center.
- Add RGB crop embeddings and a learned 3D usability score to the geometric
  residual prototype.
- Retain all-scene jobs `692149` and `692150` as the geometry-only ablation
  baseline; do not promote their direct scale-1.0 correction.
- Sweep conservative residual scales and uncertainty gates after the direct
  scale-1.0 model failed the fixed HOTA gate.
- Train and ablate the RGB-enhanced residual head after the geometric-only
  result establishes the localization gain.
- Validate with detector oracle boxes first, then detector predictions.

Deliverables:

- single-camera `track1.txt`
- error report by class, camera, range, and box size
- ablation: class priors vs learned residuals

## 3. Multi-View Fusion

Goal: turn per-camera 3D hypotheses into one scene-level detection set per
frame.

Implemented first-stage steps:

- camera-deduplicated reliability-weighted center/size/yaw fusion;
- adaptive BEV distance and 3D-IoU compatibility;
- post-fusion 3D duplicate suppression;
- per-detection camera support and cluster spread diagnostics.
- class-specific fusion radii in normal inference;
- cached per-class sweeps over fusion radius, source count, and 3D NMS
  distance.

Next steps:

- Add ray-to-center distance and camera-view compatibility to clustering.
- Reject physically impossible candidates with map bounds and class priors.
- Extend reliability weights with crop area, view angle, and learned
  uncertainty.
- Run the oracle and detector class-fusion sweeps and freeze parameters only
  when gains transfer across held-out scenes.
- Refine centers with VGCR-style reprojection/ray residual minimization.
- Quantify localization error before and after refinement by class and range.

Deliverables:

- fused per-frame detections without tracking IDs
- validation HOTA with temporary frame-local IDs
- localization error plots before and after refinement

## 4. Online 3D Tracker

Goal: award-focused online submission.

Implemented first-stage steps:

- motion-predicted global Hungarian assignment;
- costs for normalized BEV distance, 3D IoU, yaw, confidence, and distinct
  camera support;
- adaptive motion gates and velocity/shape smoothing;
- crossing-target regression test.

Next steps:

- Implement class-gated 3D Kalman/IMM tracker.
- Add association costs:
  - 3D IoU
  - Mahalanobis BEV distance
  - yaw and velocity agreement
  - appearance/ReID distance
  - camera visibility consistency
  - shape compatibility
- Add delayed track initialization and long occlusion memory.
- Add duplicate track merge and crossing-safe split handling.
- Log an online proof: each output frame only depends on frames `<= t`.

Deliverables:

- `runs/online_bonus/track1.txt`
- HOTA/DetA/AssA/LocA report
- online proof logs for paper and challenge review

## 5. Offline Linker And Self-Training

Goal: recover association fragments and distill useful corrections into the
online branch.

Steps:

- Link track fragments with a temporal graph over long gaps.
- Use spatial continuity, appearance, class shape, yaw, and camera visibility.
- Validate any pseudo-labeling/self-training against held-out validation scenes.
- Distill offline corrections into the online association and fusion thresholds.

Deliverables:

- `runs/offline_max_raw/track1.txt`
- online-vs-offline comparison table
- ablation table showing whether self-training helps

## 6. Offline Linker

Goal: diagnostic and public leaderboard branch.

Steps:

- Link track fragments with a temporal graph over long gaps.
- Use spatial continuity, appearance, class shape, yaw, and camera visibility.
- Only use this branch as the final submission if the raw HOTA advantage beats
  the online-bonus-adjusted score by a comfortable margin.

Deliverables:

- `runs/offline_max_raw/track1.txt`
- online-vs-offline comparison table

## 7. Paper Package

Goal: reproducible ECCV workshop paper after results are final.

Steps:

- Freeze the final method and ablations.
- Document dataset splits, external data, models, and training compute.
- Include online proof if submitting for bonus.
- Report failure cases: rare robots, heavy occlusion, low-light, calibration
  drift, and person/humanoid confusion.
