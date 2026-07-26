# Winning Plan: STAR-3D

## Task Reality

AI City Challenge 2026 Track 1 is not a language fine-tuning problem. It is a
multi-camera 3D detection and tracking problem evaluated with 3D HOTA. The model
must output scene-level 3D boxes and stable object IDs:

```text
scene_id class_id object_id frame_id x y z width length height yaw
```

The training and validation data are synthetic RGB videos with calibration,
2D/3D annotations, and top-down maps. The current downloaded `depth_maps`
folders are empty, so the usable data does not include depth maps. The test
contains RGB videos and calibration without GT. The key Sim2Real issue is that
the final model must learn RGB-to-geometry from 3D annotations, calibration, and
multi-view consistency.

The class set in the 2026 README includes:

```text
Person=0, Forklift=1, NovaCarter=2, Transporter=3,
FourierGR1T2=4, AgilityDigit=5, PalletTruck=6
```

The videos run at 30 FPS. There is no separate timestamp field in the GT files;
`timestamp_seconds = frame_id / 30`.

## Why This Direction Can Win

The official Track 1 page confirms that leaderboard ranking is raw 3D HOTA on
hidden test, while the award ranking applies a 10 percent multiplicative bonus
to systems proven to be online-only. The final method should therefore target
the best online score first, with an offline linker maintained as a secondary
diagnostic and public-leaderboard tool.

The 2025 top methods show four clear lessons:

1. 3D HOTA rewards association and localization, not only 2D detection.
2. 3D-native association and localization matter more than camera-level 2D AP.
3. BEV tracking and geometric center refinement are robust ways to convert
   single-camera tracks into scene-level 3D identities.
4. Online tracking gets a 10 percent award bonus in 2026, so a slightly lower
   raw HOTA online method can beat a higher raw HOTA offline method.

The proposed system combines the strongest parts of the public 2025 approaches
without depending on any single one at test time:

- ZIOVISION style 3D-native association and global trajectory linking, without
  depending on unavailable RGB-D point clouds.
- DepthTrack/SKKU style 2D detection, BoT-SORT tracklets, BEV clustering, and
  tracklet-cluster mapping as a fast online branch.
- VGCRTrack style view-aware geometric center refinement, ray consistency, and
  temporal smoothing as the real-scene localization module.
- Strong Sim2Real detector adaptation, because 2026 hidden test includes real
  warehouse videos.

## Architecture

### Branch A: RGB-Calibrated Online Tracker

This is the award-focused submission path.

#### Stage 1: RGB Detector

Train a large detector on visible 2D boxes from `2d bounding box visible`.
Candidates:

- YOLOv12/YOLOv10/RT-DETR-large for speed and mature deployment.
- DINO/Co-DETR/RT-DETR-x for an offline high-accuracy teacher ensemble.
- YOLO-World or GroundingDINO as a teacher to improve rare classes.

Training details:

- Use 2024, 2025, and 2026 Track 1 data where allowed by the current rules.
- Oversample rare classes: `AgilityDigit`, `FourierGR1T2`, `PalletTruck`.
- Use heavy Sim2Real augmentation: blur, compression, exposure, gamma,
  shadows, sensor noise, crop/truncation, motion blur, color temperature.
- Train at 1280 or 1536 long side; test with TTA at multiple scales.
- Keep per-class thresholds low initially; HOTA punishes false negatives.

#### Stage 2: RGB-To-3D Lifting

For every 2D detection, estimate a 3D hypothesis:

- Center ray from the 2D box bottom-center and mid-center through calibration.
- Class-conditioned dimensions from training statistics.
- Learned center residual or ground-plane intersection correction.
- Height prior and camera pitch correction.
- Yaw from motion when available, otherwise from a yaw head.

Train auxiliary heads:

- `center_residual_head`: predicts 3D center residuals from crop features plus
  camera metadata.
- `dimension_head`: predicts width/length/height residuals over class priors.
- `yaw_head`: circular classification/regression for yaw.
- `visibility_head`: predicts whether a 2D detection is usable for 3D lifting.

Train these heads directly from 3D GT. At test time, use RGB and calibration.

#### Stage 3: Multi-View Fusion

At each frame and class:

1. Generate candidate 3D boxes from all cameras.
2. Reject boxes that violate map extents, class dimensions, impossible height,
   or reprojection consistency.
3. Group candidates using 3D IoU, BEV distance, ray intersection distance, and
   camera visibility overlap.
4. Fuse centers using reliability weights:
   detection confidence, crop size, view angle, reprojection residual, and
   learned uncertainty.
5. Fuse dimensions with class prior regularization.

This is where most localization HOTA is won. Multi-view boxes should be more
accurate than any single-camera projection.

#### Stage 4: Online 3D Tracking

Use an online tracker for the +10 percent bonus:

- State: `(x, y, z, vx, vy, vz, width, length, height, yaw, yaw_rate)`.
- Motion model: constant velocity with class-specific process noise.
- Association cost:
  - 3D IoU
  - BEV Mahalanobis distance
  - velocity/yaw agreement
  - cross-view appearance embedding
  - camera visibility consistency
  - class and dimension compatibility
- Track management:
  - delayed initialization for low-confidence objects
  - long occlusion memory for warehouse shelving occlusions
  - merge duplicate tracks if 3D IoU and trajectory agreement are high
  - split tracks when two objects overlap then diverge

Maintain an online-only submission branch. Also build an offline linker for
analysis and possibly for public leaderboard probing, but the final award
strategy should prioritize the online bonus.

### Stage 5: Sim2Real Adaptation

The real test scenes are the decisive gap. Use:

- Synthetic style randomization during detector training.
- Low-light and contrast enhancement as deterministic preprocessing.
- Teacher-student pseudo-labeling on public real videos allowed by rules.
- Camera calibration sanity checks on every test scene.
- Scene-level auto-tuning without future frames for the online branch:
  per-camera confidence thresholds, color normalization, map masks, and class
  priors.

### Stage 6: Submission Strategy

Maintain three submissions during development:

- `oracle_val`: GT-to-submission smoke test. It should score near 1.0 locally.
- `online_bonus`: RGB-calibrated online tracker, award-focused.
- `offline_max_raw`: stronger temporal graph linker, diagnostic only unless raw
  HOTA is clearly higher than the online-bonus-adjusted score.

If raw leaderboard difference is less than about 9 percent, submit the online
method as the award-focused solution.

## Experiment Ladder

1. Oracle format smoke test: convert validation GT to `track1.txt` and score it.
2. Detector-only 2D AP on validation visible boxes.
3. Single-view geometry baseline using class priors and calibration.
4. Multi-view fusion without temporal tracking.
5. Online Kalman tracker with 3D IoU association.
6. Add learned center/dimension/yaw heads.
7. Add appearance/ReID and rare-class metric learning.
8. Add Sim2Real augmentations and test-time enhancement.
9. Add optional offline graph linker.
10. Ablate each cue for the paper.

## Implementation Milestones

### Milestone 0: Infrastructure

Status: implemented.

- Dataset inspector.
- GT-to-submission converter.
- Local 3D IoU / HOTA-like evaluator.
- YOLO label exporter.
- Frame extraction utility.
- Class-wise geometry statistics.

### Milestone 1: 2D Detector

- Export 2026 train/val labels at stride 5 to 15, with rare-class oversampling.
- Train an Ultralytics/RT-DETR baseline at 1280 resolution.
- Validate per-class AP and confusion, especially humanoid/person and
  robot/pallet-truck confusions.
- Add Sim2Real augmentation sweeps and deterministic enhancement.

### Milestone 2: RGB-To-3D Baseline

- Compute class priors from all training scenes.
- Implement single-camera ray-ground lifting.
- Fit learned residual heads for center, dimensions, yaw, and uncertainty.
- Score val with the local evaluator and then the official evaluator.

### Milestone 3: Multi-View Fusion

- Build per-frame 3D candidate clustering.
- Add VGCR-style ray/reprojection refinement.
- Add confidence-weighted fusion and duplicate suppression.
- Compare single-view, multi-view, and learned-refinement variants.

### Milestone 4: Online Tracker

- Implement 3D Kalman/IMM state.
- Add class-gated Hungarian association with IoU, distance, yaw, velocity, and
  appearance costs.
- Add online-only long occlusion memory.
- Produce proof that inference uses only current and past frames.

### Milestone 5: Offline Linker And Self-Training

- Link long trajectory fragments with a scene-level graph.
- Use high-confidence validation-style predictions as pseudo-labels only when
  ablations prove they help.
- Distill offline linker corrections into the online branch where possible.
- Implement optional offline trajectory linker for diagnosis and public probing.

## Paper Contributions

The paper should claim only what is implemented and verified:

1. A calibrated RGB-only 3D lifting module trained directly with synthetic 3D
   GT and camera geometry.
2. Uncertainty-aware multi-view center refinement for Sim2Real warehouse data.
3. An online 3D association method that wins the 10 percent bonus while
   maintaining strong raw HOTA.
4. A data-centric Sim2Real recipe for synthetic Omniverse/CT2.5 to real CCTV.
