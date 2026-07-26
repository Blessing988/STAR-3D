# ECCV Workshop Paper Outline

## Title

STAR-3D: Sim2Real Temporal Association and Ray-Guided Multi-Camera 3D Tracking

## Abstract

State the challenge: synthetic train/validation, real test, calibrated
multi-camera RGB, scene-level 3D boxes, stable IDs, and 3D HOTA. Summarize the
method and final leaderboard score once available.

## 1. Introduction

- Importance of multi-camera 3D perception in smart spaces.
- Why Sim2Real is difficult: synthetic appearance, real cameras, occlusions,
  no usable depth maps in the current data, and RGB-only real test inference.
- Challenge metric: 3D HOTA balances detection, localization, and association.
- Contributions.

## 2. Related Work

- Multi-camera tracking and ReID.
- BEV multi-view detection.
- Geometry-assisted 3D tracking.
- Sim2Real adaptation for detection/tracking.

## 3. Dataset and Evaluation

- PhysicalAI Smart Spaces 2026.
- Classes and scene splits.
- Submission format.
- 3D HOTA and online bonus.
- Clarify that `frame_id / 30` gives timestamp in seconds.

## 4. Method

- RGB detector.
- Camera-calibrated 3D lifting with learned geometry residuals.
- Uncertainty-aware multi-view fusion and geometric center refinement.
- Online 3D tracking.
- Sim2Real adaptation.

## 5. Experiments

- Validation setup.
- Detector AP.
- 3D HOTA ablations.
- Online vs offline.
- Sim2Real ablations.
- Geometry-head and center-refinement ablations.
- Rare-class analysis, including PalletTruck.
- Runtime.

## 6. Challenge Results

- Public/private leaderboard score.
- Failure cases.
- Reproducibility details.

## 7. Conclusion

Summarize the method and future work.
