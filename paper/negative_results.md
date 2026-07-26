# Negative Results

These should be included carefully as analysis, not as failed engineering.

## DepthPro/Multi-View Optimizer

Artifacts:

- `track1-yolo11x1920_resume_e29_dfine_p025_mvopt_balanced.zip`
- `track1-final_classhybrid_hota_depth_mvopt_fullclasses.zip`

Observation:

Depth/MV optimization slightly increased DetA/LocA in some variants but reduced AssA enough that the final HOTA stayed below the best BEV relink result.

Likely reasons:

- Depth estimates were not perfectly object-centric.
- Small localization shifts can perturb tracker association.
- Validation-selected classwise hybrid did not transfer to the final test distribution.

## V-DETR/ZIO-Style Point Cloud Pipeline

Observation:

The point-cloud/3D detector path did not become competitive in final submissions.

Likely reasons:

- Domain mismatch between synthetic training and real test.
- Sparse/estimated depth generated noisy point clouds.
- Rare classes had weak AP and unstable 3D proposals.
- Training/inference format mismatch risk remained high under deadline.

## RGB Residual Correction

Observation:

Residual corrections did not improve official results.

Likely reasons:

- Residual dataset had limited reliable matches.
- Learned shifts overfit validation geometry.
- Localization changes did not consistently improve HOTA thresholds.

## Paper Use

Use these as evidence that Track 1 improvement is not just "add depth". The key lesson:

> Depth and 3D detectors help only when calibration, depth scale, object segmentation, and 3D box parameterization are jointly consistent. Otherwise, association can degrade even when LocA appears locally improved.

