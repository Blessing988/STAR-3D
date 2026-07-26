# Method Notes

## Best Raw-Score Pipeline

Artifact:

`results/track1-yolo11x1920_dfine_p025_bev_gap60_cost115_v2.zip`

Pipeline:

1. Run high-resolution YOLO11x inference at 1920 resolution.
2. Fuse YOLO11x predictions with D-FINE predictions at precision threshold `p=0.25`.
3. Lift 2D detections into global 3D coordinates using official camera calibration and class-size priors.
4. Run causal online 3D tracking.
5. Apply optional offline conservative BEV tracklet relinking:
   - temporal gap: `60` frames
   - max association cost: `1.15`
   - objective: recover short ID breaks without aggressive false merges

The graph reads complete tracklets and therefore uses future frames. It is not
eligible for an online claim.

## Causal Online Pipeline

Artifact:

`results/track1-yolo11x1920_dfine_precision025_adaptive_online.zip`

The person track birth rule is confidence-conditioned:

1. score at least 0.90: emit immediately;
2. lower score: require a second matched observation;
3. never backfill or revise a processed frame.

## Contribution Framing

Main contributions to claim:

1. Geometry-association coupling under privileged-depth Sim2Real shift.
2. Association-Preserving Reliability Cascade (APRC), carrying reprojection,
   camera-support, cluster-spread, detection, and track-history reliability
   through fusion and causal association.
3. Precision-calibrated heterogeneous detector fusion for RGB-only inference.
4. Class-conditioned causal tracking with confidence-conditioned track birth.
5. Conservative offline tracklet relinking that improves AssA while preserving every detection and 3D box.
6. Scoped analysis of depth/point-cloud failure modes when depth scale, object support, calibration, and box parameterization are not jointly aligned.

Avoid claiming:

- We solved depth-based 3D detection.
- V-DETR improved the final official result.
- DepthPro improved the final official result.
- Classwise hybrid improved the final official result.

## Why Best Variant Won

The best variant did not improve LocA over the adaptive online baseline. It improved AssA:

- Adaptive online AssA: `11.7634`
- BEV relink AssA: `11.8109`
- Gain: `+0.0475`

This means the best method worked by reducing ID fragmentation, not by changing 3D box localization.

Archive-level verification:

- rows before/after: `1,414,646` / `1,414,646`;
- track IDs before/after: `23,518` / `23,269`;
- fragments merged: `249`;
- Person: `245` merges;
- Forklift: `4` merges;
- all other classes: `0` merges.
