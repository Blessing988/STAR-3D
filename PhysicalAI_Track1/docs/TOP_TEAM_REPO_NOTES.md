# Prior Team Repository Notes

These notes summarize the public 2025 Track 1 repositories provided by the
user. Treat them as design references, not code to copy. Before porting code,
check each repository license and dependency constraints.

## ZIOVISION/AIC2025_Track1_ZV

Repository: https://github.com/ZIOVISION/AIC2025_Track1_ZV

Paper direction: V-DETR and point-cloud ReID. The public paper reports a
state-of-the-art 2025 3D HOTA result by fusing multi-view RGB-D into global
point clouds, running a transformer-style 3D detector, learning 3D ReID
embeddings, and adding offline global trajectory linking.

What matters for our 2026 solution:

- Point-cloud methods show the value of solving the task in 3D space.
- 3D embeddings solve the hardest non-person association cases better than
  2D appearance alone.
- Offline global linking gives large association gains, but may lose the 2026
  online bonus.
- The approach depends heavily on depth, and our current downloaded train/val
  depth-map folders are empty, so we should not build around RGB-D.

Action items:

- Reuse the 3D-native association and global linking ideas.
- Train RGB geometry heads directly from 3D GT.
- Keep an offline linker for analysis and public leaderboard probing.

## SKKUAutoLab/AIC25_Track_01

Repository: https://github.com/SKKUAutoLab/AIC25_Track_01

Paper direction: DepthTrack, with YOLO detection, person pose/ReID,
single-view tracking, BEV tracklets, depth/point-cloud clustering, and
tracklet-cluster mapping.

What matters for our 2026 solution:

- The 2D-detector -> single-view tracklet -> BEV/3D fusion pipeline is practical
  and easier to run online than a full point-cloud detector.
- Person ReID should be pose-aware or part-aware; whole-box appearance features
  are fragile under occlusion.
- Their class handling was designed for the 2025 taxonomy; 2026 adds
  `PalletTruck=6`.
- Scene-specific low-light and contrast enhancement is worth testing because
  real camera appearance is the dominant Sim2Real gap.

Action items:

- Implement our own single-view tracklet exporter from detector outputs.
- Add class-gated BEV clustering and tracklet-cluster mapping.
- Use person-specific ReID and simpler geometry/shape embeddings for robots.
- Extend all class maps and shape priors to include PalletTruck.

## kiyotaka1102/VGCRTrack

Repository: https://github.com/kiyotaka1102/VGCRTrack

Paper direction: online multi-camera 3D tracking with view-aware geometric
center refinement, calibrated camera geometry, depth/ray cues, temporal
smoothing, and multi-cue association.

What matters for our 2026 solution:

- VGCR-style center refinement is the right localization module for real scenes
  because it uses calibration and multi-view consistency instead of assuming
  perfect depth.
- Ray consistency, reprojection error, view angle, and track motion are useful
  reliability weights for fusing camera hypotheses.
- Their online design aligns with the 2026 bonus incentive.

Action items:

- Implement ray-ground lifting and multi-view ray intersection.
- Add reprojection residual minimization for candidate centers.
- Use uncertainty-weighted temporal smoothing in the online tracker.
- Save logs proving each frame used only current and past frames.

## 2026 Strategy Adjustment

Last year, depth-assisted point-cloud methods were very strong. In the current
downloaded 2026 data, train/val `depth_maps` folders are empty and test has no
depth maps. The final system should be RGB/calibration-first:

- Use 3D GT, camera calibration, learned geometry priors, and online multi-view
  fusion.
- Use point-cloud papers as design inspiration for 3D-native association, not
  as a dependency.
- Select branch weights per scene with validation evidence, not assumptions.
