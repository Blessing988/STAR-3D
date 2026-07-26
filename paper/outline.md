# Paper Outline

## Title Options

1. STAR-3D: Association-Preserving RGB-Only Multi-Camera 3D Tracking under Sim2Real Shift
2. When Depth Disappears: Association-Preserving Multi-Camera 3D Tracking under Sim2Real Shift
3. High-Resolution Detector Fusion and Conservative Tracklet Relinking for Sim2Real 3D Perception

Recommended title:

**STAR-3D: Association-Preserving RGB-Only Multi-Camera 3D Tracking under Sim2Real Shift**

## Abstract Skeleton

See `problem_motivation.md` for the paper-ready abstract. Report two modes:

- causal online: 12.3778 3D HOTA;
- optional offline graph: 12.4128 3D HOTA, rank 9.

## Sections

1. Introduction
   - Privileged-modality Sim2Real gap: synthetic RGB-D supervision, real RGB-only inference.
   - Geometry-association coupling: 2D errors propagate into lifting, fusion, duplicate births, and identity fragmentation.
   - Thesis: association-preserving evidence control across the full pipeline.

2. Related Work
   - Multi-camera multi-object tracking.
   - BEV/tracklet graph association.
   - 2D-detector-driven 3D lifting.
   - Depth/point-cloud methods for AI City-style warehouse tracking.

3. Dataset and Evaluation
   - AI City 2026 Track 1.
   - 7 classes.
   - 3D HOTA, DetA, AssA, LocA.
   - Separate online and offline settings accurately.

4. Method
   - High-resolution detector training/inference.
   - YOLO11x1920 and D-FINE fusion.
   - Calibration-based 3D lifting.
   - Association-Preserving Reliability Cascade (APRC).
   - Confidence-conditioned causal 3D tracker.
   - Optional conservative offline BEV tracklet graph.

5. Experiments
   - Official leaderboard result.
   - Ablation table.
   - Detector base comparison.
   - Tracker/postprocess comparison.
   - Negative depth/V-DETR results.

6. Discussion
   - Why LocA is high but DetA/AssA remain limited.
   - Failure modes: class imbalance, domain shift, monocular depth mismatch, 3D box scale/yaw sensitivity.
   - What would likely close gap to top teams: native 3D detector, real depth-consistent training, stronger cross-camera association.

7. Conclusion
   - RGB-only baseline is reliable and reproducible.
   - BEV relinking gives modest but real association improvement.
   - Future work: calibrated object-centric 3D detector and depth-supervised BEV fusion.
