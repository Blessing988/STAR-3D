# Experiment Log

Use this file for decision-relevant results. Every entry must identify the
data slice, detector checkpoint, post-processing settings, and all four local
3D HOTA-style metrics.

## E001: YOLO26x Detector-To-STAR Pilot

Date: 2026-06-12

PBS job: `692131.bright04`

Status: completed successfully in 9 minutes 31 seconds on one A100 40 GB.

Detector:

- checkpoint:
  `/path/to/scratch/PhysicalAI_Track1/runs/detector/yolo26x_1536_2xh100_stride30_class_balanced_from_e1_ncclsafe/weights/best.pt`
- inference resolution: 1536
- detector confidence: 0.01
- detector NMS IoU: 0.70

Validation slice:

- scenes: `Warehouse_020`, `Warehouse_021`, `Warehouse_022`
- source frames: first 600 per scene
- frame stride: 30
- sampled GT boxes: 2,853

Pipeline counts:

- camera detections: 14,312
- lifted 3D candidates: 14,312
- fused 3D detections: 10,084
- tracked output boxes: 10,084
- output track IDs: 1,128

Metrics:

| Metric | Value |
|---|---:|
| HOTA-like | 0.083892 |
| DetA | 0.040586 |
| AssA | 0.184302 |
| LocA | 0.407331 |

Artifacts:

```text
/path/to/scratch/PhysicalAI_Track1/runs/inference/yolo26x_star_val_692131.bright04/
```

Interpretation:

- The run proves that trained detector predictions now execute through the
  complete 2D-to-3D-to-track evaluation path.
- False positives dominate: 10,084 output boxes versus 2,853 sampled GT boxes.
- The next low-cost action is confidence and association tuning on cached
  `fused.tsv`; GPU inference does not need to be repeated.
- LocA is substantially less limiting than DetA at low IoU thresholds, but
  learned center residuals and view-consistent refinement are still required
  for high-threshold HOTA.

Follow-up:

- tracker sweep PBS job: `692135.bright04`
- D-FINE detector-to-STAR pilot PBS job: `692134.bright04`
- optimized YOLO runner now selects only 713 relevant images for this slice,
  instead of processing all 10,472 validation images.

## E002: YOLO26x Cached Tracker Sweep

Date: 2026-06-13

PBS job: `692135.bright04`

Status: completed successfully in 13 minutes.

Best parameters:

- minimum fused score: 0.60
- maximum association cost: 1.20
- maximum BEV motion distance: 1.8 m

Best output:

- tracked boxes: 2,720
- track IDs: 648

| Metric | Value |
|---|---:|
| HOTA-like | 0.157854 |
| DetA | 0.110478 |
| AssA | 0.235925 |
| LocA | 0.411302 |

This is an 88 percent relative HOTA-like improvement over E001 without
rerunning detector inference. It confirms that challenge-level calibration
must optimize 3D HOTA rather than detector AP alone.

Per-class breakdown on this sampled slice:

| Class | GT boxes | Pred boxes | HOTA-like | LocA |
|---|---:|---:|---:|---:|
| Person | 2,093 | 1,934 | 0.159199 | 0.416173 |
| Forklift | 360 | 633 | 0.169269 | 0.285045 |
| PalletTruck | 400 | 153 | 0.110465 | 0.162189 |

Only these three classes occur in the fixed slice. PalletTruck has both a
recall deficit and severe 3D localization error, so aggregate tuning alone
will hide the highest-value class-specific geometry work.

## E003: D-FINE-L Detector-To-STAR Pilot

Date: 2026-06-13

PBS job: `692134.bright04`

Status: completed successfully in 20 minutes 50 seconds on one A100 40 GB.

Checkpoint:

```text
/path/to/scratch/PhysicalAI_Track1/runs/dfine/dfine_l_960_class_balanced_168h_fixed/best_stg1.pth
```

Pipeline counts:

- camera detections: 156,665
- fused 3D detections: 69,257
- tracked output boxes: 69,257

| Metric | Value |
|---|---:|
| HOTA-like | 0.024327 |
| DetA | 0.005966 |
| AssA | 0.116452 |
| LocA | 0.410386 |

Interpretation:

- The checkpoint has useful localization but its DETR query scores are not
  calibrated like YOLO confidences.
- A single global threshold is inappropriate: class score distributions differ
  by more than an order of magnitude.
- Class-specific calibration job `692148.bright04` was submitted using cached
  fused detections.

## E004: Full-Frame Oracle-2D Geometry Ceiling

Date: 2026-06-13

PBS job: `692130.bright04`

Status: completed successfully in 15 minutes 17 seconds.

Validation slice:

- scenes: `Warehouse_020`, `Warehouse_021`, `Warehouse_022`
- first 600 frames per scene
- frame stride: 1

| Metric | Value |
|---|---:|
| HOTA-like | 0.190844 |
| DetA | 0.128249 |
| AssA | 0.290353 |
| LocA | 0.522365 |

This is not directly comparable to stride-30 detector pilots.

Matched stride-30 oracle run:

- PBS job: `692146.bright04`
- status: completed successfully in 3 minutes 10 seconds
- sampled GT boxes: 2,853

| Metric | Value |
|---|---:|
| HOTA-like | 0.174598 |
| DetA | 0.114589 |
| AssA | 0.279986 |
| LocA | 0.464844 |

The matched oracle result is only 10.6 percent above the tuned YOLO E002
HOTA-like score. Detector calibration remains important, but geometry and
association now form the dominant ceiling.

The dependent cached tracker sweep, PBS job `692147.bright04`, completed in
21 minutes 7 seconds. Its best setting retained all fused detections:

- minimum score: 0.0
- maximum association cost: 1.20
- maximum motion distance: 1.8 m
- HOTA-like: 0.179001
- DetA: 0.117825
- AssA: 0.283089
- LocA: 0.465590

Tracker-only tuning improves the matched oracle HOTA-like score by just 2.5
percent. The remaining ceiling is primarily lifting and multi-view fusion.

Matched oracle per-class breakdown before tracker retuning:

| Class | GT boxes | Pred boxes | HOTA-like | LocA |
|---|---:|---:|---:|---:|
| Person | 2,093 | 3,550 | 0.163159 | 0.470237 |
| Forklift | 360 | 1,026 | 0.198420 | 0.423383 |
| PalletTruck | 400 | 666 | 0.186596 | 0.205914 |

Oracle 2D boxes improve PalletTruck HOTA substantially, but its LocA remains
only 0.206. The excessive prediction counts also show that multi-view
deduplication is a major bottleneck even with perfect 2D detections.

## E005: Geometry Residual End-To-End Smoke Test

Date: 2026-06-13

Validation:

- training scenes: `Warehouse_000`, `Warehouse_001`
- samples: 1,440
- feature dimension: 46
- target dimension: 8
- scene-held-out split: 720 train / 720 validation
- epochs: 2 on CPU

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 1.202666 | 1.094610 |
| 2 | 1.031997 | 0.914877 |

The exported NumPy model loaded successfully and produced 7,224 residual-aware
single-camera 3D candidates from the matched oracle validation detections.
This verifies dataset construction, heteroscedastic training, model export,
model loading, residual correction, and uncertainty propagation.

Full jobs:

- all-scene stride-15 dataset build: `692149.bright04`
- dependent GPU training: `692150.bright04`
- dependent matched residual validation: `692151.bright04`
- dependent residual tracker sweep: `692152.bright04`

The full dataset build completed successfully:

- training boxes scanned for priors: 8,337,925
- residual samples: 1,621,296
- feature dimension: 46
- compressed dataset size: 74 MB

The two-scene model is a wiring test only and must not be used for a score
claim. The first meaningful ablation is baseline geometry versus the
all-scene residual model on the fixed stride-30 oracle slice.

Full-scale direct residual result:

| Metric | Baseline | Residual scale 1.0 |
|---|---:|---:|
| HOTA-like | 0.174598 | 0.058587 |
| DetA | 0.114589 | 0.017278 |
| AssA | 0.279986 | 0.224406 |
| LocA | 0.464844 | 0.309436 |

The dependent tracker sweep on the same failed residual cache reached only
0.059181 HOTA-like, so tracking cannot rescue the direct residual output.

The direct residual is rejected. Scene-held-out likelihood selected a model
that does not preserve challenge-level 3D localization. The implementation now
supports bounded residual blending and uncertainty gating; the next ablation
sweeps scales `0.0, 0.05, 0.10, 0.25, 0.50, 1.0` and uncertainty cutoffs before
any learned correction is promoted into the main pipeline.

Residual scale/uncertainty sweep PBS job: `692162.bright04`.

## E006: Class-Conditioned Multi-View Fusion Sweep

Date: 2026-06-13

Motivation:

- the matched oracle baseline emits 5,242 fused detections for 2,853 GT boxes;
- Person, Forklift, and PalletTruck have different localization-error scales;
- one global clustering radius cannot simultaneously preserve close people
  and merge noisy industrial-vehicle hypotheses.

The sweep selects per-class BEV radius, minimum camera support, and 3D NMS
distance from cached lifted candidates. It evaluates 60 configurations and
exports the best fused TSV plus Track 1 submission.

Jobs:

- oracle candidates: `692153.bright04`
- YOLO26x candidates: `692154.bright04`

Both jobs completed successfully in about 10 minutes.

| Input | Previous HOTA-like | Class-fusion HOTA-like | Relative gain |
|---|---:|---:|---:|
| Oracle 2D | 0.179001 | 0.191547 | +7.0% |
| YOLO26x | 0.157854 | 0.164032 | +3.9% |

Oracle-selected parameters:

| Class | Radius | Minimum sources | NMS distance | Class HOTA-like |
|---|---:|---:|---:|---:|
| Person | 1.2 m | 2 | 0.25 m | 0.185788 |
| Forklift | 0.8 m | 2 | 0.25 m | 0.215449 |
| PalletTruck | 0.8 m | 2 | 0.25 m | 0.187532 |

YOLO-selected parameters:

| Class | Radius | Minimum sources | NMS distance | Class HOTA-like |
|---|---:|---:|---:|---:|
| Person | 1.2 m | 1 | 0.25 m | 0.162992 |
| Forklift | 2.0 m | 1 | 0.25 m | 0.187819 |
| PalletTruck | 2.5 m | 1 | 0.25 m | 0.110698 |

The detector needs larger vehicle radii than oracle boxes, directly measuring
the detector-to-world localization noise that the residual model must remove.
The `min_sources=2` oracle optimum also confirms that unsupported single-view
hypotheses are a major false-positive source.

Follow-up calibration on the improved YOLO fused cache:

- global tracker sweep: `692155.bright04`
- per-class confidence sweep: `692156.bright04`

Both completed successfully:

| Calibration | HOTA-like | DetA | AssA | LocA | Boxes |
|---|---:|---:|---:|---:|---:|
| Class fusion + global threshold | 0.164032 | 0.111958 | 0.250144 | 0.412448 | 2,693 |
| Class fusion + class thresholds | 0.164256 | 0.114652 | 0.245418 | 0.412648 | 2,495 |

Best current YOLO validation candidate:

```text
/path/to/scratch/PhysicalAI_Track1/runs/sweeps/yolo26x_class_fusion_threshold_sweep/track1_best_class_thresholds.txt
```

Selected detector-side thresholds:

- Person: 0.6
- Forklift: 0.8
- PalletTruck: 0.4

## E007: Precision Ensemble and Adaptive Online Track Birth

Date: 2026-06-15

Motivation:

- the first real-test leaderboard results showed association and detection
  errors remained coupled;
- unconditional delayed confirmation removed false person tracks but also lost
  valid first-frame detections;
- offline backfill improved validation, but it revises past outputs and is not
  eligible for the challenge's online bonus;
- high-confidence detections can be trusted immediately while uncertain person
  births can wait for one additional observation using only present and past
  information.

Detector ensemble calibration:

| Variant | YOLO weight | D-FINE weight | WBF IoU | Final NMS IoU | Min score | HOTA-like |
|---|---:|---:|---:|---:|---:|---:|
| Original ensemble | 1.00 | 0.90 | 0.62 | 0.80 | 0.01 | 0.160446 |
| Precision ensemble | 1.00 | 0.40 | 0.70 | 0.75 | 0.05 | 0.166246 |
| Recall ensemble | 1.00 | 0.80 | 0.55 | 0.85 | 0.005 | 0.159747 |
| D-FINE-heavy ensemble | 1.00 | 1.20 | 0.62 | 0.85 | 0.01 | 0.152940 |

Online birth-control sweep on the original ensemble:

| Variant | HOTA-like | DetA | AssA | LocA | Online eligible |
|---|---:|---:|---:|---:|---|
| Immediate class-aware baseline | 0.164361 | 0.115811 | 0.248648 | 0.417413 | Yes |
| Person confirmed-only | 0.163796 | 0.119134 | 0.238296 | 0.416259 | Yes |
| Person backfill | 0.168871 | 0.121846 | 0.249299 | 0.417508 | No |
| Adaptive person score 0.90 | 0.168560 | 0.121831 | 0.248559 | 0.417391 | Yes |

The adaptive online rule is:

1. Apply the class-specific detector and association gates.
2. Emit a new person track immediately when its fused score is at least 0.90.
3. Otherwise require a second matched person detection before emitting it.
4. Never revise an already processed frame.

Combined precision-ensemble and adaptive-online validation:

| Variant | HOTA-like | DetA | AssA | LocA |
|---|---:|---:|---:|---:|
| Precision ensemble, original tracker | 0.166246 | 0.122917 | 0.233434 | 0.375156 |
| Precision ensemble, adaptive score 0.85 | 0.172470 | 0.126674 | 0.242523 | 0.375352 |
| Precision ensemble, adaptive score 0.90 | **0.172634** | **0.127379** | 0.241642 | **0.375386** |
| Precision ensemble, adaptive score 0.95 | 0.169888 | 0.124741 | 0.238951 | 0.375276 |

The selected online configuration improves HOTA-like by 7.6% relative to the
original ensemble while retaining eligibility for the 10% online award bonus.

Jobs:

- online birth-control sweep: `692977.bright04`
- adaptive online sweep on original ensemble: `692980.bright04`
- adaptive online sweep on precision ensemble: `692983.bright04`
- full D-FINE test inference and base ensemble: `692801.bright04`
- dependent precision-adaptive test packaging: `692982.bright04`

The dependent test job writes the primary online archive to:

```text
/path/to/scratch/PhysicalAI_Track1/runs/submissions/test_ensemble_precision_adaptive_online/track1.zip
```

It also writes a clearly labeled offline diagnostic archive under
`offline_person_backfill/`; that archive must not be declared online.
