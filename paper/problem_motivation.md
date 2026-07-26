# Problem Motivation and Paper Positioning

## Recommended Title

**STAR-3D: Association-Preserving RGB-Only Multi-Camera 3D Tracking under Sim2Real Shift**

Alternative, more memorable title:

**When Depth Disappears: Association-Preserving Multi-Camera 3D Tracking under Sim2Real Shift**

Use the STAR-3D title for the submission. It names the system and states the
technical problem. Use "When Depth Disappears" as the opening phrase in the
Introduction, not necessarily as the final title.

STAR-3D expands to **Sim2Real Temporal Association and Ray-Grounded 3D
Tracking**.

## One-Sentence Problem

How can a calibrated multi-camera system recover accurate global 3D boxes and
persistent scene-level identities when it is trained with richly annotated
synthetic RGB-D data but must operate on real RGB video without depth at test
time?

## Central Scientific Motivation

AI City Challenge 2026 Track 1 creates a deployment mismatch that is more
fundamental than ordinary appearance domain shift. Training and validation
provide synthetic RGB, calibration, 2D/3D annotations, and optional depth,
whereas real-world test scenes provide RGB and calibration but no depth. A model
must therefore transfer both appearance and metric geometry across domains
while maintaining identities across heterogeneous, partially overlapping
cameras.

These requirements are coupled. A weak 2D box changes its bottom-center ray; a
small ray error changes the world-space center; inconsistent centers prevent
duplicate views from being fused; duplicate or missing 3D observations then
fragment tracks. Conversely, a geometry correction that improves one frame can
still reduce HOTA if it perturbs temporal association. This explains why
optimizing detector AP, monocular depth error, or framewise 3D localization in
isolation is insufficient.

The paper should formulate this as **geometry-association coupling under
privileged-modality Sim2Real shift**. Depth is privileged supervision during
training, not a guaranteed test-time sensor. The central design principle is:

> Treat test-time geometry as uncertain multi-view evidence and preserve
> association consistency when filtering, lifting, fusing, and linking that
> evidence.

## Why Existing Paradigms Are Insufficient

### RGB-D point-cloud pipelines

Point-cloud 3D detectors and 3D ReID were highly effective in the 2025 task,
where depth was available at inference. They solve detection and association in
the metric's native 3D space. The 2026 RGB-only test protocol removes this
assumption. Replacing sensor depth with monocular metric depth is not a drop-in
substitution: scale bias, foreground/background mixing, boundary noise, and
calibration inconsistency all alter the reconstructed world geometry.

### Early multi-view BEV aggregation

Learned BEV aggregation can produce strong 3D detections, but many systems are
trained for fixed camera layouts or depend on dense, accurately aligned depth.
Track 1 contains different scenes, camera counts, placements, and rendering
domains. A practical method must accept arbitrary calibrated cameras without
retraining a scene-specific BEV model.

### Late projection and tracking

Late-fusion systems are flexible across camera networks, but they amplify 2D
localization errors during projection and often use one global threshold for
classes with very different dimensions and dynamics. People, humanoids,
forklifts, AMRs, and pallet trucks should not share identical fusion radii,
motion gates, track ages, or birth rules.

### Independent module optimization

HOTA jointly evaluates detection, association, and localization. Improving one
component can reduce the final score by damaging another. Our official results
show this directly: variants with slightly higher DetA and LocA obtained lower
HOTA because AssA dropped. The research problem is therefore not merely better
detection or better depth; it is controlling error propagation through the
complete RGB-to-3D-to-identity chain.

## Proposed Thesis

STAR-3D uses a modular, calibration-first late-fusion design because it remains
valid for unseen camera configurations and RGB-only inference. It combines:

1. high-resolution, architecturally diverse 2D detections;
2. precision-calibrated weighted box fusion;
3. ray-ground lifting with class-conditioned 3D priors;
4. class-conditioned multi-view fusion and online 3D association;
5. confidence-conditioned causal track initiation; and
6. an optional conservative tracklet graph for offline fragment repair.

The key idea is not any detector in isolation. It is **association-preserving
evidence control**: only propagate a hypothesis when detector agreement,
multi-view geometry, class-specific motion, and temporal consistency make it
reliable enough for the next stage.

We name this core mechanism the **Association-Preserving Reliability Cascade
(APRC)**. APRC carries reliability through the complete pipeline instead of
discarding confidence after detection:

1. each lifted candidate is weighted by squared detector confidence,
   reprojection consistency, and geometry uncertainty;
2. each multi-view cluster receives a fused confidence based on camera support,
   spatial compactness, and single-view status;
3. online assignment rewards reliable detections, multi-camera support, and
   reliable track histories in addition to motion and 3D overlap;
4. track reliability is updated recursively from observation confidence,
   camera support, and track persistence; and
5. reliability controls whether a new track is emitted immediately or held for
   one causal confirmation.

APRC is the paper's primary method contribution. Detector fusion and the
offline graph are supporting components. The ablation must separately remove
reprojection weighting, camera support, spread penalties, track confidence,
and confidence-conditioned initiation to demonstrate that APRC is a coherent
mechanism rather than a collection of thresholds.

## Defensible Contributions

### 1. RGB-only Sim2Real 3D perception pipeline

We present a reproducible multi-camera pipeline that uses synthetic 2D/3D
supervision during development but requires only synchronized RGB and camera
calibration at inference. Unlike fixed-layout BEV systems, the same geometric
lifting and fusion procedure applies to unseen camera networks.

### 2. Association-Preserving Reliability Cascade

We introduce APRC, which propagates reliability from calibrated observations
through multi-view fusion, online assignment, track confidence, and causal
track initiation. Unlike detector confidence filtering alone, APRC uses
cross-view support and world-space consistency to determine how strongly an
observation may affect identity state.

### 3. Precision-calibrated heterogeneous detector fusion

We combine high-resolution YOLO11x predictions with D-FINE using
source-specific prefilters and reliability weights. Weighted box fusion uses
confidence-weighted coordinates and noisy-OR score aggregation, followed by
classwise NMS. Validation shows that a precision-oriented D-FINE weight is
better than recall-heavy or D-FINE-heavy alternatives.

### 4. Class-conditioned causal association

The online tracker uses class-specific confidence thresholds, spatial gates,
assignment costs, memory durations, and duplicate-birth radii. For the dominant
person class, high-confidence observations start tracks immediately, while
uncertain observations require a second causal match. This suppresses false
track births without revising past outputs.

### 5. Conservative world-space tracklet graph

For the offline configuration, fragmented tracklets form a directed acyclic
graph within each scene and class. Candidate edges are gated by temporal gap
and motion-scaled BEV distance, then scored using predicted center distance,
velocity continuity, 3D IoU, yaw consistency, and logarithmic size consistency.
A greedy one-in/one-out selection prevents cycles and many-to-one merges.

The final graph changes no box coordinates and adds no detections. Archive
analysis shows that it preserves all 1,414,646 output boxes while merging 249
track fragments: 245 person fragments and 4 forklift fragments. Relative to
the causal detector-fusion baseline, it improves official 3D HOTA from 12.3778
to 12.4128 and AssA from 11.7634 to 11.8109, while LocA remains exactly
57.0038. This isolates the gain to association rather than geometry.

### 6. Diagnostic study of privileged depth

We analyze why monocular DepthPro lifting, multi-view depth optimization,
V-DETR point-cloud detection, and learned geometric residuals did not improve
the final result. The useful conclusion is not that depth is unhelpful. It is
that depth scale, object support, camera calibration, and 3D box
parameterization must be jointly aligned; otherwise geometric changes can
increase identity fragmentation.

Contribution 5 should appear only with a compact quantitative ablation and
clear implementation details. Do not imply a universal conclusion from one
depth estimator or one dataset.

## Novelty Boundary

Safe claims:

- first application of this complete association-preserving pipeline to the
  2026 RGB-only Sim2Real protocol;
- confidence-conditioned causal track initiation for multi-camera 3D tracking;
- class-conditioned fusion/tracking calibration tied to 3D HOTA;
- a constrained world-space tracklet graph with an isolated official AssA
  gain;
- empirical analysis of when privileged synthetic depth fails to transfer.

Claims to avoid:

- first multi-camera 3D tracker;
- first detector fusion or weighted box fusion method;
- first BEV tracklet graph;
- state of the art;
- depth-free geometry learning, because the final method uses calibration and
  class priors rather than learning full metric depth;
- calling the BEV graph online. It uses future tracklets and is offline.

## Paper-Ready Abstract Draft

Multi-camera 3D tracking in smart spaces must recover metric object states and
persistent identities across heterogeneous views. AI City Challenge 2026 Track
1 makes this problem especially difficult: training data are synthetic and
include privileged depth, whereas evaluation introduces real RGB video without depth.
We present STAR-3D, an association-preserving RGB-only pipeline that controls
error propagation across detection, geometric lifting, multi-view fusion, and
tracking. STAR-3D combines high-resolution YOLO11x and D-FINE detections through
precision-calibrated weighted box fusion, lifts observations into a shared
world frame using camera calibration and class priors, and performs
class-conditioned online association with confidence-aware track initiation.
An optional conservative tracklet graph repairs short identity fragments using
motion, 3D overlap, yaw, and size consistency without altering detections or
box geometry. Our offline configuration ranks ninth in Track 1 with 12.4128 3D
HOTA, while the causal online configuration obtains 12.3778. Controlled
experiments show that estimated-depth and point-cloud variants can degrade
tracking when depth scale, object support, and calibration are not jointly
aligned.

Keywords: multi-camera tracking; 3D perception; Sim2Real; RGB-only inference;
camera calibration; HOTA

## Paper-Ready Introduction Draft

Reliable perception in warehouses, factories, hospitals, and retail spaces
requires more than detecting objects independently in each camera. A deployed
system must estimate where each person, robot, forklift, or pallet truck is in
a common metric coordinate system and preserve its identity as it moves across
overlapping and non-overlapping views. This scene-level representation supports
safety monitoring, human-robot interaction, and operational analytics, but it
is difficult to obtain from conventional CCTV. Objects are frequently small or
occluded, camera viewpoints vary widely, and the same target may look different
across cameras.

AI City Challenge 2026 Track 1 exposes an additional deployment gap. Its
training and validation corpus contains richly annotated synthetic warehouse
scenes with synchronized RGB, camera calibration, 2D and 3D boxes, and optional
depth. The test split introduces real-world scenes and provides only RGB and
calibration. Consequently,
the model must cross both an appearance gap and a modality gap: supervision can
use synthetic depth and metric 3D labels, but inference cannot assume an RGB-D
sensor. This setting differs materially from the 2025 benchmark, where
point-cloud detectors and 3D ReID could exploit depth throughout the pipeline.

The central difficulty is that detection, localization, and association errors
are coupled. A small error in a 2D box changes the image ray used for geometric
lifting. The resulting world-space error can prevent observations of the same
object from being fused across cameras, producing duplicate 3D detections.
Those duplicates create false track births and fragmented identities. The
reverse is also true: a depth or center correction that improves a framewise
localization objective may perturb temporal consistency and reduce association
accuracy. This coupling is reflected by 3D HOTA, which balances DetA, AssA, and
LocA instead of rewarding any module independently.

Existing approaches address parts of this problem but leave important gaps.
Point-cloud detectors provide native 3D reasoning but require accurate depth.
Learned early-BEV aggregation can combine views before detection, yet often
assumes fixed camera layouts or densely aligned geometric input. Conventional
late-fusion systems transfer more easily across camera networks, but their
projection and association stages are sensitive to detector confidence,
class-specific scale, and motion. Monocular metric-depth estimation appears to
restore the missing modality, but errors at object boundaries and synthetic-to-
real scale shifts can corrupt 3D fusion even when the predicted depth map is
visually plausible.

We therefore ask: how can an RGB-only multi-camera tracker preserve identity
when its test-time 3D evidence is uncertain and domain shifted? Our answer is
STAR-3D, a calibration-first pipeline built around association-preserving
evidence control. First, high-resolution YOLO11x and D-FINE predictions are
fused with source-specific confidence calibration, retaining complementary
detections while limiting low-confidence transformer proposals. Second,
detections are lifted into a common world frame through camera calibration and
class-conditioned 3D priors, then fused with class-specific spatial gates.
Third, an online tracker applies class-conditioned motion and lifetime models.
Its confidence-aware initiation rule immediately emits reliable person tracks
but requires causal confirmation for ambiguous births. Finally, an optional
offline directed tracklet graph repairs short fragments only when motion,
overlap, yaw, and object dimensions agree.

Our experiments support two findings. First, preserving association can be
more important than marginal framewise localization gains: several variants
with slightly higher DetA and LocA score lower overall because AssA decreases.
Second, adding estimated depth or a point-cloud detector is not sufficient by
itself. Without joint alignment of metric scale, foreground support,
calibration, and box parameterization, geometric corrections can destabilize
cross-view fusion and tracking. On the official real test set, STAR-3D's causal
configuration obtains 12.3778 3D HOTA. The optional offline graph preserves all
detections and box coordinates, merges 249 fragments, and improves the score to
12.4128, ranking ninth in the challenge.

Our contributions are:

1. We formulate RGB-only Sim2Real multi-camera 3D tracking as a
   geometry-association coupling problem under privileged training depth.
2. We introduce APRC, which propagates reprojection, camera-support, spatial,
   detection, and track-history reliability through multi-view fusion and
   causal online association.
3. We design a conservative world-space tracklet graph that improves AssA
   without modifying detection or localization outputs.
4. We provide a controlled analysis of detector, estimated-depth, geometric
   residual, and point-cloud variants, identifying cross-module consistency as
   the main requirement for useful RGB-to-3D transfer.

## Required Experimental Evidence

Current official results support the system paper, but stronger acceptance
odds require validation evidence beyond hidden-test submissions.

Minimum tables:

1. Detector and fusion ablation: YOLO11x, D-FINE, uncalibrated ensemble,
   precision-calibrated ensemble.
2. Association ablation: global thresholds, class-conditioned thresholds,
   delayed person confirmation, confidence-conditioned causal initiation,
   offline graph.
3. Geometry ablation: ground-plane lifting, learned residual, DepthPro lift,
   multi-view depth optimization, V-DETR fusion.
4. Online/offline comparison with exact causality labeling.
5. Per-class metrics and prediction counts, especially Person, NovaCarter,
   FourierGR1T2, AgilityDigit, and PalletTruck.
6. Runtime and hardware for detector inference, geometric fusion, tracking,
   and graph relinking.

Required qualitative figure:

- one multi-camera frame group;
- corresponding 2D detections;
- projected global BEV candidates;
- fused 3D observation;
- online fragmented trajectory;
- offline graph repair;
- one depth failure case showing foreground/background contamination.

## Reviewer-Risk Checklist

- Explain why this is more than a detector ensemble.
- Separate implemented components from attempted components.
- Report both absolute scores and deltas.
- Never tune scientific claims only from hidden-test submissions.
- State that final rank is ninth; do not claim state of the art.
- Mark `12.3778` as causal online and `12.4128` as optional offline.
- Define every class-specific threshold and how validation selected it.
- Include a code/config mapping for full reproducibility.
- Keep negative depth conclusions scoped to tested models and protocol.
