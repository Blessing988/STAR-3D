# ECCV 2026 AI City Workshop Submission Notes

## Workshop Requirements

- Deadline: July 24, 2026, Anywhere on Earth.
- Length: 4-14 pages excluding references.
- Review: not double-blind; authors and affiliations must be visible.
- Title, abstract, keywords, author names, and affiliations are fixed at submission.
- Accepted papers are published by Springer.
- Acceptance is based primarily on technical merit, not leaderboard rank.

## Template Configuration

The downloaded `main.tex` currently uses:

```tex
\usepackage[review,year=2026,ID=*****]{eccv}
```

In `eccv.sty`, review mode replaces the supplied authors and affiliations with
an anonymous placeholder. This conflicts with the AI City workshop's explicit
non-double-blind policy.

For the workshop submission, use the supported non-anonymous mode:

```tex
\usepackage{eccv}
```

Do not patch `eccv.sty` to combine review line numbering with visible authors.
The template prohibits layout/style modifications. If organizers issue a
workshop-specific line-number instruction, follow that instruction instead.

## Required Formatting Details

- Use the supplied `llncs.cls` and `eccv.sty` without margin, font, line-spacing,
  or page-layout changes.
- Abstract target: approximately 150 words.
- Include `\keywords{...}` inside the abstract environment.
- Do not cite papers or use footnotes in the abstract.
- Use Computer Modern through the template; do not switch to Times.
- Put table captions above tables.
- Put figure captions below figures.
- Cross-reference every figure, table, equation, and section in prose.
- Prefer vector diagrams. Raster line art should be at least 800 dpi.
- Keep figure text at least 6 pt.
- Number and punctuate displayed equations.
- Use `splncs04` bibliography style.
- Remove the template's example `\clearpage` before submission.
- Use normal `hyperref` for the non-review/final-style version.
- Add ORCID identifiers when available, especially for camera-ready.

## Lessons from 2025 AI City Papers

### VGCRTrack

Strong structure:

1. identifies online cross-view association and center localization failures;
2. introduces named mechanisms: trajectory-level Frechet affinity, view-wise
   3D IoU, and View-Aware Geometric Center Refinement;
3. states exact leaderboard score and rank;
4. lists modular contributions that correspond to method subsections.

Lesson for STAR-3D: APRC must be defined mathematically and ablated component by
component. The graph must remain a separate offline extension.

### DepthTrack

Strong structure:

1. contrasts camera-only and expensive point-cloud approaches;
2. names one central mechanism, Tracklet-Cluster Mapping;
3. connects each supporting component to a concrete deployment problem;
4. reports challenge performance in the abstract and Introduction.

Lesson for STAR-3D: center the paper on APRC, not the list of detectors and
experiments.

### Online Depth-Based Late Aggregation

Strong structure:

1. motivates compatibility with existing 2D tracking systems;
2. presents local-ID consistency as the association innovation;
3. clearly claims online operation;
4. separates 2D tracking from 3D recovery.

Lesson for STAR-3D: provide exact causality pseudocode and clearly separate the
12.3778 online system from the 12.4128 offline graph result.

### MCBLT

Strong structure:

1. organizes prior work into late, geometric late, and early aggregation;
2. identifies fixed-camera generalization and long-term tracking as gaps;
3. proposes early BEV aggregation plus hierarchical GNN tracking;
4. supports broad claims on multiple datasets.

Lesson for STAR-3D: do not claim broad state of the art from one challenge.
Claim camera-layout flexibility and RGB-only deployment only where supported.

### Hierarchical Multi-Modal Fusion

Strong structure:

1. frames traditional and learned methods as complementary;
2. identifies sparse labels as the constraint;
3. gives a named hierarchical fusion strategy;
4. compares each branch and their fusion quantitatively.

Lesson for STAR-3D: show why YOLO11x and D-FINE are complementary through
per-class precision/recall or disagreement statistics, not only final HOTA.

## Recommended Paper Shape

Target 9-10 pages excluding references:

1. Introduction: 1 page.
2. Related Work: 0.75-1 page.
3. Problem Setup: 0.75 page.
4. STAR-3D Method: 2.5-3 pages.
5. Experiments: 3-3.5 pages.
6. Limitations and Discussion: 0.5 page.
7. Conclusion: 0.25 page.

Essential visual elements:

- Figure 1: problem contrast, synthetic RGB-D training versus real RGB-only test.
- Figure 2: full STAR-3D/APRC pipeline.
- Figure 3: causal track initiation and optional offline tracklet graph.
- Table 1: dataset and protocol.
- Table 2: detector/fusion ablation.
- Table 3: APRC component ablation.
- Table 4: depth and 3D-detector diagnostic study.
- Table 5: official test results and online/offline distinction.

