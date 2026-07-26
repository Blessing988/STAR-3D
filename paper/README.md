# Track 1 Paper Workspace

This folder holds the paper draft material for AI City Challenge 2026 Track 1:
Multi-Camera 3D Perception (Sim2Real).

## Main Results

Offline result:

`results/track1-yolo11x1920_dfine_p025_bev_gap60_cost115_v2.zip`

Leaderboard result:

| Method | 3D HOTA (%) | DetA (%) | AssA (%) | LocA (%) | Rank |
|---|---:|---:|---:|---:|---:|
| YOLO11x1920 + D-FINE p0.25 + BEV gap60 cost1.15 | **12.4128** | 14.5213 | 11.8109 | 57.0038 | 9 |

Causal online result:

`results/track1-yolo11x1920_dfine_precision025_adaptive_online.zip`

| Method | 3D HOTA (%) | DetA (%) | AssA (%) | LocA (%) |
|---|---:|---:|---:|---:|
| YOLO11x1920 + D-FINE p0.25 + confidence-conditioned online tracking | **12.3778** | 14.5214 | 11.7634 | 57.0038 |

## Paper Position

Core thesis:

> Under RGB-only Sim2Real shift, detection, metric localization, and identity
> association must be optimized as one coupled system. STAR-3D controls how
> uncertain evidence propagates through detector fusion, calibrated lifting,
> multi-view fusion, causal tracking, and optional offline fragment repair
> through an Association-Preserving Reliability Cascade (APRC).

Do not claim DepthPro, V-DETR, or final class-hybrid improved the official result. They are useful as negative/diagnostic ablations.

Do not call the BEV tracklet graph online. It uses future tracklets. Use the
`12.3778` configuration for online claims and the `12.4128` configuration for
best raw leaderboard performance.

## Submission Constraints

- submission deadline: July 24, 2026, Anywhere on Earth;
- length: 4-14 pages, excluding references;
- review is not double-blind;
- title, abstract, keywords, author names, and affiliations must be final at submission;
- accepted papers are published by Springer.

Template warning: `eccv.sty` replaces authors with an anonymous placeholder
when loaded with the `review` option. The AI City workshop requires visible
authors, so the paper source must use the non-anonymous package mode unless the
organizers issue a different instruction.

## Draft Files

- `outline.md`: section-by-section paper plan.
- `results_table.md`: official result and ablation table.
- `method_notes.md`: concise method details and contribution framing.
- `negative_results.md`: what failed and how to discuss it scientifically.
- `problem_motivation.md`: final research framing, abstract, Introduction, and
  contribution claims.
- `submission_guidelines.md`: ECCV/AI City formatting rules and lessons from
  the 2025 workshop exemplars.
