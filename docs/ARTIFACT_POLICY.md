# Artifact Policy

This repository is intended to be a clean, reproducible paper-code release. It
does not store large data or private cluster outputs.

## Tracked

- Source code under `PhysicalAI_Track1/src/`.
- Reproducible scripts under `PhysicalAI_Track1/scripts/`.
- PBS templates under `PhysicalAI_Track1/scripts/pbs/`.
- Configuration files under `PhysicalAI_Track1/configs/`.
- Documentation and experiment logs under `PhysicalAI_Track1/docs/`.
- ECCV paper source and final paper figures under
  `eccv_paper/paper-template-Latest/ECCV-paper-template/`.

## Not Tracked

- AI City dataset files.
- Extracted RGB frames.
- Depth maps or generated depth estimates.
- Detector/V-DETR checkpoints.
- Submission zips and raw leaderboard artifacts.
- Large report folders.
- Cloned third-party repositories.
- Scheduler stdout/stderr logs.

## Recommended Release Layout

If artifacts need to be shared, use GitHub Releases or an external artifact
store:

```text
release-v1/
  checkpoints/
  submissions/
  qualitative_assets/
  validation_reports/
```

Keep the repository itself source-first. This makes cloning fast and keeps the
paper implementation auditable.
