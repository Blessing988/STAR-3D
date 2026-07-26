from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, Mapping, Tuple

from .fusion import FusedDetection3D


SceneClass = Tuple[int, int]


@dataclass
class SceneClassStats:
    frames: int = 0
    detections: int = 0
    mean_score: float = 0.0
    mean_sources: float = 0.0
    mean_spread_m: float = 0.0
    mean_reprojection_error: float = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _median(values: list[float], default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def calibrate_scene_class_thresholds(
    detections: Iterable[FusedDetection3D],
    base_class_min_scores: Mapping[int, float],
    base_class_immediate_birth_scores: Mapping[int, float] | None = None,
    warmup_frames: int = 180,
    strength: float = 0.18,
    min_threshold: float = 0.05,
    max_threshold: float = 0.98,
) -> dict:
    """Estimate test-time reliability from early fused detections.

    The estimator uses no labels. It treats multi-camera support, compact
    cluster spread, low reprojection error, and stable detection density as
    evidence that a scene/class pair can tolerate a slightly lower threshold.
    Weak support or unusually dense detections raise the threshold.
    """

    warmup_frames = max(1, int(warmup_frames))
    strength = _clamp(strength, 0.0, 0.5)

    first_frame: Dict[int, int] = {}
    frame_keys: Dict[SceneClass, set[int]] = defaultdict(set)
    scores: Dict[SceneClass, list[float]] = defaultdict(list)
    sources: Dict[SceneClass, list[float]] = defaultdict(list)
    spreads: Dict[SceneClass, list[float]] = defaultdict(list)
    reproj: Dict[SceneClass, list[float]] = defaultdict(list)

    for det in detections:
        scene_id = det.box.scene_id
        first = first_frame.setdefault(scene_id, det.box.frame_id)
        if det.box.frame_id - first >= warmup_frames:
            continue
        key = (scene_id, det.box.class_id)
        frame_keys[key].add(det.box.frame_id)
        scores[key].append(det.score)
        sources[key].append(float(det.source_count))
        spreads[key].append(max(0.0, det.cluster_spread_m))
        reproj[key].append(max(0.0, det.mean_reprojection_error))

    densities = {
        key: len(scores[key]) / max(1, len(frame_keys[key]))
        for key in scores
    }
    class_density_medians: Dict[int, float] = {}
    for class_id in {key[1] for key in scores}:
        class_density_medians[class_id] = _median(
            [density for key, density in densities.items() if key[1] == class_id],
            default=1.0,
        )

    scene_class_min_scores: Dict[SceneClass, float] = {}
    scene_class_immediate_birth_scores: Dict[SceneClass, float] = {}
    stats: Dict[str, dict] = {}

    for key in sorted(scores):
        scene_id, class_id = key
        base_threshold = float(base_class_min_scores.get(class_id, 0.0))
        if base_threshold <= 0.0:
            continue

        mean_score = mean(scores[key])
        mean_sources = mean(sources[key])
        mean_spread = mean(spreads[key]) if spreads[key] else 0.0
        mean_reproj = mean(reproj[key]) if reproj[key] else 0.0
        density = densities[key]
        median_density = max(1e-6, class_density_medians.get(class_id, density))

        support_bonus = _clamp((mean_sources - 1.0) / 2.0, 0.0, 1.0)
        score_bonus = _clamp((mean_score - base_threshold) / max(1e-6, 1.0 - base_threshold), -1.0, 1.0)
        compact_bonus = 1.0 - _clamp(mean_spread / 1.25, 0.0, 1.0)
        reproj_bonus = 1.0 - _clamp(mean_reproj / 24.0, 0.0, 1.0)
        density_penalty = _clamp((density / median_density) - 1.0, 0.0, 1.5)

        reliability = (
            0.35 * support_bonus
            + 0.25 * score_bonus
            + 0.20 * compact_bonus
            + 0.20 * reproj_bonus
            - 0.30 * density_penalty
        )
        delta = -strength * reliability
        calibrated = _clamp(base_threshold + delta, min_threshold, max_threshold)
        scene_class_min_scores[key] = calibrated

        if base_class_immediate_birth_scores and class_id in base_class_immediate_birth_scores:
            birth_base = float(base_class_immediate_birth_scores[class_id])
            scene_class_immediate_birth_scores[key] = _clamp(
                birth_base + 0.5 * delta,
                min_threshold,
                max_threshold,
            )

        stats[f"{scene_id}:{class_id}"] = {
            "base_min_score": base_threshold,
            "calibrated_min_score": calibrated,
            "density_per_frame": density,
            "median_class_density_per_frame": median_density,
            "frames": len(frame_keys[key]),
            "detections": len(scores[key]),
            "mean_score": mean_score,
            "mean_sources": mean_sources,
            "mean_spread_m": mean_spread,
            "mean_reprojection_error": mean_reproj,
            "reliability": reliability,
        }

    return {
        "warmup_frames": warmup_frames,
        "strength": strength,
        "scene_class_min_scores": scene_class_min_scores,
        "scene_class_immediate_birth_scores": scene_class_immediate_birth_scores,
        "stats": stats,
    }
