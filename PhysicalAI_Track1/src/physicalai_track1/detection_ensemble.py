from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .detections import Detection2D, read_detections, write_detections
from .detector_adapters import _iou2d, classwise_nms


@dataclass(frozen=True)
class WeightedDetection:
    detection: Detection2D
    source_index: int
    source_weight: float

    @property
    def adjusted_score(self) -> float:
        return max(0.0, self.detection.score * self.source_weight)


def _score_from_cluster(cluster: Sequence[WeightedDetection], mode: str) -> float:
    scores = [item.adjusted_score for item in cluster]
    if mode == "max":
        return max(scores)
    if mode == "mean":
        return sum(scores) / max(1, len(scores))
    if mode == "noisy_or":
        miss = 1.0
        for score in scores:
            miss *= max(0.0, 1.0 - min(1.0, score))
        return 1.0 - miss
    raise ValueError(f"Unknown score mode: {mode}")


def _fuse_cluster(cluster: Sequence[WeightedDetection], score_mode: str) -> Detection2D:
    if not cluster:
        raise ValueError("Cannot fuse an empty cluster")
    denom = sum(max(1e-6, item.adjusted_score) for item in cluster)
    x1 = sum(item.detection.x1 * max(1e-6, item.adjusted_score) for item in cluster) / denom
    y1 = sum(item.detection.y1 * max(1e-6, item.adjusted_score) for item in cluster) / denom
    x2 = sum(item.detection.x2 * max(1e-6, item.adjusted_score) for item in cluster) / denom
    y2 = sum(item.detection.y2 * max(1e-6, item.adjusted_score) for item in cluster) / denom
    best = max(cluster, key=lambda item: item.adjusted_score).detection
    return Detection2D(
        scene_name=best.scene_name,
        camera_id=best.camera_id,
        frame_id=best.frame_id,
        class_id=best.class_id,
        score=min(1.0, _score_from_cluster(cluster, score_mode)),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _weighted_box_fusion(
    detections: Sequence[WeightedDetection],
    iou_threshold: float,
    score_mode: str,
) -> list[Detection2D]:
    clusters: list[list[WeightedDetection]] = []
    representatives: list[Detection2D] = []
    for item in sorted(detections, key=lambda x: x.adjusted_score, reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, representative in enumerate(representatives):
            iou = _iou2d(item.detection, representative)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            clusters[best_idx].append(item)
            representatives[best_idx] = _fuse_cluster(clusters[best_idx], score_mode)
        else:
            clusters.append([item])
            representatives.append(item.detection)
    return [_fuse_cluster(cluster, score_mode) for cluster in clusters]


def ensemble_detections(
    inputs: Sequence[Path | str],
    out_path: Path | str,
    weights: Sequence[float] | None = None,
    wbf_iou: float = 0.65,
    final_nms_iou: float = 0.80,
    min_score: float = 0.01,
    score_mode: str = "noisy_or",
    max_per_frame_class: int | None = None,
) -> dict:
    if not inputs:
        raise ValueError("At least one detection input is required")
    source_weights = [1.0 for _ in inputs] if weights is None else [float(value) for value in weights]
    if len(source_weights) != len(inputs):
        raise ValueError(f"Expected {len(inputs)} weights, got {len(source_weights)}")

    groups: dict[tuple[str, str, int, int], list[WeightedDetection]] = defaultdict(list)
    source_counts = []
    for source_index, input_path in enumerate(inputs):
        count = 0
        for det in read_detections(input_path):
            adjusted_score = det.score * source_weights[source_index]
            if adjusted_score < min_score:
                continue
            groups[(det.scene_name, det.camera_id, det.frame_id, det.class_id)].append(
                WeightedDetection(det, source_index, source_weights[source_index])
            )
            count += 1
        source_counts.append({"input": str(input_path), "kept_after_min_score": count, "weight": source_weights[source_index]})

    fused: list[Detection2D] = []
    for key in sorted(groups):
        group_fused = _weighted_box_fusion(groups[key], iou_threshold=wbf_iou, score_mode=score_mode)
        if final_nms_iou > 0.0:
            group_fused = classwise_nms(group_fused, final_nms_iou)
        group_fused.sort(key=lambda det: det.score, reverse=True)
        if max_per_frame_class is not None:
            group_fused = group_fused[:max_per_frame_class]
        fused.extend(group_fused)

    count = write_detections(fused, out_path)
    result = {
        "inputs": source_counts,
        "groups": len(groups),
        "detections": count,
        "wbf_iou": wbf_iou,
        "final_nms_iou": final_nms_iou,
        "min_score": min_score,
        "score_mode": score_mode,
        "max_per_frame_class": max_per_frame_class,
        "output": str(out_path),
    }
    Path(str(out_path) + ".json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
