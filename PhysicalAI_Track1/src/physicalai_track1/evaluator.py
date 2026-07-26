from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .dataset import TrackBox
from .constants import ID_TO_CLASS
from .geometry import box3d_iou

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for minimal environments
    linear_sum_assignment = None


FrameKey = Tuple[int, int, int]
TrackKey = Tuple[int, int, int]
PairKey = Tuple[int, int, int, int]


@dataclass
class ThresholdResult:
    threshold: float
    hota: float
    deta: float
    assa: float
    loca: float
    tp: int
    fp: int
    fn: int


def _group_by_frame(boxes: Sequence[TrackBox]) -> Dict[FrameKey, List[int]]:
    grouped: Dict[FrameKey, List[int]] = defaultdict(list)
    for idx, box in enumerate(boxes):
        grouped[(box.scene_id, box.class_id, box.frame_id)].append(idx)
    return grouped


def _track_counts(boxes: Sequence[TrackBox]) -> Counter:
    counts = Counter()
    for box in boxes:
        counts[(box.scene_id, box.class_id, box.object_id)] += 1
    return counts


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _match_iou_matrix(iou_matrix: List[List[float]], threshold: float) -> List[Tuple[int, int, float]]:
    if not iou_matrix or not iou_matrix[0]:
        return []
    if linear_sum_assignment is not None:
        cost = [[-v for v in row] for row in iou_matrix]
        rows, cols = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(rows, cols):
            iou = float(iou_matrix[r][c])
            if iou >= threshold:
                matches.append((int(r), int(c), iou))
        return matches

    # Greedy fallback. This is not exact but keeps validation usable.
    candidates = []
    for r, row in enumerate(iou_matrix):
        for c, value in enumerate(row):
            iou = float(value)
            if iou >= threshold:
                candidates.append((iou, r, c))
    candidates.sort(reverse=True)
    used_r = set()
    used_c = set()
    matches = []
    for iou, r, c in candidates:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matches.append((r, c, iou))
    return matches


def evaluate_at_threshold(
    gt_boxes: Sequence[TrackBox],
    pred_boxes: Sequence[TrackBox],
    threshold: float,
) -> ThresholdResult:
    gt_by_frame = _group_by_frame(gt_boxes)
    pred_by_frame = _group_by_frame(pred_boxes)
    gt_track_counts = _track_counts(gt_boxes)
    pred_track_counts = _track_counts(pred_boxes)

    keys = sorted(set(gt_by_frame) | set(pred_by_frame))
    pair_counts: Counter[PairKey] = Counter()
    matched_records: List[Tuple[TrackKey, TrackKey, float]] = []
    tp = 0
    fp = 0
    fn = 0
    loc_sum = 0.0

    for key in keys:
        gt_indices = gt_by_frame.get(key, [])
        pred_indices = pred_by_frame.get(key, [])
        if not gt_indices:
            fp += len(pred_indices)
            continue
        if not pred_indices:
            fn += len(gt_indices)
            continue

        ious = [[0.0 for _ in pred_indices] for _ in gt_indices]
        for r, gi in enumerate(gt_indices):
            for c, pi in enumerate(pred_indices):
                ious[r][c] = box3d_iou(gt_boxes[gi], pred_boxes[pi])

        matches = _match_iou_matrix(ious, threshold)
        tp += len(matches)
        fp += len(pred_indices) - len(matches)
        fn += len(gt_indices) - len(matches)

        for local_g, local_p, iou in matches:
            g = gt_boxes[gt_indices[local_g]]
            p = pred_boxes[pred_indices[local_p]]
            gt_key = (g.scene_id, g.class_id, g.object_id)
            pred_key = (p.scene_id, p.class_id, p.object_id)
            pair_key = (g.scene_id, g.class_id, g.object_id, p.object_id)
            pair_counts[pair_key] += 1
            matched_records.append((gt_key, pred_key, iou))
            loc_sum += iou

    denom = tp + fp + fn
    deta = tp / denom if denom else 0.0
    loca = loc_sum / tp if tp else 0.0

    if tp:
        ass_values = []
        for gt_key, pred_key, _ in matched_records:
            scene_id, class_id, gt_id = gt_key
            pred_id = pred_key[2]
            pair_key = (scene_id, class_id, gt_id, pred_id)
            tpa = pair_counts[pair_key]
            fna = gt_track_counts[gt_key] - tpa
            fpa = pred_track_counts[pred_key] - tpa
            ass_values.append(tpa / (tpa + fna + fpa))
        assa = _mean(ass_values)
    else:
        assa = 0.0

    hota = math.sqrt(deta * assa) if deta > 0.0 and assa > 0.0 else 0.0
    return ThresholdResult(threshold, hota, deta, assa, loca, tp, fp, fn)


def evaluate_hota_like(
    gt_boxes: Sequence[TrackBox],
    pred_boxes: Sequence[TrackBox],
    thresholds: Iterable[float] | None = None,
) -> dict:
    if thresholds is None:
        thresholds = [round(i / 100, 2) for i in range(5, 100, 5)]
    results = [evaluate_at_threshold(gt_boxes, pred_boxes, float(t)) for t in thresholds]
    payload = {
        "hota_like": _mean([r.hota for r in results]),
        "deta": _mean([r.deta for r in results]),
        "assa": _mean([r.assa for r in results]),
        "loca": _mean([r.loca for r in results]),
        "thresholds": [
            {
                "threshold": r.threshold,
                "hota": r.hota,
                "deta": r.deta,
                "assa": r.assa,
                "loca": r.loca,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
            }
            for r in results
        ],
    }
    return payload


def evaluate_by_class(
    gt_boxes: Sequence[TrackBox],
    pred_boxes: Sequence[TrackBox],
    thresholds: Iterable[float] | None = None,
) -> dict:
    class_ids = sorted({box.class_id for box in gt_boxes} | {box.class_id for box in pred_boxes})
    return {
        ID_TO_CLASS.get(class_id, str(class_id)): {
            "class_id": class_id,
            "gt_boxes": sum(box.class_id == class_id for box in gt_boxes),
            "pred_boxes": sum(box.class_id == class_id for box in pred_boxes),
            **evaluate_hota_like(
                [box for box in gt_boxes if box.class_id == class_id],
                [box for box in pred_boxes if box.class_id == class_id],
                thresholds=thresholds,
            ),
        }
        for class_id in class_ids
    }


def dumps_metrics(metrics: dict) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)
