from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from .adaptive_thresholds import calibrate_scene_class_thresholds
from .association_model import AssociationScorer
from .dataset import TrackBox
from .fusion import FusedDetection3D, read_fused_detections
from .geometry import angle_distance, box3d_iou, distance_xy
from .submission import write_submission

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - keeps tracker usable without scipy
    linear_sum_assignment = None


@dataclass
class OnlineTrack:
    object_id: int
    class_id: int
    scene_id: int
    x: float
    y: float
    z: float
    width: float
    length: float
    height: float
    yaw: float
    vx: float
    vy: float
    last_frame: int
    misses: int = 0
    hits: int = 1
    score: float = 1.0
    source_count: int = 1
    cameras: Tuple[str, ...] = ()
    confirmed: bool = True
    pending_outputs: List[TrackBox] = field(default_factory=list)
    track_confidence: float = 0.5
    class_votes: Dict[int, float] = field(default_factory=dict)

    def predict_xy(self, frame_id: int) -> Tuple[float, float]:
        dt = max(0, frame_id - self.last_frame)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def predicted_box(self, frame_id: int) -> TrackBox:
        px, py = self.predict_xy(frame_id)
        return TrackBox(
            scene_id=self.scene_id,
            class_id=self.class_id,
            object_id=self.object_id,
            frame_id=frame_id,
            x=px,
            y=py,
            z=self.z,
            width=self.width,
            length=self.length,
            height=self.height,
            yaw=self.yaw,
            score=self.score,
        )


def _yaw_from_velocity(vx: float, vy: float, fallback: float) -> float:
    speed = (vx * vx + vy * vy) ** 0.5
    if speed < 0.05:
        return fallback
    # Dataset yaw=0 points roughly along negative y in the object-centered figure.
    return math.atan2(vx, -vy)


def _dynamic_gate(base: float, track: OnlineTrack, frame_id: int) -> float:
    dt = max(1, frame_id - track.last_frame)
    size_gate = 0.35 * max(track.width, track.length, 0.5)
    return max(base + size_gate, base * min(3.0, math.sqrt(dt)))


def _class_float(
    values: Mapping[int, float] | None,
    class_id: int,
    default: float,
) -> float:
    if not values:
        return default
    return values.get(class_id, default)


def _scene_class_float(
    values: Mapping[Tuple[int, int], float] | None,
    scene_id: int,
    class_id: int,
    default: float,
) -> float:
    if not values:
        return default
    return values.get((scene_id, class_id), default)


def _class_int(
    values: Mapping[int, int] | None,
    class_id: int,
    default: int,
) -> int:
    if not values:
        return default
    return values.get(class_id, default)


def _association_cost(
    track: OnlineTrack,
    det: FusedDetection3D,
    max_distance_m: float,
    distance_weight: float,
    iou_weight: float,
    yaw_weight: float,
    score_weight: float,
    source_weight: float,
    association_scorer: AssociationScorer | None = None,
    association_weight: float = 0.0,
    association_min_probability: float | None = None,
) -> Tuple[float, float]:
    predicted = track.predicted_box(det.box.frame_id)
    dist = distance_xy(predicted, det.box)
    gate = _dynamic_gate(max_distance_m, track, det.box.frame_id)
    if dist > gate:
        return float("inf"), dist
    iou = box3d_iou(predicted, det.box)
    yaw_cost = angle_distance(predicted.yaw, det.box.yaw) / math.pi
    camera_support = min(1.0, len(det.cameras) / 3.0)
    track_reliability = min(1.0, max(0.0, track.track_confidence))
    cost = (
        distance_weight * (dist / max(1e-6, gate))
        + iou_weight * (1.0 - iou)
        + yaw_weight * yaw_cost
        - score_weight * det.score
        - source_weight * camera_support
        - 0.10 * track_reliability
    )
    if association_scorer is not None and association_weight > 0.0:
        probability = association_scorer.predict_proba(predicted, det.box)
        if association_min_probability is not None and probability < association_min_probability:
            return float("inf"), dist
        learned_cost = 1.0 - probability
        weight = min(1.0, max(0.0, association_weight))
        cost = (1.0 - weight) * cost + weight * learned_cost
    return cost, dist


def _match_detections_to_tracks(
    tracks: List[OnlineTrack],
    detections: List[FusedDetection3D],
    max_distance_m: float,
    max_cost: float,
    distance_weight: float,
    iou_weight: float,
    yaw_weight: float,
    score_weight: float,
    source_weight: float,
    association_scorer: AssociationScorer | None = None,
    association_weight: float = 0.0,
    association_min_probability: float | None = None,
) -> List[Tuple[int, int]]:
    if not tracks or not detections:
        return []
    costs: List[List[float]] = []
    for trk in tracks:
        row = []
        for det in detections:
            cost, _ = _association_cost(
                trk,
                det,
                max_distance_m=max_distance_m,
                distance_weight=distance_weight,
                iou_weight=iou_weight,
                yaw_weight=yaw_weight,
                score_weight=score_weight,
                source_weight=source_weight,
                association_scorer=association_scorer,
                association_weight=association_weight,
                association_min_probability=association_min_probability,
            )
            row.append(cost)
        costs.append(row)

    matches: List[Tuple[int, int]] = []
    if linear_sum_assignment is not None:
        finite = [[v if math.isfinite(v) else 1e6 for v in row] for row in costs]
        rows, cols = linear_sum_assignment(finite)
        for r, c in zip(rows, cols):
            if costs[int(r)][int(c)] <= max_cost:
                matches.append((int(r), int(c)))
        return matches

    candidates: List[Tuple[float, int, int]] = []
    for r, row in enumerate(costs):
        for c, cost in enumerate(row):
            if cost <= max_cost:
                candidates.append((cost, r, c))
    candidates.sort()
    used_tracks = set()
    used_dets = set()
    for _, r, c in candidates:
        if r in used_tracks or c in used_dets:
            continue
        used_tracks.add(r)
        used_dets.add(c)
        matches.append((r, c))
    return matches


def _track_output_box(track: OnlineTrack, det: FusedDetection3D) -> TrackBox:
    class_id = max(track.class_votes.items(), key=lambda item: item[1])[0] if track.class_votes else det.box.class_id
    return TrackBox(
        scene_id=det.box.scene_id,
        class_id=class_id,
        object_id=track.object_id,
        frame_id=det.box.frame_id,
        x=track.x,
        y=track.y,
        z=track.z,
        width=track.width,
        length=track.length,
        height=track.height,
        yaw=track.yaw,
        score=det.score,
    )


def _update_track(
    track: OnlineTrack,
    det: FusedDetection3D,
    position_alpha: float,
    velocity_alpha: float,
) -> TrackBox:
    dt = max(1, det.box.frame_id - track.last_frame)
    pred_x, pred_y = track.predict_xy(det.box.frame_id)
    measured_vx = (det.box.x - track.x) / dt
    measured_vy = (det.box.y - track.y) / dt
    alpha = min(1.0, max(0.0, position_alpha))
    track.x = alpha * det.box.x + (1.0 - alpha) * pred_x
    track.y = alpha * det.box.y + (1.0 - alpha) * pred_y
    track.z = alpha * det.box.z + (1.0 - alpha) * track.z
    track.width = alpha * det.box.width + (1.0 - alpha) * track.width
    track.length = alpha * det.box.length + (1.0 - alpha) * track.length
    track.height = alpha * det.box.height + (1.0 - alpha) * track.height
    track.vx = velocity_alpha * track.vx + (1.0 - velocity_alpha) * measured_vx
    track.vy = velocity_alpha * track.vy + (1.0 - velocity_alpha) * measured_vy
    track.yaw = _yaw_from_velocity(track.vx, track.vy, det.box.yaw)
    track.last_frame = det.box.frame_id
    track.misses = 0
    track.hits += 1
    support_bonus = min(0.12, 0.04 * max(0, det.source_count - 1))
    score_signal = min(1.0, max(0.0, det.score + support_bonus))
    hit_signal = min(1.0, track.hits / 5.0)
    track.track_confidence = min(1.0, 0.82 * track.track_confidence + 0.18 * (0.65 * score_signal + 0.35 * hit_signal))
    track.score = max(track.score * 0.95, det.score)
    track.source_count = det.source_count
    track.cameras = det.cameras
    for class_id in list(track.class_votes):
        track.class_votes[class_id] *= 0.92
    track.class_votes[det.box.class_id] = track.class_votes.get(det.box.class_id, 0.0) + max(0.05, det.score)
    track.class_id = max(track.class_votes.items(), key=lambda item: item[1])[0]
    return _track_output_box(track, det)


def _new_track_output(track: OnlineTrack, det: FusedDetection3D) -> TrackBox:
    class_id = max(track.class_votes.items(), key=lambda item: item[1])[0] if track.class_votes else det.box.class_id
    return TrackBox(
        scene_id=det.box.scene_id,
        class_id=class_id,
        object_id=track.object_id,
        frame_id=det.box.frame_id,
        x=det.box.x,
        y=det.box.y,
        z=det.box.z,
        width=det.box.width,
        length=det.box.length,
        height=det.box.height,
        yaw=det.box.yaw,
        score=det.score,
    )


def _emit_confirmed_box(
    track: OnlineTrack,
    box: TrackBox,
    outputs: List[TrackBox],
    required_hits: int,
    confirmation_mode: str,
) -> None:
    if required_hits <= 1 or confirmation_mode == "immediate":
        track.confirmed = True
        outputs.append(box)
        return
    if track.confirmed:
        outputs.append(box)
        return

    track.pending_outputs.append(box)
    if track.hits < required_hits:
        return

    track.confirmed = True
    if confirmation_mode == "backfill":
        outputs.extend(track.pending_outputs)
    elif confirmation_mode == "confirmed_only":
        outputs.append(box)
    else:
        raise ValueError(f"Unsupported confirmation mode: {confirmation_mode}")
    track.pending_outputs.clear()


def _is_duplicate_birth(
    det: FusedDetection3D,
    active_tracks: List[OnlineTrack],
    duplicate_birth_distance_m: float,
    duplicate_birth_iou: float,
) -> bool:
    if duplicate_birth_distance_m <= 0.0 and duplicate_birth_iou <= 0.0:
        return False
    for track in active_tracks:
        if track.last_frame != det.box.frame_id:
            continue
        predicted = track.predicted_box(det.box.frame_id)
        if duplicate_birth_distance_m > 0.0 and distance_xy(predicted, det.box) <= duplicate_birth_distance_m:
            return True
        if duplicate_birth_iou > 0.0 and box3d_iou(predicted, det.box) >= duplicate_birth_iou:
            return True
    return False


def _birth_allowed(
    det: FusedDetection3D,
    frame_active_tracks: List[OnlineTrack],
    min_track_confidence: float,
    adaptive_birth_score: float | None,
    adaptive_birth_min_sources: int | None,
) -> bool:
    if adaptive_birth_score is None and adaptive_birth_min_sources is None:
        return True
    if adaptive_birth_score is not None and det.score >= adaptive_birth_score:
        return True
    if adaptive_birth_min_sources is not None and det.source_count >= adaptive_birth_min_sources:
        return True
    for track in frame_active_tracks:
        if track.track_confidence < min_track_confidence:
            continue
        if distance_xy(track.predicted_box(det.box.frame_id), det.box) <= max(0.75, 0.4 * max(track.width, track.length, 1.0)):
            return True
    return False


def online_track(
    detections: Iterable[FusedDetection3D],
    max_distance_m: float = 2.5,
    max_age: int = 45,
    min_score: float = 0.0,
    class_min_scores: Mapping[int, float] | None = None,
    class_max_distances_m: Mapping[int, float] | None = None,
    class_max_costs: Mapping[int, float] | None = None,
    class_max_ages: Mapping[int, int] | None = None,
    class_confirmation_hits: Mapping[int, int] | None = None,
    class_duplicate_birth_distances_m: Mapping[int, float] | None = None,
    class_immediate_birth_scores: Mapping[int, float] | None = None,
    scene_class_min_scores: Mapping[Tuple[int, int], float] | None = None,
    scene_class_immediate_birth_scores: Mapping[Tuple[int, int], float] | None = None,
    max_cost: float = 1.35,
    distance_weight: float = 1.0,
    iou_weight: float = 0.35,
    yaw_weight: float = 0.08,
    score_weight: float = 0.18,
    source_weight: float = 0.12,
    position_alpha: float = 0.85,
    velocity_alpha: float = 0.70,
    association_model_path: Path | str | None = None,
    association_weight: float = 0.0,
    association_min_probability: float | None = None,
    confirmation_hits: int = 1,
    confirmation_mode: str = "immediate",
    duplicate_birth_distance_m: float = 0.0,
    duplicate_birth_iou: float = 0.0,
    immediate_birth_score: float | None = None,
    immediate_birth_min_sources: int | None = None,
    min_track_confidence: float = 0.0,
    adaptive_birth_score: float | None = None,
    adaptive_birth_min_sources: int | None = None,
) -> List[TrackBox]:
    if confirmation_hits < 1:
        raise ValueError("confirmation_hits must be at least 1")
    if confirmation_mode not in {"immediate", "confirmed_only", "backfill"}:
        raise ValueError("confirmation_mode must be immediate, confirmed_only, or backfill")
    association_scorer = AssociationScorer.load(association_model_path) if association_model_path else None
    by_scene_frame: Dict[Tuple[int, int], List[FusedDetection3D]] = {}
    for det in detections:
        class_threshold = _class_float(class_min_scores, det.box.class_id, min_score)
        threshold = _scene_class_float(
            scene_class_min_scores,
            det.box.scene_id,
            det.box.class_id,
            class_threshold,
        )
        if det.score < threshold:
            continue
        b = det.box
        by_scene_frame.setdefault((b.scene_id, b.frame_id), []).append(det)

    tracks: Dict[Tuple[int, int], List[OnlineTrack]] = {}
    next_id: Dict[Tuple[int, int], int] = {}
    outputs: List[TrackBox] = []

    for scene_id, frame_id in sorted(by_scene_frame):
        frame_dets = sorted(by_scene_frame[(scene_id, frame_id)], key=lambda d: d.score, reverse=True)
        scene_keys = {key for key in tracks if key[0] == scene_id} | {(scene_id, d.box.class_id) for d in frame_dets}
        for key in scene_keys:
            tracks.setdefault(key, [])
            next_id.setdefault(key, 1)
            for trk in tracks[key]:
                if trk.last_frame < frame_id:
                    trk.misses = frame_id - trk.last_frame
                    trk.track_confidence *= 0.92 ** max(1, trk.misses)

        for key in sorted(scene_keys):
            class_id = key[1]
            class_max_distance = _class_float(class_max_distances_m, class_id, max_distance_m)
            class_max_cost = _class_float(class_max_costs, class_id, max_cost)
            class_max_age = _class_int(class_max_ages, class_id, max_age)
            required_hits = max(1, _class_int(class_confirmation_hits, class_id, confirmation_hits))
            class_duplicate_distance = _class_float(
                class_duplicate_birth_distances_m,
                class_id,
                duplicate_birth_distance_m,
            )
            class_immediate_score = immediate_birth_score
            if class_immediate_birth_scores and class_id in class_immediate_birth_scores:
                class_immediate_score = class_immediate_birth_scores[class_id]
            if scene_class_immediate_birth_scores and (scene_id, class_id) in scene_class_immediate_birth_scores:
                class_immediate_score = scene_class_immediate_birth_scores[(scene_id, class_id)]
            class_dets = [det for det in frame_dets if (scene_id, det.box.class_id) == key]
            active_tracks = [trk for trk in tracks[key] if trk.misses <= class_max_age]
            matches = _match_detections_to_tracks(
                active_tracks,
                class_dets,
                max_distance_m=class_max_distance,
                max_cost=class_max_cost,
                distance_weight=distance_weight,
                iou_weight=iou_weight,
                yaw_weight=yaw_weight,
                score_weight=score_weight,
                source_weight=source_weight,
                association_scorer=association_scorer,
                association_weight=association_weight,
                association_min_probability=association_min_probability,
            )
            matched_dets = set()
            for track_idx, det_idx in matches:
                trk = active_tracks[track_idx]
                det = class_dets[det_idx]
                matched_dets.add(det_idx)
                box = _update_track(trk, det, position_alpha=position_alpha, velocity_alpha=velocity_alpha)
                _emit_confirmed_box(
                    trk,
                    box,
                    outputs,
                    required_hits=required_hits,
                    confirmation_mode=confirmation_mode,
                )

            for det_idx, det in enumerate(class_dets):
                if det_idx in matched_dets:
                    continue
                if _is_duplicate_birth(
                    det,
                    active_tracks,
                    duplicate_birth_distance_m=class_duplicate_distance,
                    duplicate_birth_iou=duplicate_birth_iou,
                ):
                    continue
                if not _birth_allowed(
                    det,
                    active_tracks,
                    min_track_confidence=min_track_confidence,
                    adaptive_birth_score=adaptive_birth_score,
                    adaptive_birth_min_sources=adaptive_birth_min_sources,
                ):
                    continue
                object_id = next_id[key]
                next_id[key] += 1
                confirmed = required_hits <= 1 or confirmation_mode == "immediate"
                if class_immediate_score is not None and det.score >= class_immediate_score:
                    confirmed = True
                if immediate_birth_min_sources is not None and det.source_count >= immediate_birth_min_sources:
                    confirmed = True
                trk = OnlineTrack(
                    object_id=object_id,
                    class_id=det.box.class_id,
                    scene_id=scene_id,
                    x=det.box.x,
                    y=det.box.y,
                    z=det.box.z,
                    width=det.box.width,
                    length=det.box.length,
                    height=det.box.height,
                    yaw=det.box.yaw,
                    vx=0.0,
                    vy=0.0,
                    last_frame=frame_id,
                    score=det.score,
                    source_count=det.source_count,
                    cameras=det.cameras,
                    confirmed=confirmed,
                    track_confidence=min(1.0, max(0.05, det.score + 0.04 * max(0, det.source_count - 1))),
                    class_votes={det.box.class_id: max(0.05, det.score)},
                )
                tracks[key].append(trk)
                active_tracks.append(trk)
                _emit_confirmed_box(
                    trk,
                    _new_track_output(trk, det),
                    outputs,
                    required_hits=required_hits,
                    confirmation_mode=confirmation_mode,
                )

            tracks[key] = [
                trk for trk in tracks[key]
                if trk.misses <= class_max_age and trk.track_confidence >= min_track_confidence
            ]

    outputs.sort(key=lambda box: (box.scene_id, box.frame_id, box.class_id, box.object_id))
    return outputs


def track_fused_file(
    fused_path: Path | str,
    out_path: Path | str,
    max_distance_m: float = 2.5,
    max_age: int = 45,
    min_score: float = 0.0,
    class_min_scores: Mapping[int, float] | None = None,
    class_max_distances_m: Mapping[int, float] | None = None,
    class_max_costs: Mapping[int, float] | None = None,
    class_max_ages: Mapping[int, int] | None = None,
    class_confirmation_hits: Mapping[int, int] | None = None,
    class_duplicate_birth_distances_m: Mapping[int, float] | None = None,
    class_immediate_birth_scores: Mapping[int, float] | None = None,
    adaptive_calibration: bool = False,
    adaptive_warmup_frames: int = 180,
    adaptive_strength: float = 0.18,
    adaptive_report_path: Path | str | None = None,
    decimals: int = 2,
    max_cost: float = 1.35,
    distance_weight: float = 1.0,
    iou_weight: float = 0.35,
    yaw_weight: float = 0.08,
    score_weight: float = 0.18,
    source_weight: float = 0.12,
    position_alpha: float = 0.85,
    velocity_alpha: float = 0.70,
    association_model_path: Path | str | None = None,
    association_weight: float = 0.0,
    association_min_probability: float | None = None,
    confirmation_hits: int = 1,
    confirmation_mode: str = "immediate",
    duplicate_birth_distance_m: float = 0.0,
    duplicate_birth_iou: float = 0.0,
    immediate_birth_score: float | None = None,
    immediate_birth_min_sources: int | None = None,
    min_track_confidence: float = 0.0,
    adaptive_birth_score: float | None = None,
    adaptive_birth_min_sources: int | None = None,
) -> dict:
    detections = list(read_fused_detections(fused_path))
    scene_class_min_scores = None
    scene_class_immediate_birth_scores = None
    adaptive_report = None
    if adaptive_calibration:
        adaptive_report = calibrate_scene_class_thresholds(
            detections,
            base_class_min_scores=class_min_scores or {},
            base_class_immediate_birth_scores=class_immediate_birth_scores,
            warmup_frames=adaptive_warmup_frames,
            strength=adaptive_strength,
        )
        scene_class_min_scores = adaptive_report["scene_class_min_scores"]
        scene_class_immediate_birth_scores = adaptive_report["scene_class_immediate_birth_scores"]
        if adaptive_report_path:
            report_out = Path(adaptive_report_path)
            report_out.parent.mkdir(parents=True, exist_ok=True)
            serializable = dict(adaptive_report)
            serializable["scene_class_min_scores"] = {
                f"{scene}:{class_id}": value
                for (scene, class_id), value in scene_class_min_scores.items()
            }
            serializable["scene_class_immediate_birth_scores"] = {
                f"{scene}:{class_id}": value
                for (scene, class_id), value in scene_class_immediate_birth_scores.items()
            }
            report_out.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")

    tracks = online_track(
        detections,
        max_distance_m=max_distance_m,
        max_age=max_age,
        min_score=min_score,
        class_min_scores=class_min_scores,
        class_max_distances_m=class_max_distances_m,
        class_max_costs=class_max_costs,
        class_max_ages=class_max_ages,
        class_confirmation_hits=class_confirmation_hits,
        class_duplicate_birth_distances_m=class_duplicate_birth_distances_m,
        class_immediate_birth_scores=class_immediate_birth_scores,
        scene_class_min_scores=scene_class_min_scores,
        scene_class_immediate_birth_scores=scene_class_immediate_birth_scores,
        max_cost=max_cost,
        distance_weight=distance_weight,
        iou_weight=iou_weight,
        yaw_weight=yaw_weight,
        score_weight=score_weight,
        source_weight=source_weight,
        position_alpha=position_alpha,
        velocity_alpha=velocity_alpha,
        association_model_path=association_model_path,
        association_weight=association_weight,
        association_min_probability=association_min_probability,
        confirmation_hits=confirmation_hits,
        confirmation_mode=confirmation_mode,
        duplicate_birth_distance_m=duplicate_birth_distance_m,
        duplicate_birth_iou=duplicate_birth_iou,
        immediate_birth_score=immediate_birth_score,
        immediate_birth_min_sources=immediate_birth_min_sources,
        min_track_confidence=min_track_confidence,
        adaptive_birth_score=adaptive_birth_score,
        adaptive_birth_min_sources=adaptive_birth_min_sources,
    )
    count = write_submission(tracks, out_path, decimals=decimals)
    return {
        "output": str(out_path),
        "boxes": count,
        "max_distance_m": max_distance_m,
        "max_age": max_age,
        "min_score": min_score,
        "class_min_scores": dict(sorted((class_min_scores or {}).items())),
        "class_max_distances_m": dict(sorted((class_max_distances_m or {}).items())),
        "class_max_costs": dict(sorted((class_max_costs or {}).items())),
        "class_max_ages": dict(sorted((class_max_ages or {}).items())),
        "class_confirmation_hits": dict(sorted((class_confirmation_hits or {}).items())),
        "class_duplicate_birth_distances_m": dict(sorted((class_duplicate_birth_distances_m or {}).items())),
        "class_immediate_birth_scores": dict(sorted((class_immediate_birth_scores or {}).items())),
        "adaptive_calibration": adaptive_calibration,
        "adaptive_warmup_frames": adaptive_warmup_frames,
        "adaptive_strength": adaptive_strength,
        "adaptive_report_path": str(adaptive_report_path) if adaptive_report_path else None,
        "scene_class_min_scores": {
            f"{scene}:{class_id}": value
            for (scene, class_id), value in sorted((scene_class_min_scores or {}).items())
        },
        "max_cost": max_cost,
        "distance_weight": distance_weight,
        "iou_weight": iou_weight,
        "yaw_weight": yaw_weight,
        "score_weight": score_weight,
        "source_weight": source_weight,
        "position_alpha": position_alpha,
        "velocity_alpha": velocity_alpha,
        "association_model_path": str(association_model_path) if association_model_path else None,
        "association_weight": association_weight,
        "association_min_probability": association_min_probability,
        "confirmation_hits": confirmation_hits,
        "confirmation_mode": confirmation_mode,
        "duplicate_birth_distance_m": duplicate_birth_distance_m,
        "duplicate_birth_iou": duplicate_birth_iou,
        "immediate_birth_score": immediate_birth_score,
        "immediate_birth_min_sources": immediate_birth_min_sources,
        "min_track_confidence": min_track_confidence,
        "adaptive_birth_score": adaptive_birth_score,
        "adaptive_birth_min_sources": adaptive_birth_min_sources,
    }
