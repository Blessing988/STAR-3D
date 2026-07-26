from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .detections import Detection2D, read_detections, write_detections
from .detector_adapters import _iou2d, classwise_nms


@dataclass
class TrackletState:
    track_id: int
    last: Detection2D
    velocity: tuple[float, float, float, float]
    hits: int = 1


def _bbox(det: Detection2D) -> tuple[float, float, float, float]:
    return det.x1, det.y1, det.x2, det.y2


def _make_detection_like(
    reference: Detection2D,
    frame_id: int,
    score: float,
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
) -> Detection2D:
    x1 = min(max(float(bbox[0]), 0.0), float(frame_width))
    y1 = min(max(float(bbox[1]), 0.0), float(frame_height))
    x2 = min(max(float(bbox[2]), 0.0), float(frame_width))
    y2 = min(max(float(bbox[3]), 0.0), float(frame_height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Detection2D(
        scene_name=reference.scene_name,
        camera_id=reference.camera_id,
        frame_id=frame_id,
        class_id=reference.class_id,
        score=max(0.0, min(1.0, float(score))),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        oracle_object_id=reference.oracle_object_id,
    )


def _predict(track: TrackletState, frame_id: int, frame_width: int, frame_height: int) -> Detection2D:
    dt = max(0, int(frame_id) - int(track.last.frame_id))
    bbox = tuple(value + delta * dt for value, delta in zip(_bbox(track.last), track.velocity))
    return _make_detection_like(track.last, frame_id, track.last.score, bbox, frame_width, frame_height)


def _center_distance(a: Detection2D, b: Detection2D) -> float:
    ax = (a.x1 + a.x2) * 0.5
    ay = (a.y1 + a.y2) * 0.5
    bx = (b.x1 + b.x2) * 0.5
    by = (b.y1 + b.y2) * 0.5
    return math.hypot(ax - bx, ay - by)


def _center_gate_scale(a: Detection2D, b: Detection2D) -> float:
    aw = max(1.0, a.x2 - a.x1)
    ah = max(1.0, a.y2 - a.y1)
    bw = max(1.0, b.x2 - b.x1)
    bh = max(1.0, b.y2 - b.y1)
    return max(1.0, 0.5 * (math.hypot(aw, ah) + math.hypot(bw, bh)))


def _smooth_detection(
    det: Detection2D,
    predicted: Detection2D,
    alpha: float,
    frame_width: int,
    frame_height: int,
) -> Detection2D:
    alpha = max(0.0, min(1.0, alpha))
    bbox = tuple(alpha * cur + (1.0 - alpha) * pred for cur, pred in zip(_bbox(det), _bbox(predicted)))
    return _make_detection_like(det, det.frame_id, det.score, bbox, frame_width, frame_height)


def _interpolate_gap(
    previous: Detection2D,
    current: Detection2D,
    frame_step: int,
    bridge_score_decay: float,
    frame_width: int,
    frame_height: int,
) -> list[Detection2D]:
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    gap = current.frame_id - previous.frame_id
    if gap <= frame_step:
        return []
    bridge_frames = list(range(previous.frame_id + frame_step, current.frame_id, frame_step))
    if not bridge_frames:
        return []

    out: list[Detection2D] = []
    prev_bbox = _bbox(previous)
    curr_bbox = _bbox(current)
    base_score = min(previous.score, current.score)
    for frame_id in bridge_frames:
        ratio = (frame_id - previous.frame_id) / gap
        bbox = tuple((1.0 - ratio) * a + ratio * b for a, b in zip(prev_bbox, curr_bbox))
        score = base_score * (bridge_score_decay ** min(len(bridge_frames), 8))
        out.append(_make_detection_like(current, frame_id, score, bbox, frame_width, frame_height))
    return out


def _match_tracks(
    tracks: Sequence[TrackletState],
    detections: Sequence[Detection2D],
    frame_id: int,
    min_iou: float,
    center_gate: float,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, Detection2D]]:
    candidates: list[tuple[float, float, int, int, Detection2D]] = []
    for track_idx, track in enumerate(tracks):
        predicted = _predict(track, frame_id, frame_width, frame_height)
        for det_idx, det in enumerate(detections):
            iou = _iou2d(predicted, det)
            center_norm = _center_distance(predicted, det) / _center_gate_scale(predicted, det)
            if iou < min_iou and center_norm > center_gate:
                continue
            cost = (1.0 - iou) + 0.35 * center_norm - 0.05 * det.score
            candidates.append((cost, -det.score, track_idx, det_idx, predicted))

    assigned_tracks: set[int] = set()
    assigned_detections: set[int] = set()
    matches: list[tuple[int, int, Detection2D]] = []
    for _cost, _neg_score, track_idx, det_idx, predicted in sorted(candidates):
        if track_idx in assigned_tracks or det_idx in assigned_detections:
            continue
        assigned_tracks.add(track_idx)
        assigned_detections.add(det_idx)
        matches.append((track_idx, det_idx, predicted))
    return matches


def stabilize_detection_list(
    detections: Iterable[Detection2D],
    *,
    min_iou: float = 0.20,
    center_gate: float = 0.85,
    max_gap_frames: int = 45,
    smoothing_alpha: float = 0.80,
    velocity_alpha: float = 0.55,
    min_hits_for_smoothing: int = 2,
    bridge_max_gap_frames: int = 0,
    bridge_min_score: float = 0.35,
    bridge_score_decay: float = 0.85,
    frame_step: int = 1,
    final_nms_iou: float = 0.80,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> tuple[list[Detection2D], dict]:
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    if bridge_max_gap_frames < 0:
        raise ValueError("bridge_max_gap_frames must be non-negative")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    groups: dict[tuple[str, str, int], list[Detection2D]] = defaultdict(list)
    input_count = 0
    for det in detections:
        groups[(det.scene_name, det.camera_id, det.class_id)].append(det)
        input_count += 1

    output: list[Detection2D] = []
    smoothed_count = 0
    bridged_count = 0
    track_count = 0

    for _group_key, group_dets in sorted(groups.items()):
        by_frame: dict[int, list[Detection2D]] = defaultdict(list)
        for det in group_dets:
            by_frame[det.frame_id].append(det)

        active: list[TrackletState] = []
        for frame_id in sorted(by_frame):
            frame_dets = sorted(by_frame[frame_id], key=lambda item: item.score, reverse=True)
            active = [track for track in active if frame_id - track.last.frame_id <= max_gap_frames]
            matches = _match_tracks(
                active,
                frame_dets,
                frame_id,
                min_iou=min_iou,
                center_gate=center_gate,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            matched_tracks = {track_idx for track_idx, _det_idx, _predicted in matches}
            matched_dets = {det_idx for _track_idx, det_idx, _predicted in matches}

            for track_idx, det_idx, predicted in matches:
                track = active[track_idx]
                det = frame_dets[det_idx]
                previous = track.last
                current = det
                if track.hits >= min_hits_for_smoothing:
                    current = _smooth_detection(det, predicted, smoothing_alpha, frame_width, frame_height)
                    smoothed_count += 1

                gap = current.frame_id - previous.frame_id
                if (
                    bridge_max_gap_frames > 0
                    and frame_step < gap <= bridge_max_gap_frames
                    and min(previous.score, current.score) >= bridge_min_score
                ):
                    bridges = _interpolate_gap(
                        previous,
                        current,
                        frame_step=frame_step,
                        bridge_score_decay=bridge_score_decay,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    output.extend(bridges)
                    bridged_count += len(bridges)

                dt = max(1, current.frame_id - previous.frame_id)
                observed_velocity = tuple((cur - prev) / dt for cur, prev in zip(_bbox(current), _bbox(previous)))
                track.velocity = tuple(
                    velocity_alpha * obs + (1.0 - velocity_alpha) * old
                    for obs, old in zip(observed_velocity, track.velocity)
                )
                track.last = current
                track.hits += 1
                output.append(current)

            for det_idx, det in enumerate(frame_dets):
                if det_idx in matched_dets:
                    continue
                track_count += 1
                active.append(TrackletState(track_count, det, (0.0, 0.0, 0.0, 0.0)))
                output.append(det)

            if matched_tracks:
                # Keep deterministic order and drop any stale tracks that were not already filtered.
                active = sorted(active, key=lambda track: track.track_id)

    output.sort(key=lambda det: (det.scene_name, det.camera_id, det.frame_id, det.class_id, -det.score))
    before_nms = len(output)
    if final_nms_iou > 0.0:
        output = classwise_nms(output, final_nms_iou)
        output.sort(key=lambda det: (det.scene_name, det.camera_id, det.frame_id, det.class_id, -det.score))

    stats = {
        "input_detections": input_count,
        "output_detections": len(output),
        "groups": len(groups),
        "tracks_created": track_count,
        "smoothed_detections": smoothed_count,
        "bridged_detections": bridged_count,
        "removed_by_final_nms": before_nms - len(output),
        "params": {
            "min_iou": min_iou,
            "center_gate": center_gate,
            "max_gap_frames": max_gap_frames,
            "smoothing_alpha": smoothing_alpha,
            "velocity_alpha": velocity_alpha,
            "min_hits_for_smoothing": min_hits_for_smoothing,
            "bridge_max_gap_frames": bridge_max_gap_frames,
            "bridge_min_score": bridge_min_score,
            "bridge_score_decay": bridge_score_decay,
            "frame_step": frame_step,
            "final_nms_iou": final_nms_iou,
            "frame_width": frame_width,
            "frame_height": frame_height,
        },
    }
    return output, stats


def stabilize_detections_file(
    input_path: Path | str,
    out_path: Path | str,
    **kwargs,
) -> dict:
    detections, stats = stabilize_detection_list(read_detections(input_path), **kwargs)
    stats["output"] = str(out_path)
    stats["input"] = str(input_path)
    stats["written"] = write_detections(detections, out_path)
    Path(str(out_path) + ".json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats
