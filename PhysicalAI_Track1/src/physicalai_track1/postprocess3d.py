from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from .dataset import TrackBox
from .priors import load_priors, prior_for_class
from .submission import read_submission, write_submission


def _angle_lerp(a: float, b: float, t: float) -> float:
    delta = math.atan2(math.sin(b - a), math.cos(b - a))
    return a + t * delta


def _velocity_yaw(prev: TrackBox, cur: TrackBox, fallback: float, min_speed: float) -> float:
    dt = max(1, cur.frame_id - prev.frame_id)
    vx = (cur.x - prev.x) / dt
    vy = (cur.y - prev.y) / dt
    speed = math.hypot(vx, vy)
    if speed < min_speed:
        return fallback
    return math.atan2(vx, -vy)


def _blend(value: float, target: float, strength: float) -> float:
    strength = max(0.0, min(1.0, strength))
    return (1.0 - strength) * value + strength * target


def _class_strength(values: Mapping[int, float] | None, class_id: int, default: float) -> float:
    if not values:
        return default
    return float(values.get(class_id, default))


def parse_class_strengths(value: str | None) -> dict[int, float] | None:
    if not value:
        return None
    parsed: dict[int, float] = {}
    for item in value.split(","):
        class_id, score = item.split(":", 1)
        parsed[int(class_id)] = float(score)
    return parsed


def apply_geometry_priors(
    boxes: Iterable[TrackBox],
    priors: Mapping[int, Mapping[str, float]],
    *,
    size_strength: float = 0.35,
    z_strength: float = 0.50,
    class_size_strengths: Mapping[int, float] | None = None,
    class_z_strengths: Mapping[int, float] | None = None,
    min_dimension_m: float = 0.05,
) -> list[TrackBox]:
    out: list[TrackBox] = []
    for box in boxes:
        prior = prior_for_class(priors, box.class_id)
        s_size = _class_strength(class_size_strengths, box.class_id, size_strength)
        s_z = _class_strength(class_z_strengths, box.class_id, z_strength)
        out.append(
            replace(
                box,
                z=_blend(box.z, float(prior["z"]), s_z),
                width=max(min_dimension_m, _blend(box.width, float(prior["width"]), s_size)),
                length=max(min_dimension_m, _blend(box.length, float(prior["length"]), s_size)),
                height=max(min_dimension_m, _blend(box.height, float(prior["height"]), s_size)),
            )
        )
    return out


def smooth_track_yaw(
    boxes: Iterable[TrackBox],
    *,
    yaw_alpha: float = 0.75,
    velocity_yaw_alpha: float = 0.35,
    velocity_min_speed_mpf: float = 0.03,
) -> list[TrackBox]:
    grouped: dict[tuple[int, int, int], list[TrackBox]] = defaultdict(list)
    for box in boxes:
        grouped[(box.scene_id, box.class_id, box.object_id)].append(box)

    out: list[TrackBox] = []
    yaw_alpha = max(0.0, min(1.0, yaw_alpha))
    velocity_yaw_alpha = max(0.0, min(1.0, velocity_yaw_alpha))
    for key in sorted(grouped):
        track = sorted(grouped[key], key=lambda b: b.frame_id)
        if not track:
            continue
        previous_out = track[0]
        out.append(previous_out)
        for cur in track[1:]:
            smoothed = _angle_lerp(previous_out.yaw, cur.yaw, yaw_alpha)
            motion_yaw = _velocity_yaw(previous_out, cur, smoothed, velocity_min_speed_mpf)
            final_yaw = _angle_lerp(smoothed, motion_yaw, velocity_yaw_alpha)
            updated = replace(cur, yaw=final_yaw)
            out.append(updated)
            previous_out = updated
    out.sort(key=lambda b: (b.scene_id, b.frame_id, b.class_id, b.object_id))
    return out


def interpolate_short_gaps(
    boxes: Iterable[TrackBox],
    *,
    max_gap_frames: int = 0,
    min_track_length: int = 3,
    max_step_distance_m: float = 0.75,
) -> tuple[list[TrackBox], int]:
    if max_gap_frames <= 1:
        return sorted(boxes, key=lambda b: (b.scene_id, b.frame_id, b.class_id, b.object_id)), 0

    grouped: dict[tuple[int, int, int], list[TrackBox]] = defaultdict(list)
    for box in boxes:
        grouped[(box.scene_id, box.class_id, box.object_id)].append(box)

    out: list[TrackBox] = []
    added = 0
    for key in sorted(grouped):
        track = sorted(grouped[key], key=lambda b: b.frame_id)
        out.extend(track)
        if len(track) < min_track_length:
            continue
        for prev, cur in zip(track, track[1:]):
            gap = cur.frame_id - prev.frame_id
            if gap <= 1 or gap > max_gap_frames:
                continue
            step_distance = math.hypot(cur.x - prev.x, cur.y - prev.y) / gap
            if step_distance > max_step_distance_m:
                continue
            for frame_id in range(prev.frame_id + 1, cur.frame_id):
                t = (frame_id - prev.frame_id) / gap
                out.append(
                    TrackBox(
                        scene_id=prev.scene_id,
                        class_id=prev.class_id,
                        object_id=prev.object_id,
                        frame_id=frame_id,
                        x=(1.0 - t) * prev.x + t * cur.x,
                        y=(1.0 - t) * prev.y + t * cur.y,
                        z=(1.0 - t) * prev.z + t * cur.z,
                        width=(1.0 - t) * prev.width + t * cur.width,
                        length=(1.0 - t) * prev.length + t * cur.length,
                        height=(1.0 - t) * prev.height + t * cur.height,
                        yaw=_angle_lerp(prev.yaw, cur.yaw, t),
                        score=min(prev.score, cur.score) * 0.95,
                    )
                )
                added += 1
    out.sort(key=lambda b: (b.scene_id, b.frame_id, b.class_id, b.object_id))
    return out, added


def postprocess_submission_file(
    input_path: Path | str,
    out_path: Path | str,
    *,
    priors_path: Path | str | None = None,
    size_strength: float = 0.35,
    z_strength: float = 0.50,
    yaw_alpha: float = 0.75,
    velocity_yaw_alpha: float = 0.35,
    velocity_min_speed_mpf: float = 0.03,
    interpolate_max_gap_frames: int = 0,
    interpolate_min_track_length: int = 3,
    interpolate_max_step_distance_m: float = 0.75,
    class_size_strengths: Mapping[int, float] | None = None,
    class_z_strengths: Mapping[int, float] | None = None,
    decimals: int = 2,
) -> dict:
    boxes = read_submission(input_path)
    priors = load_priors(priors_path)
    boxes = apply_geometry_priors(
        boxes,
        priors,
        size_strength=size_strength,
        z_strength=z_strength,
        class_size_strengths=class_size_strengths,
        class_z_strengths=class_z_strengths,
    )
    boxes = smooth_track_yaw(
        boxes,
        yaw_alpha=yaw_alpha,
        velocity_yaw_alpha=velocity_yaw_alpha,
        velocity_min_speed_mpf=velocity_min_speed_mpf,
    )
    boxes, interpolated = interpolate_short_gaps(
        boxes,
        max_gap_frames=interpolate_max_gap_frames,
        min_track_length=interpolate_min_track_length,
        max_step_distance_m=interpolate_max_step_distance_m,
    )
    written = write_submission(boxes, out_path, decimals=decimals)
    stats = {
        "input": str(input_path),
        "output": str(out_path),
        "input_boxes": len(read_submission(input_path)),
        "output_boxes": written,
        "interpolated_boxes": interpolated,
        "params": {
            "priors_path": str(priors_path) if priors_path else None,
            "size_strength": size_strength,
            "z_strength": z_strength,
            "yaw_alpha": yaw_alpha,
            "velocity_yaw_alpha": velocity_yaw_alpha,
            "velocity_min_speed_mpf": velocity_min_speed_mpf,
            "interpolate_max_gap_frames": interpolate_max_gap_frames,
            "interpolate_min_track_length": interpolate_min_track_length,
            "interpolate_max_step_distance_m": interpolate_max_step_distance_m,
            "class_size_strengths": dict(class_size_strengths or {}),
            "class_z_strengths": dict(class_z_strengths or {}),
            "decimals": decimals,
        },
    }
    Path(str(out_path) + ".json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats
