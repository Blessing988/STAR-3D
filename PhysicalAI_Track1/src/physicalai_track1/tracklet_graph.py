from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .association_model import AssociationScorer
from .dataset import TrackBox
from .geometry import angle_distance, box3d_iou, distance_xy
from .submission import read_submission, write_submission


@dataclass(frozen=True)
class Tracklet:
    index: int
    scene_id: int
    class_id: int
    old_object_id: int
    boxes: tuple[TrackBox, ...]
    start_frame: int
    end_frame: int
    first_box: TrackBox
    last_box: TrackBox
    vx: float
    vy: float

    @property
    def hits(self) -> int:
        return len(self.boxes)


def _estimate_velocity(boxes: tuple[TrackBox, ...], tail: bool, window: int = 5) -> tuple[float, float]:
    if len(boxes) < 2:
        return 0.0, 0.0
    subset = boxes[-window:] if tail else boxes[:window]
    if len(subset) < 2:
        subset = boxes
    first = subset[0]
    last = subset[-1]
    dt = max(1, last.frame_id - first.frame_id)
    return (last.x - first.x) / dt, (last.y - first.y) / dt


def _yaw_from_velocity(vx: float, vy: float, fallback: float) -> float:
    if math.hypot(vx, vy) < 0.02:
        return fallback
    return math.atan2(vx, -vy)


def _make_tracklets(boxes: Iterable[TrackBox], min_tracklet_hits: int) -> tuple[list[Tracklet], list[TrackBox]]:
    grouped: dict[tuple[int, int, int], list[TrackBox]] = {}
    passthrough: list[TrackBox] = []
    for box in boxes:
        grouped.setdefault((box.scene_id, box.class_id, box.object_id), []).append(box)

    tracklets: list[Tracklet] = []
    for key in sorted(grouped):
        trajectory = tuple(sorted(grouped[key], key=lambda item: item.frame_id))
        if len(trajectory) < max(1, min_tracklet_hits):
            passthrough.extend(trajectory)
            continue
        vx, vy = _estimate_velocity(trajectory, tail=True)
        tracklets.append(
            Tracklet(
                index=len(tracklets),
                scene_id=key[0],
                class_id=key[1],
                old_object_id=key[2],
                boxes=trajectory,
                start_frame=trajectory[0].frame_id,
                end_frame=trajectory[-1].frame_id,
                first_box=trajectory[0],
                last_box=trajectory[-1],
                vx=vx,
                vy=vy,
            )
        )
    return tracklets, passthrough


def _predict_end(tracklet: Tracklet, frame_id: int) -> TrackBox:
    dt = max(0, frame_id - tracklet.end_frame)
    yaw = _yaw_from_velocity(tracklet.vx, tracklet.vy, tracklet.last_box.yaw)
    return TrackBox(
        scene_id=tracklet.scene_id,
        class_id=tracklet.class_id,
        object_id=tracklet.old_object_id,
        frame_id=frame_id,
        x=tracklet.last_box.x + tracklet.vx * dt,
        y=tracklet.last_box.y + tracklet.vy * dt,
        z=tracklet.last_box.z,
        width=tracklet.last_box.width,
        length=tracklet.last_box.length,
        height=tracklet.last_box.height,
        yaw=yaw,
        score=tracklet.last_box.score,
    )


def _safe_log_ratio(a: float, b: float) -> float:
    return abs(math.log(max(a, 1e-6) / max(b, 1e-6)))


def _size_cost(a: TrackBox, b: TrackBox) -> float:
    return (
        _safe_log_ratio(a.width, b.width)
        + _safe_log_ratio(a.length, b.length)
        + _safe_log_ratio(a.height, b.height)
    ) / 3.0


def _velocity_cost(a: Tracklet, b: Tracklet, dt: int) -> float:
    required_vx = (b.first_box.x - a.last_box.x) / max(1, dt)
    required_vy = (b.first_box.y - a.last_box.y) / max(1, dt)
    b_vx, b_vy = _estimate_velocity(b.boxes, tail=False)
    scale = max(0.02, math.hypot(required_vx, required_vy), math.hypot(a.vx, a.vy), math.hypot(b_vx, b_vy))
    tail_delta = math.hypot(required_vx - a.vx, required_vy - a.vy) / scale
    head_delta = math.hypot(required_vx - b_vx, required_vy - b_vy) / scale
    return min(3.0, 0.5 * (tail_delta + head_delta))


def _transition_cost(
    a: Tracklet,
    b: Tracklet,
    *,
    max_gap_frames: int,
    max_distance_m: float,
    frame_step: int,
    distance_weight: float,
    velocity_weight: float,
    iou_weight: float,
    yaw_weight: float,
    size_weight: float,
    association_scorer: AssociationScorer | None,
    association_weight: float,
    association_min_probability: float | None,
) -> float:
    if a.scene_id != b.scene_id or a.class_id != b.class_id:
        return float("inf")
    gap = b.start_frame - a.end_frame
    if gap <= 0 or gap > max_gap_frames:
        return float("inf")

    predicted = _predict_end(a, b.start_frame)
    dist = distance_xy(predicted, b.first_box)
    step_scale = math.sqrt(max(1.0, gap / max(1, frame_step)))
    gate = max(max_distance_m, max_distance_m * min(3.0, step_scale))
    gate += 0.25 * max(a.last_box.width, a.last_box.length, b.first_box.width, b.first_box.length)
    if dist > gate:
        return float("inf")

    yaw_cost = angle_distance(predicted.yaw, b.first_box.yaw) / math.pi
    iou_cost = 1.0 - box3d_iou(predicted, b.first_box)
    size = _size_cost(a.last_box, b.first_box)
    velocity = _velocity_cost(a, b, gap)
    cost = (
        distance_weight * (dist / max(1e-6, gate))
        + velocity_weight * velocity
        + iou_weight * iou_cost
        + yaw_weight * yaw_cost
        + size_weight * size
    )
    if association_scorer is not None and association_weight > 0.0:
        probability = association_scorer.predict_proba(predicted, b.first_box)
        if association_min_probability is not None and probability < association_min_probability:
            return float("inf")
        weight = max(0.0, min(1.0, association_weight))
        cost = (1.0 - weight) * cost + weight * (1.0 - probability)
    return cost


def _creates_cycle(outgoing: dict[int, int], src: int, dst: int) -> bool:
    current = dst
    seen: set[int] = set()
    while current in outgoing:
        if current == src:
            return True
        if current in seen:
            return True
        seen.add(current)
        current = outgoing[current]
    return current == src


def _select_edges(
    tracklets: list[Tracklet],
    *,
    max_gap_frames: int,
    max_distance_m: float,
    max_cost: float,
    frame_step: int,
    distance_weight: float,
    velocity_weight: float,
    iou_weight: float,
    yaw_weight: float,
    size_weight: float,
    association_scorer: AssociationScorer | None,
    association_weight: float,
    association_min_probability: float | None,
) -> tuple[dict[int, int], list[dict[str, float | int]]]:
    by_group: dict[tuple[int, int], list[Tracklet]] = {}
    for tracklet in tracklets:
        by_group.setdefault((tracklet.scene_id, tracklet.class_id), []).append(tracklet)

    candidates: list[tuple[float, int, int]] = []
    for group in by_group.values():
        ordered = sorted(group, key=lambda item: (item.start_frame, item.end_frame, item.old_object_id))
        for a in ordered:
            for b in ordered:
                if b.start_frame <= a.end_frame:
                    continue
                if b.start_frame - a.end_frame > max_gap_frames:
                    continue
                cost = _transition_cost(
                    a,
                    b,
                    max_gap_frames=max_gap_frames,
                    max_distance_m=max_distance_m,
                    frame_step=frame_step,
                    distance_weight=distance_weight,
                    velocity_weight=velocity_weight,
                    iou_weight=iou_weight,
                    yaw_weight=yaw_weight,
                    size_weight=size_weight,
                    association_scorer=association_scorer,
                    association_weight=association_weight,
                    association_min_probability=association_min_probability,
                )
                if cost <= max_cost:
                    candidates.append((cost, a.index, b.index))

    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    accepted: list[dict[str, float | int]] = []
    for cost, src, dst in sorted(candidates):
        if src in outgoing or dst in incoming:
            continue
        if _creates_cycle(outgoing, src, dst):
            continue
        outgoing[src] = dst
        incoming[dst] = src
        accepted.append({"src": src, "dst": dst, "cost": cost})
    return outgoing, accepted


def _root_for(index: int, incoming: dict[int, int]) -> int:
    current = index
    seen: set[int] = set()
    while current in incoming and current not in seen:
        seen.add(current)
        current = incoming[current]
    return current


def relink_tracklets(
    boxes: Iterable[TrackBox],
    *,
    max_gap_frames: int = 90,
    max_distance_m: float = 2.0,
    max_cost: float = 1.25,
    frame_step: int = 1,
    min_tracklet_hits: int = 2,
    distance_weight: float = 1.0,
    velocity_weight: float = 0.20,
    iou_weight: float = 0.25,
    yaw_weight: float = 0.08,
    size_weight: float = 0.25,
    association_model_path: Path | str | None = None,
    association_weight: float = 0.0,
    association_min_probability: float | None = None,
) -> tuple[list[TrackBox], dict]:
    all_boxes = list(boxes)
    association_scorer = AssociationScorer.load(association_model_path) if association_model_path else None
    tracklets, passthrough = _make_tracklets(all_boxes, min_tracklet_hits=min_tracklet_hits)
    outgoing, accepted = _select_edges(
        tracklets,
        max_gap_frames=max_gap_frames,
        max_distance_m=max_distance_m,
        max_cost=max_cost,
        frame_step=frame_step,
        distance_weight=distance_weight,
        velocity_weight=velocity_weight,
        iou_weight=iou_weight,
        yaw_weight=yaw_weight,
        size_weight=size_weight,
        association_scorer=association_scorer,
        association_weight=association_weight,
        association_min_probability=association_min_probability,
    )
    incoming = {dst: src for src, dst in outgoing.items()}

    chains: dict[tuple[int, int, int], list[Tracklet]] = {}
    for tracklet in tracklets:
        root = _root_for(tracklet.index, incoming)
        root_tracklet = tracklets[root]
        chains.setdefault((root_tracklet.scene_id, root_tracklet.class_id, root), []).append(tracklet)

    chain_ids: dict[tuple[int, int, int], int] = {}
    next_ids: dict[tuple[int, int], int] = {}
    for chain_key, chain_tracklets in sorted(
        chains.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            min(t.start_frame for t in item[1]),
            min(t.old_object_id for t in item[1]),
        ),
    ):
        scene_id, class_id, _root = chain_key
        next_ids.setdefault((scene_id, class_id), 1)
        chain_ids[chain_key] = next_ids[(scene_id, class_id)]
        next_ids[(scene_id, class_id)] += 1

    relinked: list[TrackBox] = []
    for chain_key, chain_tracklets in chains.items():
        new_id = chain_ids[chain_key]
        for tracklet in chain_tracklets:
            for box in tracklet.boxes:
                relinked.append(
                    TrackBox(
                        scene_id=box.scene_id,
                        class_id=box.class_id,
                        object_id=new_id,
                        frame_id=box.frame_id,
                        x=box.x,
                        y=box.y,
                        z=box.z,
                        width=box.width,
                        length=box.length,
                        height=box.height,
                        yaw=box.yaw,
                        score=box.score,
                    )
                )

    for box in passthrough:
        key = (box.scene_id, box.class_id)
        next_ids.setdefault(key, 1)
        new_id = next_ids[key]
        next_ids[key] += 1
        relinked.append(
            TrackBox(
                scene_id=box.scene_id,
                class_id=box.class_id,
                object_id=new_id,
                frame_id=box.frame_id,
                x=box.x,
                y=box.y,
                z=box.z,
                width=box.width,
                length=box.length,
                height=box.height,
                yaw=box.yaw,
                score=box.score,
            )
        )

    relinked.sort(key=lambda box: (box.scene_id, box.frame_id, box.class_id, box.object_id))
    stats = {
        "input_boxes": len(all_boxes),
        "output_boxes": len(relinked),
        "tracklets": len(tracklets),
        "passthrough_boxes": len(passthrough),
        "accepted_edges": len(accepted),
        "objects_before": len({(b.scene_id, b.class_id, b.object_id) for b in all_boxes}),
        "objects_after": len({(b.scene_id, b.class_id, b.object_id) for b in relinked}),
        "params": {
            "max_gap_frames": max_gap_frames,
            "max_distance_m": max_distance_m,
            "max_cost": max_cost,
            "frame_step": frame_step,
            "min_tracklet_hits": min_tracklet_hits,
            "distance_weight": distance_weight,
            "velocity_weight": velocity_weight,
            "iou_weight": iou_weight,
            "yaw_weight": yaw_weight,
            "size_weight": size_weight,
            "association_model_path": str(association_model_path) if association_model_path else None,
            "association_weight": association_weight,
            "association_min_probability": association_min_probability,
        },
        "accepted_edge_preview": accepted[:200],
    }
    return relinked, stats


def relink_tracklets_file(
    input_path: Path | str,
    out_path: Path | str,
    *,
    decimals: int = 2,
    **kwargs,
) -> dict:
    relinked, stats = relink_tracklets(read_submission(input_path), **kwargs)
    stats["input"] = str(input_path)
    stats["output"] = str(out_path)
    stats["decimals"] = decimals
    stats["written"] = write_submission(relinked, out_path, decimals=decimals)
    Path(str(out_path) + ".json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats
