from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from physicalai_track1.calibration import CameraCalibration, load_scene_cameras
from physicalai_track1.dataset import TrackBox, scene_name_to_id, year_dir
from physicalai_track1.detections import Detection2D, read_detections
from physicalai_track1.fusion import FusedDetection3D, read_fused_detections, write_fused_detections
from physicalai_track1.geometry import circular_mean
from physicalai_track1.geometry_residual import _depth_npz_file, _load_depth_npz, _region_mask_for_class
from physicalai_track1.priors import load_priors, prior_for_class


@dataclass(frozen=True)
class DepthObjectCandidate:
    scene_id: int
    scene_name: str
    frame_id: int
    class_id: int
    camera_id: str
    x: float
    y: float
    z: float
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    pixel_count: int
    xy_spread_m: float
    depth_spread_m: float


@dataclass(frozen=True)
class DepthObjectCluster:
    box: TrackBox
    cameras: tuple[str, ...]
    candidates: tuple[DepthObjectCandidate, ...]
    source_count: int
    spread_m: float


@dataclass
class MergeStats:
    base_rows: int = 0
    frames_with_depth_clusters: int = 0
    depth_candidates: int = 0
    depth_clusters: int = 0
    corrected_rows: int = 0
    mean_xy_shift_sum: float = 0.0
    max_xy_shift: float = 0.0

    def as_dict(self) -> dict:
        return {
            "base_rows": self.base_rows,
            "frames_with_depth_clusters": self.frames_with_depth_clusters,
            "depth_candidates": self.depth_candidates,
            "depth_clusters": self.depth_clusters,
            "corrected_rows": self.corrected_rows,
            "mean_xy_shift": self.mean_xy_shift_sum / max(1, self.corrected_rows),
            "max_xy_shift": self.max_xy_shift,
        }


def parse_class_float_map(text: str | None) -> dict[int, float]:
    out: dict[int, float] = {}
    if not text:
        return out
    for item in text.replace(";", ",").split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        out[int(key)] = float(value)
    return out


def class_value(values: Mapping[int, float], class_id: int, default: float) -> float:
    return float(values.get(class_id, default))


def camera_name(camera_id: str) -> str:
    if str(camera_id).startswith("Camera_"):
        return str(camera_id)
    try:
        return f"Camera_{int(camera_id):04d}"
    except ValueError:
        return str(camera_id)


class DepthCache:
    def __init__(self, depth_root: Path, split: str, max_items: int = 64):
        self.depth_root = Path(depth_root)
        self.split = split
        self.max_items = max_items
        self.cache: OrderedDict[tuple[str, str, int], np.ndarray | None] = OrderedDict()

    def get(self, scene_name: str, camera_id: str, frame_id: int) -> np.ndarray | None:
        key = (scene_name, camera_name(camera_id), int(frame_id))
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        path = _depth_npz_file(self.depth_root, self.split, scene_name, key[1], frame_id)
        value = _load_depth_npz(path) if path.exists() else None
        self.cache[key] = value
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


def backproject_many(camera: CameraCalibration, u_img: np.ndarray, v_img: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    k = np.asarray(camera.intrinsic_matrix, dtype=np.float64)
    e = np.asarray(camera.extrinsic_matrix, dtype=np.float64)
    inv_k = np.linalg.inv(k)
    r = e[:, :3]
    t = e[:, 3]
    inv_r = np.linalg.inv(r)
    pixels = np.stack([u_img, v_img, np.ones_like(u_img)], axis=0).astype(np.float64)
    rays = inv_k @ pixels
    camera_points = rays * depth_m.astype(np.float64)[None, :]
    world = inv_r @ (camera_points - t[:, None])
    return world.T.astype(np.float32)


def box_corners_world(box: TrackBox) -> list[tuple[float, float, float]]:
    hw = 0.5 * float(box.width)
    hl = 0.5 * float(box.length)
    hh = 0.5 * float(box.height)
    c = math.cos(float(box.yaw))
    s = math.sin(float(box.yaw))
    corners: list[tuple[float, float, float]] = []
    for lx in (-hw, hw):
        for ly in (-hl, hl):
            wx = box.x + c * lx - s * ly
            wy = box.y + s * lx + c * ly
            for lz in (-hh, hh):
                corners.append((wx, wy, box.z + lz))
    return corners


def projected_bbox(camera: CameraCalibration, box: TrackBox) -> tuple[float, float, float, float] | None:
    points: list[tuple[float, float]] = []
    for x, y, z in box_corners_world(box):
        uv = camera.world_to_image(x, y, z)
        if uv is None:
            continue
        u, v = uv
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        points.append((u, v))
    if len(points) < 4:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou_2d(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def reprojection_loss(
    box: TrackBox,
    support: Iterable[DepthObjectCandidate],
    cameras: Mapping[str, Mapping[str, CameraCalibration]],
) -> float:
    losses: list[float] = []
    weights: list[float] = []
    for cand in support:
        camera = cameras.get(cand.scene_name, {}).get(cand.camera_id)
        if camera is None:
            continue
        pred = projected_bbox(camera, box)
        if pred is None:
            losses.append(4.0)
            weights.append(max(0.05, cand.score))
            continue
        target = (cand.x1, cand.y1, cand.x2, cand.y2)
        iou = bbox_iou_2d(pred, target)
        pcx = 0.5 * (pred[0] + pred[2])
        pcy = 0.5 * (pred[1] + pred[3])
        tcx = 0.5 * (target[0] + target[2])
        tcy = 0.5 * (target[1] + target[3])
        tw = max(1.0, target[2] - target[0])
        th = max(1.0, target[3] - target[1])
        center = math.hypot((pcx - tcx) / tw, (pcy - tcy) / th)
        pw = max(1.0, pred[2] - pred[0])
        ph = max(1.0, pred[3] - pred[1])
        size = abs(math.log(pw / tw)) + abs(math.log(ph / th))
        losses.append((1.0 - iou) + 0.20 * center + 0.08 * size)
        weights.append(max(0.05, cand.score))
    if not losses:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(losses, dtype=np.float64)
    return float((v * w).sum() / max(1e-9, w.sum()))


def replace_box_xy(box: TrackBox, x: float, y: float) -> TrackBox:
    return TrackBox(
        scene_id=box.scene_id,
        class_id=box.class_id,
        object_id=box.object_id,
        frame_id=box.frame_id,
        x=float(x),
        y=float(y),
        z=box.z,
        width=box.width,
        length=box.length,
        height=box.height,
        yaw=box.yaw,
        score=box.score,
    )


def detection_depth_candidate(
    det: Detection2D,
    camera: CameraCalibration,
    depth_map: np.ndarray,
    score: float,
    min_pixels: int,
    max_pixels: int,
    depth_band_m: float,
) -> DepthObjectCandidate | None:
    depth_h, depth_w = depth_map.shape[:2]
    sx = depth_w / max(1.0, float(camera.frame_width))
    sy = depth_h / max(1.0, float(camera.frame_height))
    x1 = max(0, min(depth_w - 1, int(math.floor(det.x1 * sx))))
    x2 = max(x1 + 1, min(depth_w, int(math.ceil(det.x2 * sx))))
    y1 = max(0, min(depth_h - 1, int(math.floor(det.y1 * sy))))
    y2 = max(y1 + 1, min(depth_h, int(math.ceil(det.y2 * sy))))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = np.asarray(depth_map[y1:y2, x1:x2], dtype=np.float32)
    valid = np.isfinite(crop) & (crop > 0)
    if int(valid.sum()) < min_pixels:
        return None

    region = _region_mask_for_class(det.class_id, crop.shape)
    selected = valid & region
    if int(selected.sum()) < min_pixels:
        selected = valid
    values = crop[selected]
    if values.size < min_pixels:
        return None

    if det.class_id in {1, 6}:
        percentile = 35.0
    elif det.class_id == 0:
        percentile = 45.0
    else:
        percentile = 50.0
    ref_depth = float(np.percentile(values, percentile))
    band = max(depth_band_m, 0.08 * ref_depth)
    selected = selected & (np.abs(crop - ref_depth) <= band)
    if int(selected.sum()) < min_pixels:
        diff = np.abs(crop - ref_depth)
        valid_y, valid_x = np.nonzero(valid)
        order = np.argsort(diff[valid_y, valid_x])[: max(min_pixels, min(max_pixels, len(valid_y)))]
        ys = valid_y[order]
        xs = valid_x[order]
    else:
        ys, xs = np.nonzero(selected)
        if len(xs) > max_pixels:
            stride = max(1, len(xs) // max_pixels)
            xs = xs[::stride][:max_pixels]
            ys = ys[::stride][:max_pixels]
    if len(xs) < min_pixels:
        return None

    depth_values = crop[ys, xs]
    u_depth = xs.astype(np.float32) + float(x1)
    v_depth = ys.astype(np.float32) + float(y1)
    u_img = u_depth / max(1e-6, sx)
    v_img = v_depth / max(1e-6, sy)
    points = backproject_many(camera, u_img, v_img, depth_values)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < min_pixels:
        return None
    xy = points[:, :2]
    center_xy = np.median(xy, axis=0)
    dists = np.linalg.norm(xy - center_xy[None, :], axis=1)
    keep = dists <= max(0.25, float(np.percentile(dists, 80)))
    if int(keep.sum()) >= min_pixels:
        points = points[keep]
        dists = dists[keep]
    center = np.median(points, axis=0)
    spread_xy = float(np.median(dists)) if len(dists) else 0.0
    spread_depth = float(np.percentile(depth_values, 90) - np.percentile(depth_values, 10))
    if not np.isfinite(center).all():
        return None
    return DepthObjectCandidate(
        scene_id=scene_name_to_id(det.scene_name),
        scene_name=det.scene_name,
        frame_id=det.frame_id,
        class_id=det.class_id,
        camera_id=camera_name(det.camera_id),
        x=float(center[0]),
        y=float(center[1]),
        z=float(center[2]),
        score=score,
        x1=float(det.x1),
        y1=float(det.y1),
        x2=float(det.x2),
        y2=float(det.y2),
        pixel_count=int(points.shape[0]),
        xy_spread_m=spread_xy,
        depth_spread_m=spread_depth,
    )


def candidate_weight(cand: DepthObjectCandidate) -> float:
    spread_weight = 1.0 / (1.0 + max(0.0, cand.xy_spread_m))
    return max(1e-4, cand.score) ** 2 * spread_weight * min(1.0, cand.pixel_count / 64.0)


def cluster_candidates(
    candidates: Iterable[DepthObjectCandidate],
    priors: Mapping[int, Mapping[str, float]],
    cluster_gates: Mapping[int, float],
    default_gate: float,
    min_sources: int,
) -> dict[tuple[int, int, int], list[DepthObjectCluster]]:
    groups: dict[tuple[int, int, int], list[DepthObjectCandidate]] = defaultdict(list)
    for cand in candidates:
        groups[(cand.scene_id, cand.frame_id, cand.class_id)].append(cand)

    out: dict[tuple[int, int, int], list[DepthObjectCluster]] = {}
    for key, rows in groups.items():
        class_id = key[2]
        gate = class_value(cluster_gates, class_id, default_gate)
        clusters: list[list[DepthObjectCandidate]] = []
        for cand in sorted(rows, key=lambda x: x.score, reverse=True):
            placed = False
            for cluster in clusters:
                if any(math.hypot(cand.x - item.x, cand.y - item.y) <= gate for item in cluster):
                    cluster.append(cand)
                    placed = True
                    break
            if not placed:
                clusters.append([cand])
        fused: list[DepthObjectCluster] = []
        for cluster in clusters:
            best_by_camera: dict[str, DepthObjectCandidate] = {}
            for cand in cluster:
                prev = best_by_camera.get(cand.camera_id)
                if prev is None or candidate_weight(cand) > candidate_weight(prev):
                    best_by_camera[cand.camera_id] = cand
            unique = list(best_by_camera.values())
            if len(unique) < min_sources:
                continue
            weights = np.asarray([candidate_weight(c) for c in unique], dtype=np.float64)
            weights = weights / max(1e-9, float(weights.sum()))
            xs = np.asarray([c.x for c in unique], dtype=np.float64)
            ys = np.asarray([c.y for c in unique], dtype=np.float64)
            x = float((xs * weights).sum())
            y = float((ys * weights).sum())
            spread = float((np.hypot(xs - x, ys - y) * weights).sum())
            prior = prior_for_class(priors, class_id)
            score = min(1.0, max(c.score for c in unique) * (0.92 + 0.08 * min(3, len(unique))))
            box = TrackBox(
                scene_id=key[0],
                class_id=class_id,
                object_id=-1,
                frame_id=key[1],
                x=x,
                y=y,
                z=float(prior["z"]),
                width=float(prior["width"]),
                length=float(prior["length"]),
                height=float(prior["height"]),
                yaw=circular_mean([float(prior.get("yaw", 0.0)) for _ in unique]),
                score=score,
            )
            fused.append(
                DepthObjectCluster(
                    box=box,
                    cameras=tuple(sorted(best_by_camera)),
                    candidates=tuple(unique),
                    source_count=len(unique),
                    spread_m=spread,
                )
            )
        if fused:
            out[key] = fused
    return out


def iter_depth_candidates(args: argparse.Namespace) -> tuple[list[DepthObjectCandidate], dict, dict[str, dict[str, CameraCalibration]]]:
    priors = load_priors(args.priors)
    root = year_dir(args.data_root, args.year) / args.split
    camera_cache: dict[str, dict[str, CameraCalibration]] = {}
    depth_cache = DepthCache(args.depth_root, args.split, max_items=args.depth_cache_size)
    class_min_scores = parse_class_float_map(args.class_min_scores)
    candidates: list[DepthObjectCandidate] = []
    stats = {
        "detections_seen": 0,
        "detections_depth_frame": 0,
        "detections_score_kept": 0,
        "missing_depth": 0,
        "candidate_fail": 0,
    }
    for det in read_detections(args.detections):
        stats["detections_seen"] += 1
        if args.depth_frame_stride > 1 and det.frame_id % args.depth_frame_stride != 0:
            continue
        stats["detections_depth_frame"] += 1
        threshold = class_value(class_min_scores, det.class_id, args.min_score)
        if det.score < threshold:
            continue
        stats["detections_score_kept"] += 1
        if det.scene_name not in camera_cache:
            camera_cache[det.scene_name] = load_scene_cameras(root / det.scene_name)
        camera = camera_cache[det.scene_name].get(str(det.camera_id)) or camera_cache[det.scene_name].get(camera_name(det.camera_id))
        if camera is None:
            stats["candidate_fail"] += 1
            continue
        depth = depth_cache.get(det.scene_name, det.camera_id, det.frame_id)
        if depth is None:
            stats["missing_depth"] += 1
            continue
        cand = detection_depth_candidate(
            det,
            camera,
            depth,
            det.score,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            depth_band_m=args.depth_band_m,
        )
        if cand is None:
            stats["candidate_fail"] += 1
            continue
        candidates.append(cand)
    stats["depth_candidates"] = len(candidates)
    return candidates, stats, camera_cache


def optimized_correction(
    det: FusedDetection3D,
    cluster: DepthObjectCluster,
    cameras: Mapping[str, Mapping[str, CameraCalibration]],
    *,
    alpha: float,
    match_gate: float,
    max_shift_m: float,
    search_radius_m: float,
    min_loss_improvement: float,
    max_loss_increase: float,
    depth_weight: float,
    movement_weight: float,
) -> tuple[TrackBox, float, float, bool]:
    base = det.box
    support = cluster.candidates
    base_loss = reprojection_loss(base, support, cameras)
    local_alpha = max(0.0, min(1.0, alpha))
    dist = math.hypot(base.x - cluster.box.x, base.y - cluster.box.y)
    local_alpha *= max(0.10, 1.0 - dist / max(1e-6, match_gate))
    if cluster.source_count >= 2:
        local_alpha *= min(1.35, 1.0 + 0.15 * (cluster.source_count - 1))
    local_alpha = min(1.0, local_alpha)

    dx = cluster.box.x - base.x
    dy = cluster.box.y - base.y
    candidates: list[TrackBox] = [base]
    for scale in (0.20, 0.35, 0.50, 0.75, 1.00):
        step = min(local_alpha * scale, 1.0)
        x = base.x + step * dx
        y = base.y + step * dy
        if math.hypot(x - base.x, y - base.y) <= max_shift_m:
            candidates.append(replace_box_xy(base, x, y))
    if search_radius_m > 0.0:
        center_x = base.x + local_alpha * dx
        center_y = base.y + local_alpha * dy
        for ox, oy in (
            (search_radius_m, 0.0),
            (-search_radius_m, 0.0),
            (0.0, search_radius_m),
            (0.0, -search_radius_m),
            (0.707 * search_radius_m, 0.707 * search_radius_m),
            (-0.707 * search_radius_m, 0.707 * search_radius_m),
            (0.707 * search_radius_m, -0.707 * search_radius_m),
            (-0.707 * search_radius_m, -0.707 * search_radius_m),
        ):
            x = center_x + ox
            y = center_y + oy
            if math.hypot(x - base.x, y - base.y) <= max_shift_m:
                candidates.append(replace_box_xy(base, x, y))

    best_box = base
    best_obj = float("inf")
    best_loss = base_loss
    for cand_box in candidates:
        shift = math.hypot(cand_box.x - base.x, cand_box.y - base.y)
        depth_dist = math.hypot(cand_box.x - cluster.box.x, cand_box.y - cluster.box.y) / max(1e-6, match_gate)
        loss = reprojection_loss(cand_box, support, cameras)
        objective = loss + depth_weight * depth_dist + movement_weight * (shift / max(1e-6, max_shift_m))
        if objective < best_obj:
            best_obj = objective
            best_box = cand_box
            best_loss = loss

    shift = math.hypot(best_box.x - base.x, best_box.y - base.y)
    if shift <= 1e-9:
        return base, 0.0, base_loss, False
    gate = base_loss - min_loss_improvement if base_loss > min_loss_improvement else base_loss + max_loss_increase
    accepted = best_loss <= gate
    return (best_box if accepted else base), shift if accepted else 0.0, best_loss, accepted


def corrected_group(
    group: list[FusedDetection3D],
    clusters: list[DepthObjectCluster],
    match_gate: float,
    alpha: float,
    stats: MergeStats,
    cameras: Mapping[str, Mapping[str, CameraCalibration]],
    *,
    optimizer: bool,
    max_shift_m: float,
    search_radius_m: float,
    min_loss_improvement: float,
    max_loss_increase: float,
    depth_weight: float,
    movement_weight: float,
) -> list[FusedDetection3D]:
    if not group or not clusters:
        return group
    used: set[int] = set()
    out: list[FusedDetection3D] = []
    order = sorted(range(len(group)), key=lambda i: group[i].score, reverse=True)
    corrections: dict[int, FusedDetection3D] = {}
    for i in order:
        det = group[i]
        best_j = -1
        best_dist = float("inf")
        for j, cluster in enumerate(clusters):
            if j in used:
                continue
            dist = math.hypot(det.box.x - cluster.box.x, det.box.y - cluster.box.y)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j < 0 or best_dist > match_gate:
            continue
        cluster = clusters[best_j]
        used.add(best_j)
        if optimizer:
            box, shift, _loss, accepted = optimized_correction(
                det,
                cluster,
                cameras,
                alpha=alpha,
                match_gate=match_gate,
                max_shift_m=max_shift_m,
                search_radius_m=search_radius_m,
                min_loss_improvement=min_loss_improvement,
                max_loss_increase=max_loss_increase,
                depth_weight=depth_weight,
                movement_weight=movement_weight,
            )
            if not accepted:
                continue
        else:
            local_alpha = max(0.0, min(1.0, alpha)) * max(0.10, 1.0 - best_dist / max(1e-6, match_gate))
            if cluster.source_count >= 2:
                local_alpha *= min(1.35, 1.0 + 0.15 * (cluster.source_count - 1))
            local_alpha = min(1.0, local_alpha)
            x = (1.0 - local_alpha) * det.box.x + local_alpha * cluster.box.x
            y = (1.0 - local_alpha) * det.box.y + local_alpha * cluster.box.y
            shift = math.hypot(x - det.box.x, y - det.box.y)
            if shift > max_shift_m:
                continue
            box = replace_box_xy(det.box, x, y)
        if shift <= 1e-9:
            continue
        corrections[i] = FusedDetection3D(
            box=box,
            score=det.score,
            cameras=tuple(sorted(set(det.cameras) | set(cluster.cameras))),
            source_count=max(det.source_count, cluster.source_count),
            cluster_spread_m=min(det.cluster_spread_m, cluster.spread_m) if det.cluster_spread_m > 0 else cluster.spread_m,
            mean_reprojection_error=det.mean_reprojection_error,
        )
        stats.corrected_rows += 1
        stats.mean_xy_shift_sum += shift
        stats.max_xy_shift = max(stats.max_xy_shift, shift)
    for idx, det in enumerate(group):
        out.append(corrections.get(idx, det))
    return out


def iter_corrected_fused(
    base_fused: Path,
    clusters: Mapping[tuple[int, int, int], list[DepthObjectCluster]],
    match_gates: Mapping[int, float],
    default_match_gate: float,
    alpha: float,
    stats: MergeStats,
    cameras: Mapping[str, Mapping[str, CameraCalibration]],
    *,
    optimizer: bool,
    max_shift_m: float,
    search_radius_m: float,
    min_loss_improvement: float,
    max_loss_increase: float,
    depth_weight: float,
    movement_weight: float,
) -> Iterable[FusedDetection3D]:
    prev_key = None
    group: list[FusedDetection3D] = []

    def flush(rows: list[FusedDetection3D]) -> Iterable[FusedDetection3D]:
        if not rows:
            return []
        key = (rows[0].box.scene_id, rows[0].box.frame_id, rows[0].box.class_id)
        gate = class_value(match_gates, key[2], default_match_gate)
        return corrected_group(
            rows,
            list(clusters.get(key, ())),
            gate,
            alpha,
            stats,
            cameras,
            optimizer=optimizer,
            max_shift_m=max_shift_m,
            search_radius_m=search_radius_m,
            min_loss_improvement=min_loss_improvement,
            max_loss_increase=max_loss_increase,
            depth_weight=depth_weight,
            movement_weight=movement_weight,
        )

    for det in read_fused_detections(base_fused):
        stats.base_rows += 1
        key = (det.box.scene_id, det.box.frame_id, det.box.class_id)
        if prev_key is not None and key != prev_key:
            yield from flush(group)
            group = []
        prev_key = key
        group.append(det)
    if group:
        yield from flush(group)


def run(args: argparse.Namespace) -> dict:
    priors = load_priors(args.priors)
    candidates, candidate_stats, camera_cache = iter_depth_candidates(args)
    clusters = cluster_candidates(
        candidates,
        priors=priors,
        cluster_gates=parse_class_float_map(args.class_cluster_gates_m),
        default_gate=args.cluster_gate_m,
        min_sources=args.min_sources,
    )
    stats = MergeStats(
        frames_with_depth_clusters=len(clusters),
        depth_candidates=len(candidates),
        depth_clusters=sum(len(v) for v in clusters.values()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = write_fused_detections(
        iter_corrected_fused(
            args.base_fused,
            clusters,
            parse_class_float_map(args.class_match_gates_m),
            args.match_gate_m,
            args.alpha,
            stats,
            camera_cache,
            optimizer=args.optimizer,
            max_shift_m=args.max_shift_m,
            search_radius_m=args.search_radius_m,
            min_loss_improvement=args.min_loss_improvement,
            max_loss_increase=args.max_loss_increase,
            depth_weight=args.depth_weight,
            movement_weight=args.movement_weight,
        ),
        args.out,
    )
    payload = {
        "out": str(args.out),
        "written": count,
        "candidate_stats": candidate_stats,
        "merge_stats": stats.as_dict(),
        "params": {
            "min_sources": args.min_sources,
            "alpha": args.alpha,
            "min_score": args.min_score,
            "class_min_scores": args.class_min_scores,
            "cluster_gate_m": args.cluster_gate_m,
            "match_gate_m": args.match_gate_m,
            "optimizer": args.optimizer,
            "max_shift_m": args.max_shift_m,
            "search_radius_m": args.search_radius_m,
            "min_loss_improvement": args.min_loss_improvement,
            "max_loss_increase": args.max_loss_increase,
            "depth_weight": args.depth_weight,
            "movement_weight": args.movement_weight,
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object-centric multi-view DepthPro fusion for Track 1 fused TSVs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--split", default="test")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--base-fused", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--priors", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--depth-frame-stride", type=int, default=30)
    parser.add_argument("--depth-cache-size", type=int, default=64)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--class-min-scores", default="0:0.70,1:0.75,2:0.35,3:0.35,4:0.35,5:0.35,6:0.45")
    parser.add_argument("--min-pixels", type=int, default=12)
    parser.add_argument("--max-pixels", type=int, default=96)
    parser.add_argument("--depth-band-m", type=float, default=0.35)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--cluster-gate-m", type=float, default=1.40)
    parser.add_argument("--class-cluster-gates-m", default="0:0.90,1:1.80,2:1.20,3:1.50,4:1.00,5:1.00,6:1.70")
    parser.add_argument("--match-gate-m", type=float, default=1.20)
    parser.add_argument("--class-match-gates-m", default="0:0.75,1:1.50,2:1.00,3:1.20,4:0.90,5:0.90,6:1.40")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--optimizer", action="store_true", help="Use reprojection-gated object-centric correction search")
    parser.add_argument("--max-shift-m", type=float, default=0.30)
    parser.add_argument("--search-radius-m", type=float, default=0.10)
    parser.add_argument("--min-loss-improvement", type=float, default=0.005)
    parser.add_argument("--max-loss-increase", type=float, default=0.000)
    parser.add_argument("--depth-weight", type=float, default=0.08)
    parser.add_argument("--movement-weight", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
