from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from .calibration import CameraCalibration, load_scene_cameras
from .dataset import TrackBox, scene_name_to_id, year_dir
from .detections import Detection2D, read_detections
from .depth_pointcloud import StaticDepthCache, estimate_foreground_depth_lift
from .geometry_residual import (
    GeometryResidualPredictor,
    _depth_file,
    _depth_npz_file,
    _empty_depth_features,
    _frame_depth_key,
    _load_depth_npz,
    depth_crop_features,
    geometry_features,
    object_aware_depth_features,
)
from .priors import load_priors, prior_for_class


def _robust_depth_value(
    depth_dataset,
    box: tuple[float, float, float, float],
    depth_scale: float,
    region: str = "lower",
    percentile: float = 35.0,
    source_width: float | None = None,
    source_height: float | None = None,
) -> float | None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    height, width = depth_dataset.shape[:2]
    x1, y1, x2, y2 = box
    if source_width is not None and source_height is not None:
        sx = width / max(1.0, float(source_width))
        sy = height / max(1.0, float(source_height))
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy
    x1_i = max(0, min(width - 1, int(math.floor(x1))))
    x2_i = max(0, min(width, int(math.ceil(x2))))
    y1_i = max(0, min(height - 1, int(math.floor(y1))))
    y2_i = max(0, min(height, int(math.ceil(y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    if region == "lower":
        y1_i = y1_i + int(round((y2_i - y1_i) * 0.55))
    elif region == "center":
        cx1 = x1_i + int(round((x2_i - x1_i) * 0.35))
        cx2 = x1_i + int(round((x2_i - x1_i) * 0.65))
        cy1 = y1_i + int(round((y2_i - y1_i) * 0.35))
        cy2 = y1_i + int(round((y2_i - y1_i) * 0.65))
        x1_i, x2_i, y1_i, y2_i = cx1, max(cx1 + 1, cx2), cy1, max(cy1 + 1, cy2)
    crop = np.asarray(depth_dataset[y1_i:y2_i, x1_i:x2_i], dtype=np.float32)
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size < 4:
        return None
    return float(np.percentile(valid, percentile) * depth_scale)


def _depth_backproject_xy(
    det: Detection2D,
    camera: CameraCalibration,
    depth_dataset,
    depth_scale: float,
    percentile: float,
) -> tuple[float, float, float] | None:
    depth = _robust_depth_value(
        depth_dataset,
        (det.x1, det.y1, det.x2, det.y2),
        depth_scale=depth_scale,
        region="lower",
        percentile=percentile,
        source_width=camera.frame_width,
        source_height=camera.frame_height,
    )
    if depth is None:
        return None
    u = 0.5 * (det.x1 + det.x2)
    v = det.y1 + 0.82 * (det.y2 - det.y1)
    point = camera.image_depth_to_world(u, v, depth)
    if point is None:
        return None
    return point


@dataclass(frozen=True)
class LiftedCandidate:
    box: TrackBox
    camera_id: str
    score: float
    source_object_id: Optional[int] = None
    reprojection_error: float = 0.0
    geometry_uncertainty: float = 0.0


def lift_detection(
    det: Detection2D,
    camera: CameraCalibration,
    class_prior: dict,
    object_id: int = -1,
    residual_predictor: GeometryResidualPredictor | None = None,
    residual_scale: float = 1.0,
    max_residual_uncertainty: float | None = None,
    depth_features: list[float] | None = None,
    depth_world_point: tuple[float, float, float] | None = None,
    depth_lift_mode: str = "none",
    depth_blend_alpha: float = 0.65,
    depth_max_reprojection_px: float = 80.0,
    depth_yaw: float | None = None,
    depth_width_extent: float | None = None,
    depth_length_extent: float | None = None,
) -> LiftedCandidate:
    u, v = det.bottom_center
    x, y = camera.image_to_ground(u, v)
    base_x, base_y = x, y
    width = float(class_prior["width"])
    length = float(class_prior["length"])
    height = float(class_prior["height"])
    z = float(class_prior["z"])
    yaw = float(class_prior.get("yaw", 0.0))
    geometry_uncertainty = 0.0
    if depth_world_point is not None and depth_lift_mode in {"backproject", "blend", "foreground-pointcloud"}:
        depth_x, depth_y, _depth_z = depth_world_point
        if depth_lift_mode in {"backproject", "foreground-pointcloud"}:
            x, y = depth_x, depth_y
        else:
            alpha = max(0.0, min(1.0, depth_blend_alpha))
            x = (1.0 - alpha) * x + alpha * depth_x
            y = (1.0 - alpha) * y + alpha * depth_y
        projected_depth = camera.ground_to_image(x, y)
        depth_reproj = ((projected_depth[0] - u) ** 2 + (projected_depth[1] - v) ** 2) ** 0.5
        if depth_reproj > depth_max_reprojection_px:
            x, y = base_x, base_y
        elif depth_lift_mode == "foreground-pointcloud":
            if depth_yaw is not None and det.class_id in {1, 2, 3, 6}:
                yaw = depth_yaw
            if depth_width_extent is not None and 0.25 * width <= depth_width_extent <= 2.5 * width:
                width = 0.75 * width + 0.25 * depth_width_extent
            if depth_length_extent is not None and 0.25 * length <= depth_length_extent <= 2.5 * length:
                length = 0.75 * length + 0.25 * depth_length_extent
    if residual_predictor is not None:
        feature, _ = geometry_features(det, camera, class_prior, depth_features)
        residual = residual_predictor.predict(feature)
        uncertainty = float(residual["center_uncertainty"])
        effective_scale = max(0.0, min(1.0, residual_scale))
        if max_residual_uncertainty is not None and uncertainty > max_residual_uncertainty:
            effective_scale = 0.0
        x += effective_scale * residual["dx"]
        y += effective_scale * residual["dy"]
        z += effective_scale * residual["dz"]
        width *= math.exp(effective_scale * residual["dlog_width"])
        length *= math.exp(effective_scale * residual["dlog_length"])
        height *= math.exp(effective_scale * residual["dlog_height"])
        yaw_delta = math.atan2(
            math.sin(residual["yaw"] - yaw),
            math.cos(residual["yaw"] - yaw),
        )
        yaw += effective_scale * yaw_delta
        geometry_uncertainty = effective_scale * uncertainty
    projected = camera.ground_to_image(x, y)
    reproj = ((projected[0] - u) ** 2 + (projected[1] - v) ** 2) ** 0.5
    return LiftedCandidate(
        box=TrackBox(
            scene_id=scene_name_to_id(det.scene_name),
            class_id=det.class_id,
            object_id=object_id,
            frame_id=det.frame_id,
            x=x,
            y=y,
            z=z,
            width=width,
            length=length,
            height=height,
            yaw=yaw,
            score=det.score,
        ),
        camera_id=det.camera_id,
        score=det.score,
        source_object_id=det.oracle_object_id,
        reprojection_error=reproj,
        geometry_uncertainty=geometry_uncertainty,
    )


def iter_lifted_candidates(
    detections_path: Path | str,
    data_root: Path | str,
    year: int,
    priors_path: Path | str | None = None,
    split: str = "val",
    use_oracle_ids: bool = False,
    residual_model_path: Path | str | None = None,
    residual_scale: float = 1.0,
    max_residual_uncertainty: float | None = None,
    use_depth: bool = False,
    depth_scale: float = 0.001,
    depth_lift_mode: str = "none",
    depth_blend_alpha: float = 0.65,
    depth_percentile: float = 35.0,
    depth_max_reprojection_px: float = 80.0,
    depth_root: Path | str | None = None,
    static_depth_root: Path | str | None = None,
) -> Iterator[LiftedCandidate]:
    if (use_depth or depth_lift_mode != "none") and depth_root is None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("h5py is required for depth-aware lift-2d") from exc
    else:
        h5py = None
    priors = load_priors(priors_path)
    camera_cache: Dict[str, Dict[str, CameraCalibration]] = {}
    depth_cache = {}
    npz_depth_cache: OrderedDict[tuple[str, str, int], object | None] = OrderedDict()
    static_depth_cache = StaticDepthCache(static_root=static_depth_root)
    root = year_dir(data_root, year) / split
    depth_root_path = Path(depth_root) if depth_root is not None else None
    residual_predictor = GeometryResidualPredictor(residual_model_path) if residual_model_path else None
    detections = list(read_detections(detections_path))
    frame_groups: Dict[tuple[str, str, int], list[Detection2D]] = {}
    if depth_lift_mode == "foreground-pointcloud":
        for item in detections:
            frame_groups.setdefault((item.scene_name, item.camera_id, item.frame_id), []).append(item)
    try:
        for det in detections:
            if det.scene_name not in camera_cache:
                camera_cache[det.scene_name] = load_scene_cameras(root / det.scene_name)
            camera = camera_cache[det.scene_name].get(det.camera_id)
            if camera is None:
                continue
            depth_feature = None
            depth_world_point = None
            depth_yaw = None
            depth_width_extent = None
            depth_length_extent = None
            if use_depth or depth_lift_mode != "none":
                scene_dir = root / det.scene_name
                handle = None
                frame_key = _frame_depth_key(det.frame_id)
                depth_dataset = None
                if depth_root_path is not None:
                    cache_key = (det.scene_name, det.camera_id, det.frame_id)
                    if cache_key in npz_depth_cache:
                        depth_dataset = npz_depth_cache.pop(cache_key)
                        npz_depth_cache[cache_key] = depth_dataset
                    else:
                        path = _depth_npz_file(depth_root_path, split, det.scene_name, det.camera_id, det.frame_id)
                        depth_dataset = _load_depth_npz(path) if path.exists() else None
                        npz_depth_cache[cache_key] = depth_dataset
                        if len(npz_depth_cache) > 96:
                            npz_depth_cache.popitem(last=False)
                    source_depth_scale = 1.0
                else:
                    cache_key = (det.scene_name, det.camera_id)
                    if cache_key not in depth_cache:
                        path = _depth_file(scene_dir, det.camera_id)
                        depth_cache[cache_key] = h5py.File(path, "r") if path.exists() else None
                    handle = depth_cache.get(cache_key)
                    if handle is not None and frame_key in handle:
                        depth_dataset = handle[frame_key]
                    source_depth_scale = depth_scale
                if depth_dataset is not None:
                    if use_depth and residual_predictor is not None:
                        depth_feature = object_aware_depth_features(
                            depth_dataset,
                            (det.x1, det.y1, det.x2, det.y2),
                            det.class_id,
                            camera,
                            camera.image_to_ground(*det.bottom_center),
                            scene_name=scene_dir.name,
                            camera_id=det.camera_id,
                            static_depth_root=static_depth_root,
                            depth_scale=source_depth_scale,
                        )
                    elif use_depth:
                        depth_feature = depth_crop_features(
                            depth_dataset,
                            (det.x1, det.y1, det.x2, det.y2),
                            depth_scale=source_depth_scale,
                        )
                    else:
                        depth_feature = None
                    if depth_lift_mode != "none":
                        if depth_lift_mode == "foreground-pointcloud":
                            path = _depth_file(scene_dir, det.camera_id)
                            background = None
                            if handle is not None:
                                background = static_depth_cache.get(path, handle)
                            result = estimate_foreground_depth_lift(
                                det,
                                camera,
                                depth_dataset,
                                background,
                                frame_detections=frame_groups.get((det.scene_name, det.camera_id, det.frame_id), ()),
                                depth_scale=source_depth_scale,
                                max_reprojection_px=depth_max_reprojection_px,
                            )
                            if result is not None:
                                depth_world_point = (result.x, result.y, result.z_observed)
                                depth_yaw = result.yaw
                                depth_width_extent = result.width_extent
                                depth_length_extent = result.length_extent
                        else:
                            depth_world_point = _depth_backproject_xy(
                                det,
                                camera,
                                depth_dataset,
                                depth_scale=source_depth_scale,
                                percentile=depth_percentile,
                            )
                else:
                    depth_feature = _empty_depth_features() if use_depth else None
            object_id = det.oracle_object_id if use_oracle_ids and det.oracle_object_id is not None else -1
            yield lift_detection(
                det,
                camera,
                dict(prior_for_class(priors, det.class_id)),
                object_id=object_id,
                residual_predictor=residual_predictor,
                residual_scale=residual_scale,
                max_residual_uncertainty=max_residual_uncertainty,
                depth_features=depth_feature,
                depth_world_point=depth_world_point,
                depth_lift_mode=depth_lift_mode,
                depth_blend_alpha=depth_blend_alpha,
                depth_max_reprojection_px=depth_max_reprojection_px,
                depth_yaw=depth_yaw,
                depth_width_extent=depth_width_extent,
                depth_length_extent=depth_length_extent,
            )
    finally:
        for handle in depth_cache.values():
            if handle is not None:
                handle.close()


def write_lifted_candidates(candidates: Iterable[LiftedCandidate], out_path: Path | str) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        f.write(
            "scene_id\tframe_id\tclass_id\tx\ty\tz\twidth\tlength\theight\tyaw\tscore\tcamera_id\tsource_object_id\treprojection_error\tgeometry_uncertainty\n"
        )
        for cand in candidates:
            b = cand.box
            source = "" if cand.source_object_id is None else str(cand.source_object_id)
            f.write(
                "\t".join(
                    [
                        str(b.scene_id),
                        str(b.frame_id),
                        str(b.class_id),
                        f"{b.x:.6f}",
                        f"{b.y:.6f}",
                        f"{b.z:.6f}",
                        f"{b.width:.6f}",
                        f"{b.length:.6f}",
                        f"{b.height:.6f}",
                        f"{b.yaw:.6f}",
                        f"{cand.score:.6f}",
                        cand.camera_id,
                        source,
                        f"{cand.reprojection_error:.6f}",
                        f"{cand.geometry_uncertainty:.6f}",
                    ]
                )
            )
            f.write("\n")
            count += 1
    return count


def read_lifted_candidates(path: Path | str) -> Iterator[LiftedCandidate]:
    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline()
        for line_no, line in enumerate(f, start=2):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 12:
                raise ValueError(f"{path}:{line_no}: expected lifted candidate TSV")
            source = int(parts[12]) if len(parts) > 12 and parts[12] else None
            box = TrackBox(
                scene_id=int(parts[0]),
                frame_id=int(parts[1]),
                class_id=int(parts[2]),
                object_id=-1,
                x=float(parts[3]),
                y=float(parts[4]),
                z=float(parts[5]),
                width=float(parts[6]),
                length=float(parts[7]),
                height=float(parts[8]),
                yaw=float(parts[9]),
                score=float(parts[10]),
            )
            yield LiftedCandidate(
                box=box,
                camera_id=parts[11],
                score=float(parts[10]),
                source_object_id=source,
                reprojection_error=float(parts[13]) if len(parts) > 13 and parts[13] else 0.0,
                geometry_uncertainty=float(parts[14]) if len(parts) > 14 and parts[14] else 0.0,
            )
