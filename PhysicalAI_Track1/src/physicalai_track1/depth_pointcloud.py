from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .calibration import CameraCalibration
from .detections import Detection2D
from .geometry_residual import _depth_file, _frame_depth_key


@dataclass(frozen=True)
class DepthLiftResult:
    x: float
    y: float
    z_observed: float
    yaw: float | None
    width_extent: float | None
    length_extent: float | None
    pixel_count: int
    valid_ratio: float
    dynamic_ratio: float
    depth_median: float
    depth_spread: float
    reprojection_error: float


def _box_indices(det: Detection2D, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    height, width = shape
    x1 = max(0, min(width - 1, int(math.floor(det.x1))))
    x2 = max(0, min(width, int(math.ceil(det.x2))))
    y1 = max(0, min(height - 1, int(math.floor(det.y1))))
    y2 = max(0, min(height, int(math.ceil(det.y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _overlap_mask(det: Detection2D, detections: Iterable[Detection2D], shape: tuple[int, int]) -> np.ndarray | None:
    box = _box_indices(det, shape)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for other in detections:
        if other is det or other.camera_id != det.camera_id or other.frame_id != det.frame_id:
            continue
        if other.x1 == det.x1 and other.y1 == det.y1 and other.x2 == det.x2 and other.y2 == det.y2:
            continue
        ox1 = max(x1, int(math.floor(other.x1)))
        ox2 = min(x2, int(math.ceil(other.x2)))
        oy1 = max(y1, int(math.floor(other.y1)))
        oy2 = min(y2, int(math.ceil(other.y2)))
        if ox2 > ox1 and oy2 > oy1:
            mask[oy1 - y1 : oy2 - y1, ox1 - x1 : ox2 - x1] = True
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover
        return mask
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = int(np.argmax(sizes))
    return labels == keep


class StaticDepthCache:
    def __init__(
        self,
        max_cameras: int = 2,
        max_samples: int = 48,
        sample_stride: int = 30,
        static_root: Path | str | None = None,
    ) -> None:
        self.max_cameras = max(1, int(max_cameras))
        self.max_samples = max(1, int(max_samples))
        self.sample_stride = max(1, int(sample_stride))
        self.static_root = Path(static_root) if static_root else None
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    def static_path(self, depth_path: Path) -> Path | None:
        if self.static_root is None:
            return None
        depth_path = Path(depth_path)
        scene_name = depth_path.parent.parent.name
        return self.static_root / scene_name / f"{depth_path.stem}.npy"

    def get(self, depth_path: Path, handle) -> np.ndarray:
        depth_path = Path(depth_path)
        cached = self._cache.get(depth_path)
        if cached is not None:
            self._cache.move_to_end(depth_path)
            return cached
        static_path = self.static_path(depth_path)
        if static_path is not None and static_path.exists():
            background = np.load(static_path).astype(np.uint16)
            self._cache[depth_path] = background
            while len(self._cache) > self.max_cameras:
                self._cache.popitem(last=False)
            return background
        keys = sorted(handle.keys())
        sampled = keys[:: self.sample_stride][: self.max_samples]
        if not sampled:
            raise ValueError(f"{depth_path}: no depth frames")
        stack = [np.asarray(handle[key], dtype=np.uint16) for key in sampled]
        background = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint16)
        if static_path is not None:
            static_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(static_path, background)
        self._cache[depth_path] = background
        while len(self._cache) > self.max_cameras:
            self._cache.popitem(last=False)
        return background


def estimate_foreground_depth_lift(
    det: Detection2D,
    camera: CameraCalibration,
    depth_dataset,
    background_depth: np.ndarray | None,
    frame_detections: Iterable[Detection2D] = (),
    depth_scale: float = 0.001,
    foreground_delta_m: float = 0.25,
    min_pixels: int = 12,
    max_reprojection_px: float = 80.0,
) -> DepthLiftResult | None:
    depth = np.asarray(depth_dataset, dtype=np.float32)
    box = _box_indices(det, depth.shape[:2])
    if box is None:
        return None
    x1, y1, x2, y2 = box
    crop = depth[y1:y2, x1:x2]
    valid = np.isfinite(crop) & (crop > 0)
    if background_depth is not None and background_depth.shape == depth.shape:
        bg = background_depth[y1:y2, x1:x2].astype(np.float32)
        dynamic = valid & (np.abs(crop - bg) * depth_scale >= foreground_delta_m)
    else:
        dynamic = valid.copy()
    overlap = _overlap_mask(det, frame_detections, depth.shape[:2])
    if overlap is not None:
        dynamic &= ~overlap
    if dynamic.sum() < min_pixels:
        # Keep lower object pixels before giving up; for small/far objects static
        # subtraction can erase the full crop.
        lower = np.zeros_like(dynamic)
        lower[int(round(lower.shape[0] * 0.55)) :, :] = True
        dynamic = valid & lower
    if dynamic.sum() < min_pixels:
        return None
    component = _largest_component(dynamic)
    if component.sum() >= min_pixels:
        dynamic = component
    ys, xs = np.nonzero(dynamic)
    depths_m = crop[ys, xs] * depth_scale
    good = np.isfinite(depths_m) & (depths_m > 0)
    if good.sum() < min_pixels:
        return None
    xs = xs[good] + x1
    ys = ys[good] + y1
    depths_m = depths_m[good]
    median = float(np.median(depths_m))
    spread = float(np.percentile(depths_m, 90) - np.percentile(depths_m, 10))
    # Use robust lower/near-center reference pixel. It is more stable than the
    # full crop centroid for partially visible objects.
    q = np.percentile(depths_m, 40)
    ref_mask = depths_m <= q
    if ref_mask.sum() < min_pixels:
        ref_mask = np.ones_like(depths_m, dtype=bool)
    ref_x = float(np.median(xs[ref_mask]))
    ref_y = float(np.percentile(ys[ref_mask], 70))
    ref_depth = float(np.median(depths_m[ref_mask]))
    point = camera.image_depth_to_world(ref_x, ref_y, ref_depth)
    if point is None:
        return None
    projected = camera.ground_to_image(point[0], point[1])
    reproj = float(((projected[0] - ref_x) ** 2 + (projected[1] - ref_y) ** 2) ** 0.5)
    if reproj > max_reprojection_px:
        return None
    world_points = []
    # Sparse sample for extent/yaw. Avoid huge point clouds.
    step = max(1, len(xs) // 256)
    for px, py, pd in zip(xs[::step], ys[::step], depths_m[::step]):
        p = camera.image_depth_to_world(float(px), float(py), float(pd))
        if p is not None:
            world_points.append(p)
    yaw = None
    width_extent = None
    length_extent = None
    if len(world_points) >= 8:
        pts = np.asarray(world_points, dtype=np.float32)
        xy = pts[:, :2]
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) >= 8:
            centered = xy - np.median(xy, axis=0, keepdims=True)
            cov = centered.T @ centered / max(1, len(centered) - 1)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            major = vecs[:, order[0]]
            yaw = float(math.atan2(float(major[1]), float(major[0])))
            proj = centered @ vecs[:, order]
            extent = np.percentile(proj, 90, axis=0) - np.percentile(proj, 10, axis=0)
            length_extent = float(max(extent[0], 0.0))
            width_extent = float(max(extent[1], 0.0))
    area = max(1, crop.size)
    return DepthLiftResult(
        x=float(point[0]),
        y=float(point[1]),
        z_observed=float(point[2]),
        yaw=yaw,
        width_extent=width_extent,
        length_extent=length_extent,
        pixel_count=int(good.sum()),
        valid_ratio=float(valid.sum() / area),
        dynamic_ratio=float(dynamic.sum() / area),
        depth_median=median,
        depth_spread=spread,
        reprojection_error=reproj,
    )
