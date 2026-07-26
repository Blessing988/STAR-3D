from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from physicalai_track1.constants import GT_KEY_ALIASES
from physicalai_track1.dataset import iter_gt_frames, year_dir
from generate_zio_pcd_from_depth_pro_2026 import load_camera_params


def _get_any(record: dict, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


def _camera_name(name: str) -> str:
    if name.startswith("Camera_"):
        return name
    if name.startswith("camera_"):
        return "Camera_" + name.rsplit("_", 1)[-1]
    try:
        return f"Camera_{int(name):04d}"
    except ValueError:
        return name


def _bbox_xyxy(value) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        keys = [
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        ]
        for names in keys:
            if all(k in value for k in names):
                return tuple(float(value[k]) for k in names)  # type: ignore[return-value]
        if all(k in value for k in ("x", "y", "width", "height")):
            x = float(value["x"])
            y = float(value["y"])
            return x, y, x + float(value["width"]), y + float(value["height"])
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])
    return None


def _parse_depth_roots(values: list[str]) -> list[tuple[str, Path]]:
    roots = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
            roots.append((label, Path(path)))
        else:
            path = Path(value)
            roots.append((path.name, path))
    return roots


def _load_depth_npz(path: Path) -> tuple[np.ndarray, dict[str, float]] | None:
    if not path.exists():
        return None
    data = np.load(path)
    depth = data["depth_m"].astype(np.float32)
    meta = {}
    for key in ("focal_px", "focal_px_pred", "focal_px_used", "source_width", "source_height", "image_width", "image_height"):
        if key in data:
            meta[key] = float(np.asarray(data[key]).reshape(()))
    return depth, meta


def _scaled_k(camera: dict, depth_w: int, depth_h: int) -> np.ndarray:
    k = np.asarray(camera["intrinsic"], dtype=np.float64).copy()
    sx = depth_w / float(camera["width"])
    sy = depth_h / float(camera["height"])
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def _sample_bbox_points(
    depth: np.ndarray,
    camera: dict,
    bbox: tuple[float, float, float, float],
    sample_grid: int,
    crop_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_h, depth_w = depth.shape[:2]
    sx = depth_w / float(camera["width"])
    sy = depth_h / float(camera["height"])
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * max(1.0, x2 - x1) * crop_fraction
    half_h = 0.5 * max(1.0, y2 - y1) * crop_fraction
    x1d = np.clip((cx - half_w) * sx, 0, depth_w - 1)
    x2d = np.clip((cx + half_w) * sx, 0, depth_w - 1)
    y1d = np.clip((cy - half_h) * sy, 0, depth_h - 1)
    y2d = np.clip((cy + half_h) * sy, 0, depth_h - 1)
    if x2d <= x1d or y2d <= y1d:
        return np.empty(0), np.empty(0), np.empty(0)
    us = np.linspace(x1d, x2d, sample_grid)
    vs = np.linspace(y1d, y2d, sample_grid)
    uu, vv = np.meshgrid(us, vs)
    ui = np.clip(np.rint(uu).astype(np.int32), 0, depth_w - 1)
    vi = np.clip(np.rint(vv).astype(np.int32), 0, depth_h - 1)
    z = depth[vi, ui].reshape(-1)
    u = uu.reshape(-1)
    v = vv.reshape(-1)
    valid = np.isfinite(z) & (z > 0.05) & (z < 120.0)
    return u[valid], v[valid], z[valid]


def _backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, camera: dict, mode: str) -> np.ndarray:
    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float64)
    depth_h = int(camera["_depth_h"])
    depth_w = int(camera["_depth_w"])
    k = _scaled_k(camera, depth_w, depth_h)
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam = np.stack([x_cam, y_cam, z], axis=1)
    e = np.asarray(camera["extrinsic"], dtype=np.float64)
    r = e[:3, :3]
    t = e[:3, 3]
    if mode == "inverse":
        world = (np.linalg.inv(r) @ (cam - t).T).T
    elif mode == "direct":
        world = (r @ cam.T).T + t
    else:
        raise ValueError(mode)
    return world


def _inside_loose_box(points: np.ndarray, location: np.ndarray, scale: np.ndarray, margin: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=bool)
    half = 0.5 * np.maximum(scale, 0.01) + margin
    return np.all(np.abs(points - location.reshape(1, 3)) <= half.reshape(1, 3), axis=1)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p75": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def audit_root(args: argparse.Namespace, label: str, depth_root: Path) -> dict:
    root = year_dir(args.data_root, args.year)
    scene_filter = set(args.scenes or [])
    mode_stats = {
        mode: {
            "items": 0,
            "points": 0,
            "items_with_inside_points": 0,
            "inside_points": 0,
            "min_center_dist_m": [],
            "median_center_dist_m": [],
        }
        for mode in ("inverse", "direct")
    }
    focal_ratios = []
    focal_abs_delta = []
    missing_depth = 0
    sampled_items = 0
    for scene_dir in sorted(path for path in (root / args.split).iterdir() if path.is_dir()):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        camera_params = load_camera_params(scene_dir)
        for frame_key, items in iter_gt_frames(scene_dir, max_frames=args.max_frames_per_scene):
            frame_id = int(frame_key)
            if frame_id % args.frame_stride != 0:
                continue
            for item in items:
                bbox_map = _get_any(item, "bbox2d")
                if not isinstance(bbox_map, dict):
                    continue
                location = np.asarray(_get_any(item, "location"), dtype=np.float64).reshape(3)
                scale = np.asarray(_get_any(item, "scale"), dtype=np.float64).reshape(3)
                for raw_cam, raw_bbox in bbox_map.items():
                    if sampled_items >= args.max_items:
                        break
                    cam = _camera_name(str(raw_cam))
                    if cam not in camera_params:
                        continue
                    bbox = _bbox_xyxy(raw_bbox)
                    if bbox is None:
                        continue
                    depth_path = depth_root / args.split / scene_dir.name / cam / f"{frame_id:06d}.npz"
                    loaded = _load_depth_npz(depth_path)
                    if loaded is None:
                        missing_depth += 1
                        continue
                    depth, meta = loaded
                    pred = meta.get("focal_px_pred")
                    used = meta.get("focal_px_used")
                    if pred is not None and used is not None and math.isfinite(pred) and math.isfinite(used) and used > 0:
                        focal_ratios.append(pred / used)
                        focal_abs_delta.append(abs(pred - used))
                    camera = dict(camera_params[cam])
                    camera["_depth_h"] = depth.shape[0]
                    camera["_depth_w"] = depth.shape[1]
                    u, v, z = _sample_bbox_points(
                        depth,
                        camera,
                        bbox,
                        sample_grid=args.sample_grid,
                        crop_fraction=args.crop_fraction,
                    )
                    if len(z) == 0:
                        continue
                    sampled_items += 1
                    for mode in ("inverse", "direct"):
                        points = _backproject(u, v, z, camera, mode)
                        center_dist = np.linalg.norm(points - location.reshape(1, 3), axis=1)
                        inside = _inside_loose_box(points, location, scale, args.box_margin_m)
                        stats = mode_stats[mode]
                        stats["items"] += 1
                        stats["points"] += int(len(points))
                        stats["items_with_inside_points"] += int(bool(np.any(inside)))
                        stats["inside_points"] += int(np.count_nonzero(inside))
                        stats["min_center_dist_m"].append(float(np.min(center_dist)))
                        stats["median_center_dist_m"].append(float(np.median(center_dist)))
                if sampled_items >= args.max_items:
                    break
            if sampled_items >= args.max_items:
                break
        if sampled_items >= args.max_items:
            break

    out_modes = {}
    for mode, stats in mode_stats.items():
        items = int(stats["items"])
        points = int(stats["points"])
        out_modes[mode] = {
            "items": items,
            "points": points,
            "item_inside_rate": float(stats["items_with_inside_points"] / items) if items else None,
            "point_inside_rate": float(stats["inside_points"] / points) if points else None,
            "min_center_dist_m": _percentiles(stats["min_center_dist_m"]),
            "median_center_dist_m": _percentiles(stats["median_center_dist_m"]),
        }
    return {
        "label": label,
        "depth_root": str(depth_root),
        "split": args.split,
        "frame_stride": args.frame_stride,
        "sampled_items": sampled_items,
        "missing_depth": missing_depth,
        "focal_pred_over_used": _percentiles(focal_ratios),
        "focal_abs_delta_px": _percentiles(focal_abs_delta),
        "modes": out_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DepthPro metric geometry against Track1 GT 3D boxes.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--depth-roots", nargs="+", required=True, help="label=/path or /path")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--frame-stride", type=int, default=300)
    parser.add_argument("--max-frames-per-scene", type=int, default=900)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--sample-grid", type=int, default=7)
    parser.add_argument("--crop-fraction", type=float, default=0.70)
    parser.add_argument("--box-margin-m", type=float, default=0.75)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    reports = [audit_root(args, label, path) for label, path in _parse_depth_roots(args.depth_roots)]
    payload = {"reports": reports}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
