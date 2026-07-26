from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from physicalai_track1.constants import CLASS_TO_ID, GT_KEY_ALIASES
from physicalai_track1.dataset import iter_gt_frames, year_dir
from generate_zio_pcd_2026 import _camera_name, write_binary_ply, write_gt


def _get_any(record: dict, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


def load_camera_params(scene_dir: Path) -> dict[str, dict[str, np.ndarray | tuple[float, float, float, float]]]:
    data = json.loads((scene_dir / "calibration.json").read_text(encoding="utf-8"))
    params = {}
    for sensor in data["sensors"]:
        name = _camera_name(str(sensor["id"]))
        intrinsic = np.asarray(sensor["intrinsicMatrix"], dtype=np.float64).reshape(3, 3)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :4] = np.asarray(sensor["extrinsicMatrix"], dtype=np.float64).reshape(3, 4)
        width = 1920
        height = 1080
        for attr in sensor.get("attributes", []):
            if attr.get("name") == "frameWidth":
                width = int(float(attr.get("value")))
            if attr.get("name") == "frameHeight":
                height = int(float(attr.get("value")))
        params[name] = {
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "width": width,
            "height": height,
        }
    return params


def load_depth(depth_root: Path, split: str, scene_name: str, camera_name: str, frame_id: int) -> np.ndarray | None:
    path = depth_root / split / scene_name / camera_name / f"{frame_id:06d}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["depth_m"].astype(np.float32)


def rgbd_to_world_pcd(
    rgb_by_camera: dict[str, np.ndarray],
    depth_by_camera: dict[str, np.ndarray | None],
    camera_params: dict,
    pixel_stride: int,
    voxel_size: float,
    max_points_before_downsample: int,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    point_chunks = []
    color_chunks = []
    for cam, rgb_bgr_full in rgb_by_camera.items():
        depth_m = depth_by_camera.get(cam)
        if depth_m is None or cam not in camera_params:
            continue
        depth_h, depth_w = depth_m.shape[:2]
        rgb_bgr = cv2.resize(rgb_bgr_full, (depth_w, depth_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        p = camera_params[cam]
        k = np.asarray(p["intrinsic"], dtype=np.float64).copy()
        sx = depth_w / float(p["width"])
        sy = depth_h / float(p["height"])
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])

        ys, xs = np.mgrid[0:depth_h:pixel_stride, 0:depth_w:pixel_stride]
        z = depth_m[ys, xs]
        valid = np.isfinite(z) & (z > depth_min_m) & (z < depth_max_m)
        if not np.any(valid):
            continue
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        x_cam = (xs - cx) * z / fx
        y_cam = (ys - cy) * z / fy
        camera_points = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
        world = (np.linalg.inv(np.asarray(p["extrinsic"], dtype=np.float64)) @ camera_points.T).T[:, :3]
        colors = rgb[ys.astype(np.int32), xs.astype(np.int32)]
        point_chunks.append(world.astype(np.float32))
        color_chunks.append(colors.astype(np.uint8))
    if not point_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    if len(points) > max_points_before_downsample:
        quantized = np.floor(points / float(voxel_size)).astype(np.int64)
        _, keep = np.unique(quantized, axis=0, return_index=True)
        if len(keep) > max_points_before_downsample:
            rng = np.random.default_rng(2026)
            keep = rng.choice(keep, size=max_points_before_downsample, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points, colors


def process_scene(args: argparse.Namespace, scene_dir: Path, split: str) -> dict:
    camera_params = load_camera_params(scene_dir)
    video_dir = scene_dir / "videos"
    camera_names = sorted(_camera_name(path.stem) for path in video_dir.glob("*.mp4"))
    captures = {}
    for cam in camera_names:
        path = video_dir / f"{cam}.mp4"
        if path.exists():
            captures[cam] = cv2.VideoCapture(str(path))
    out_pcd_dir = args.out_dir / split / "pcd"
    out_pcd_dir.mkdir(parents=True, exist_ok=True)
    produced_frames = []
    total_frames = int(min((cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in captures.values()), default=0))
    max_frame = total_frames if args.max_frames is None else min(total_frames, args.max_frames)
    written = 0
    try:
        for frame_id in range(0, max_frame, args.frame_stride):
            rgb_by_camera = {}
            depth_by_camera = {}
            for cam, cap in captures.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb_by_camera[cam] = frame
                depth_by_camera[cam] = load_depth(args.depth_root, split, scene_dir.name, cam, frame_id)
            out_path = out_pcd_dir / f"{scene_dir.name}_{frame_id:05d}.ply"
            if rgb_by_camera and not out_path.exists():
                start = time.time()
                points, colors = rgbd_to_world_pcd(
                    rgb_by_camera,
                    depth_by_camera,
                    camera_params,
                    pixel_stride=args.pixel_stride,
                    voxel_size=args.voxel_size,
                    max_points_before_downsample=args.max_points_before_downsample,
                    depth_min_m=args.depth_min_m,
                    depth_max_m=args.depth_max_m,
                )
                write_binary_ply(out_path, points, colors)
                print(f"WROTE {out_path} points={len(points)} seconds={time.time() - start:.2f}", flush=True)
            produced_frames.append(frame_id)
            written += 1
    finally:
        for cap in captures.values():
            cap.release()
    gt_written = 0
    if split != "test":
        gt_written = write_gt(scene_dir, args.out_dir / split / "gt", produced_frames)
    return {"split": split, "scene": scene_dir.name, "frames": written, "gt": gt_written, "cameras": len(captures)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ZIO-style point clouds from Depth Pro estimated depth.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-points-before-downsample", type=int, default=750_000)
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    scene_filter = set(args.scenes or [])
    root = year_dir(args.data_root, args.year)
    for split in args.splits:
        for scene_dir in sorted(path for path in (root / split).iterdir() if path.is_dir()):
            if scene_filter and scene_dir.name not in scene_filter:
                continue
            summaries.append(process_scene(args, scene_dir, split))
    print(json.dumps({"summaries": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
