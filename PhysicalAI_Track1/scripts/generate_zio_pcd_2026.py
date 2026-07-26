from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

from physicalai_track1.constants import CLASS_TO_ID, GT_KEY_ALIASES
from physicalai_track1.dataset import iter_gt_frames, year_dir


def _get_any(record: dict, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


def _camera_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("Camera_"):
        try:
            return f"Camera_{int(stem.split('_')[1]):04d}"
        except Exception:
            return stem
    try:
        return f"Camera_{int(stem):04d}"
    except Exception:
        return stem


def load_camera_params(scene_dir: Path) -> dict[str, dict[str, np.ndarray | tuple[float, float, float, float]]]:
    data = json.loads((scene_dir / "calibration.json").read_text(encoding="utf-8"))
    params = {}
    for sensor in data["sensors"]:
        name = _camera_name(str(sensor["id"]))
        intrinsic = np.asarray(sensor["intrinsicMatrix"], dtype=np.float64).reshape(3, 3)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :4] = np.asarray(sensor["extrinsicMatrix"], dtype=np.float64).reshape(3, 4)
        params[name] = {
            "intrinsic": (intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]),
            "extrinsic": extrinsic,
        }
    return params


def rgbd_to_world_pcd(
    rgb_by_camera: dict[str, np.ndarray],
    depth_by_camera: dict[str, np.ndarray | None],
    camera_params: dict,
    pixel_stride: int,
    voxel_size: float,
    max_points_before_downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    point_chunks = []
    color_chunks = []
    for camera_name, rgb_bgr in rgb_by_camera.items():
        depth = depth_by_camera.get(camera_name)
        if depth is None or camera_name not in camera_params:
            continue
        depth_m = np.asarray(depth, dtype=np.float32) / 1000.0
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        fx, fy, cx, cy = camera_params[camera_name]["intrinsic"]
        ys, xs = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
        z = depth_m[ys, xs]
        valid = np.isfinite(z) & (z > 0.05) & (z < 100.0)
        if not np.any(valid):
            continue
        xs = xs[valid].astype(np.float32)
        ys = ys[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        x_cam = (xs - float(cx)) * z / float(fx)
        y_cam = (ys - float(cy)) * z / float(fy)
        camera_points = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=1)
        world = (np.linalg.inv(camera_params[camera_name]["extrinsic"]) @ camera_points.T).T[:, :3]
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


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    payload = np.empty(len(points), dtype=dtype)
    payload["x"] = points[:, 0]
    payload["y"] = points[:, 1]
    payload["z"] = points[:, 2]
    payload["red"] = colors[:, 0]
    payload["green"] = colors[:, 1]
    payload["blue"] = colors[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        payload.tofile(handle)


def write_gt(scene_dir: Path, out_gt_dir: Path, frame_ids: list[int]) -> int:
    if not (scene_dir / "ground_truth.json").exists():
        return 0
    frames = {int(frame_key): items for frame_key, items in iter_gt_frames(scene_dir)}
    written = 0
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    for frame_id in frame_ids:
        items = frames.get(frame_id, [])
        lines = []
        for item in items:
            class_id = CLASS_TO_ID[_get_any(item, "object_type")]
            object_id = int(_get_any(item, "object_id"))
            location = list(map(float, _get_any(item, "location")))
            scale = list(map(float, _get_any(item, "scale")))
            rotation = list(map(float, _get_any(item, "rotation")))
            lines.append(
                f"{class_id} {object_id} {location[0]} {location[1]} {location[2]} "
                f"{scale[0]} {scale[1]} {scale[2]} {rotation[0]} {rotation[1]} {rotation[2]}"
            )
        (out_gt_dir / f"{scene_dir.name}_{frame_id:05d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    return written


def process_scene(args: argparse.Namespace, scene_dir: Path, split: str) -> dict:
    camera_params = load_camera_params(scene_dir)
    video_dir = scene_dir / "videos"
    depth_dir = scene_dir / "depth_maps"
    camera_names = sorted(_camera_name(path.stem) for path in video_dir.glob("*.mp4"))
    captures = {}
    depths = {}
    for camera_name in camera_names:
        video_path = video_dir / f"{camera_name}.mp4"
        depth_path = depth_dir / f"{camera_name}.h5"
        if not video_path.exists():
            continue
        captures[camera_name] = cv2.VideoCapture(str(video_path))
        if depth_path.exists():
            depths[camera_name] = h5py.File(depth_path, "r")

    out_pcd_dir = args.out_dir / split / "pcd"
    out_pcd_dir.mkdir(parents=True, exist_ok=True)
    produced_frames = []
    total_frames = int(min((cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in captures.values()), default=0))
    max_frame = total_frames if args.max_frames is None else min(total_frames, args.max_frames)
    frame_id = 0
    written = 0
    try:
        while frame_id < max_frame:
            rgb_by_camera = {}
            depth_by_camera = {}
            for camera_name, cap in captures.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb_by_camera[camera_name] = frame
                key = f"distance_to_image_plane_{frame_id:05d}.png"
                handle = depths.get(camera_name)
                depth_by_camera[camera_name] = handle[key][:] if handle is not None and key in handle else None

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
                )
                write_binary_ply(out_path, points, colors)
                print(f"WROTE {out_path} points={len(points)} seconds={time.time() - start:.2f}", flush=True)
            produced_frames.append(frame_id)
            written += 1
            frame_id += args.frame_stride
    finally:
        for cap in captures.values():
            cap.release()
        for handle in depths.values():
            handle.close()

    gt_written = 0
    if split != "test":
        gt_written = write_gt(scene_dir, args.out_dir / split / "gt", produced_frames)
    return {"scene": scene_dir.name, "frames": written, "gt": gt_written, "cameras": len(captures)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ZIO-style fused RGB-D point clouds for Track1 2026.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--max-points-before-downsample", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    scene_filter = set(args.scenes or [])
    root = year_dir(args.data_root, args.year)
    for split in args.splits:
        split_dir = root / split
        for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if scene_filter and scene_dir.name not in scene_filter:
                continue
            summaries.append(process_scene(args, scene_dir, split))
    print(json.dumps({"summaries": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
