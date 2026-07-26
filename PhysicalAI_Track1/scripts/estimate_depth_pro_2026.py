from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch


def scene_id(scene_name: str) -> int:
    return int(scene_name.rsplit("_", 1)[-1])


def camera_name(video_path: Path) -> str:
    stem = video_path.stem
    if stem.startswith("Camera_"):
        return stem
    if stem.startswith("camera_"):
        return "Camera_" + stem.rsplit("_", 1)[-1]
    return stem


def _sensor_attrs(sensor: dict) -> dict[str, str]:
    return {str(attr.get("name")): str(attr.get("value")) for attr in sensor.get("attributes", [])}


def _float_attr(attrs: dict[str, str], key: str, default: float) -> float:
    value = attrs.get(key)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() in {"none", "null", "nan"}:
        return default
    return float(value)


def load_camera_focals(scene: Path) -> dict[str, dict[str, float]]:
    path = scene / "calibration.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    focals: dict[str, dict[str, float]] = {}
    for sensor in data.get("sensors", []):
        if sensor.get("type") != "camera":
            continue
        sensor_id = str(sensor.get("id"))
        cam = sensor_id if sensor_id.startswith("Camera_") else f"Camera_{sensor_id}"
        intrinsic = np.asarray(sensor["intrinsicMatrix"], dtype=np.float64).reshape(3, 3)
        attrs = _sensor_attrs(sensor)
        frame_width = _float_attr(attrs, "frameWidth", 1920.0)
        frame_height = _float_attr(attrs, "frameHeight", 1080.0)
        focals[cam] = {
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "frame_width": float(frame_width),
            "frame_height": float(frame_height),
        }
    return focals


def scaled_focal_px(camera_focal: dict[str, float], image_w: int, image_h: int) -> float:
    sx = float(image_w) / float(camera_focal["frame_width"])
    sy = float(image_h) / float(camera_focal["frame_height"])
    fx = float(camera_focal["fx"]) * sx
    fy = float(camera_focal["fy"]) * sy
    return 0.5 * (fx + fy)


def iter_scenes(dataset_root: Path, split: str, scenes: list[str] | None) -> Iterator[Path]:
    split_root = dataset_root / "MTMC_Tracking_2026" / split
    if not split_root.exists():
        raise FileNotFoundError(split_root)
    requested = set(scenes or [])
    for path in sorted(split_root.iterdir()):
        if not path.is_dir():
            continue
        if requested and path.name not in requested and str(scene_id(path.name)) not in requested:
            continue
        yield path


def load_depth_pro(repo_root: Path | None):
    if repo_root is not None:
        import sys

        sys.path.insert(0, str(repo_root / "src"))
        sys.path.insert(0, str(repo_root))
    import depth_pro

    old_cwd = Path.cwd()
    if repo_root is not None:
        os.chdir(repo_root)
    try:
        model, transform = depth_pro.create_model_and_transforms()
    finally:
        os.chdir(old_cwd)
    model.eval().cuda()
    return depth_pro, model, transform


def infer_frame(depth_pro, model, transform, frame_bgr: np.ndarray, f_px: float | None = None) -> tuple[np.ndarray, float]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = depth_pro.load_rgb_from_array(frame_rgb)[0] if hasattr(depth_pro, "load_rgb_from_array") else None
    if image is None:
        from PIL import Image

        image = Image.fromarray(frame_rgb)
    image = transform(image)
    image_cuda = image.cuda()
    f_px_tensor = None if f_px is None else torch.as_tensor(float(f_px), dtype=image_cuda.dtype, device=image_cuda.device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        pred = model.infer(image_cuda, f_px=f_px_tensor)
    depth = pred["depth"].detach().float().cpu().numpy()
    focal = pred.get("focallength_px")
    if focal is None:
        focal_px = float("nan")
    elif hasattr(focal, "detach"):
        focal_px = float(focal.detach().float().cpu().item())
    else:
        focal_px = float(focal)
    return depth, focal_px


def write_npz(
    out_path: Path,
    depth_m: np.ndarray,
    focal_px: float,
    focal_px_pred: float,
    focal_px_used: float,
    source_width: int,
    source_height: int,
    image_width: int,
    image_height: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
    np.savez_compressed(
        out_path,
        depth_m=depth,
        focal_px=np.float32(focal_px),
        focal_px_pred=np.float32(focal_px_pred),
        focal_px_used=np.float32(focal_px_used),
        source_width=np.int32(source_width),
        source_height=np.int32(source_height),
        image_width=np.int32(image_width),
        image_height=np.int32(image_height),
    )


def process_video(args, depth_pro, model, transform, split: str, scene: Path, video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = total if args.max_frames is None else min(total, args.max_frames)
    cam = camera_name(video)
    camera_focals = load_camera_focals(scene) if args.use_calibration_focal else {}
    camera_focal = camera_focals.get(cam)
    if args.use_calibration_focal and camera_focal is None:
        raise KeyError(f"No camera calibration focal for {scene.name}/{cam}")
    out_dir = args.out_dir / split / scene.name / cam
    done = 0
    skipped = 0
    for frame_id in range(0, max_frame, args.frame_stride):
        out_file = out_dir / f"{frame_id:06d}.npz"
        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        source_h, source_w = frame.shape[:2]
        if args.resize_max_side and max(frame.shape[:2]) > args.resize_max_side:
            h, w = frame.shape[:2]
            scale = args.resize_max_side / float(max(h, w))
            frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        image_h, image_w = frame.shape[:2]
        focal_px_used = scaled_focal_px(camera_focal, image_w, image_h) if camera_focal is not None else float("nan")
        depth, focal_px_pred = infer_frame(
            depth_pro,
            model,
            transform,
            frame,
            f_px=None if camera_focal is None else focal_px_used,
        )
        focal_px = focal_px_used if camera_focal is not None else focal_px_pred
        write_npz(
            out_file,
            depth,
            focal_px=focal_px,
            focal_px_pred=focal_px_pred,
            focal_px_used=focal_px_used,
            source_width=source_w,
            source_height=source_h,
            image_width=image_w,
            image_height=image_h,
        )
        done += 1
    cap.release()
    return {
        "split": split,
        "scene": scene.name,
        "camera": cam,
        "video": str(video),
        "frames_total": total,
        "frames_limit": max_frame,
        "stride": args.frame_stride,
        "written": done,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate metric depth from RGB videos with Apple Depth Pro.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth-pro-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names or numeric ids to include.")
    parser.add_argument("--frame-stride", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resize-max-side", type=int, default=1536)
    parser.add_argument("--use-calibration-focal", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    depth_pro, model, transform = load_depth_pro(args.depth_pro_root)
    reports = []
    for split in args.splits:
        for scene in iter_scenes(args.dataset_root, split, args.scenes):
            video_dir = scene / "videos"
            for video in sorted(video_dir.glob("*.mp4")):
                report = process_video(args, depth_pro, model, transform, split, scene, video)
                print(json.dumps(report), flush=True)
                reports.append(report)
    manifest = args.manifest or (args.out_dir / "depth_pro_manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
