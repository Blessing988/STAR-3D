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


def load_camera_params(scene: Path) -> dict[str, dict[str, np.ndarray | float]]:
    data = json.loads((scene / "calibration.json").read_text(encoding="utf-8"))
    params: dict[str, dict[str, np.ndarray | float]] = {}
    for sensor in data.get("sensors", []):
        if sensor.get("type") != "camera":
            continue
        sensor_id = str(sensor.get("id"))
        cam = sensor_id if sensor_id.startswith("Camera_") else f"Camera_{sensor_id}"
        attrs = _sensor_attrs(sensor)
        params[cam] = {
            "intrinsic": np.asarray(sensor["intrinsicMatrix"], dtype=np.float32).reshape(3, 3),
            "frame_width": _float_attr(attrs, "frameWidth", 1920.0),
            "frame_height": _float_attr(attrs, "frameHeight", 1080.0),
        }
    return params


def scaled_intrinsic(camera: dict[str, np.ndarray | float], image_w: int, image_h: int) -> np.ndarray:
    k = np.asarray(camera["intrinsic"], dtype=np.float32).copy()
    sx = float(image_w) / float(camera["frame_width"])
    sy = float(image_h) / float(camera["frame_height"])
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


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


def write_npz(
    out_path: Path,
    depth_m: np.ndarray,
    model_name: str,
    intrinsic_scaled: np.ndarray,
    source_width: int,
    source_height: int,
    image_width: int,
    image_height: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
    focal_px = float(0.5 * (intrinsic_scaled[0, 0] + intrinsic_scaled[1, 1]))
    np.savez_compressed(
        out_path,
        depth_m=depth,
        model_name=np.asarray(model_name),
        focal_px=np.float32(focal_px),
        focal_px_used=np.float32(focal_px),
        focal_px_pred=np.float32(focal_px),
        intrinsic_scaled=intrinsic_scaled.astype(np.float32),
        source_width=np.int32(source_width),
        source_height=np.int32(source_height),
        image_width=np.int32(image_width),
        image_height=np.int32(image_height),
    )


class UniDepthV2Runner:
    def __init__(self, model_id: str, repo_root: Path | None = None):
        if repo_root is not None:
            import sys

            sys.path.insert(0, str(repo_root))
        from unidepth.models import UniDepthV2
        from unidepth.utils.camera import Pinhole

        self.Pinhole = Pinhole
        self.model = UniDepthV2.from_pretrained(model_id).cuda().eval()

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, intrinsic_scaled: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(frame_rgb).permute(2, 0, 1).cuda()
        k = torch.from_numpy(intrinsic_scaled).float().cuda()
        camera = self.Pinhole(K=k)
        pred = self.model.infer(rgb, camera)
        depth = pred["depth"]
        if isinstance(depth, (list, tuple)):
            depth = depth[0]
        depth_np = depth.detach().float().squeeze().cpu().numpy()
        return depth_np


class Metric3DRunner:
    def __init__(self, repo_or_dir: str, hub_model: str):
        self.model = torch.hub.load(repo_or_dir, hub_model, pretrain=True, source="local" if Path(repo_or_dir).exists() else "github")
        self.model.cuda().eval()
        self.input_size = (616, 1064) if "vit" in hub_model else (544, 1216)

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, intrinsic_scaled: np.ndarray) -> np.ndarray:
        rgb_origin = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb_origin.shape[:2]
        input_h, input_w = self.input_size
        scale = min(input_h / h0, input_w / w0)
        rgb = cv2.resize(rgb_origin, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
        k = intrinsic_scaled.copy()
        k[0, :] *= scale
        k[1, :] *= scale

        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_h - h
        pad_w = input_w - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(
            rgb,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding,
        )
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        tensor = torch.div((tensor - mean), std)[None].cuda()
        pred_depth, _confidence, _output_dict = self.model.inference({"input": tensor})
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_h_half : pred_depth.shape[0] - (pad_h - pad_h_half),
            pad_w_half : pred_depth.shape[1] - (pad_w - pad_w_half),
        ]
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None], (h0, w0), mode="bilinear", align_corners=False
        ).squeeze()
        canonical_to_real_scale = float(k[0, 0]) / 1000.0
        pred_depth = torch.clamp(pred_depth * canonical_to_real_scale, 0.0, 300.0)
        return pred_depth.detach().float().cpu().numpy()


def load_runner(args: argparse.Namespace):
    if args.model == "unidepth_v2":
        return UniDepthV2Runner(args.model_id, args.repo_root)
    if args.model == "metric3d":
        return Metric3DRunner(str(args.metric3d_repo), args.hub_model)
    raise ValueError(args.model)


def process_video(args, runner, split: str, scene: Path, video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = total if args.max_frames is None else min(total, args.max_frames)
    cam = camera_name(video)
    camera_params = load_camera_params(scene)
    if cam not in camera_params:
        raise KeyError(f"No camera calibration for {scene.name}/{cam}")
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
        k = scaled_intrinsic(camera_params[cam], image_w, image_h)
        depth = runner.infer(frame, k)
        if depth.shape[:2] != (image_h, image_w):
            depth = cv2.resize(depth.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        write_npz(
            out_file,
            depth,
            model_name=f"{args.model}:{args.model_id if args.model == 'unidepth_v2' else args.hub_model}",
            intrinsic_scaled=k,
            source_width=source_w,
            source_height=source_h,
            image_width=image_w,
            image_height=image_h,
        )
        done += 1
    cap.release()
    return {
        "model": args.model,
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
    parser = argparse.ArgumentParser(description="Estimate metric depth from RGB with UniDepthV2 or Metric3D.")
    parser.add_argument("--model", choices=["unidepth_v2", "metric3d"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--model-id", default="lpiccinelli/unidepth-v2-vitl14")
    parser.add_argument("--metric3d-repo", default="yvanyin/metric3d")
    parser.add_argument("--hub-model", default="metric3d_vit_large")
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names or numeric ids to include.")
    parser.add_argument("--frame-stride", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resize-max-side", type=int, default=1536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = load_runner(args)
    reports = []
    for split in args.splits:
        for scene in iter_scenes(args.dataset_root, split, args.scenes):
            video_dir = scene / "videos"
            for video in sorted(video_dir.glob("*.mp4")):
                report = process_video(args, runner, split, scene, video)
                print(json.dumps(report), flush=True)
                reports.append(report)
    manifest = args.manifest or (args.out_dir / f"{args.model}_manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
