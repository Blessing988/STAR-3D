from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from physicalai_track1.calibration import load_scene_cameras
from physicalai_track1.dataset import year_dir
from physicalai_track1.depth_pointcloud import StaticDepthCache, estimate_foreground_depth_lift
from physicalai_track1.detections import Detection2D, read_detections
from physicalai_track1.geometry_residual import _depth_file, _frame_depth_key


def _image_path(image_root: Path, det: Detection2D) -> Path:
    return image_root / f"{det.scene_name}_{det.camera_id}_{det.frame_id:06d}.jpg"


def _draw_rgb(image: Image.Image, det: Detection2D, depth_xy: tuple[float, float] | None) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=(255, 80, 40), width=4)
    bottom = det.bottom_center
    draw.ellipse([bottom[0] - 5, bottom[1] - 5, bottom[0] + 5, bottom[1] + 5], fill=(60, 220, 80))
    if depth_xy is not None:
        draw.ellipse([depth_xy[0] - 6, depth_xy[1] - 6, depth_xy[0] + 6, depth_xy[1] + 6], fill=(80, 140, 255))
    return out


def _depth_vis(depth: np.ndarray, det: Detection2D) -> np.ndarray:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0, 1
    norm = np.clip((depth.astype(np.float32) - lo) / max(1.0, hi - lo), 0, 1)
    rgb = (plt.cm.viridis(norm)[..., :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=(255, 255, 255), width=4)
    return np.asarray(image)


def visualize(
    data_root: Path,
    image_root: Path,
    detections: Path,
    out_dir: Path,
    year: int,
    split: str,
    max_samples: int,
    depth_scale: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_root = year_dir(data_root, year) / split
    camera_cache = {}
    depth_handles = {}
    static_cache = StaticDepthCache(max_cameras=2, max_samples=24, sample_stride=60)
    count = 0
    try:
        for det in read_detections(detections):
            image_path = _image_path(image_root, det)
            if not image_path.exists():
                continue
            if det.scene_name not in camera_cache:
                camera_cache[det.scene_name] = load_scene_cameras(scene_root / det.scene_name)
            camera = camera_cache[det.scene_name].get(det.camera_id)
            if camera is None:
                continue
            depth_path = _depth_file(scene_root / det.scene_name, det.camera_id)
            if depth_path not in depth_handles:
                depth_handles[depth_path] = h5py.File(depth_path, "r") if depth_path.exists() else None
            handle = depth_handles[depth_path]
            key = _frame_depth_key(det.frame_id)
            if handle is None or key not in handle:
                continue
            depth = handle[key]
            background = static_cache.get(depth_path, handle)
            result = estimate_foreground_depth_lift(
                det,
                camera,
                depth,
                background,
                frame_detections=(),
                depth_scale=depth_scale,
                max_reprojection_px=99999.0,
            )
            bottom = det.bottom_center
            hom_x, hom_y = camera.image_to_ground(*bottom)
            depth_xy = None
            projected_depth = None
            if result is not None:
                projected_depth = camera.ground_to_image(result.x, result.y)
                depth_xy = projected_depth
            image = Image.open(image_path).convert("RGB")
            rgb_overlay = _draw_rgb(image, det, depth_xy)
            depth_overlay = _depth_vis(np.asarray(depth), det)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(rgb_overlay)
            axes[0].set_title("RGB bbox: green=bottom-center, blue=depth reproj")
            axes[0].axis("off")
            axes[1].imshow(depth_overlay)
            axes[1].set_title("Depth map + bbox")
            axes[1].axis("off")
            axes[2].scatter([hom_x], [hom_y], c="lime", label="homography", s=80)
            if result is not None:
                axes[2].scatter([result.x], [result.y], c="dodgerblue", label="depth", s=80)
                axes[2].set_title(
                    f"BEV depth pixels={result.pixel_count} reproj={result.reprojection_error:.1f}px"
                )
            else:
                axes[2].set_title("BEV depth failed")
            axes[2].set_aspect("equal", adjustable="box")
            axes[2].grid(True)
            axes[2].legend()
            axes[2].invert_yaxis()
            fig.suptitle(
                f"{det.scene_name} {det.camera_id} frame={det.frame_id} class={det.class_id} score={det.score:.2f}"
            )
            fig.tight_layout()
            out = out_dir / f"{count:03d}_{det.scene_name}_{det.camera_id}_{det.frame_id:06d}_c{det.class_id}.png"
            fig.savefig(out, dpi=130)
            plt.close(fig)
            count += 1
            if count >= max_samples:
                break
    finally:
        for handle in depth_handles.values():
            if handle is not None:
                handle.close()
    print({"outputs": count, "out_dir": str(out_dir)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    args = parser.parse_args()
    visualize(**vars(args))


if __name__ == "__main__":
    main()
