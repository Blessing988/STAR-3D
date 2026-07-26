from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constants import CLASS_TO_ID, GT_KEY_ALIASES
from .dataset import iter_gt_frames, iter_scene_dirs, load_calibration


def _get_any(record: Mapping, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


def _sensor_frame_size(sensor: Mapping) -> Tuple[int, int]:
    width = None
    height = None
    for attr in sensor.get("attributes", []):
        if attr.get("name") == "frameWidth":
            width = int(attr.get("value"))
        elif attr.get("name") == "frameHeight":
            height = int(attr.get("value"))
    if width is None or height is None:
        width, height = 1920, 1080
    return width, height


def camera_sizes(scene_dir: Path) -> Dict[str, Tuple[int, int]]:
    cal = load_calibration(scene_dir)
    sizes = {}
    for sensor in cal.get("sensors", []):
        if sensor.get("type") == "camera":
            sizes[str(sensor["id"])] = _sensor_frame_size(sensor)
    return sizes


def _xyxy_to_yolo(xyxy: Sequence[float], frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = min(max(x1, 0.0), frame_w - 1.0)
    x2 = min(max(x2, 0.0), frame_w - 1.0)
    y1 = min(max(y1, 0.0), frame_h - 1.0)
    y2 = min(max(y2, 0.0), frame_h - 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw * 0.5
    cy = y1 + bh * 0.5
    return cx / frame_w, cy / frame_h, bw / frame_w, bh / frame_h


def _frame_image_name(scene: str, camera: str, frame_id: int) -> str:
    return f"{scene}_{camera}_{frame_id:06d}.jpg"


def export_yolo_labels(
    data_root: Path | str,
    year: int,
    split: str,
    output_dir: Path | str,
    scenes: Optional[Sequence[str]] = None,
    frame_stride: int = 30,
    max_frames_per_scene: Optional[int] = None,
    min_box_area: float = 16.0,
) -> dict:
    """Export YOLO labels from visible 2D boxes.

    This writes labels and manifests only. Use the manifest to extract matching
    video frames with ffmpeg/OpenCV in a separate job.
    """
    output = Path(output_dir)
    labels_root = output / "labels" / split
    images_root = output / "images" / split
    labels_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    scene_filter = set(scenes or [])
    manifest_rows: List[str] = []
    label_count = 0
    image_count = 0
    boxes_by_class = Counter()

    for scene_dir in iter_scene_dirs(data_root, year, split):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        gt_path = scene_dir / "ground_truth.json"
        if not gt_path.exists():
            continue
        sizes = camera_sizes(scene_dir)
        camera_frame_labels: Dict[Tuple[str, int], List[str]] = defaultdict(list)

        for frame_key, frame_items in iter_gt_frames(scene_dir, max_frames=max_frames_per_scene):
            frame_id = int(frame_key)
            if frame_stride > 1 and frame_id % frame_stride != 0:
                continue
            for item in frame_items:
                class_name = _get_any(item, "object_type")
                class_id = CLASS_TO_ID[class_name]
                visible = _get_any(item, "bbox2d")
                for camera_id, xyxy in visible.items():
                    if camera_id not in sizes:
                        continue
                    frame_w, frame_h = sizes[camera_id]
                    x1, y1, x2, y2 = map(float, xyxy)
                    if max(0.0, x2 - x1) * max(0.0, y2 - y1) < min_box_area:
                        continue
                    cx, cy, bw, bh = _xyxy_to_yolo(xyxy, frame_w, frame_h)
                    if bw <= 0.0 or bh <= 0.0:
                        continue
                    camera_frame_labels[(camera_id, frame_id)].append(
                        f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
                    )
                    boxes_by_class[class_name] += 1
                    label_count += 1

        for (camera_id, frame_id), lines in sorted(camera_frame_labels.items()):
            stem = _frame_image_name(scene_dir.name, camera_id, frame_id).replace(".jpg", "")
            label_path = labels_root / f"{stem}.txt"
            image_path = images_root / f"{stem}.jpg"
            video_path = scene_dir / "videos" / f"{camera_id}.mp4"
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest_rows.append(f"{video_path}\t{frame_id}\t{image_path}\t{label_path}")
            image_count += 1

    manifest = output / f"{split}_frames.tsv"
    manifest.write_text(
        "video_path\tframe_id\timage_path\tlabel_path\n" + "\n".join(manifest_rows) + "\n",
        encoding="utf-8",
    )
    data_yaml = output / f"aicity_track1_{year}.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output}",
                f"train: images/train",
                f"val: images/val",
                "names:",
                *[f"  {idx}: {name}" for name, idx in sorted(CLASS_TO_ID.items(), key=lambda kv: kv[1])],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output),
        "split": split,
        "images_referenced": image_count,
        "labels_written": image_count,
        "boxes": label_count,
        "boxes_by_class": dict(sorted(boxes_by_class.items())),
        "manifest": str(manifest),
        "data_yaml": str(data_yaml),
        "note": "Frames are not extracted by this command; use the TSV manifest for an ffmpeg/OpenCV extraction job.",
    }
