from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .constants import ID_TO_CLASS


def _iter_images_from_split(dataset_dir: Path, split: str) -> list[Path]:
    return sorted((dataset_dir / "images" / split).glob("*.jpg"))


def _iter_images_from_list(list_path: Path) -> list[Path]:
    images: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            images.append(Path(stripped))
    return images


def _read_yolo_label(label_path: Path, frame_w: int, frame_h: int) -> list[tuple[int, list[float], float]]:
    anns: list[tuple[int, list[float], float]] = []
    if not label_path.exists():
        return anns
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(value) for value in parts[1:5]]
        width = bw * frame_w
        height = bh * frame_h
        x = (cx * frame_w) - width * 0.5
        y = (cy * frame_h) - height * 0.5
        x = max(0.0, min(x, frame_w - 1.0))
        y = max(0.0, min(y, frame_h - 1.0))
        width = max(0.0, min(width, frame_w - x))
        height = max(0.0, min(height, frame_h - y))
        if width <= 0.0 or height <= 0.0:
            continue
        anns.append((class_id + 1, [x, y, width, height], width * height))
    return anns


def export_coco_from_yolo(
    dataset_dir: Path | str,
    split: str,
    out_json: Path | str,
    image_list: Path | str | None = None,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> dict:
    """Export YOLO labels/images to COCO detection JSON.

    If ``image_list`` is provided, every row becomes an image entry. This allows
    repeated class-balanced training lists while keeping the same JPEG files on
    disk. Validation exports should usually omit ``image_list``.
    """
    dataset = Path(dataset_dir)
    out_path = Path(out_json)
    images = _iter_images_from_list(Path(image_list)) if image_list else _iter_images_from_split(dataset, split)
    labels_root = dataset / "labels" / split

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    class_counts: Counter[int] = Counter()
    image_rows_with_labels = 0
    ann_id = 1

    for image_id, image_path in enumerate(images, start=1):
        stem = image_path.stem
        label_path = labels_root / f"{stem}.txt"
        anns = _read_yolo_label(label_path, frame_width, frame_height)
        if anns:
            image_rows_with_labels += 1
        coco_images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": frame_width,
                "height": frame_height,
            }
        )
        for category_id, bbox, area in anns:
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(value, 4) for value in bbox],
                    "area": round(area, 4),
                    "iscrowd": 0,
                }
            )
            class_counts[category_id - 1] += 1
            ann_id += 1

    categories = [
        {"id": class_id + 1, "name": name, "supercategory": "object"}
        for class_id, name in sorted(ID_TO_CLASS.items())
    ]
    payload = {
        "info": {"description": "AI City 2026 Track 1 YOLO-to-COCO export"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    return {
        "dataset_dir": str(dataset),
        "split": split,
        "image_list": str(image_list) if image_list else None,
        "out_json": str(out_path),
        "images": len(coco_images),
        "images_with_labels": image_rows_with_labels,
        "annotations": len(coco_annotations),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "instances_by_class": {
            ID_TO_CLASS[class_id]: class_counts.get(class_id, 0) for class_id in sorted(ID_TO_CLASS)
        },
    }
