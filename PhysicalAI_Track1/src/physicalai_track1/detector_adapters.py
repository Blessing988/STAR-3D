from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from .constants import ID_TO_CLASS
from .detections import Detection2D, write_detections


@dataclass(frozen=True)
class FrameMetadata:
    scene_name: str
    camera_id: str
    frame_id: int
    image_name: str
    width: int
    height: int


def _portable_path(path: str) -> PurePosixPath:
    return PurePosixPath(path.replace("\\", "/"))


def read_frame_manifest(
    manifest_path: Path | str,
    frame_width: int = 1920,
    frame_height: int = 1080,
) -> dict[str, FrameMetadata]:
    frames: dict[str, FrameMetadata] = {}
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"video_path", "frame_id", "image_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{manifest_path}: expected manifest columns {sorted(required)}")
        for row in reader:
            video_path = _portable_path(row["video_path"])
            image_path = _portable_path(row["image_path"])
            if len(video_path.parts) < 3:
                raise ValueError(f"{manifest_path}: cannot infer scene from {row['video_path']!r}")
            metadata = FrameMetadata(
                scene_name=video_path.parent.parent.name,
                camera_id=video_path.stem,
                frame_id=int(row["frame_id"]),
                image_name=image_path.name,
                width=int(frame_width),
                height=int(frame_height),
            )
            if metadata.image_name in frames and frames[metadata.image_name] != metadata:
                raise ValueError(f"{manifest_path}: duplicate image name {metadata.image_name!r}")
            frames[metadata.image_name] = metadata
    return frames


def _xywhn_to_xyxy(
    cx: float,
    cy: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    box_width = width * frame_width
    box_height = height * frame_height
    x1 = (cx * frame_width) - box_width * 0.5
    y1 = (cy * frame_height) - box_height * 0.5
    x2 = x1 + box_width
    y2 = y1 + box_height
    return _clip_xyxy(x1, y1, x2, y2, frame_width, frame_height)


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float]:
    x1 = min(max(float(x1), 0.0), float(frame_width))
    x2 = min(max(float(x2), 0.0), float(frame_width))
    y1 = min(max(float(y1), 0.0), float(frame_height))
    y2 = min(max(float(y2), 0.0), float(frame_height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou2d(a: Detection2D, b: Detection2D) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


def classwise_nms(detections: Iterable[Detection2D], iou_threshold: float) -> list[Detection2D]:
    groups: dict[tuple[str, str, int, int], list[Detection2D]] = defaultdict(list)
    for det in detections:
        groups[(det.scene_name, det.camera_id, det.frame_id, det.class_id)].append(det)

    kept: list[Detection2D] = []
    for key in sorted(groups):
        selected: list[Detection2D] = []
        for det in sorted(groups[key], key=lambda item: item.score, reverse=True):
            if all(_iou2d(det, previous) < iou_threshold for previous in selected):
                selected.append(det)
        kept.extend(selected)
    return kept


def _frame_selected(
    metadata: FrameMetadata,
    scenes: Sequence[str] | None,
    max_frame_id: int | None,
) -> bool:
    if scenes and metadata.scene_name not in set(scenes):
        return False
    return max_frame_id is None or metadata.frame_id <= max_frame_id


def import_yolo_predictions(
    manifest_path: Path | str,
    labels_dir: Path | str,
    out_path: Path | str,
    frame_width: int = 1920,
    frame_height: int = 1080,
    min_score: float = 0.01,
    nms_iou: float = 0.70,
    scenes: Sequence[str] | None = None,
    max_frame_id: int | None = None,
) -> dict:
    frames = read_frame_manifest(manifest_path, frame_width=frame_width, frame_height=frame_height)
    raw: list[Detection2D] = []
    matched_files = 0
    unmatched_files = 0

    for label_path in sorted(Path(labels_dir).rglob("*.txt")):
        metadata = frames.get(f"{label_path.stem}.jpg")
        if metadata is None:
            unmatched_files += 1
            continue
        if not _frame_selected(metadata, scenes, max_frame_id):
            continue
        matched_files += 1
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"{label_path}:{line_no}: expected class cx cy width height [score]")
            class_id = int(float(parts[0]))
            if class_id not in ID_TO_CLASS:
                continue
            cx, cy, width, height = map(float, parts[1:5])
            score = float(parts[5]) if len(parts) > 5 else 1.0
            if score < min_score:
                continue
            x1, y1, x2, y2 = _xywhn_to_xyxy(
                cx,
                cy,
                width,
                height,
                metadata.width,
                metadata.height,
            )
            if x2 <= x1 or y2 <= y1:
                continue
            raw.append(
                Detection2D(
                    scene_name=metadata.scene_name,
                    camera_id=metadata.camera_id,
                    frame_id=metadata.frame_id,
                    class_id=class_id,
                    score=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    kept = classwise_nms(raw, nms_iou)
    count = write_detections(kept, out_path)
    return {
        "format": "yolo",
        "manifest_frames": len(frames),
        "matched_prediction_files": matched_files,
        "unmatched_prediction_files": unmatched_files,
        "raw_detections": len(raw),
        "detections": count,
        "min_score": min_score,
        "nms_iou": nms_iou,
        "output": str(out_path),
    }


def _prediction_rows(payload: object) -> list[Mapping]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "predictions", "annotations"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
    raise ValueError("COCO predictions must be a list or contain results/predictions/annotations")


def import_coco_predictions(
    predictions_path: Path | str,
    annotations_path: Path | str,
    manifest_path: Path | str,
    out_path: Path | str,
    category_offset: int = 1,
    min_score: float = 0.01,
    nms_iou: float = 0.70,
    frame_width: int = 1920,
    frame_height: int = 1080,
    scenes: Sequence[str] | None = None,
    max_frame_id: int | None = None,
) -> dict:
    frames = read_frame_manifest(manifest_path, frame_width=frame_width, frame_height=frame_height)
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    image_metadata: dict[int, tuple[FrameMetadata, int, int]] = {}
    for image in annotations.get("images", []):
        metadata = frames.get(_portable_path(str(image["file_name"])).name)
        if metadata is None or not _frame_selected(metadata, scenes, max_frame_id):
            continue
        image_metadata[int(image["id"])] = (
            metadata,
            int(image.get("width", metadata.width)),
            int(image.get("height", metadata.height)),
        )

    payload = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    rows = _prediction_rows(payload)
    raw: list[Detection2D] = []
    skipped_images = 0
    for row in rows:
        image = image_metadata.get(int(row["image_id"]))
        if image is None:
            skipped_images += 1
            continue
        metadata, width, height = image
        score = float(row.get("score", 1.0))
        if score < min_score:
            continue
        class_id = int(row["category_id"]) - category_offset
        if class_id not in ID_TO_CLASS:
            continue
        x, y, box_width, box_height = map(float, row["bbox"])
        x1, y1, x2, y2 = _clip_xyxy(x, y, x + box_width, y + box_height, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        raw.append(
            Detection2D(
                scene_name=metadata.scene_name,
                camera_id=metadata.camera_id,
                frame_id=metadata.frame_id,
                class_id=class_id,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    kept = classwise_nms(raw, nms_iou)
    count = write_detections(kept, out_path)
    return {
        "format": "coco",
        "manifest_frames": len(frames),
        "mapped_images": len(image_metadata),
        "prediction_rows": len(rows),
        "skipped_prediction_images": skipped_images,
        "raw_detections": len(raw),
        "detections": count,
        "category_offset": category_offset,
        "min_score": min_score,
        "nms_iou": nms_iou,
        "output": str(out_path),
    }
