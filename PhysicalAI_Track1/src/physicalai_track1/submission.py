from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .dataset import TrackBox


def write_submission(boxes: Iterable[TrackBox], output_path: Path | str, decimals: int = 2) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for box in boxes:
            f.write(" ".join(box.submission_fields(decimals=decimals)))
            f.write("\n")
            count += 1
    return count


def read_submission(path: Path | str) -> List[TrackBox]:
    boxes: List[TrackBox] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 11:
                raise ValueError(f"{path}:{line_no}: expected 11 fields, got {len(parts)}")
            scene_id, class_id, object_id, frame_id = map(int, parts[:4])
            x, y, z, width, length, height, yaw = map(float, parts[4:])
            if width <= 0 or length <= 0 or height <= 0:
                raise ValueError(f"{path}:{line_no}: non-positive box dimension")
            boxes.append(
                TrackBox(
                    scene_id=scene_id,
                    class_id=class_id,
                    object_id=object_id,
                    frame_id=frame_id,
                    x=x,
                    y=y,
                    z=z,
                    width=width,
                    length=length,
                    height=height,
                    yaw=yaw,
                )
            )
    return boxes


def validate_submission(path: Path | str) -> dict:
    boxes = read_submission(path)
    scene_ids = {b.scene_id for b in boxes}
    class_ids = {b.class_id for b in boxes}
    object_ids = {b.object_id for b in boxes}
    frame_ids = {b.frame_id for b in boxes}
    return {
        "path": str(path),
        "boxes": len(boxes),
        "scenes": len(scene_ids),
        "classes": sorted(class_ids),
        "objects": len(object_ids),
        "frame_min": min(frame_ids) if frame_ids else None,
        "frame_max": max(frame_ids) if frame_ids else None,
    }
