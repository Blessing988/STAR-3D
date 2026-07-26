from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

from .constants import CLASS_TO_ID, GT_KEY_ALIASES
from .dataset import iter_gt_frames, iter_scene_dirs


def _get_any(record: Mapping, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


@dataclass(frozen=True)
class Detection2D:
    scene_name: str
    camera_id: str
    frame_id: int
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    oracle_object_id: Optional[int] = None

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) * 0.5, self.y2


HEADER = "scene_name\tcamera_id\tframe_id\tclass_id\tscore\tx1\ty1\tx2\ty2\toracle_object_id"


def write_detections(detections: Iterable[Detection2D], out_path: Path | str) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for det in detections:
            oid = "" if det.oracle_object_id is None else str(det.oracle_object_id)
            f.write(
                "\t".join(
                    [
                        det.scene_name,
                        det.camera_id,
                        str(det.frame_id),
                        str(det.class_id),
                        f"{det.score:.6f}",
                        f"{det.x1:.3f}",
                        f"{det.y1:.3f}",
                        f"{det.x2:.3f}",
                        f"{det.y2:.3f}",
                        oid,
                    ]
                )
            )
            f.write("\n")
            count += 1
    return count


def read_detections(path: Path | str) -> Iterator[Detection2D]:
    with Path(path).open("r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
        has_header = first.startswith("scene_name\t")
        rows = f if has_header else [first, *f]
        for line_no, line in enumerate(rows, start=2 if has_header else 1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 9:
                raise ValueError(f"{path}:{line_no}: expected at least 9 TSV fields")
            oid = None
            if len(parts) >= 10 and parts[9] != "":
                oid = int(parts[9])
            yield Detection2D(
                scene_name=parts[0],
                camera_id=parts[1],
                frame_id=int(parts[2]),
                class_id=int(parts[3]),
                score=float(parts[4]),
                x1=float(parts[5]),
                y1=float(parts[6]),
                x2=float(parts[7]),
                y2=float(parts[8]),
                oracle_object_id=oid,
            )


def iter_gt_2d_detections(
    data_root: Path | str,
    year: int,
    split: str,
    scenes: Optional[Sequence[str]] = None,
    frame_stride: int = 1,
    max_frames_per_scene: Optional[int] = None,
    min_box_area: float = 16.0,
) -> Iterator[Detection2D]:
    scene_filter = set(scenes or [])
    for scene_dir in iter_scene_dirs(data_root, year, split):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        if not (scene_dir / "ground_truth.json").exists():
            continue
        for frame_key, frame_items in iter_gt_frames(scene_dir, max_frames=max_frames_per_scene):
            frame_id = int(frame_key)
            if frame_stride > 1 and frame_id % frame_stride != 0:
                continue
            for item in frame_items:
                class_id = CLASS_TO_ID[_get_any(item, "object_type")]
                object_id = int(_get_any(item, "object_id"))
                visible = _get_any(item, "bbox2d")
                for camera_id, xyxy in visible.items():
                    x1, y1, x2, y2 = map(float, xyxy)
                    if max(0.0, x2 - x1) * max(0.0, y2 - y1) < min_box_area:
                        continue
                    yield Detection2D(
                        scene_name=scene_dir.name,
                        camera_id=str(camera_id),
                        frame_id=frame_id,
                        class_id=class_id,
                        score=1.0,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        oracle_object_id=object_id,
                    )


def export_gt_2d_detections(
    data_root: Path | str,
    year: int,
    split: str,
    out_path: Path | str,
    scenes: Optional[Sequence[str]] = None,
    frame_stride: int = 1,
    max_frames_per_scene: Optional[int] = None,
    min_box_area: float = 16.0,
) -> dict:
    count = write_detections(
        iter_gt_2d_detections(
            data_root=data_root,
            year=year,
            split=split,
            scenes=scenes,
            frame_stride=frame_stride,
            max_frames_per_scene=max_frames_per_scene,
            min_box_area=min_box_area,
        ),
        out_path,
    )
    return {"output": str(out_path), "detections": count, "split": split}

