from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .constants import CLASS_TO_ID, GT_KEY_ALIASES


@dataclass(frozen=True)
class TrackBox:
    scene_id: int
    class_id: int
    object_id: int
    frame_id: int
    x: float
    y: float
    z: float
    width: float
    length: float
    height: float
    yaw: float
    score: float = 1.0

    def submission_fields(self, decimals: int = 2) -> List[str]:
        fmt = f"{{:.{decimals}f}}"
        return [
            str(int(self.scene_id)),
            str(int(self.class_id)),
            str(int(self.object_id)),
            str(int(self.frame_id)),
            fmt.format(self.x),
            fmt.format(self.y),
            fmt.format(self.z),
            fmt.format(self.width),
            fmt.format(self.length),
            fmt.format(self.height),
            fmt.format(self.yaw),
        ]


def _get_any(record: Mapping, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


def scene_name_to_id(scene_name: str) -> int:
    """Convert scene folder names like Warehouse_023 or scene_061 to IDs."""
    match = re.search(r"(\d+)$", scene_name)
    if not match:
        raise ValueError(f"Cannot infer scene_id from scene name: {scene_name}")
    return int(match.group(1))


def year_dir(data_root: Path | str, year: int) -> Path:
    return Path(data_root) / f"MTMC_Tracking_{year}"


def iter_scene_dirs(data_root: Path | str, year: int, split: str) -> Iterator[Path]:
    root = year_dir(data_root, year) / split
    if not root.exists():
        raise FileNotFoundError(root)
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_calibration(scene_dir: Path | str) -> Dict:
    return load_json(Path(scene_dir) / "calibration.json")


def load_ground_truth(scene_dir: Path | str) -> Dict:
    return load_json(Path(scene_dir) / "ground_truth.json")


def iter_top_level_json_items(path: Path | str, max_items: Optional[int] = None) -> Iterator[tuple[str, object]]:
    """Stream key/value pairs from a top-level JSON object.

    The GT files are large objects keyed by frame id. For smoke tests and
    sampled exports, this avoids loading the full scene file.
    """
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        pos = 0
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            if eof:
                return
            chunk = f.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def compact() -> None:
            nonlocal buffer, pos
            if pos > chunk_size:
                buffer = buffer[pos:]
                pos = 0

        def ensure() -> bool:
            while pos >= len(buffer) and not eof:
                fill()
            return pos < len(buffer)

        def skip_ws() -> None:
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                fill()

        fill()
        skip_ws()
        if not ensure() or buffer[pos] != "{":
            raise ValueError(f"{path} is not a top-level JSON object")
        pos += 1
        yielded = 0

        while True:
            compact()
            skip_ws()
            while not ensure():
                fill()
            if buffer[pos] == "}":
                return
            if buffer[pos] == ",":
                pos += 1
                skip_ws()

            while True:
                try:
                    key, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            pos = end
            skip_ws()
            while not ensure():
                fill()
            if buffer[pos] != ":":
                raise ValueError(f"Expected ':' after key {key!r} in {path}")
            pos += 1
            skip_ws()

            # Values are arrays in the GT, but this scanner supports objects too.
            while not ensure():
                fill()
            start = pos
            first = buffer[pos]
            if first not in "[{":
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, pos)
                        pos = end
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        fill()
            else:
                depth = 0
                in_string = False
                escape = False
                while True:
                    while pos >= len(buffer) and not eof:
                        fill()
                    if pos >= len(buffer):
                        raise ValueError(f"Unexpected EOF while parsing value for {key!r}")
                    ch = buffer[pos]
                    if in_string:
                        if escape:
                            escape = False
                        elif ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch in "[{":
                            depth += 1
                        elif ch in "]}":
                            depth -= 1
                            if depth == 0:
                                pos += 1
                                raw = buffer[start:pos]
                                value = json.loads(raw)
                                break
                    pos += 1

            yield str(key), value
            yielded += 1
            if max_items is not None and yielded >= max_items:
                return


def iter_gt_frames(scene_dir: Path | str, max_frames: Optional[int] = None) -> Iterator[tuple[str, list]]:
    gt_path = Path(scene_dir) / "ground_truth.json"
    if max_frames is not None:
        yield from iter_top_level_json_items(gt_path, max_items=max_frames)
        return
    gt = load_json(gt_path)
    for frame_key in sorted(gt.keys(), key=lambda x: int(x)):
        yield frame_key, gt[frame_key]


def iter_gt_boxes(
    scene_dir: Path | str,
    scene_id: Optional[int] = None,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> Iterator[TrackBox]:
    scene_path = Path(scene_dir)
    if scene_id is None:
        scene_id = scene_name_to_id(scene_path.name)
    for frame_key, frame_items in iter_gt_frames(scene_path, max_frames=max_frames):
        frame_id = int(frame_key)
        if frame_stride > 1 and frame_id % frame_stride != 0:
            continue
        for item in frame_items:
            object_type = _get_any(item, "object_type")
            if object_type not in CLASS_TO_ID:
                raise KeyError(f"Unknown class {object_type!r} in {scene_path}")
            location = _get_any(item, "location")
            scale = _get_any(item, "scale")
            rotation = _get_any(item, "rotation")
            yield TrackBox(
                scene_id=scene_id,
                class_id=CLASS_TO_ID[object_type],
                object_id=int(_get_any(item, "object_id")),
                frame_id=frame_id,
                x=float(location[0]),
                y=float(location[1]),
                z=float(location[2]),
                width=float(scale[0]),
                length=float(scale[1]),
                height=float(scale[2]),
                yaw=float(rotation[2]),
            )


def load_gt_boxes_for_split(
    data_root: Path | str,
    year: int,
    split: str,
    scenes: Optional[Sequence[str]] = None,
    max_frames_per_scene: Optional[int] = None,
    frame_stride: int = 1,
) -> List[TrackBox]:
    boxes: List[TrackBox] = []
    scene_filter = set(scenes or [])
    for scene_dir in iter_scene_dirs(data_root, year, split):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        gt_path = scene_dir / "ground_truth.json"
        if not gt_path.exists():
            continue
        boxes.extend(
            iter_gt_boxes(
                scene_dir,
                max_frames=max_frames_per_scene,
                frame_stride=frame_stride,
            )
        )
    return boxes


def summarize_dataset(data_root: Path | str, year: int, deep: bool = False) -> Dict:
    root = year_dir(data_root, year)
    summary = {
        "root": str(root),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        split_summary = {
            "scenes": 0,
            "cameras": 0,
            "frames_with_gt": 0,
            "gt_boxes": 0,
            "gt_files": 0,
            "objects_by_class": defaultdict(set),
            "boxes_by_class": Counter(),
            "scene_names": [],
        }
        for scene_dir in iter_scene_dirs(data_root, year, split):
            split_summary["scenes"] += 1
            split_summary["scene_names"].append(scene_dir.name)
            videos = scene_dir / "videos"
            if videos.exists():
                split_summary["cameras"] += len(list(videos.glob("*.mp4")))
            gt_path = scene_dir / "ground_truth.json"
            if gt_path.exists():
                split_summary["gt_files"] += 1
                if deep:
                    gt = load_json(gt_path)
                    split_summary["frames_with_gt"] += len(gt)
                    for frame_items in gt.values():
                        split_summary["gt_boxes"] += len(frame_items)
                        for item in frame_items:
                            cls = _get_any(item, "object_type")
                            oid = int(_get_any(item, "object_id"))
                            split_summary["boxes_by_class"][cls] += 1
                            split_summary["objects_by_class"][cls].add(oid)
        split_summary["objects_by_class"] = {
            cls: len(ids) for cls, ids in sorted(split_summary["objects_by_class"].items())
        }
        split_summary["boxes_by_class"] = dict(sorted(split_summary["boxes_by_class"].items()))
        summary["splits"][split] = split_summary
    return summary
