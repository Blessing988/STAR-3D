from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from .constants import ID_TO_CLASS
from .dataset import iter_gt_boxes, iter_scene_dirs


def _summarize(values):
    if not values:
        return {"count": 0}
    vals = sorted(float(v) for v in values)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": vals[0],
        "p05": vals[int(0.05 * (n - 1))],
        "p50": vals[int(0.50 * (n - 1))],
        "p95": vals[int(0.95 * (n - 1))],
        "max": vals[-1],
    }


def class_box_stats(
    data_root: Path | str,
    year: int,
    split: str,
    scenes: Optional[Sequence[str]] = None,
    max_frames_per_scene: Optional[int] = None,
) -> dict:
    scene_filter = set(scenes or [])
    values = defaultdict(lambda: defaultdict(list))
    object_ids = defaultdict(set)
    total = 0
    for scene_dir in iter_scene_dirs(data_root, year, split):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        if not (scene_dir / "ground_truth.json").exists():
            continue
        for box in iter_gt_boxes(scene_dir, max_frames=max_frames_per_scene):
            cls = ID_TO_CLASS.get(box.class_id, str(box.class_id))
            values[cls]["width"].append(box.width)
            values[cls]["length"].append(box.length)
            values[cls]["height"].append(box.height)
            values[cls]["z"].append(box.z)
            values[cls]["yaw"].append(box.yaw)
            object_ids[cls].add((box.scene_id, box.object_id))
            total += 1
    return {
        "split": split,
        "boxes": total,
        "classes": {
            cls: {
                "objects": len(object_ids[cls]),
                "width": _summarize(fields["width"]),
                "length": _summarize(fields["length"]),
                "height": _summarize(fields["height"]),
                "z": _summarize(fields["z"]),
                "yaw": _summarize(fields["yaw"]),
            }
            for cls, fields in sorted(values.items())
        },
    }

