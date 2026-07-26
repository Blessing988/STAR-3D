from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .detections import Detection2D, read_detections, write_detections


def _class_value(values: Mapping[int, float] | None, class_id: int, default: float) -> float:
    if values is None:
        return default
    return float(values.get(class_id, default))


def filter_detections_file(
    input_path: Path | str,
    output_path: Path | str,
    min_area: float = 0.0,
    min_width: float = 0.0,
    min_height: float = 0.0,
    class_min_area: Mapping[int, float] | None = None,
    class_min_width: Mapping[int, float] | None = None,
    class_min_height: Mapping[int, float] | None = None,
) -> dict:
    kept: list[Detection2D] = []
    total = 0
    dropped = 0
    dropped_by_class: dict[int, int] = {}
    kept_by_class: dict[int, int] = {}

    for det in read_detections(input_path):
        total += 1
        width = max(0.0, det.x2 - det.x1)
        height = max(0.0, det.y2 - det.y1)
        area = width * height
        keep = (
            area >= _class_value(class_min_area, det.class_id, min_area)
            and width >= _class_value(class_min_width, det.class_id, min_width)
            and height >= _class_value(class_min_height, det.class_id, min_height)
        )
        if keep:
            kept.append(det)
            kept_by_class[det.class_id] = kept_by_class.get(det.class_id, 0) + 1
        else:
            dropped += 1
            dropped_by_class[det.class_id] = dropped_by_class.get(det.class_id, 0) + 1

    written = write_detections(kept, output_path)
    result = {
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "kept": written,
        "dropped": dropped,
        "min_area": min_area,
        "min_width": min_width,
        "min_height": min_height,
        "class_min_area": dict(sorted((class_min_area or {}).items())),
        "class_min_width": dict(sorted((class_min_width or {}).items())),
        "class_min_height": dict(sorted((class_min_height or {}).items())),
        "kept_by_class": dict(sorted(kept_by_class.items())),
        "dropped_by_class": dict(sorted(dropped_by_class.items())),
    }
    Path(str(output_path) + ".json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
