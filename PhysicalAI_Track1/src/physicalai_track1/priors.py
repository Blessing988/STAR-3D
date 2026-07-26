from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

from .constants import ID_TO_CLASS
from .stats import class_box_stats


FALLBACK_PRIORS = {
    0: {"width": 0.58, "length": 0.53, "height": 1.84, "z": 0.92, "yaw": 0.0},
    1: {"width": 1.14, "length": 2.05, "height": 2.60, "z": 1.30, "yaw": 0.0},
    2: {"width": 0.75, "length": 1.00, "height": 1.10, "z": 0.55, "yaw": 0.0},
    3: {"width": 0.80, "length": 1.20, "height": 1.40, "z": 0.70, "yaw": 0.0},
    4: {"width": 0.65, "length": 0.65, "height": 1.75, "z": 0.88, "yaw": 0.0},
    5: {"width": 0.65, "length": 0.65, "height": 1.70, "z": 0.85, "yaw": 0.0},
    6: {"width": 0.75, "length": 1.75, "height": 1.60, "z": 0.80, "yaw": 0.0},
}


def build_priors(
    data_root: Path | str,
    year: int,
    split: str,
    out_path: Path | str,
    max_frames_per_scene: int | None = None,
) -> dict:
    stats = class_box_stats(data_root, year, split, max_frames_per_scene=max_frames_per_scene)
    priors: Dict[str, dict] = {}
    for class_id, name in ID_TO_CLASS.items():
        cls_stats = stats["classes"].get(name)
        fallback = FALLBACK_PRIORS[class_id]
        if not cls_stats:
            priors[str(class_id)] = {"class_name": name, **fallback, "source": "fallback"}
            continue
        priors[str(class_id)] = {
            "class_name": name,
            "width": cls_stats["width"].get("p50", fallback["width"]),
            "length": cls_stats["length"].get("p50", fallback["length"]),
            "height": cls_stats["height"].get("p50", fallback["height"]),
            "z": cls_stats["z"].get("p50", fallback["z"]),
            "yaw": cls_stats["yaw"].get("p50", fallback["yaw"]),
            "source": split,
            "count": cls_stats["width"].get("count", 0),
        }
    payload = {"year": year, "split": split, "priors": priors, "stats": stats}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"output": str(out), "classes": len(priors), "boxes": stats["boxes"]}


def load_priors(path: Path | str | None) -> Dict[int, dict]:
    if path is None:
        return {class_id: dict(values) for class_id, values in FALLBACK_PRIORS.items()}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("priors", payload)
    priors: Dict[int, dict] = {}
    for class_id, fallback in FALLBACK_PRIORS.items():
        item = raw.get(str(class_id), {})
        priors[class_id] = {
            "width": float(item.get("width", fallback["width"])),
            "length": float(item.get("length", fallback["length"])),
            "height": float(item.get("height", fallback["height"])),
            "z": float(item.get("z", fallback["z"])),
            "yaw": float(item.get("yaw", fallback["yaw"])),
        }
    return priors


def prior_for_class(priors: Mapping[int, Mapping[str, float]], class_id: int) -> Mapping[str, float]:
    return priors.get(class_id, FALLBACK_PRIORS[class_id])

