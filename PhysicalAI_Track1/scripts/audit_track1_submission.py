from __future__ import annotations

import argparse
import re
from pathlib import Path


FLOAT_RE = re.compile(r"^-?[0-9]+\.[0-9]{2}$")


def audit(path: Path) -> dict[str, int | dict[int, int]]:
    field_counts: dict[int, int] = {}
    scene_object_class: dict[tuple[int, int], int] = {}
    lines = 0
    class_conflicts = 0
    bad_nan_inf = 0
    bad_float_decimals = 0
    nonpositive_object_ids = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            lines += 1
            field_counts[len(parts)] = field_counts.get(len(parts), 0) + 1
            if len(parts) != 11:
                continue
            scene_id = int(parts[0])
            class_id = int(parts[1])
            object_id = int(parts[2])
            if object_id <= 0:
                nonpositive_object_ids += 1
            key = (scene_id, object_id)
            previous_class = scene_object_class.setdefault(key, class_id)
            if previous_class != class_id:
                class_conflicts += 1
            for value in parts[4:]:
                lower = value.lower()
                if "nan" in lower or "inf" in lower:
                    bad_nan_inf += 1
                if FLOAT_RE.match(value) is None:
                    bad_float_decimals += 1

    return {
        "lines": lines,
        "field_counts": field_counts,
        "scene_object_class_conflicts": class_conflicts,
        "bad_nan_inf": bad_nan_inf,
        "bad_float_decimals": bad_float_decimals,
        "nonpositive_object_ids": nonpositive_object_ids,
        "unique_scene_object_ids": len(scene_object_class),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an AI City 2026 Track 1 submission file.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    for key, value in audit(args.path).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
