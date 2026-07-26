from __future__ import annotations

import argparse
from pathlib import Path


def remap_ids(input_path: Path, output_path: Path) -> dict[str, int]:
    next_id_by_scene: dict[int, int] = {}
    id_map: dict[tuple[int, int, int], int] = {}
    lines = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            parts = line.split()
            if len(parts) != 11:
                raise ValueError(f"Expected 11 fields, got {len(parts)} on line {lines + 1}")
            scene_id = int(parts[0])
            class_id = int(parts[1])
            old_object_id = int(parts[2])
            key = (scene_id, class_id, old_object_id)
            if key not in id_map:
                next_id = next_id_by_scene.get(scene_id, 1)
                id_map[key] = next_id
                next_id_by_scene[scene_id] = next_id + 1
            parts[2] = str(id_map[key])
            dst.write(" ".join(parts) + "\n")
            lines += 1

    return {
        "lines": lines,
        "old_tracks": len(id_map),
        "scenes": len(next_id_by_scene),
        "max_scene_object_id": max(next_id_by_scene.values(), default=1) - 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap Track 1 object IDs to be unique across classes within each scene."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stats = remap_ids(args.input, args.output)
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
