from __future__ import annotations

import argparse
import json
from pathlib import Path

from physicalai_track1.constants import CLASS_TO_ID, GT_KEY_ALIASES
from physicalai_track1.dataset import year_dir


def _get_any(record: dict, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


def parse_stem(stem: str) -> tuple[str, int]:
    scene, frame = stem.rsplit("_", 1)
    return scene, int(frame)


def load_gt(scene_dir: Path) -> dict:
    path = scene_dir / "ground_truth.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def row_for_item(item: dict) -> str:
    class_id = CLASS_TO_ID[_get_any(item, "object_type")]
    object_id = int(_get_any(item, "object_id"))
    location = list(map(float, _get_any(item, "location")))
    scale = list(map(float, _get_any(item, "scale")))
    rotation = list(map(float, _get_any(item, "rotation")))
    return (
        f"{class_id} {object_id} {location[0]} {location[1]} {location[2]} "
        f"{scale[0]} {scale[1]} {scale[2]} {rotation[0]} {rotation[1]} {rotation[2]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write ZIO-style GT txt files for existing PCD stems.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pcd-root", type=Path, required=True)
    parser.add_argument("--out-gt-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    split_root = year_dir(args.data_root, args.year) / args.split
    args.out_gt_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    written = 0
    missing_frames = 0
    for ply in sorted(args.pcd_root.glob("*.ply")):
        scene, frame_id = parse_stem(ply.stem)
        if scene not in cache:
            cache[scene] = load_gt(split_root / scene)
        items = cache[scene].get(str(frame_id), cache[scene].get(f"{frame_id:05d}", []))
        lines = [row_for_item(item) for item in items]
        if not lines:
            missing_frames += 1
        out = args.out_gt_dir / f"{ply.stem}.txt"
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    print({"written": written, "missing_or_empty_frames": missing_frames, "out_gt_dir": str(args.out_gt_dir)})


if __name__ == "__main__":
    main()
