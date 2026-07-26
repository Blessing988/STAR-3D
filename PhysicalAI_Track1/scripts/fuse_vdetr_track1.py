from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from physicalai_track1.dataset import TrackBox, scene_name_to_id
from physicalai_track1.geometry import distance_xy
from physicalai_track1.submission import read_submission, validate_submission, write_submission


@dataclass(frozen=True)
class VDetrBox:
    scene_id: int
    frame_id: int
    group_id: int
    score: float
    x: float
    y: float
    z: float
    width: float
    length: float
    height: float
    yaw: float


def parse_slice_id(slice_id: str) -> tuple[int, int]:
    parts = slice_id.split("_")
    if len(parts) < 5:
        raise ValueError(f"bad slice_id={slice_id!r}")
    scene_name = "_".join(parts[:-3])
    return scene_name_to_id(scene_name), int(parts[-3])


def base_group(class_id: int, mode: str) -> int | None:
    if mode == "merged4":
        if class_id == 0:
            return 0
        if class_id == 1:
            return 1
        if class_id in {2, 3, 6}:
            return 2
        if class_id in {4, 5}:
            return 3
        return None
    return class_id


def box_to_trackbox(v: VDetrBox, class_id: int = 0, object_id: int = 1) -> TrackBox:
    return TrackBox(
        scene_id=v.scene_id,
        class_id=class_id,
        object_id=object_id,
        frame_id=v.frame_id,
        x=v.x,
        y=v.y,
        z=v.z,
        width=v.width,
        length=v.length,
        height=v.height,
        yaw=v.yaw,
        score=v.score,
    )


def compatible_gate(base: TrackBox, vdetr: VDetrBox, gate_m: float) -> bool:
    dx = base.x - vdetr.x
    dy = base.y - vdetr.y
    dz = abs(base.z - vdetr.z)
    diag = 0.35 * max(base.width, base.length, vdetr.width, vdetr.length, 1.0)
    return math.hypot(dx, dy) <= gate_m + diag and dz <= 2.0


def nms_vdetr(boxes: list[VDetrBox], distance_m: float) -> list[VDetrBox]:
    kept: list[VDetrBox] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        duplicate = False
        tb = box_to_trackbox(box)
        for old in kept:
            if old.group_id != box.group_id:
                continue
            if distance_xy(tb, box_to_trackbox(old)) <= distance_m:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def iter_vdetr_groups(path: Path, *, min_score: float, allowed_groups: set[int]) -> tuple[tuple[int, int], list[VDetrBox]]:
    current_key: tuple[int, int] | None = None
    current: list[VDetrBox] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = float(row["score"])
            if score < min_score:
                continue
            scene_id, frame_id = parse_slice_id(row["slice_id"])
            group_id = int(row["class_id"])
            if group_id not in allowed_groups:
                continue
            key = (scene_id, frame_id)
            if current_key is not None and key != current_key:
                yield current_key, current
                current = []
            current_key = key
            current.append(
                VDetrBox(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    group_id=group_id,
                    score=score,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]),
                    width=float(row["width"]),
                    length=float(row["length"]),
                    height=float(row["height"]),
                    yaw=float(row["yaw"]),
                )
            )
    if current_key is not None:
        yield current_key, current


def log_ratio(target: float, source: float) -> float:
    return math.log(max(target, 1e-4) / max(source, 1e-4))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def corrected_box(box: TrackBox, residual: dict[str, float], strength: float) -> TrackBox:
    return TrackBox(
        scene_id=box.scene_id,
        class_id=box.class_id,
        object_id=box.object_id,
        frame_id=box.frame_id,
        x=box.x + strength * residual["dx"],
        y=box.y + strength * residual["dy"],
        z=box.z + strength * residual["dz"],
        width=box.width * math.exp(strength * residual["dw"]),
        length=box.length * math.exp(strength * residual["dl"]),
        height=box.height * math.exp(strength * residual["dh"]),
        yaw=box.yaw,
        score=box.score,
    )


def fuse(args: argparse.Namespace) -> dict:
    boxes = read_submission(args.base)
    by_frame: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, box in enumerate(boxes):
        by_frame[(box.scene_id, box.frame_id)].append(idx)

    allowed_groups = {int(x) for x in args.allowed_groups.split(",") if x.strip()}
    residuals: dict[tuple[int, int, int], list[dict[str, float]]] = defaultdict(list)
    matched_indices: set[int] = set()
    matched_frames = 0
    vdetr_groups = 0
    vdetr_after_nms = 0

    for key, vdetr_boxes in iter_vdetr_groups(args.vdetr, min_score=args.min_score, allowed_groups=allowed_groups):
        base_indices = by_frame.get(key)
        if not base_indices:
            continue
        vdetr_groups += 1
        candidates = nms_vdetr(vdetr_boxes, distance_m=args.vdetr_nms_distance_m)
        vdetr_after_nms += len(candidates)
        pairs: list[tuple[float, int, VDetrBox]] = []
        for idx in base_indices:
            base = boxes[idx]
            group_id = base_group(base.class_id, args.class_mode)
            if group_id is None:
                continue
            for vdetr in candidates:
                if vdetr.group_id != group_id:
                    continue
                if compatible_gate(base, vdetr, args.match_gate_m):
                    pairs.append((math.hypot(base.x - vdetr.x, base.y - vdetr.y), idx, vdetr))
        used_base: set[int] = set()
        used_vdetr: set[int] = set()
        frame_matches = 0
        for _dist, idx, vdetr in sorted(pairs, key=lambda item: item[0]):
            marker = id(vdetr)
            if idx in used_base or marker in used_vdetr:
                continue
            base = boxes[idx]
            used_base.add(idx)
            used_vdetr.add(marker)
            matched_indices.add(idx)
            frame_matches += 1
            residuals[(base.scene_id, base.class_id, base.object_id)].append(
                {
                    "dx": clamp(vdetr.x - base.x, -args.max_offset_m, args.max_offset_m),
                    "dy": clamp(vdetr.y - base.y, -args.max_offset_m, args.max_offset_m),
                    "dz": clamp(vdetr.z - base.z, -args.max_z_offset_m, args.max_z_offset_m),
                    "dw": clamp(log_ratio(vdetr.width, base.width), -args.max_log_scale, args.max_log_scale),
                    "dl": clamp(log_ratio(vdetr.length, base.length), -args.max_log_scale, args.max_log_scale),
                    "dh": clamp(log_ratio(vdetr.height, base.height), -args.max_log_scale, args.max_log_scale),
                }
            )
        if frame_matches:
            matched_frames += 1

    track_residuals: dict[tuple[int, int, int], dict[str, float]] = {}
    for key, values in residuals.items():
        if len(values) < args.min_track_matches:
            continue
        track_residuals[key] = {name: median(v[name] for v in values) for name in ("dx", "dy", "dz", "dw", "dl", "dh")}

    out: list[TrackBox] = []
    corrected = 0
    for box in boxes:
        key = (box.scene_id, box.class_id, box.object_id)
        residual = track_residuals.get(key)
        if residual is None:
            out.append(box)
            continue
        out.append(corrected_box(box, residual, args.strength))
        corrected += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_submission(out, args.out, decimals=args.decimals)
    params = {}
    for key, value in vars(args).items():
        params[key] = str(value) if isinstance(value, Path) else value
    stats = {
        "base": str(args.base),
        "vdetr": str(args.vdetr),
        "out": str(args.out),
        "input_boxes": len(boxes),
        "written": written,
        "matched_sample_boxes": len(matched_indices),
        "matched_frames": matched_frames,
        "vdetr_frame_groups": vdetr_groups,
        "vdetr_after_nms": vdetr_after_nms,
        "tracks_with_samples": len(residuals),
        "tracks_corrected": len(track_residuals),
        "boxes_corrected": corrected,
        "params": params,
        "validation": validate_submission(args.out),
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    if args.zip_out:
        args.zip_out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname="track1.txt")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse V-DETR 3D geometry corrections into a Track1 submission.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--vdetr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip-out", type=Path, default=None)
    parser.add_argument("--class-mode", choices=["merged4", "seven"], default="merged4")
    parser.add_argument("--allowed-groups", default="0,1,2")
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--vdetr-nms-distance-m", type=float, default=0.75)
    parser.add_argument("--match-gate-m", type=float, default=1.75)
    parser.add_argument("--min-track-matches", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--max-offset-m", type=float, default=1.25)
    parser.add_argument("--max-z-offset-m", type=float, default=0.75)
    parser.add_argument("--max-log-scale", type=float, default=0.25)
    parser.add_argument("--decimals", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(fuse(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
