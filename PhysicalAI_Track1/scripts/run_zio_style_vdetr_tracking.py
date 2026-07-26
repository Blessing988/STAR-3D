from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from physicalai_track1.dataset import TrackBox, scene_name_to_id
from physicalai_track1.submission import validate_submission, write_submission


@dataclass(frozen=True)
class Det3D:
    scene_id: int
    frame_id: int
    class_id: int
    score: float
    x: float
    y: float
    z: float
    width: float
    length: float
    height: float
    yaw: float


def parse_slice_id(slice_id: str, scene_offset: int = 0) -> tuple[int, int]:
    parts = slice_id.split("_")
    if len(parts) < 5:
        raise ValueError(f"bad slice_id={slice_id!r}")
    scene_name = "_".join(parts[:-3])
    frame_id = int(parts[-3])
    return scene_name_to_id(scene_name) + scene_offset, frame_id


def class_for_group(group_id: int, mode: str, default_vehicle_class: int, default_humanoid_class: int) -> int | None:
    if mode == "merged4":
        if group_id == 0:
            return 0
        if group_id == 1:
            return 1
        if group_id == 2:
            return default_vehicle_class
        if group_id == 3:
            return default_humanoid_class
        return None
    if 0 <= group_id <= 6:
        return group_id
    return None


def center_distance(a: Det3D, b: Det3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + 0.25 * (a.z - b.z) ** 2)


def nms_detections(dets: list[Det3D], distance_m: float) -> list[Det3D]:
    if distance_m <= 0:
        return sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Det3D] = []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        duplicate = False
        for old in kept:
            if old.class_id == det.class_id and center_distance(old, det) <= distance_m:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def load_vdetr_frames(
    path: Path,
    *,
    class_mode: str,
    min_score: float,
    allowed_groups: set[int],
    scene_offset: int,
    default_vehicle_class: int,
    default_humanoid_class: int,
    nms_distance_m: float,
    topk_per_class: int,
    topk_total: int,
) -> dict[tuple[int, int], list[Det3D]]:
    by_frame_raw: dict[tuple[int, int], list[Det3D]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = float(row["score"])
            if score < min_score:
                continue
            group_id = int(row["class_id"])
            if allowed_groups and group_id not in allowed_groups:
                continue
            class_id = class_for_group(group_id, class_mode, default_vehicle_class, default_humanoid_class)
            if class_id is None:
                continue
            scene_id, frame_id = parse_slice_id(row["slice_id"], scene_offset=scene_offset)
            det = Det3D(
                scene_id=scene_id,
                frame_id=frame_id,
                class_id=class_id,
                score=score,
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                width=max(0.05, float(row["width"])),
                length=max(0.05, float(row["length"])),
                height=max(0.05, float(row["height"])),
                yaw=float(row["yaw"]),
            )
            by_frame_raw[(scene_id, frame_id)].append(det)

    by_frame: dict[tuple[int, int], list[Det3D]] = {}
    for key, dets in by_frame_raw.items():
        dets = nms_detections(dets, nms_distance_m)
        if topk_per_class > 0:
            grouped: dict[int, list[Det3D]] = defaultdict(list)
            for det in dets:
                grouped[det.class_id].append(det)
            dets = []
            for group in grouped.values():
                dets.extend(sorted(group, key=lambda d: d.score, reverse=True)[:topk_per_class])
        if topk_total > 0 and len(dets) > topk_total:
            dets = sorted(dets, key=lambda d: d.score, reverse=True)[:topk_total]
        by_frame[key] = sorted(dets, key=lambda d: (d.class_id, -d.score))
    return by_frame


class FallbackTrack:
    def __init__(self, detection: np.ndarray, track_id: int):
        self.id = track_id
        self.box_params = detection[:7].astype(float)
        self.score = float(detection[7])
        self.cls_id = int(detection[8])
        self.row_idx = int(detection[9])
        self.vx = 0.0
        self.vy = 0.0
        self.time_since_update = 0
        self.hits = 1

    def predict(self) -> None:
        self.box_params[0] += self.vx
        self.box_params[1] += self.vy
        self.time_since_update += 1

    def update(self, detection: np.ndarray) -> None:
        new_box = detection[:7].astype(float)
        self.vx = 0.7 * self.vx + 0.3 * (new_box[0] - self.box_params[0])
        self.vy = 0.7 * self.vy + 0.3 * (new_box[1] - self.box_params[1])
        self.box_params = new_box
        self.score = float(detection[7])
        self.cls_id = int(detection[8])
        self.row_idx = int(detection[9])
        self.time_since_update = 0
        self.hits += 1


def axis_aligned_iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    amin = np.array([a[0] - a[3] / 2, a[1] - a[4] / 2, a[2] - a[5] / 2])
    amax = np.array([a[0] + a[3] / 2, a[1] + a[4] / 2, a[2] + a[5] / 2])
    bmin = np.array([b[0] - b[3] / 2, b[1] - b[4] / 2, b[2] - b[5] / 2])
    bmax = np.array([b[0] + b[3] / 2, b[1] + b[4] / 2, b[2] + b[5] / 2])
    inter = np.maximum(0.0, np.minimum(amax, bmax) - np.maximum(amin, bmin))
    inter_vol = float(inter[0] * inter[1] * inter[2])
    if inter_vol <= 0:
        return 0.0
    avol = float(np.prod(np.maximum(0.0, amax - amin)))
    bvol = float(np.prod(np.maximum(0.0, bmax - bmin)))
    return inter_vol / max(avol + bvol - inter_vol, 1e-6)


def fallback_gate(class_id: int) -> float:
    return {
        0: 1.40,
        1: 2.20,
        2: 2.00,
        3: 2.00,
        4: 1.60,
        5: 1.60,
        6: 2.00,
    }.get(class_id, 1.80)


class FallbackDeepSort3D:
    """Dependency-safe ZIO-style tracker fallback.

    It keeps ZIO's tracker contract but avoids filterpy/shapely. Association uses
    class-gated axis-aligned 3D IoU plus center distance and constant velocity.
    """

    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[FallbackTrack] = []
        self.next_id = 0

    def update(self, detections_raw: np.ndarray, features=None) -> np.ndarray:
        from scipy.optimize import linear_sum_assignment

        for track in self.tracks:
            track.predict()

        detections_raw = np.asarray(detections_raw, dtype=np.float32)
        if detections_raw.size == 0:
            detections_raw = detections_raw.reshape(0, 10)

        num_tracks = len(self.tracks)
        num_dets = len(detections_raw)
        cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32) * 1e4
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections_raw):
                if int(det[8]) != track.cls_id:
                    continue
                iou = axis_aligned_iou_3d(track.box_params, det[:7])
                dist = math.hypot(float(track.box_params[0] - det[0]), float(track.box_params[1] - det[1]))
                gate = fallback_gate(track.cls_id)
                if iou >= self.iou_threshold or dist <= gate:
                    cost_matrix[t, d] = (1.0 - iou) + 0.35 * (dist / gate)

        matched = []
        unmatched_tracks = set(range(num_tracks))
        unmatched_dets = set(range(num_dets))
        if num_tracks > 0 and num_dets > 0:
            rows, cols = linear_sum_assignment(cost_matrix)
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < 1.25:
                    matched.append((int(r), int(c)))
                    unmatched_tracks.discard(int(r))
                    unmatched_dets.discard(int(c))

        for r, c in matched:
            self.tracks[r].update(detections_raw[c])

        for d_idx in sorted(unmatched_dets):
            self.tracks.append(FallbackTrack(detections_raw[d_idx], self.next_id))
            self.next_id += 1

        active_tracks = []
        final_results = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                active_tracks.append(track)
                if track.hits >= self.min_hits:
                    final_results.append(
                        np.concatenate(
                            (
                                track.box_params,
                                np.array([track.score, track.id, track.cls_id, track.row_idx], dtype=float),
                            )
                        )
                    )
        self.tracks = active_tracks
        return np.array(final_results, dtype=np.float32)


def import_zio_tracker(zio_root: Path):
    tracker_dir = zio_root / "Tracker"
    sys.path.insert(0, str(tracker_dir))
    try:
        from tracker import DeepSort3D  # type: ignore

        print("tracker_backend=zio_deepsort3d", flush=True)
        return DeepSort3D
    except ModuleNotFoundError as exc:
        print(f"tracker_backend=fallback_deepsort3d reason={exc}", flush=True)
        return FallbackDeepSort3D


def run_short_tracking(
    frames: dict[tuple[int, int], list[Det3D]],
    *,
    zio_root: Path,
    max_age: int,
    min_hits: int,
    iou_threshold: float,
    score_threshold: float,
    per_class_tracker: bool,
) -> list[TrackBox]:
    DeepSort3D = import_zio_tracker(zio_root)
    trackers = {}
    outputs: list[TrackBox] = []
    row_idx = 0

    scene_ids = sorted({scene_id for scene_id, _ in frames})
    for scene_id in scene_ids:
        frame_ids = sorted(frame_id for s, frame_id in frames if s == scene_id)
        if not frame_ids:
            continue
        for frame_id in range(min(frame_ids), max(frame_ids) + 1):
            dets = [d for d in frames.get((scene_id, frame_id), []) if d.score >= score_threshold]
            if per_class_tracker:
                existing = {cls for (scene_key, cls) in trackers if scene_key == scene_id}
                class_ids = sorted(existing | {d.class_id for d in dets})
            else:
                class_ids = [-1]
            if not class_ids:
                continue
            for class_key in class_ids:
                if per_class_tracker:
                    tracker_key = (scene_id, class_key)
                    use_dets = [d for d in dets if d.class_id == class_key]
                else:
                    tracker_key = (scene_id, -1)
                    use_dets = dets
                tracker = trackers.setdefault(
                    tracker_key,
                    DeepSort3D(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold),
                )
                arr_rows = []
                for det in use_dets:
                    arr_rows.append(
                        [
                            det.x,
                            det.y,
                            det.z,
                            det.width,
                            det.length,
                            det.height,
                            det.yaw,
                            det.score,
                            det.class_id,
                            row_idx,
                        ]
                    )
                    row_idx += 1
                if arr_rows:
                    tracked = tracker.update(np.array(arr_rows, dtype=np.float32))
                else:
                    tracked = tracker.update(np.zeros((0, 10), dtype=np.float32))
                if tracked.size == 0:
                    continue
                if tracked.ndim == 1:
                    tracked = tracked.reshape(1, -1)
                for obj in tracked:
                    cls_id = int(obj[9])
                    track_id = int(obj[8]) + 1
                    outputs.append(
                        TrackBox(
                            scene_id=scene_id,
                            class_id=cls_id,
                            object_id=track_id,
                            frame_id=frame_id,
                            x=float(obj[0]),
                            y=float(obj[1]),
                            z=float(obj[2]),
                            width=max(0.05, float(obj[3])),
                            length=max(0.05, float(obj[4])),
                            height=max(0.05, float(obj[5])),
                            yaw=float(obj[6]),
                            score=float(obj[7]),
                        )
                    )
    return outputs


def speed_gate(class_id: int, gap: int) -> float:
    base = {
        0: 0.10,
        1: 0.18,
        2: 0.22,
        3: 0.22,
        4: 0.14,
        5: 0.14,
        6: 0.20,
    }.get(class_id, 0.18)
    return 1.0 + base * gap


def relink_tracklets(boxes: list[TrackBox], *, max_gap: int, max_distance: float, min_len: int) -> list[TrackBox]:
    by_key: dict[tuple[int, int, int], list[TrackBox]] = defaultdict(list)
    for box in boxes:
        by_key[(box.scene_id, box.class_id, box.object_id)].append(box)

    tracks = []
    for (scene_id, class_id, object_id), seq in by_key.items():
        seq = sorted(seq, key=lambda b: b.frame_id)
        if len(seq) >= min_len:
            tracks.append(
                {
                    "scene_id": scene_id,
                    "class_id": class_id,
                    "old_id": object_id,
                    "boxes": seq,
                    "parent": None,
                }
            )

    by_scene_class: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, trk in enumerate(tracks):
        by_scene_class[(trk["scene_id"], trk["class_id"])].append(idx)

    parent = list(range(len(tracks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, indices in by_scene_class.items():
        ordered = sorted(indices, key=lambda i: tracks[i]["boxes"][0].frame_id)
        for i in ordered:
            last = tracks[i]["boxes"][-1]
            best_j = None
            best_cost = float("inf")
            for j in ordered:
                if i == j:
                    continue
                first = tracks[j]["boxes"][0]
                gap = first.frame_id - last.frame_id
                if gap <= 0 or gap > max_gap:
                    continue
                dist = math.hypot(first.x - last.x, first.y - last.y)
                if dist > min(max_distance, speed_gate(last.class_id, gap)):
                    continue
                size_delta = abs(math.log(first.width / max(last.width, 1e-4))) + abs(
                    math.log(first.length / max(last.length, 1e-4))
                )
                cost = dist + 0.02 * gap + 0.5 * size_delta
                if cost < best_cost:
                    best_cost = cost
                    best_j = j
            if best_j is not None:
                union(i, best_j)

    grouped: dict[int, list[TrackBox]] = defaultdict(list)
    for idx, trk in enumerate(tracks):
        grouped[find(idx)].extend(trk["boxes"])

    relinked: list[TrackBox] = []
    next_ids: dict[tuple[int, int], int] = defaultdict(lambda: 1)
    for _, seq in sorted(grouped.items(), key=lambda kv: (kv[1][0].scene_id, kv[1][0].class_id, kv[1][0].frame_id)):
        seq = sorted(seq, key=lambda b: b.frame_id)
        scene_class = (seq[0].scene_id, seq[0].class_id)
        new_id = next_ids[scene_class]
        next_ids[scene_class] += 1
        seen_frames = set()
        for box in seq:
            if box.frame_id in seen_frames:
                continue
            seen_frames.add(box.frame_id)
            relinked.append(
                TrackBox(
                    scene_id=box.scene_id,
                    class_id=box.class_id,
                    object_id=new_id,
                    frame_id=box.frame_id,
                    x=box.x,
                    y=box.y,
                    z=box.z,
                    width=box.width,
                    length=box.length,
                    height=box.height,
                    yaw=box.yaw,
                    score=box.score,
                )
            )
    return relinked


def write_zip(txt_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(txt_path, arcname="track1.txt")


def summarize(boxes: list[TrackBox]) -> dict:
    by_class = Counter(b.class_id for b in boxes)
    by_scene = Counter(b.scene_id for b in boxes)
    ids = {(b.scene_id, b.class_id, b.object_id) for b in boxes}
    return {
        "boxes": len(boxes),
        "objects_scene_class": len(ids),
        "classes": dict(sorted(by_class.items())),
        "scenes": dict(sorted(by_scene.items())),
    }


def remap_object_ids_scene_unique(boxes: list[TrackBox]) -> list[TrackBox]:
    """Keep track identity, but avoid reusing object_id across classes in a scene."""
    grouped: dict[tuple[int, int, int], list[TrackBox]] = defaultdict(list)
    for box in boxes:
        grouped[(box.scene_id, box.class_id, box.object_id)].append(box)

    next_id_by_scene: dict[int, int] = defaultdict(lambda: 1)
    remapped: list[TrackBox] = []
    for key in sorted(grouped, key=lambda item: (item[0], min(b.frame_id for b in grouped[item]), item[1], item[2])):
        scene_id, _, _ = key
        new_id = next_id_by_scene[scene_id]
        next_id_by_scene[scene_id] += 1
        for box in grouped[key]:
            remapped.append(
                TrackBox(
                    scene_id=box.scene_id,
                    class_id=box.class_id,
                    object_id=new_id,
                    frame_id=box.frame_id,
                    x=box.x,
                    y=box.y,
                    z=box.z,
                    width=box.width,
                    length=box.length,
                    height=box.height,
                    yaw=box.yaw,
                    score=box.score,
                )
            )
    return remapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ZIO-style V-DETR-primary 3D tracker and Track1 export.")
    parser.add_argument("--vdetr", type=Path, required=True)
    parser.add_argument("--zio-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip-out", type=Path, required=True)
    parser.add_argument("--class-mode", choices=["merged4", "7cls"], default="merged4")
    parser.add_argument("--allowed-groups", default="0,1,2,3")
    parser.add_argument("--default-vehicle-class", type=int, default=6)
    parser.add_argument("--default-humanoid-class", type=int, default=4)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--track-score-threshold", type=float, default=0.30)
    parser.add_argument("--nms-distance-m", type=float, default=0.75)
    parser.add_argument("--topk-per-class", type=int, default=60)
    parser.add_argument("--topk-total", type=int, default=180)
    parser.add_argument("--max-age", type=int, default=15)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.08)
    parser.add_argument("--single-tracker", action="store_true")
    parser.add_argument("--global-relink", action="store_true")
    parser.add_argument("--relink-max-gap", type=int, default=90)
    parser.add_argument("--relink-max-distance", type=float, default=4.0)
    parser.add_argument("--relink-min-len", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_groups = {int(x) for x in args.allowed_groups.split(",") if x.strip()}
    frames = load_vdetr_frames(
        args.vdetr,
        class_mode=args.class_mode,
        min_score=args.min_score,
        allowed_groups=allowed_groups,
        scene_offset=args.scene_offset,
        default_vehicle_class=args.default_vehicle_class,
        default_humanoid_class=args.default_humanoid_class,
        nms_distance_m=args.nms_distance_m,
        topk_per_class=args.topk_per_class,
        topk_total=args.topk_total,
    )
    boxes = run_short_tracking(
        frames,
        zio_root=args.zio_root,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        score_threshold=args.track_score_threshold,
        per_class_tracker=not args.single_tracker,
    )
    if args.global_relink:
        boxes = relink_tracklets(
            boxes,
            max_gap=args.relink_max_gap,
            max_distance=args.relink_max_distance,
            min_len=args.relink_min_len,
        )
    boxes = remap_object_ids_scene_unique(boxes)
    boxes = sorted(boxes, key=lambda b: (b.scene_id, b.frame_id, b.class_id, b.object_id))
    count = write_submission(boxes, args.out)
    write_zip(args.out, args.zip_out)
    args_dict = vars(args).copy()
    args_dict.update(
        {
            "vdetr": str(args.vdetr),
            "zio_root": str(args.zio_root),
            "out": str(args.out),
            "zip_out": str(args.zip_out),
        }
    )
    stats = {
        "count": count,
        "frames": len(frames),
        "vdetr": str(args.vdetr),
        "out": str(args.out),
        "zip_out": str(args.zip_out),
        "summary": summarize(boxes),
        "validation": validate_submission(args.out),
        "args": args_dict,
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
