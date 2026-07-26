from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
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


@dataclass
class InjectTrack:
    object_id: int
    scene_id: int
    class_id: int
    group_id: int
    boxes: list[TrackBox] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def last(self) -> TrackBox:
        return self.boxes[-1]

    @property
    def hits(self) -> int:
        return len(self.boxes)


def parse_slice_id(slice_id: str, scene_offset: int = 0) -> tuple[int, int]:
    parts = slice_id.split('_')
    if len(parts) < 5:
        raise ValueError(f'bad slice_id={slice_id!r}')
    scene_name = '_'.join(parts[:-3])
    frame_id = int(parts[-3])
    return scene_name_to_id(scene_name) + scene_offset, frame_id


def group_for_class(class_id: int, mode: str) -> int | None:
    if mode == 'merged4':
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


def class_for_group(group_id: int, mode: str, default_vehicle_class: int = 6) -> int | None:
    if mode == 'merged4':
        if group_id == 0:
            return 0
        if group_id == 1:
            return 1
        if group_id == 2:
            return default_vehicle_class
        if group_id == 3:
            return 4
        return None
    return group_id


def box_to_trackbox(v: VDetrBox, class_id: int, object_id: int) -> TrackBox:
    return TrackBox(
        scene_id=v.scene_id,
        class_id=class_id,
        object_id=object_id,
        frame_id=v.frame_id,
        x=v.x,
        y=v.y,
        z=v.z,
        width=max(0.05, v.width),
        length=max(0.05, v.length),
        height=max(0.05, v.height),
        yaw=v.yaw,
        score=v.score,
    )


def log_ratio(target: float, source: float) -> float:
    return math.log(max(target, 1e-4) / max(source, 1e-4))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compatible_gate(base: TrackBox, vdetr: VDetrBox, gate_m: float, z_gate_m: float) -> bool:
    dx = base.x - vdetr.x
    dy = base.y - vdetr.y
    dz = abs(base.z - vdetr.z)
    diag = 0.35 * max(base.width, base.length, vdetr.width, vdetr.length, 1.0)
    return math.hypot(dx, dy) <= gate_m + diag and dz <= z_gate_m


def nms_vdetr(boxes: list[VDetrBox], distance_m: float) -> list[VDetrBox]:
    kept: list[VDetrBox] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        duplicate = False
        tb = box_to_trackbox(box, class_id=0, object_id=1)
        for old in kept:
            if old.group_id != box.group_id:
                continue
            if distance_xy(tb, box_to_trackbox(old, class_id=0, object_id=1)) <= distance_m:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def prune_vdetr_frame(
    boxes: list[VDetrBox],
    *,
    topk_per_group: int,
    topk_total: int,
) -> list[VDetrBox]:
    if not boxes:
        return []
    if topk_per_group > 0:
        by_group: dict[int, list[VDetrBox]] = defaultdict(list)
        for box in boxes:
            by_group[box.group_id].append(box)
        boxes = []
        for group_boxes in by_group.values():
            boxes.extend(sorted(group_boxes, key=lambda item: item.score, reverse=True)[:topk_per_group])
    if topk_total > 0 and len(boxes) > topk_total:
        boxes = sorted(boxes, key=lambda item: item.score, reverse=True)[:topk_total]
    return boxes


def iter_vdetr_groups(
    path: Path,
    *,
    min_score: float,
    allowed_groups: set[int],
    scene_offset: int = 0,
    topk_per_group: int = 128,
    topk_total: int = 512,
):
    current_key: tuple[int, int] | None = None
    current: list[VDetrBox] = []
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            score = float(row['score'])
            if score < min_score:
                continue
            scene_id, frame_id = parse_slice_id(row['slice_id'], scene_offset=scene_offset)
            group_id = int(row['class_id'])
            if group_id not in allowed_groups:
                continue
            key = (scene_id, frame_id)
            if current_key is not None and key != current_key:
                yield current_key, prune_vdetr_frame(
                    current,
                    topk_per_group=topk_per_group,
                    topk_total=topk_total,
                )
                current = []
            current_key = key
            current.append(
                VDetrBox(
                    scene_id=scene_id,
                    frame_id=frame_id,
                    group_id=group_id,
                    score=score,
                    x=float(row['x']),
                    y=float(row['y']),
                    z=float(row['z']),
                    width=float(row['width']),
                    length=float(row['length']),
                    height=float(row['height']),
                    yaw=float(row['yaw']),
                )
            )
    if current_key is not None:
        yield current_key, prune_vdetr_frame(
            current,
            topk_per_group=topk_per_group,
            topk_total=topk_total,
        )


def corrected_box(box: TrackBox, residual: dict[str, float], strength: float, center_only: bool) -> TrackBox:
    if center_only:
        return TrackBox(
            scene_id=box.scene_id, class_id=box.class_id, object_id=box.object_id, frame_id=box.frame_id,
            x=box.x + strength * residual['dx'], y=box.y + strength * residual['dy'], z=box.z + strength * residual['dz'],
            width=box.width, length=box.length, height=box.height, yaw=box.yaw, score=box.score,
        )
    return TrackBox(
        scene_id=box.scene_id, class_id=box.class_id, object_id=box.object_id, frame_id=box.frame_id,
        x=box.x + strength * residual['dx'], y=box.y + strength * residual['dy'], z=box.z + strength * residual['dz'],
        width=box.width * math.exp(strength * residual['dw']),
        length=box.length * math.exp(strength * residual['dl']),
        height=box.height * math.exp(strength * residual['dh']),
        yaw=box.yaw, score=box.score,
    )


def next_ids(boxes: list[TrackBox]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = defaultdict(int)
    for b in boxes:
        key = (b.scene_id, b.class_id)
        out[key] = max(out[key], b.object_id)
    return out


def interpolate_track(track: InjectTrack, max_gap: int) -> list[TrackBox]:
    boxes = sorted(track.boxes, key=lambda b: b.frame_id)
    if not boxes:
        return []
    out: list[TrackBox] = [boxes[0]]
    for a, b in zip(boxes, boxes[1:]):
        gap = b.frame_id - a.frame_id
        if 1 < gap <= max_gap:
            for frame in range(a.frame_id + 1, b.frame_id):
                alpha = (frame - a.frame_id) / gap
                out.append(
                    TrackBox(
                        scene_id=a.scene_id, class_id=a.class_id, object_id=a.object_id, frame_id=frame,
                        x=a.x + alpha * (b.x - a.x), y=a.y + alpha * (b.y - a.y), z=a.z + alpha * (b.z - a.z),
                        width=a.width + alpha * (b.width - a.width),
                        length=a.length + alpha * (b.length - a.length),
                        height=a.height + alpha * (b.height - a.height),
                        yaw=a.yaw, score=min(a.score, b.score),
                    )
                )
        out.append(b)
    return out


def build_injection_tracks(
    unmatched: dict[tuple[int, int], list[VDetrBox]],
    *,
    class_mode: str,
    inject_groups: set[int],
    default_vehicle_class: int,
    base_boxes: list[TrackBox],
    track_gate_m: float,
    max_track_gap: int,
    min_track_hits: int,
    interpolate_gap: int,
) -> tuple[list[TrackBox], dict]:
    ids = next_ids(base_boxes)
    active: dict[tuple[int, int, int], list[InjectTrack]] = defaultdict(list)
    completed: list[InjectTrack] = []
    new_tracks = 0
    for key in sorted(unmatched):
        scene_id, frame_id = key
        by_class: dict[int, list[VDetrBox]] = defaultdict(list)
        for v in unmatched[key]:
            if v.group_id not in inject_groups:
                continue
            class_id = class_for_group(v.group_id, class_mode, default_vehicle_class=default_vehicle_class)
            if class_id is None:
                continue
            by_class[class_id].append(v)
        for class_id, detections in by_class.items():
            tkey = (scene_id, class_id, group_for_class(class_id, class_mode) or class_id)
            tracks = active[tkey]
            still_active: list[InjectTrack] = []
            for trk in tracks:
                if frame_id - trk.last.frame_id <= max_track_gap:
                    still_active.append(trk)
                else:
                    completed.append(trk)
            tracks = still_active
            used_tracks: set[int] = set()
            for det in sorted(detections, key=lambda d: d.score, reverse=True):
                best_i = None
                best_dist = float('inf')
                det_box = box_to_trackbox(det, class_id=class_id, object_id=1)
                for i, trk in enumerate(tracks):
                    if i in used_tracks:
                        continue
                    dist = distance_xy(trk.last, det_box)
                    if dist < best_dist and dist <= track_gate_m:
                        best_i = i
                        best_dist = dist
                if best_i is None:
                    ids[(scene_id, class_id)] += 1
                    trk = InjectTrack(
                        object_id=ids[(scene_id, class_id)], scene_id=scene_id, class_id=class_id, group_id=det.group_id,
                    )
                    trk.boxes.append(box_to_trackbox(det, class_id=class_id, object_id=trk.object_id))
                    trk.scores.append(det.score)
                    tracks.append(trk)
                    used_tracks.add(len(tracks) - 1)
                    new_tracks += 1
                else:
                    trk = tracks[best_i]
                    trk.boxes.append(box_to_trackbox(det, class_id=class_id, object_id=trk.object_id))
                    trk.scores.append(det.score)
                    used_tracks.add(best_i)
            active[tkey] = tracks
    for tracks in active.values():
        completed.extend(tracks)
    accepted = [t for t in completed if t.hits >= min_track_hits]
    out: list[TrackBox] = []
    for trk in accepted:
        if interpolate_gap > 1:
            out.extend(interpolate_track(trk, interpolate_gap))
        else:
            out.extend(trk.boxes)
    return out, {
        'injection_tracks_created': new_tracks,
        'injection_tracks_accepted': len(accepted),
        'injection_boxes_observed': sum(t.hits for t in accepted),
        'injection_boxes_written': len(out),
    }


def run(args: argparse.Namespace) -> dict:
    boxes = read_submission(args.base)
    by_frame: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, box in enumerate(boxes):
        by_frame[(box.scene_id, box.frame_id)].append(idx)

    allowed_groups = {int(x) for x in args.allowed_groups.split(',') if x.strip()}
    inject_groups = {int(x) for x in args.inject_groups.split(',') if x.strip()}
    residuals: dict[tuple[int, int, int], list[dict[str, float]]] = defaultdict(list)
    matched_indices: set[int] = set()
    unmatched_for_injection: dict[tuple[int, int], list[VDetrBox]] = defaultdict(list)
    vdetr_frame_groups = 0
    vdetr_after_nms = 0
    matched_frames = 0
    matched_sample_boxes = 0
    unmatched_candidates = 0

    for key, vdetr_boxes in iter_vdetr_groups(
        args.vdetr,
        min_score=args.min_score,
        allowed_groups=allowed_groups,
        scene_offset=args.scene_offset,
        topk_per_group=args.vdetr_frame_topk_per_group,
        topk_total=args.vdetr_frame_topk_total,
    ):
        base_indices = by_frame.get(key, [])
        if not base_indices and not args.inject_without_base_frame:
            continue
        vdetr_frame_groups += 1
        candidates = nms_vdetr(vdetr_boxes, distance_m=args.vdetr_nms_distance_m)
        vdetr_after_nms += len(candidates)
        pairs: list[tuple[float, int, int]] = []
        for idx in base_indices:
            base = boxes[idx]
            group_id = group_for_class(base.class_id, args.class_mode)
            if group_id is None:
                continue
            for vi, vdetr in enumerate(candidates):
                if vdetr.group_id != group_id:
                    continue
                if compatible_gate(base, vdetr, args.match_gate_m, args.match_z_gate_m):
                    pairs.append((math.hypot(base.x - vdetr.x, base.y - vdetr.y), idx, vi))
        used_base: set[int] = set()
        used_vdetr: set[int] = set()
        frame_matches = 0
        for _dist, idx, vi in sorted(pairs, key=lambda item: item[0]):
            if idx in used_base or vi in used_vdetr:
                continue
            vdetr = candidates[vi]
            base = boxes[idx]
            used_base.add(idx)
            used_vdetr.add(vi)
            matched_indices.add(idx)
            frame_matches += 1
            matched_sample_boxes += 1
            residuals[(base.scene_id, base.class_id, base.object_id)].append(
                {
                    'dx': clamp(vdetr.x - base.x, -args.max_offset_m, args.max_offset_m),
                    'dy': clamp(vdetr.y - base.y, -args.max_offset_m, args.max_offset_m),
                    'dz': clamp(vdetr.z - base.z, -args.max_z_offset_m, args.max_z_offset_m),
                    'dw': clamp(log_ratio(vdetr.width, base.width), -args.max_log_scale, args.max_log_scale),
                    'dl': clamp(log_ratio(vdetr.length, base.length), -args.max_log_scale, args.max_log_scale),
                    'dh': clamp(log_ratio(vdetr.height, base.height), -args.max_log_scale, args.max_log_scale),
                }
            )
        if frame_matches:
            matched_frames += 1
        for vi, vdetr in enumerate(candidates):
            if vi in used_vdetr:
                continue
            if vdetr.score < args.inject_min_score:
                continue
            if vdetr.group_id not in inject_groups:
                continue
            unmatched_for_injection[key].append(vdetr)
            unmatched_candidates += 1

    track_residuals: dict[tuple[int, int, int], dict[str, float]] = {}
    for key, values in residuals.items():
        if len(values) < args.min_track_matches:
            continue
        track_residuals[key] = {name: median(v[name] for v in values) for name in ('dx', 'dy', 'dz', 'dw', 'dl', 'dh')}

    out: list[TrackBox] = []
    corrected = 0
    for box in boxes:
        residual = track_residuals.get((box.scene_id, box.class_id, box.object_id))
        if residual is None or args.correction_strength <= 0.0:
            out.append(box)
        else:
            out.append(corrected_box(box, residual, args.correction_strength, args.center_only_correction))
            corrected += 1

    injected, inject_stats = build_injection_tracks(
        unmatched_for_injection,
        class_mode=args.class_mode,
        inject_groups=inject_groups,
        default_vehicle_class=args.default_vehicle_class,
        base_boxes=boxes,
        track_gate_m=args.inject_track_gate_m,
        max_track_gap=args.inject_max_track_gap,
        min_track_hits=args.inject_min_track_hits,
        interpolate_gap=args.inject_interpolate_gap,
    )
    out.extend(injected)
    out.sort(key=lambda b: (b.scene_id, b.frame_id, b.class_id, b.object_id))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_submission(out, args.out, decimals=args.decimals)
    if args.zip_out:
        args.zip_out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip_out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname='track1.txt')
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    stats = {
        'base': str(args.base),
        'vdetr': str(args.vdetr),
        'out': str(args.out),
        'input_boxes': len(boxes),
        'written': written,
        'vdetr_frame_groups': vdetr_frame_groups,
        'vdetr_after_nms': vdetr_after_nms,
        'matched_frames': matched_frames,
        'matched_sample_boxes': matched_sample_boxes,
        'tracks_with_samples': len(residuals),
        'tracks_corrected': len(track_residuals),
        'boxes_corrected': corrected,
        'unmatched_injection_candidates': unmatched_candidates,
        **inject_stats,
        'params': params,
        'validation': validate_submission(args.out),
    }
    args.out.with_suffix(args.out.suffix + '.json').write_text(json.dumps(stats, indent=2, sort_keys=True), encoding='utf-8')
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fuse V-DETR geometry and inject high-confidence V-DETR-only tracklets.')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--vdetr', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--zip-out', type=Path, default=None)
    parser.add_argument('--class-mode', choices=['merged4', 'seven'], default='merged4')
    parser.add_argument('--allowed-groups', default='0,1,2')
    parser.add_argument('--inject-groups', default='0,1')
    parser.add_argument('--default-vehicle-class', type=int, default=6)
    parser.add_argument('--min-score', type=float, default=0.025)
    parser.add_argument('--vdetr-nms-distance-m', type=float, default=0.75)
    parser.add_argument('--vdetr-frame-topk-per-group', type=int, default=128)
    parser.add_argument('--vdetr-frame-topk-total', type=int, default=512)
    parser.add_argument('--match-gate-m', type=float, default=1.5)
    parser.add_argument('--match-z-gate-m', type=float, default=2.0)
    parser.add_argument('--min-track-matches', type=int, default=3)
    parser.add_argument('--correction-strength', type=float, default=0.05)
    parser.add_argument('--center-only-correction', action='store_true')
    parser.add_argument('--max-offset-m', type=float, default=0.70)
    parser.add_argument('--max-z-offset-m', type=float, default=0.40)
    parser.add_argument('--max-log-scale', type=float, default=0.12)
    parser.add_argument('--inject-min-score', type=float, default=0.04)
    parser.add_argument('--inject-track-gate-m', type=float, default=1.5)
    parser.add_argument('--inject-max-track-gap', type=int, default=45)
    parser.add_argument('--inject-min-track-hits', type=int, default=2)
    parser.add_argument('--inject-interpolate-gap', type=int, default=30)
    parser.add_argument('--inject-without-base-frame', action='store_true')
    parser.add_argument('--scene-offset', type=int, default=0)
    parser.add_argument('--decimals', type=int, default=2)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
