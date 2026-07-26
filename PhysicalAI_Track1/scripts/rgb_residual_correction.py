from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from physicalai_track1.dataset import TrackBox, load_gt_boxes_for_split, scene_name_to_id
from physicalai_track1.evaluator import _match_iou_matrix
from physicalai_track1.geometry import box3d_iou
from physicalai_track1.submission import read_submission, validate_submission, write_submission


TARGETS = ("dx", "dy", "dz", "dw", "dl", "dh")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def log_ratio(target: float, source: float) -> float:
    return math.log(max(target, 1e-4) / max(source, 1e-4))


def feature_vector(box: TrackBox) -> list[float]:
    radius = math.hypot(box.x, box.y)
    dims = [max(box.width, 1e-3), max(box.length, 1e-3), max(box.height, 1e-3)]
    one_hot = [1.0 if box.class_id == i else 0.0 for i in range(7)]
    return [
        1.0,
        box.x / 100.0,
        box.y / 100.0,
        box.z / 5.0,
        math.log1p(radius) / 5.0,
        math.log(dims[0]),
        math.log(dims[1]),
        math.log(dims[2]),
        math.sin(box.yaw),
        math.cos(box.yaw),
        box.frame_id / 9000.0,
        *one_hot,
    ]


def residual(pred: TrackBox, target: TrackBox, *, max_offset: float, max_z: float, max_log_scale: float) -> list[float]:
    return [
        clamp(target.x - pred.x, -max_offset, max_offset),
        clamp(target.y - pred.y, -max_offset, max_offset),
        clamp(target.z - pred.z, -max_z, max_z),
        clamp(log_ratio(target.width, pred.width), -max_log_scale, max_log_scale),
        clamp(log_ratio(target.length, pred.length), -max_log_scale, max_log_scale),
        clamp(log_ratio(target.height, pred.height), -max_log_scale, max_log_scale),
    ]


def group_frame_class(boxes: list[TrackBox]) -> dict[tuple[int, int, int], list[int]]:
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, box in enumerate(boxes):
        grouped[(box.scene_id, box.class_id, box.frame_id)].append(idx)
    return grouped


def match_pred_gt(pred: list[TrackBox], gt: list[TrackBox], iou_threshold: float) -> list[tuple[int, int, float]]:
    pred_by = group_frame_class(pred)
    gt_by = group_frame_class(gt)
    matches: list[tuple[int, int, float]] = []
    for key in sorted(set(pred_by) & set(gt_by)):
        pred_idx = pred_by[key]
        gt_idx = gt_by[key]
        matrix = [[box3d_iou(gt[g], pred[p]) for p in pred_idx] for g in gt_idx]
        for local_g, local_p, iou in _match_iou_matrix(matrix, iou_threshold):
            matches.append((pred_idx[local_p], gt_idx[local_g], iou))
    return matches


def merged_group(class_id: int) -> int | None:
    if class_id == 0:
        return 0
    if class_id == 1:
        return 1
    if class_id in {2, 3, 6}:
        return 2
    if class_id in {4, 5}:
        return 3
    return None


def parse_slice_id(slice_id: str) -> tuple[int, int]:
    parts = slice_id.split("_")
    scene_name = "_".join(parts[:-3])
    return scene_name_to_id(scene_name), int(parts[-3])


def load_teacher_boxes(path: Path | None, min_score: float) -> dict[tuple[int, int, int], list[TrackBox]]:
    out: dict[tuple[int, int, int], list[TrackBox]] = defaultdict(list)
    if path is None or not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = float(row["score"])
            if score < min_score:
                continue
            scene_id, frame_id = parse_slice_id(row["slice_id"])
            group_id = int(row["class_id"])
            box = TrackBox(
                scene_id=scene_id,
                class_id=group_id,
                object_id=1,
                frame_id=frame_id,
                x=float(row["x"]),
                y=float(row["y"]),
                z=float(row["z"]),
                width=float(row["width"]),
                length=float(row["length"]),
                height=float(row["height"]),
                yaw=float(row["yaw"]),
                score=score,
            )
            out[(scene_id, group_id, frame_id)].append(box)
    for key, values in list(out.items()):
        values.sort(key=lambda item: item.score, reverse=True)
        out[key] = values[:80]
    return out


def nearest_teacher(pred: TrackBox, teacher: dict[tuple[int, int, int], list[TrackBox]], gate_m: float) -> TrackBox | None:
    group_id = merged_group(pred.class_id)
    if group_id is None:
        return None
    candidates = teacher.get((pred.scene_id, group_id, pred.frame_id), [])
    best = None
    best_dist = float("inf")
    for item in candidates:
        dist = math.hypot(pred.x - item.x, pred.y - item.y)
        if dist < best_dist and dist <= gate_m:
            best = item
            best_dist = dist
    return best


def fit_ridge(features: np.ndarray, targets: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[:, 1:].mean(axis=0)
    std = features[:, 1:].std(axis=0)
    std[std < 1e-6] = 1.0
    x = features.copy()
    x[:, 1:] = (x[:, 1:] - mean) / std
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + ridge * eye, x.T @ targets)
    return coef, mean, std


def train(args: argparse.Namespace) -> dict:
    pred = read_submission(args.pred)
    gt = load_gt_boxes_for_split(
        args.data_root,
        2026,
        args.split,
        scenes=args.scenes.split() if args.scenes else None,
        max_frames_per_scene=args.max_frames,
        frame_stride=args.frame_stride,
    )
    matches = match_pred_gt(pred, gt, args.iou_threshold)
    teacher = load_teacher_boxes(args.teacher_vdetr, args.teacher_min_score)

    xs = []
    ys = []
    teacher_used = 0
    teacher_better = 0
    for pred_idx, gt_idx, _iou in matches:
        p = pred[pred_idx]
        g = gt[gt_idx]
        target = residual(p, g, max_offset=args.max_offset_m, max_z=args.max_z_offset_m, max_log_scale=args.max_log_scale)
        t = nearest_teacher(p, teacher, args.teacher_gate_m)
        if t is not None and args.teacher_weight > 0:
            teacher_used += 1
            teacher_res = residual(p, t, max_offset=args.max_offset_m, max_z=args.max_z_offset_m, max_log_scale=args.max_log_scale)
            pred_err = math.hypot(p.x - g.x, p.y - g.y) + abs(p.z - g.z)
            teacher_err = math.hypot(t.x - g.x, t.y - g.y) + abs(t.z - g.z)
            if teacher_err < pred_err:
                teacher_better += 1
                weight = args.teacher_weight
                target = [(1.0 - weight) * a + weight * b for a, b in zip(target, teacher_res)]
        xs.append(feature_vector(p))
        ys.append(target)

    if len(xs) < 10:
        raise SystemExit(f"not enough matches: {len(xs)}")
    features = np.asarray(xs, dtype=np.float64)
    targets = np.asarray(ys, dtype=np.float64)
    coef, mean, std = fit_ridge(features, targets, args.ridge)
    pred_targets = features.copy()
    pred_targets[:, 1:] = (pred_targets[:, 1:] - mean) / std
    pred_targets = pred_targets @ coef
    mae = np.abs(pred_targets - targets).mean(axis=0)
    payload = {
        "coef": coef.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "targets": list(TARGETS),
        "feature_count": int(features.shape[1]),
        "params": {
            "max_offset_m": args.max_offset_m,
            "max_z_offset_m": args.max_z_offset_m,
            "max_log_scale": args.max_log_scale,
        },
        "stats": {
            "matches": len(matches),
            "train_rows": len(xs),
            "teacher_used": teacher_used,
            "teacher_better": teacher_better,
            "mae": {name: float(value) for name, value in zip(TARGETS, mae)},
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def apply(args: argparse.Namespace) -> dict:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    coef = np.asarray(model["coef"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    max_offset = float(model["params"]["max_offset_m"])
    max_z = float(model["params"]["max_z_offset_m"])
    max_log_scale = float(model["params"]["max_log_scale"])

    boxes = read_submission(args.input)
    out: list[TrackBox] = []
    corrected = 0
    for box in boxes:
        x = np.asarray(feature_vector(box), dtype=np.float64)
        x[1:] = (x[1:] - mean) / std
        r = x @ coef
        dx = clamp(float(r[0]) * args.strength, -max_offset, max_offset)
        dy = clamp(float(r[1]) * args.strength, -max_offset, max_offset)
        dz = clamp(float(r[2]) * args.strength, -max_z, max_z)
        dw = clamp(float(r[3]) * args.strength, -max_log_scale, max_log_scale)
        dl = clamp(float(r[4]) * args.strength, -max_log_scale, max_log_scale)
        dh = clamp(float(r[5]) * args.strength, -max_log_scale, max_log_scale)
        out.append(
            TrackBox(
                scene_id=box.scene_id,
                class_id=box.class_id,
                object_id=box.object_id,
                frame_id=box.frame_id,
                x=box.x + dx,
                y=box.y + dy,
                z=box.z + dz,
                width=box.width * math.exp(dw),
                length=box.length * math.exp(dl),
                height=box.height * math.exp(dh),
                yaw=box.yaw,
                score=box.score,
            )
        )
        corrected += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = write_submission(out, args.out, decimals=args.decimals)
    if args.zip_out:
        with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname="track1.txt")
    stats = {
        "input": str(args.input),
        "model": str(args.model),
        "out": str(args.out),
        "written": written,
        "corrected": corrected,
        "strength": args.strength,
        "validation": validate_submission(args.out),
    }
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/apply RGB-only 3D residual correction distilled from GT/V-DETR.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--pred", type=Path, required=True)
    tr.add_argument("--data-root", type=Path, required=True)
    tr.add_argument("--split", default="val")
    tr.add_argument("--scenes", default="Warehouse_020 Warehouse_021 Warehouse_022")
    tr.add_argument("--max-frames", type=int, default=600)
    tr.add_argument("--frame-stride", type=int, default=30)
    tr.add_argument("--teacher-vdetr", type=Path, default=None)
    tr.add_argument("--teacher-min-score", type=float, default=0.20)
    tr.add_argument("--teacher-gate-m", type=float, default=2.0)
    tr.add_argument("--teacher-weight", type=float, default=0.25)
    tr.add_argument("--iou-threshold", type=float, default=0.05)
    tr.add_argument("--max-offset-m", type=float, default=1.25)
    tr.add_argument("--max-z-offset-m", type=float, default=0.75)
    tr.add_argument("--max-log-scale", type=float, default=0.25)
    tr.add_argument("--ridge", type=float, default=10.0)
    tr.add_argument("--model-out", type=Path, required=True)
    tr.set_defaults(func=train)

    ap = sub.add_parser("apply")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--zip-out", type=Path, default=None)
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--decimals", type=int, default=2)
    ap.set_defaults(func=apply)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(args.func(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
