from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .dataset import TrackBox, iter_scene_dirs, load_gt_boxes_for_split
from .geometry import angle_distance, box3d_iou, distance_xy

FEATURE_NAMES = [
    "dt",
    "log_dt",
    "dist_xy",
    "speed_xy",
    "dist_over_size",
    "abs_dz",
    "yaw_diff_norm",
    "iou3d",
    "abs_log_width_ratio",
    "abs_log_length_ratio",
    "abs_log_height_ratio",
    "width_a",
    "length_a",
    "height_a",
    "width_b",
    "length_b",
    "height_b",
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
]


def _safe_log_ratio(a: float, b: float) -> float:
    return abs(math.log(max(a, 1e-6) / max(b, 1e-6)))


def association_features(a: TrackBox, b: TrackBox) -> list[float]:
    dt = max(1, int(b.frame_id) - int(a.frame_id))
    dist = distance_xy(a, b)
    size = max(0.5, 0.25 * (a.width + a.length + b.width + b.length))
    features = [
        float(dt),
        math.log1p(float(dt)),
        dist,
        dist / float(dt),
        dist / size,
        abs(a.z - b.z),
        angle_distance(a.yaw, b.yaw) / math.pi,
        box3d_iou(a, b),
        _safe_log_ratio(a.width, b.width),
        _safe_log_ratio(a.length, b.length),
        _safe_log_ratio(a.height, b.height),
        a.width,
        a.length,
        a.height,
        b.width,
        b.length,
        b.height,
    ]
    features.extend(1.0 if a.class_id == class_id else 0.0 for class_id in range(7))
    return features


@dataclass
class AssociationScorer:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float

    @classmethod
    def load(cls, path: Path | str) -> "AssociationScorer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            feature_names=list(payload["feature_names"]),
            mean=[float(v) for v in payload["mean"]],
            std=[float(v) for v in payload["std"]],
            weights=[float(v) for v in payload["weights"]],
            bias=float(payload["bias"]),
        )

    def predict_proba(self, a: TrackBox, b: TrackBox) -> float:
        raw = association_features(a, b)
        z = self.bias
        for value, mean, std, weight in zip(raw, self.mean, self.std, self.weights):
            z += ((value - mean) / max(std, 1e-6)) * weight
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _reservoir_add(items: list[list[float]], item: list[float], seen: int, limit: int, rng: random.Random) -> None:
    if len(items) < limit:
        items.append(item)
        return
    idx = rng.randint(0, seen)
    if idx < limit:
        items[idx] = item


def _load_boxes_by_keys(
    data_root: Path | str,
    year: int,
    split: str,
    scenes: Sequence[str] | None,
    max_frames_per_scene: int | None,
    frame_stride: int,
) -> tuple[dict[tuple[int, int, int], list[TrackBox]], dict[tuple[int, int, int], list[TrackBox]]]:
    boxes = load_gt_boxes_for_split(
        data_root=data_root,
        year=year,
        split=split,
        scenes=scenes,
        max_frames_per_scene=max_frames_per_scene,
        frame_stride=frame_stride,
    )
    by_object: dict[tuple[int, int, int], list[TrackBox]] = defaultdict(list)
    by_frame: dict[tuple[int, int, int], list[TrackBox]] = defaultdict(list)
    for box in boxes:
        by_object[(box.scene_id, box.class_id, box.object_id)].append(box)
        by_frame[(box.scene_id, box.class_id, box.frame_id)].append(box)
    for trajectory in by_object.values():
        trajectory.sort(key=lambda b: b.frame_id)
    return by_object, by_frame


def build_association_dataset(
    data_root: Path | str,
    year: int,
    split: str,
    out_path: Path | str,
    scenes: Sequence[str] | None = None,
    max_frames_per_scene: int | None = None,
    frame_stride: int = 1,
    positive_steps: Sequence[int] = (1, 5, 15, 30),
    negative_frame_tolerance: int = 2,
    negatives_per_positive: int = 2,
    max_samples_per_class_label: int = 50000,
    seed: int = 2026,
) -> dict:
    rng = random.Random(seed)
    by_object, by_frame = _load_boxes_by_keys(
        data_root=data_root,
        year=year,
        split=split,
        scenes=scenes,
        max_frames_per_scene=max_frames_per_scene,
        frame_stride=frame_stride,
    )

    samples: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    seen: dict[tuple[int, int], int] = defaultdict(int)
    stats: dict[str, object] = {
        "split": split,
        "frame_stride": frame_stride,
        "positive_steps": list(positive_steps),
        "negative_frame_tolerance": negative_frame_tolerance,
        "negatives_per_positive": negatives_per_positive,
        "max_samples_per_class_label": max_samples_per_class_label,
        "objects": len(by_object),
        "by_class_label_seen": defaultdict(int),
    }

    for (scene_id, class_id, object_id), trajectory in by_object.items():
        if len(trajectory) < 2:
            continue
        for index, current in enumerate(trajectory):
            for step in positive_steps:
                next_index = index + int(step)
                if next_index >= len(trajectory):
                    continue
                future = trajectory[next_index]
                if future.frame_id <= current.frame_id:
                    continue
                positive_key = (class_id, 1)
                positive = [1.0, *association_features(current, future)]
                seen[positive_key] += 1
                _reservoir_add(samples[positive_key], positive, seen[positive_key], max_samples_per_class_label, rng)

                candidate_frames = range(
                    future.frame_id - negative_frame_tolerance,
                    future.frame_id + negative_frame_tolerance + 1,
                )
                negatives: list[TrackBox] = []
                for frame_id in candidate_frames:
                    for other in by_frame.get((scene_id, class_id, frame_id), []):
                        if other.object_id != object_id:
                            negatives.append(other)
                negatives.sort(key=lambda box: distance_xy(future, box))
                if negatives_per_positive > 0:
                    hard = negatives[: max(negatives_per_positive * 4, negatives_per_positive)]
                    rng.shuffle(hard)
                    for other in hard[:negatives_per_positive]:
                        negative_key = (class_id, 0)
                        negative = [0.0, *association_features(current, other)]
                        seen[negative_key] += 1
                        _reservoir_add(
                            samples[negative_key],
                            negative,
                            seen[negative_key],
                            max_samples_per_class_label,
                            rng,
                        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["label", *FEATURE_NAMES])
        for key in sorted(samples):
            for row in samples[key]:
                writer.writerow([f"{value:.8g}" for value in row])
                rows += 1

    stats["rows"] = rows
    stats["samples"] = {f"class_{cls}_label_{label}": len(values) for (cls, label), values in sorted(samples.items())}
    stats["seen"] = {f"class_{cls}_label_{label}": count for (cls, label), count in sorted(seen.items())}
    stats_path = out.with_suffix(out.suffix + ".json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return {"output": str(out), "stats": str(stats_path), "rows": rows}


def _load_tsv(path: Path | str):
    import numpy as np

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header != ["label", *FEATURE_NAMES]:
            raise ValueError(f"Unexpected association dataset header: {header}")
        rows = [[float(value) for value in row] for row in reader if row]
    data = np.asarray(rows, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"No samples found in {path}")
    return data[:, 1:], data[:, 0]


def train_association_model(
    dataset_path: Path | str,
    out_path: Path | str,
    epochs: int = 80,
    batch_size: int = 4096,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    seed: int = 2026,
) -> dict:
    import numpy as np

    x, y = _load_tsv(dataset_path)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    val_size = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    weights = np.zeros(x_train.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    history = []

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            xb = x_train[batch]
            yb = y_train[batch]
            pred = sigmoid(xb @ weights + bias)
            grad = pred - yb
            weights -= learning_rate * ((xb.T @ grad) / len(batch) + weight_decay * weights)
            bias -= learning_rate * float(grad.mean())

        train_pred = sigmoid(x_train @ weights + bias)
        val_pred = sigmoid(x_val @ weights + bias)
        train_loss = float(
            -np.mean(y_train * np.log(train_pred + 1e-8) + (1.0 - y_train) * np.log(1.0 - train_pred + 1e-8))
        )
        val_loss = float(-np.mean(y_val * np.log(val_pred + 1e-8) + (1.0 - y_val) * np.log(1.0 - val_pred + 1e-8)))
        val_acc = float(np.mean((val_pred >= 0.5) == (y_val >= 0.5)))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

    payload = {
        "feature_names": FEATURE_NAMES,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "bias": float(bias),
        "metadata": {
            "dataset": str(dataset_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "val_fraction": val_fraction,
            "seed": seed,
            "samples": int(len(y)),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "final": history[-1],
        },
        "history": history,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"output": str(out), "samples": int(len(y)), "final": history[-1]}
