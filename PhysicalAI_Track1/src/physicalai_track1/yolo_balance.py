from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .constants import ID_TO_CLASS


def _read_label_classes(label_path: Path) -> list[int]:
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        classes.append(int(float(stripped.split()[0])))
    return classes


def _stochastic_round(value: float, rng: random.Random) -> int:
    base = int(value)
    if rng.random() < value - base:
        base += 1
    return max(1, base)


def _write_yolo_yaml(dataset_dir: Path, train_list: Path, val_path: str, out_yaml: Path) -> None:
    lines = [
        f"path: {dataset_dir}",
        f"train: {train_list}",
        f"val: {val_path}",
        "names:",
        *[f"  {idx}: {name}" for idx, name in sorted(ID_TO_CLASS.items())],
        "",
    ]
    out_yaml.write_text("\n".join(lines), encoding="utf-8")


def build_balanced_yolo_train_list(
    dataset_dir: Path | str,
    split: str = "train",
    out_list: Path | str | None = None,
    out_yaml: Path | str | None = None,
    val_path: str = "images/val",
    power: float = 0.5,
    max_repeat: float = 4.0,
    seed: int = 2026,
) -> dict:
    """Build an Ultralytics-compatible repeated image list for class balancing.

    The repeat factor is computed from class instance counts:

        repeat_c = min(max_repeat, max(1, (max_count / count_c) ** power))

    Each image receives the maximum repeat factor of the classes in that image.
    This is deliberately data-level balancing; it avoids patching detector loss
    internals and keeps the original labels unchanged.
    """
    dataset = Path(dataset_dir)
    labels_root = dataset / "labels" / split
    images_root = dataset / "images" / split
    if not labels_root.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_root}")
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images directory: {images_root}")

    out_list_path = Path(out_list) if out_list else dataset / f"{split}_class_balanced.txt"
    out_yaml_path = Path(out_yaml) if out_yaml else dataset / "aicity_track1_2026_class_balanced.yaml"

    image_labels: list[tuple[Path, list[int]]] = []
    raw_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()

    for label_path in sorted(labels_root.glob("*.txt")):
        classes = _read_label_classes(label_path)
        image_path = images_root / f"{label_path.stem}.jpg"
        if not image_path.exists() or not classes:
            continue
        image_labels.append((image_path, classes))
        raw_counts.update(classes)
        image_counts.update(set(classes))

    if not image_labels:
        raise RuntimeError(f"No labeled images found under {labels_root}")

    max_count = max(raw_counts.values())
    repeat_by_class: dict[int, float] = {}
    for class_id in sorted(ID_TO_CLASS):
        count = raw_counts.get(class_id, 0)
        if count <= 0:
            repeat_by_class[class_id] = 1.0
        else:
            repeat_by_class[class_id] = min(max_repeat, max(1.0, (max_count / count) ** power))

    rng = random.Random(seed)
    repeated_images: list[str] = []
    effective_counts: Counter[int] = Counter()
    effective_image_counts: Counter[int] = Counter()
    repeat_hist: Counter[int] = Counter()

    for image_path, classes in image_labels:
        repeat_float = max(repeat_by_class[class_id] for class_id in set(classes))
        repeat = _stochastic_round(repeat_float, rng)
        repeat_hist[repeat] += 1
        image_str = str(image_path)
        for _ in range(repeat):
            repeated_images.append(image_str)
            effective_counts.update(classes)
            effective_image_counts.update(set(classes))

    out_list_path.parent.mkdir(parents=True, exist_ok=True)
    out_list_path.write_text("\n".join(repeated_images) + "\n", encoding="utf-8")
    _write_yolo_yaml(dataset, out_list_path, val_path, out_yaml_path)

    result = {
        "dataset_dir": str(dataset),
        "split": split,
        "source_images": len(image_labels),
        "balanced_rows": len(repeated_images),
        "out_list": str(out_list_path),
        "out_yaml": str(out_yaml_path),
        "power": power,
        "max_repeat": max_repeat,
        "seed": seed,
        "repeat_by_class": {
            ID_TO_CLASS[class_id]: round(repeat_by_class[class_id], 4) for class_id in sorted(ID_TO_CLASS)
        },
        "image_counts_by_class": {
            ID_TO_CLASS[class_id]: image_counts.get(class_id, 0) for class_id in sorted(ID_TO_CLASS)
        },
        "raw_instances_by_class": {
            ID_TO_CLASS[class_id]: raw_counts.get(class_id, 0) for class_id in sorted(ID_TO_CLASS)
        },
        "effective_instances_by_class": {
            ID_TO_CLASS[class_id]: effective_counts.get(class_id, 0) for class_id in sorted(ID_TO_CLASS)
        },
        "effective_images_by_class": {
            ID_TO_CLASS[class_id]: effective_image_counts.get(class_id, 0) for class_id in sorted(ID_TO_CLASS)
        },
        "repeat_histogram": dict(sorted(repeat_hist.items())),
    }

    stats_path = out_list_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["stats_json"] = str(stats_path)
    return result
