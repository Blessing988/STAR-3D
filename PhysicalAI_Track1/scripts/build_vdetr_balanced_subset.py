from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def labels_in_gt(path: Path) -> set[int]:
    labels: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                labels.add(int(parts[0]))
    return labels


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_split(src_root: Path, split: str) -> dict[str, set[int]]:
    gt_dir = src_root / split / "gt"
    result = {}
    for gt in sorted(gt_dir.glob("*.txt")):
        pcd = src_root / split / "pcd" / f"{gt.stem}.ply"
        if not pcd.exists():
            continue
        labels = labels_in_gt(gt)
        if labels:
            result[gt.stem] = labels
    return result


def choose_weight(labels: set[int], class_weights: dict[int, int], default_weight: int) -> int:
    return max([default_weight] + [class_weights.get(label, default_weight) for label in labels])


def materialize(
    src_root: Path,
    out_root: Path,
    split: str,
    selected: list[tuple[str, int]],
    mode: str,
) -> Counter[int]:
    class_counts: Counter[int] = Counter()
    for stem, copy_idx in selected:
        src_pcd = src_root / split / "pcd" / f"{stem}.ply"
        src_gt = src_root / split / "gt" / f"{stem}.txt"
        suffix = "" if copy_idx == 0 else f"__dup{copy_idx:02d}"
        dst_stem = f"{stem}{suffix}"
        link_or_copy(src_pcd, out_root / split / "pcd" / f"{dst_stem}.ply", mode)
        link_or_copy(src_gt, out_root / split / "gt" / f"{dst_stem}.txt", mode)
        class_counts.update(labels_in_gt(src_gt))
    return class_counts


def parse_class_weights(items: list[str]) -> dict[int, int]:
    weights = {}
    for item in items:
        key, value = item.split(":", 1)
        weights[int(key)] = max(1, int(value))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rare-balanced V-DETR dataset with symlink duplication.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--max-train", type=int, default=60000)
    parser.add_argument("--max-person-only", type=int, default=12000)
    parser.add_argument("--default-weight", type=int, default=1)
    parser.add_argument(
        "--class-weights",
        nargs="+",
        default=["0:1", "1:2", "2:8", "3:6", "4:8", "5:8", "6:4"],
        help="class_id:duplication_weight pairs.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    class_weights = parse_class_weights(args.class_weights)
    train_labels = collect_split(args.src_root, "train")
    val_labels = collect_split(args.src_root, "val")

    by_key = defaultdict(list)
    for stem, labels in train_labels.items():
        if labels == {0}:
            by_key["person_only"].append(stem)
        elif labels & {2, 3, 4, 5}:
            by_key["rare"].append(stem)
        elif labels & {1, 6}:
            by_key["vehicle"].append(stem)
        else:
            by_key["mixed"].append(stem)

    person_only = by_key["person_only"]
    random.shuffle(person_only)
    allowed_person = set(person_only[: args.max_person_only])

    selected: list[tuple[str, int]] = []
    source_presence: Counter[int] = Counter()
    selected_presence: Counter[int] = Counter()
    for stem, labels in train_labels.items():
        source_presence.update(labels)
        if labels == {0} and stem not in allowed_person:
            continue
        weight = choose_weight(labels, class_weights, args.default_weight)
        for dup in range(weight):
            selected.append((stem, dup))
            selected_presence.update(labels)
    random.shuffle(selected)
    selected = selected[: args.max_train]

    args.out_root.mkdir(parents=True, exist_ok=True)
    train_counts = materialize(args.src_root, args.out_root, "train", selected, args.mode)
    val_selected = [(stem, 0) for stem in sorted(val_labels)]
    val_counts = materialize(args.src_root, args.out_root, "val", val_selected, args.mode)

    report = {
        "source_train_files": len(train_labels),
        "source_val_files": len(val_labels),
        "selected_train_entries": len(selected),
        "selected_val_entries": len(val_selected),
        "class_weights": class_weights,
        "source_train_class_presence": dict(sorted(source_presence.items())),
        "selected_train_class_presence": dict(sorted(selected_presence.items())),
        "materialized_train_class_presence": dict(sorted(train_counts.items())),
        "materialized_val_class_presence": dict(sorted(val_counts.items())),
    }
    print(report)


if __name__ == "__main__":
    main()
