from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def labels_in_gt(path: Path) -> set[int]:
    labels: set[int] = set()
    if not path.exists():
        return labels
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


def select_train_files(
    gt_dir: Path,
    rare_labels: set[int],
    max_common: int,
    max_total: int,
    seed: int,
) -> list[str]:
    rare: list[str] = []
    common: list[str] = []
    class_counts: Counter[int] = Counter()
    file_classes: dict[str, set[int]] = {}

    for gt_path in sorted(gt_dir.glob("*.txt")):
        labels = labels_in_gt(gt_path)
        if not labels:
            continue
        file_classes[gt_path.stem] = labels
        class_counts.update(labels)
        if labels & rare_labels:
            rare.append(gt_path.stem)
        else:
            common.append(gt_path.stem)

    rng = random.Random(seed)
    rng.shuffle(common)
    selected = list(dict.fromkeys(rare + common[:max_common]))

    if len(selected) > max_total:
        rare_set = set(rare)
        selected_rare = [x for x in selected if x in rare_set]
        selected_common = [x for x in selected if x not in rare_set]
        rng.shuffle(selected_common)
        selected = selected_rare + selected_common[: max(0, max_total - len(selected_rare))]

    selected_counts: Counter[int] = Counter()
    for stem in selected:
        selected_counts.update(file_classes.get(stem, set()))

    print(f"source_files={len(file_classes)}")
    print(f"rare_files={len(rare)}")
    print(f"common_files={len(common)}")
    print(f"selected_files={len(selected)}")
    print(f"source_class_presence={dict(sorted(class_counts.items()))}")
    print(f"selected_class_presence={dict(sorted(selected_counts.items()))}")
    return sorted(selected)


def materialize_split(src_root: Path, out_root: Path, split: str, stems: list[str] | None, mode: str) -> int:
    src_pcd = src_root / split / "pcd"
    src_gt = src_root / split / "gt"
    out_pcd = out_root / split / "pcd"
    out_gt = out_root / split / "gt"
    if stems is None:
        stems = sorted(p.stem for p in src_pcd.glob("*.ply"))

    count = 0
    for stem in stems:
        ply = src_pcd / f"{stem}.ply"
        gt = src_gt / f"{stem}.txt"
        if not ply.exists() or not gt.exists():
            continue
        link_or_copy(ply, out_pcd / ply.name, mode)
        link_or_copy(gt, out_gt / gt.name, mode)
        count += 1
    print(f"{split}_written={count}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small rare-class-balanced V-DETR slice dataset.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--rare-labels", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--max-common", type=int, default=12000)
    parser.add_argument("--max-total", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_stems = select_train_files(
        args.src_root / "train" / "gt",
        set(args.rare_labels),
        args.max_common,
        args.max_total,
        args.seed,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    materialize_split(args.src_root, args.out_root, "train", train_stems, args.mode)
    materialize_split(args.src_root, args.out_root, "val", None, args.mode)


if __name__ == "__main__":
    main()
