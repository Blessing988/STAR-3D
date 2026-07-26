#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def replace_link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            raise SystemExit(f"Refusing to replace real directory: {link}")
        link.unlink()
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RF-DETR COCO folder layout from Track1 YOLO/COCO export.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, default=None)
    parser.add_argument("--val-json", type=Path, default=None)
    args = parser.parse_args()

    source = args.source_dir
    out = args.out_dir
    train_images = source / "images" / "train"
    val_images = source / "images" / "val"
    train_json = args.train_json or source / "annotations" / "instances_train_class_balanced.json"
    val_json = args.val_json or source / "annotations" / "instances_val.json"

    for path in (train_images, val_images, train_json, val_json):
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_json, train_images / "_annotations.coco.json")
    shutil.copy2(val_json, val_images / "_annotations.coco.json")
    replace_link(out / "train", train_images)
    replace_link(out / "valid", val_images)

    print(f"dataset_dir={out}")
    print(f"train={os.path.realpath(out / 'train')}")
    print(f"valid={os.path.realpath(out / 'valid')}")
    print(f"train_annotations={train_images / '_annotations.coco.json'}")
    print(f"valid_annotations={val_images / '_annotations.coco.json'}")


if __name__ == "__main__":
    main()
