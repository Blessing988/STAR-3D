#!/usr/bin/env python
import json
import os


SRC_ROOT = "/path/to/scratch/PhysicalAI_Track1/datasets/yolo_2026_stride30"
OUT_ROOT = "/path/to/scratch/PhysicalAI_Track1/datasets/rfdetr_2026_balanced_zero"


def write_split(split: str, src_ann: str) -> None:
    with open(os.path.join(SRC_ROOT, "annotations", src_ann), "r") as f:
        data = json.load(f)
    for category in data["categories"]:
        category["id"] = int(category["id"]) - 1
    for ann in data["annotations"]:
        ann["category_id"] = int(ann["category_id"]) - 1
    out_ann = os.path.join(OUT_ROOT, f"_{split}_annotations.coco.json")
    with open(out_ann, "w") as f:
        json.dump(data, f)
    labels = [ann["category_id"] for ann in data["annotations"]]
    print(split, len(data["images"]), len(data["annotations"]), min(labels), max(labels))


def relink(path: str, target: str) -> None:
    if os.path.lexists(path):
        os.unlink(path)
    os.symlink(target, path)


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    write_split("train", "instances_train_class_balanced.json")
    write_split("valid", "instances_val.json")
    write_split("test", "instances_val.json")
    relink(os.path.join(OUT_ROOT, "train"), os.path.join(SRC_ROOT, "images", "train"))
    relink(os.path.join(OUT_ROOT, "valid"), os.path.join(SRC_ROOT, "images", "val"))
    relink(os.path.join(OUT_ROOT, "test"), os.path.join(SRC_ROOT, "images", "val"))
    relink(
        os.path.join(SRC_ROOT, "images", "train", "_annotations.coco.json"),
        os.path.join(OUT_ROOT, "_train_annotations.coco.json"),
    )
    relink(
        os.path.join(SRC_ROOT, "images", "val", "_annotations.coco.json"),
        os.path.join(OUT_ROOT, "_valid_annotations.coco.json"),
    )


if __name__ == "__main__":
    main()
