from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from physicalai_track1.dataset import TrackBox, load_gt_boxes_for_split
from physicalai_track1.evaluator import evaluate_by_class, evaluate_hota_like
from physicalai_track1.submission import read_submission, validate_submission, write_submission


def parse_variant(text: str) -> tuple[str, Path, Path, Path | None]:
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise ValueError("--variant must be name:val_track1:test_track1[:eval_json]")
    name, val_path, test_path = parts[:3]
    eval_path = Path(parts[3]) if len(parts) == 4 and parts[3] else None
    return name, Path(val_path), Path(test_path), eval_path


def class_id_from_metrics(payload: dict) -> int:
    class_id = payload.get("class_id")
    if class_id is None:
        raise KeyError(f"missing class_id in {payload}")
    return int(class_id)


def select_by_class(
    gt_boxes: list[TrackBox],
    variants: list[tuple[str, Path, Path, Path | None]],
    metric: str,
    deta_weight: float,
    assa_weight: float,
    loca_weight: float,
) -> tuple[dict[int, str], dict]:
    scores_by_class: dict[int, dict[str, dict]] = defaultdict(dict)
    overall: dict[str, dict] = {}

    for name, val_path, _, eval_path in variants:
        if eval_path is not None and eval_path.exists():
            cached = json.loads(eval_path.read_text(encoding="utf-8"))
            overall[name] = {
                key: cached.get(key, 0.0)
                for key in ("hota_like", "deta", "assa", "loca")
            }
            per_class = cached.get("per_class", {})
        else:
            pred_boxes = read_submission(val_path)
            overall[name] = evaluate_hota_like(gt_boxes, pred_boxes)
            per_class = evaluate_by_class(gt_boxes, pred_boxes)
        for _, metrics in per_class.items():
            class_id = class_id_from_metrics(metrics)
            score = float(metrics.get(metric, 0.0))
            score += deta_weight * float(metrics.get("deta", 0.0))
            score += assa_weight * float(metrics.get("assa", 0.0))
            score += loca_weight * float(metrics.get("loca", 0.0))
            scores_by_class[class_id][name] = {
                "score": score,
                "metrics": metrics,
                "val_path": str(val_path),
            }

    selected: dict[int, str] = {}
    for class_id, options in sorted(scores_by_class.items()):
        selected[class_id] = max(options.items(), key=lambda item: item[1]["score"])[0]

    return selected, {"overall": overall, "per_class_options": scores_by_class}


def merge_test_boxes(
    variants: list[tuple[str, Path, Path, Path | None]],
    selected: dict[int, str],
) -> list[TrackBox]:
    by_name = {name: test_path for name, _, test_path, _ in variants}
    by_class: dict[int, list[TrackBox]] = defaultdict(list)
    for class_id, name in selected.items():
        for box in read_submission(by_name[name]):
            if int(box.class_id) == int(class_id):
                by_class[class_id].append(box)

    boxes: list[TrackBox] = []
    for class_id in sorted(by_class):
        boxes.extend(by_class[class_id])

    return sorted(boxes, key=lambda b: (b.scene_id, b.frame_id, b.class_id, b.object_id))


def write_zip(track_path: Path) -> Path:
    zip_path = track_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(track_path, arcname="track1.txt")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classwise Track 1 hybrid from val-selected variants.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--split", default="val")
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-frames-per-scene", type=int, default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="name:val_track1:test_track1[:eval_json]",
    )
    parser.add_argument("--metric", default="hota_like", choices=["hota_like", "deta", "assa", "loca"])
    parser.add_argument("--deta-weight", type=float, default=0.0)
    parser.add_argument("--assa-weight", type=float, default=0.0)
    parser.add_argument("--loca-weight", type=float, default=0.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()

    variants = [parse_variant(item) for item in args.variant]
    all_cached = all(item[3] is not None and item[3].exists() for item in variants)
    gt_boxes = []
    if not all_cached:
        gt_boxes = load_gt_boxes_for_split(
            args.data_root,
            args.year,
            args.split,
            scenes=args.scenes,
            max_frames_per_scene=args.max_frames_per_scene,
            frame_stride=args.frame_stride,
        )
    selected, report = select_by_class(
        gt_boxes=gt_boxes,
        variants=variants,
        metric=args.metric,
        deta_weight=args.deta_weight,
        assa_weight=args.assa_weight,
        loca_weight=args.loca_weight,
    )
    boxes = merge_test_boxes(variants, selected)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    track_path = out_dir / "track1.txt"
    count = write_submission(boxes, track_path, decimals=args.decimals)
    zip_path = write_zip(track_path)

    report["selected"] = selected
    report["output"] = {
        "track1": str(track_path),
        "zip": str(zip_path),
        "boxes": count,
        "validate": validate_submission(track_path),
    }
    (out_dir / "classwise_hybrid_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["output"], indent=2, sort_keys=True))
    print(json.dumps({"selected": selected}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
