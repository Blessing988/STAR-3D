from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .dataset import load_gt_boxes_for_split
from .evaluator import evaluate_hota_like
from .fusion import fuse_candidates, read_fused_detections, write_fused_detections
from .lifting import read_lifted_candidates
from .submission import write_submission
from .tracker import online_track


def sweep_tracker_parameters(
    fused_path: Path | str,
    data_root: Path | str,
    year: int,
    split: str,
    scenes: Sequence[str],
    out_dir: Path | str,
    min_scores: Sequence[float],
    max_costs: Sequence[float],
    max_distances_m: Sequence[float],
    max_age: int = 45,
    frame_stride: int = 1,
    max_frames_per_scene: int | None = None,
    decimals: int = 6,
) -> dict:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    detections = list(read_fused_detections(fused_path))
    gt_boxes = load_gt_boxes_for_split(
        data_root=data_root,
        year=year,
        split=split,
        scenes=scenes,
        max_frames_per_scene=max_frames_per_scene,
        frame_stride=frame_stride,
    )

    results = []
    best_summary = None
    best_metrics = None
    best_tracks = None
    for min_score in min_scores:
        for max_cost in max_costs:
            for max_distance_m in max_distances_m:
                tracks = online_track(
                    detections,
                    max_distance_m=float(max_distance_m),
                    max_age=max_age,
                    min_score=float(min_score),
                    max_cost=float(max_cost),
                )
                metrics = evaluate_hota_like(gt_boxes, tracks)
                summary = {
                    "min_score": float(min_score),
                    "max_cost": float(max_cost),
                    "max_distance_m": float(max_distance_m),
                    "boxes": len(tracks),
                    "objects": len(
                        {
                            (box.scene_id, box.class_id, box.object_id)
                            for box in tracks
                        }
                    ),
                    "hota_like": metrics["hota_like"],
                    "deta": metrics["deta"],
                    "assa": metrics["assa"],
                    "loca": metrics["loca"],
                }
                results.append(summary)
                if best_summary is None or summary["hota_like"] > best_summary["hota_like"]:
                    best_summary = summary
                    best_metrics = metrics
                    best_tracks = tracks

    if best_summary is None or best_metrics is None or best_tracks is None:
        raise ValueError("The parameter sweep is empty")

    results.sort(key=lambda item: item["hota_like"], reverse=True)
    best_submission = output / "track1_best.txt"
    write_submission(best_tracks, best_submission, decimals=decimals)
    payload = {
        "fused": str(fused_path),
        "gt_boxes": len(gt_boxes),
        "fused_detections": len(detections),
        "trials": len(results),
        "best": best_summary,
        "best_submission": str(best_submission),
        "best_metrics": best_metrics,
        "results": results,
    }
    results_path = output / "sweep_results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "results": str(results_path),
        "best_submission": str(best_submission),
        "trials": len(results),
        "best": best_summary,
    }


def sweep_class_thresholds(
    fused_path: Path | str,
    data_root: Path | str,
    year: int,
    split: str,
    scenes: Sequence[str],
    out_dir: Path | str,
    thresholds: Sequence[float],
    max_costs: Sequence[float],
    max_distances_m: Sequence[float],
    max_age: int = 45,
    frame_stride: int = 1,
    max_frames_per_scene: int | None = None,
    decimals: int = 6,
) -> dict:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    detections = list(read_fused_detections(fused_path))
    gt_boxes = load_gt_boxes_for_split(
        data_root=data_root,
        year=year,
        split=split,
        scenes=scenes,
        max_frames_per_scene=max_frames_per_scene,
        frame_stride=frame_stride,
    )
    class_ids = sorted({box.class_id for box in gt_boxes} | {det.box.class_id for det in detections})
    trials = []
    best_payload = None
    best_tracks = None
    best_metrics = None

    for max_cost in max_costs:
        for max_distance_m in max_distances_m:
            selected_thresholds: dict[int, float] = {}
            selected_class_metrics: dict[int, dict] = {}
            combined_tracks = []

            for class_id in class_ids:
                class_detections = [det for det in detections if det.box.class_id == class_id]
                class_gt = [box for box in gt_boxes if box.class_id == class_id]
                class_best = None
                class_best_tracks = None
                for threshold in thresholds:
                    tracks = online_track(
                        class_detections,
                        max_distance_m=float(max_distance_m),
                        max_age=max_age,
                        min_score=float(threshold),
                        max_cost=float(max_cost),
                    )
                    metrics = evaluate_hota_like(class_gt, tracks)
                    summary = {
                        "threshold": float(threshold),
                        "boxes": len(tracks),
                        "objects": len(
                            {
                                (box.scene_id, box.class_id, box.object_id)
                                for box in tracks
                            }
                        ),
                        "hota_like": metrics["hota_like"],
                        "deta": metrics["deta"],
                        "assa": metrics["assa"],
                        "loca": metrics["loca"],
                    }
                    if class_best is None or summary["hota_like"] > class_best["hota_like"]:
                        class_best = summary
                        class_best_tracks = tracks

                if class_best is None or class_best_tracks is None:
                    continue
                selected_thresholds[class_id] = class_best["threshold"]
                selected_class_metrics[class_id] = class_best
                combined_tracks.extend(class_best_tracks)

            metrics = evaluate_hota_like(gt_boxes, combined_tracks)
            summary = {
                "max_cost": float(max_cost),
                "max_distance_m": float(max_distance_m),
                "class_min_scores": selected_thresholds,
                "per_class": selected_class_metrics,
                "boxes": len(combined_tracks),
                "objects": len(
                    {
                        (box.scene_id, box.class_id, box.object_id)
                        for box in combined_tracks
                    }
                ),
                "hota_like": metrics["hota_like"],
                "deta": metrics["deta"],
                "assa": metrics["assa"],
                "loca": metrics["loca"],
            }
            trials.append(summary)
            if best_payload is None or summary["hota_like"] > best_payload["hota_like"]:
                best_payload = summary
                best_tracks = combined_tracks
                best_metrics = metrics

    if best_payload is None or best_tracks is None or best_metrics is None:
        raise ValueError("The class-threshold sweep is empty")

    trials.sort(key=lambda item: item["hota_like"], reverse=True)
    best_submission = output / "track1_best_class_thresholds.txt"
    write_submission(best_tracks, best_submission, decimals=decimals)
    payload = {
        "fused": str(fused_path),
        "gt_boxes": len(gt_boxes),
        "fused_detections": len(detections),
        "trials": len(trials),
        "candidate_thresholds": [float(value) for value in thresholds],
        "best": best_payload,
        "best_submission": str(best_submission),
        "best_metrics": best_metrics,
        "results": trials,
    }
    results_path = output / "class_threshold_sweep_results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "results": str(results_path),
        "best_submission": str(best_submission),
        "trials": len(trials),
        "best": best_payload,
    }


def sweep_class_fusion_parameters(
    lifted_path: Path | str,
    data_root: Path | str,
    year: int,
    split: str,
    scenes: Sequence[str],
    out_dir: Path | str,
    distances_m: Sequence[float],
    min_sources_values: Sequence[int],
    nms_distances_m: Sequence[float],
    tracker_min_score: float = 0.0,
    tracker_max_cost: float = 1.20,
    tracker_max_distance_m: float = 1.8,
    max_age: int = 45,
    merge_iou: float = 0.08,
    nms_iou: float = 0.35,
    single_camera_score_factor: float = 0.92,
    frame_stride: int = 1,
    max_frames_per_scene: int | None = None,
    decimals: int = 6,
) -> dict:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = list(read_lifted_candidates(lifted_path))
    gt_boxes = load_gt_boxes_for_split(
        data_root=data_root,
        year=year,
        split=split,
        scenes=scenes,
        max_frames_per_scene=max_frames_per_scene,
        frame_stride=frame_stride,
    )

    best_by_class: dict[int, dict] = {}
    best_fused = []
    best_tracks = []
    total_trials = 0
    for class_id in sorted({box.class_id for box in gt_boxes}):
        class_candidates = [candidate for candidate in candidates if candidate.box.class_id == class_id]
        class_gt = [box for box in gt_boxes if box.class_id == class_id]
        class_best = None
        class_best_fused = None
        class_best_tracks = None

        for distance_m in distances_m:
            for min_sources in min_sources_values:
                for nms_distance_m in nms_distances_m:
                    fused = fuse_candidates(
                        class_candidates,
                        distance_m=float(distance_m),
                        min_sources=int(min_sources),
                        merge_iou=merge_iou,
                        nms_iou=nms_iou,
                        nms_distance_m=float(nms_distance_m),
                        single_camera_score_factor=single_camera_score_factor,
                    )
                    tracks = online_track(
                        fused,
                        max_distance_m=tracker_max_distance_m,
                        max_age=max_age,
                        min_score=tracker_min_score,
                        max_cost=tracker_max_cost,
                    )
                    metrics = evaluate_hota_like(class_gt, tracks)
                    summary = {
                        "class_id": class_id,
                        "distance_m": float(distance_m),
                        "min_sources": int(min_sources),
                        "nms_distance_m": float(nms_distance_m),
                        "candidates": len(class_candidates),
                        "fused_detections": len(fused),
                        "boxes": len(tracks),
                        "hota_like": metrics["hota_like"],
                        "deta": metrics["deta"],
                        "assa": metrics["assa"],
                        "loca": metrics["loca"],
                    }
                    total_trials += 1
                    rank = (summary["hota_like"], -summary["boxes"])
                    best_rank = (
                        (class_best["hota_like"], -class_best["boxes"])
                        if class_best is not None
                        else None
                    )
                    if best_rank is None or rank > best_rank:
                        class_best = summary
                        class_best_fused = fused
                        class_best_tracks = tracks

        if class_best is None or class_best_fused is None or class_best_tracks is None:
            raise ValueError(f"No fusion trials were produced for class {class_id}")
        best_by_class[class_id] = class_best
        best_fused.extend(class_best_fused)
        best_tracks.extend(class_best_tracks)

    if not best_by_class:
        raise ValueError("The class fusion sweep is empty")

    metrics = evaluate_hota_like(gt_boxes, best_tracks)
    best_fused_path = output / "fused_best.tsv"
    best_submission = output / "track1_best_class_fusion.txt"
    write_fused_detections(best_fused, best_fused_path)
    write_submission(best_tracks, best_submission, decimals=decimals)
    payload = {
        "lifted": str(lifted_path),
        "gt_boxes": len(gt_boxes),
        "candidates": len(candidates),
        "trials": total_trials,
        "tracker": {
            "min_score": tracker_min_score,
            "max_cost": tracker_max_cost,
            "max_distance_m": tracker_max_distance_m,
            "max_age": max_age,
        },
        "best_by_class": best_by_class,
        "class_distance_m": {
            class_id: summary["distance_m"] for class_id, summary in best_by_class.items()
        },
        "best_fused": str(best_fused_path),
        "best_submission": str(best_submission),
        "best_metrics": metrics,
    }
    results_path = output / "class_fusion_sweep_results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "results": str(results_path),
        "best_fused": str(best_fused_path),
        "best_submission": str(best_submission),
        "trials": total_trials,
        "best_metrics": metrics,
        "best_by_class": best_by_class,
    }
