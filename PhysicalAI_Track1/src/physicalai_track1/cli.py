from __future__ import annotations

import argparse
import json
from pathlib import Path

from .association_model import build_association_dataset, train_association_model
from .coco_export import export_coco_from_yolo
from .dataset import load_gt_boxes_for_split, summarize_dataset
from .detection_ensemble import ensemble_detections
from .detection_filters import filter_detections_file
from .dfine_inference import infer_dfine_manifest
from .detector_adapters import import_coco_predictions, import_yolo_predictions
from .detections import export_gt_2d_detections
from .evaluator import dumps_metrics, evaluate_by_class, evaluate_hota_like
from .frame_extract import extract_frames_from_manifest
from .frame_manifest import export_frame_manifest
from .fusion import fuse_lifted_file, fuse_lifted_file_streaming_sorted
from .geometry_residual import build_geometry_residual_dataset, train_geometry_residual_model
from .lifting import iter_lifted_candidates, write_lifted_candidates
from .postprocess3d import parse_class_strengths, postprocess_submission_file
from .priors import build_priors
from .stats import class_box_stats
from .submission import read_submission, validate_submission, write_submission
from .sweeps import (
    sweep_class_fusion_parameters,
    sweep_class_thresholds,
    sweep_tracker_parameters,
)
from .tracklet_stabilization import stabilize_detections_file
from .tracklet_graph import relink_tracklets_file
from .tracker import track_fused_file
from .yolo_balance import build_balanced_yolo_train_list
from .yolo_export import export_yolo_labels
from .yolo_inference import infer_yolo_manifest


def cmd_inspect(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_dataset(args.data_root, args.year, deep=args.deep), indent=2, sort_keys=True))


def cmd_gt_to_submission(args: argparse.Namespace) -> None:
    boxes = load_gt_boxes_for_split(
        args.data_root,
        args.year,
        args.split,
        args.scenes,
        max_frames_per_scene=args.max_frames_per_scene,
        frame_stride=args.frame_stride,
    )
    count = write_submission(boxes, args.out, decimals=args.decimals)
    print(json.dumps({"output": str(args.out), "boxes": count, "decimals": args.decimals}, indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    print(json.dumps(validate_submission(args.submission), indent=2, sort_keys=True))


def cmd_eval(args: argparse.Namespace) -> None:
    gt_boxes = load_gt_boxes_for_split(
        args.data_root,
        args.year,
        args.split,
        args.scenes,
        max_frames_per_scene=args.max_frames_per_scene,
        frame_stride=args.frame_stride,
    )
    pred_boxes = read_submission(args.pred)
    metrics = evaluate_hota_like(gt_boxes, pred_boxes)
    if args.by_class:
        metrics["per_class"] = evaluate_by_class(gt_boxes, pred_boxes)
    print(dumps_metrics(metrics))


def cmd_export_yolo(args: argparse.Namespace) -> None:
    result = export_yolo_labels(
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        output_dir=args.output_dir,
        scenes=args.scenes,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        min_box_area=args.min_box_area,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_export_frame_manifest(args: argparse.Namespace) -> None:
    result = export_frame_manifest(
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        output_dir=args.output_dir,
        scenes=args.scenes,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        image_extension=args.image_extension,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_extract_frames(args: argparse.Namespace) -> None:
    result = extract_frames_from_manifest(args.manifest, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_balance_yolo(args: argparse.Namespace) -> None:
    result = build_balanced_yolo_train_list(
        dataset_dir=args.dataset_dir,
        split=args.split,
        out_list=args.out_list,
        out_yaml=args.out_yaml,
        val_path=args.val_path,
        power=args.power,
        max_repeat=args.max_repeat,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_export_coco(args: argparse.Namespace) -> None:
    result = export_coco_from_yolo(
        dataset_dir=args.dataset_dir,
        split=args.split,
        out_json=args.out,
        image_list=args.image_list,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_filter_detections(args: argparse.Namespace) -> None:
    result = filter_detections_file(
        input_path=args.input,
        output_path=args.out,
        min_area=args.min_area,
        min_width=args.min_width,
        min_height=args.min_height,
        class_min_area=parse_class_scores(args.class_min_area),
        class_min_width=parse_class_scores(args.class_min_width),
        class_min_height=parse_class_scores(args.class_min_height),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_stats(args: argparse.Namespace) -> None:
    result = class_box_stats(
        args.data_root,
        args.year,
        args.split,
        scenes=args.scenes,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_build_priors(args: argparse.Namespace) -> None:
    result = build_priors(
        args.data_root,
        args.year,
        args.split,
        args.out,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_build_geometry_dataset(args: argparse.Namespace) -> None:
    result = build_geometry_residual_dataset(
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        out_path=args.out,
        priors_path=args.priors,
        scenes=args.scenes,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        min_box_area=args.min_box_area,
        use_depth=args.use_depth,
        depth_scale=args.depth_scale,
        depth_root=args.depth_root,
        static_depth_root=args.static_depth_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_train_geometry_residual(args: argparse.Namespace) -> None:
    result = train_geometry_residual_model(
        dataset_path=args.dataset,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_build_association_dataset(args: argparse.Namespace) -> None:
    result = build_association_dataset(
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        out_path=args.out,
        scenes=args.scenes,
        max_frames_per_scene=args.max_frames_per_scene,
        frame_stride=args.frame_stride,
        positive_steps=args.positive_steps,
        negative_frame_tolerance=args.negative_frame_tolerance,
        negatives_per_positive=args.negatives_per_positive,
        max_samples_per_class_label=args.max_samples_per_class_label,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_train_association_model(args: argparse.Namespace) -> None:
    result = train_association_model(
        dataset_path=args.dataset,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_export_gt_2d(args: argparse.Namespace) -> None:
    result = export_gt_2d_detections(
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        out_path=args.out,
        scenes=args.scenes,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        min_box_area=args.min_box_area,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_import_yolo_predictions(args: argparse.Namespace) -> None:
    result = import_yolo_predictions(
        manifest_path=args.manifest,
        labels_dir=args.labels_dir,
        out_path=args.out,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        min_score=args.min_score,
        nms_iou=args.nms_iou,
        scenes=args.scenes,
        max_frame_id=args.max_frame_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_import_coco_predictions(args: argparse.Namespace) -> None:
    result = import_coco_predictions(
        predictions_path=args.predictions,
        annotations_path=args.annotations,
        manifest_path=args.manifest,
        out_path=args.out,
        category_offset=args.category_offset,
        min_score=args.min_score,
        nms_iou=args.nms_iou,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        scenes=args.scenes,
        max_frame_id=args.max_frame_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_ensemble_detections(args: argparse.Namespace) -> None:
    result = ensemble_detections(
        inputs=args.inputs,
        out_path=args.out,
        weights=args.weights,
        wbf_iou=args.wbf_iou,
        final_nms_iou=args.final_nms_iou,
        min_score=args.min_score,
        score_mode=args.score_mode,
        max_per_frame_class=args.max_per_frame_class,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_stabilize_detections(args: argparse.Namespace) -> None:
    result = stabilize_detections_file(
        input_path=args.input,
        out_path=args.out,
        min_iou=args.min_iou,
        center_gate=args.center_gate,
        max_gap_frames=args.max_gap_frames,
        smoothing_alpha=args.smoothing_alpha,
        velocity_alpha=args.velocity_alpha,
        min_hits_for_smoothing=args.min_hits_for_smoothing,
        bridge_max_gap_frames=args.bridge_max_gap_frames,
        bridge_min_score=args.bridge_min_score,
        bridge_score_decay=args.bridge_score_decay,
        frame_step=args.frame_step,
        final_nms_iou=args.final_nms_iou,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_infer_yolo_manifest(args: argparse.Namespace) -> None:
    result = infer_yolo_manifest(
        model_path=args.model,
        manifest_path=args.manifest,
        images_root=args.images_root,
        out_path=args.out,
        scenes=args.scenes,
        max_frame_id=args.max_frame_id,
        imgsz=args.imgsz,
        confidence=args.confidence,
        model_iou=args.model_iou,
        post_nms_iou=args.post_nms_iou,
        max_det=args.max_det,
        batch_size=args.batch_size,
        device=args.device,
        half=args.half,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_full_image=args.tile_full_image,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_infer_dfine_manifest(args: argparse.Namespace) -> None:
    result = infer_dfine_manifest(
        dfine_root=args.dfine_root,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        images_root=args.images_root,
        out_path=args.out,
        scenes=args.scenes,
        max_frame_id=args.max_frame_id,
        input_size=args.input_size,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        batch_size=args.batch_size,
        device=args.device,
        amp=args.amp,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_full_image=args.tile_full_image,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_lift_2d(args: argparse.Namespace) -> None:
    count = write_lifted_candidates(
        iter_lifted_candidates(
            detections_path=args.detections,
            data_root=args.data_root,
            year=args.year,
            priors_path=args.priors,
            split=args.split,
            use_oracle_ids=args.use_oracle_ids,
            residual_model_path=args.residual_model,
            residual_scale=args.residual_scale,
            max_residual_uncertainty=args.max_residual_uncertainty,
            use_depth=args.use_depth,
            depth_scale=args.depth_scale,
            depth_lift_mode=args.depth_lift_mode,
            depth_blend_alpha=args.depth_blend_alpha,
            depth_percentile=args.depth_percentile,
            depth_max_reprojection_px=args.depth_max_reprojection_px,
            depth_root=args.depth_root,
            static_depth_root=args.static_depth_root,
        ),
        args.out,
    )
    print(json.dumps({"output": str(args.out), "candidates": count}, indent=2, sort_keys=True))


def cmd_fuse_3d(args: argparse.Namespace) -> None:
    fuse_fn = fuse_lifted_file_streaming_sorted if args.streaming_sorted else fuse_lifted_file
    result = fuse_fn(
        args.lifted,
        args.out,
        distance_m=args.distance_m,
        class_distance_m=parse_class_scores(args.class_distance_m),
        min_sources=args.min_sources,
        merge_iou=args.merge_iou,
        nms_iou=args.nms_iou,
        nms_distance_m=args.nms_distance_m,
        single_camera_score_factor=args.single_camera_score_factor,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_track_online(args: argparse.Namespace) -> None:
    result = track_fused_file(
        args.fused,
        args.out,
        max_distance_m=args.max_distance_m,
        max_age=args.max_age,
        min_score=args.min_score,
        class_min_scores=parse_class_scores(args.class_min_scores),
        class_max_distances_m=parse_class_scores(args.class_max_distances_m),
        class_max_costs=parse_class_scores(args.class_max_costs),
        class_max_ages=parse_class_ints(args.class_max_ages),
        class_confirmation_hits=parse_class_ints(args.class_confirmation_hits),
        class_duplicate_birth_distances_m=parse_class_scores(args.class_duplicate_birth_distances_m),
        class_immediate_birth_scores=parse_class_scores(args.class_immediate_birth_scores),
        adaptive_calibration=args.adaptive_calibration,
        adaptive_warmup_frames=args.adaptive_warmup_frames,
        adaptive_strength=args.adaptive_strength,
        adaptive_report_path=args.adaptive_report,
        decimals=args.decimals,
        max_cost=args.max_cost,
        distance_weight=args.distance_weight,
        iou_weight=args.iou_weight,
        yaw_weight=args.yaw_weight,
        score_weight=args.score_weight,
        source_weight=args.source_weight,
        position_alpha=args.position_alpha,
        velocity_alpha=args.velocity_alpha,
        association_model_path=args.association_model,
        association_weight=args.association_weight,
        association_min_probability=args.association_min_probability,
        confirmation_hits=args.confirmation_hits,
        confirmation_mode=args.confirmation_mode,
        duplicate_birth_distance_m=args.duplicate_birth_distance_m,
        duplicate_birth_iou=args.duplicate_birth_iou,
        immediate_birth_score=args.immediate_birth_score,
        immediate_birth_min_sources=args.immediate_birth_min_sources,
        min_track_confidence=args.min_track_confidence,
        adaptive_birth_score=args.adaptive_birth_score,
        adaptive_birth_min_sources=args.adaptive_birth_min_sources,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_relink_tracklets(args: argparse.Namespace) -> None:
    result = relink_tracklets_file(
        input_path=args.input,
        out_path=args.out,
        decimals=args.decimals,
        max_gap_frames=args.max_gap_frames,
        max_distance_m=args.max_distance_m,
        max_cost=args.max_cost,
        frame_step=args.frame_step,
        min_tracklet_hits=args.min_tracklet_hits,
        distance_weight=args.distance_weight,
        velocity_weight=args.velocity_weight,
        iou_weight=args.iou_weight,
        yaw_weight=args.yaw_weight,
        size_weight=args.size_weight,
        association_model_path=args.association_model,
        association_weight=args.association_weight,
        association_min_probability=args.association_min_probability,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_postprocess_3d(args: argparse.Namespace) -> None:
    result = postprocess_submission_file(
        input_path=args.input,
        out_path=args.out,
        priors_path=args.priors,
        size_strength=args.size_strength,
        z_strength=args.z_strength,
        yaw_alpha=args.yaw_alpha,
        velocity_yaw_alpha=args.velocity_yaw_alpha,
        velocity_min_speed_mpf=args.velocity_min_speed_mpf,
        interpolate_max_gap_frames=args.interpolate_max_gap_frames,
        interpolate_min_track_length=args.interpolate_min_track_length,
        interpolate_max_step_distance_m=args.interpolate_max_step_distance_m,
        class_size_strengths=parse_class_strengths(args.class_size_strengths),
        class_z_strengths=parse_class_strengths(args.class_z_strengths),
        decimals=args.decimals,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_sweep_tracker(args: argparse.Namespace) -> None:
    result = sweep_tracker_parameters(
        fused_path=args.fused,
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        scenes=args.scenes,
        out_dir=args.out_dir,
        min_scores=args.min_scores,
        max_costs=args.max_costs,
        max_distances_m=args.max_distances_m,
        max_age=args.max_age,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        decimals=args.decimals,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_sweep_class_thresholds(args: argparse.Namespace) -> None:
    result = sweep_class_thresholds(
        fused_path=args.fused,
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        scenes=args.scenes,
        out_dir=args.out_dir,
        thresholds=args.thresholds,
        max_costs=args.max_costs,
        max_distances_m=args.max_distances_m,
        max_age=args.max_age,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        decimals=args.decimals,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_sweep_class_fusion(args: argparse.Namespace) -> None:
    result = sweep_class_fusion_parameters(
        lifted_path=args.lifted,
        data_root=args.data_root,
        year=args.year,
        split=args.split,
        scenes=args.scenes,
        out_dir=args.out_dir,
        distances_m=args.distances_m,
        min_sources_values=args.min_sources,
        nms_distances_m=args.nms_distances_m,
        tracker_min_score=args.tracker_min_score,
        tracker_max_cost=args.tracker_max_cost,
        tracker_max_distance_m=args.tracker_max_distance_m,
        max_age=args.max_age,
        merge_iou=args.merge_iou,
        nms_iou=args.nms_iou,
        single_camera_score_factor=args.single_camera_score_factor,
        frame_stride=args.frame_stride,
        max_frames_per_scene=args.max_frames_per_scene,
        decimals=args.decimals,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_class_scores(value: str | None) -> dict[int, float] | None:
    if not value:
        return None
    parsed: dict[int, float] = {}
    for item in value.split(","):
        class_id, score = item.split(":", 1)
        parsed[int(class_id)] = float(score)
    return parsed


def parse_class_ints(value: str | None) -> dict[int, int] | None:
    if not value:
        return None
    parsed: dict[int, int] = {}
    for item in value.split(","):
        class_id, score = item.split(":", 1)
        parsed[int(class_id)] = int(score)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="physicalai_track1")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Summarize dataset splits and labels")
    inspect.add_argument("--data-root", required=True)
    inspect.add_argument("--year", type=int, default=2026)
    inspect.add_argument("--deep", action="store_true", help="Read every GT frame and count labels")
    inspect.set_defaults(func=cmd_inspect)

    gt = sub.add_parser("gt-to-submission", help="Convert GT JSON to Track 1 text format")
    gt.add_argument("--data-root", required=True)
    gt.add_argument("--year", type=int, default=2026)
    gt.add_argument("--split", default="val")
    gt.add_argument("--scenes", nargs="*", default=None)
    gt.add_argument("--max-frames-per-scene", type=int, default=None)
    gt.add_argument("--frame-stride", type=int, default=1)
    gt.add_argument("--decimals", type=int, default=2, help="Float decimals; official submissions require 2")
    gt.add_argument("--out", required=True)
    gt.set_defaults(func=cmd_gt_to_submission)

    val = sub.add_parser("validate", help="Validate Track 1 text format")
    val.add_argument("--submission", required=True)
    val.set_defaults(func=cmd_validate)

    ev = sub.add_parser("eval", help="Run local 3D HOTA-style validation metric")
    ev.add_argument("--data-root", required=True)
    ev.add_argument("--year", type=int, default=2026)
    ev.add_argument("--split", default="val")
    ev.add_argument("--scenes", nargs="*", default=None)
    ev.add_argument("--max-frames-per-scene", type=int, default=None)
    ev.add_argument("--frame-stride", type=int, default=1)
    ev.add_argument("--by-class", action="store_true")
    ev.add_argument("--pred", required=True)
    ev.set_defaults(func=cmd_eval)

    yolo = sub.add_parser("export-yolo", help="Export YOLO labels from visible 2D boxes")
    yolo.add_argument("--data-root", required=True)
    yolo.add_argument("--year", type=int, default=2026)
    yolo.add_argument("--split", default="train")
    yolo.add_argument("--scenes", nargs="*", default=None)
    yolo.add_argument("--output-dir", required=True)
    yolo.add_argument("--frame-stride", type=int, default=30)
    yolo.add_argument("--max-frames-per-scene", type=int, default=None)
    yolo.add_argument("--min-box-area", type=float, default=16.0)
    yolo.set_defaults(func=cmd_export_yolo)

    frame_manifest = sub.add_parser(
        "export-frame-manifest",
        help="Export a video-frame manifest without requiring GT labels",
    )
    frame_manifest.add_argument("--data-root", required=True)
    frame_manifest.add_argument("--year", type=int, default=2026)
    frame_manifest.add_argument("--split", default="test")
    frame_manifest.add_argument("--scenes", nargs="*", default=None)
    frame_manifest.add_argument("--output-dir", required=True)
    frame_manifest.add_argument("--frame-stride", type=int, default=1)
    frame_manifest.add_argument("--max-frames-per-scene", type=int, default=None)
    frame_manifest.add_argument("--image-extension", default="jpg")
    frame_manifest.set_defaults(func=cmd_export_frame_manifest)

    frames = sub.add_parser("extract-frames", help="Extract images listed in a YOLO manifest TSV")
    frames.add_argument("--manifest", required=True)
    frames.add_argument("--overwrite", action="store_true")
    frames.set_defaults(func=cmd_extract_frames)

    balance = sub.add_parser("balance-yolo", help="Build a class-balanced YOLO image list and YAML")
    balance.add_argument("--dataset-dir", required=True)
    balance.add_argument("--split", default="train")
    balance.add_argument("--out-list", default=None)
    balance.add_argument("--out-yaml", default=None)
    balance.add_argument("--val-path", default="images/val")
    balance.add_argument("--power", type=float, default=0.5, help="Inverse-frequency power; 0.5 is sqrt balancing")
    balance.add_argument("--max-repeat", type=float, default=4.0)
    balance.add_argument("--seed", type=int, default=2026)
    balance.set_defaults(func=cmd_balance_yolo)

    coco = sub.add_parser("export-coco", help="Export YOLO labels/images to COCO detection JSON")
    coco.add_argument("--dataset-dir", required=True)
    coco.add_argument("--split", required=True, choices=["train", "val"])
    coco.add_argument("--out", required=True)
    coco.add_argument("--image-list", default=None, help="Optional repeated image list for class-balanced training")
    coco.add_argument("--frame-width", type=int, default=1920)
    coco.add_argument("--frame-height", type=int, default=1080)
    coco.set_defaults(func=cmd_export_coco)

    stats = sub.add_parser("stats", help="Compute class-wise 3D box priors")
    stats.add_argument("--data-root", required=True)
    stats.add_argument("--year", type=int, default=2026)
    stats.add_argument("--split", default="train")
    stats.add_argument("--scenes", nargs="*", default=None)
    stats.add_argument("--max-frames-per-scene", type=int, default=None)
    stats.set_defaults(func=cmd_stats)

    priors = sub.add_parser("build-priors", help="Build class-wise 3D geometry priors from GT")
    priors.add_argument("--data-root", required=True)
    priors.add_argument("--year", type=int, default=2026)
    priors.add_argument("--split", default="train")
    priors.add_argument("--max-frames-per-scene", type=int, default=None)
    priors.add_argument("--out", required=True)
    priors.set_defaults(func=cmd_build_priors)

    geometry_data = sub.add_parser(
        "build-geometry-dataset",
        help="Build supervised homography-residual targets from 2D/3D GT",
    )
    geometry_data.add_argument("--data-root", required=True)
    geometry_data.add_argument("--year", type=int, default=2026)
    geometry_data.add_argument("--split", default="train")
    geometry_data.add_argument("--priors", default=None)
    geometry_data.add_argument("--scenes", nargs="*", default=None)
    geometry_data.add_argument("--frame-stride", type=int, default=30)
    geometry_data.add_argument("--max-frames-per-scene", type=int, default=None)
    geometry_data.add_argument("--min-box-area", type=float, default=16.0)
    geometry_data.add_argument("--use-depth", action="store_true")
    geometry_data.add_argument("--depth-scale", type=float, default=0.001)
    geometry_data.add_argument("--depth-root", default=None)
    geometry_data.add_argument("--static-depth-root", default=None)
    geometry_data.add_argument("--out", required=True)
    geometry_data.set_defaults(func=cmd_build_geometry_dataset)

    geometry_train = sub.add_parser(
        "train-geometry-residual",
        help="Train and export a NumPy-compatible uncertainty-aware residual MLP",
    )
    geometry_train.add_argument("--dataset", required=True)
    geometry_train.add_argument("--epochs", type=int, default=50)
    geometry_train.add_argument("--batch-size", type=int, default=2048)
    geometry_train.add_argument("--learning-rate", type=float, default=1e-3)
    geometry_train.add_argument("--weight-decay", type=float, default=1e-4)
    geometry_train.add_argument("--val-fraction", type=float, default=0.2)
    geometry_train.add_argument("--seed", type=int, default=2026)
    geometry_train.add_argument("--device", default="cuda")
    geometry_train.add_argument("--patience", type=int, default=8)
    geometry_train.add_argument("--min-delta", type=float, default=1e-4)
    geometry_train.add_argument("--out", required=True)
    geometry_train.set_defaults(func=cmd_train_geometry_residual)

    association_data = sub.add_parser(
        "build-association-dataset",
        help="Build pairwise same-object/different-object association samples from GT trajectories",
    )
    association_data.add_argument("--data-root", required=True)
    association_data.add_argument("--year", type=int, default=2026)
    association_data.add_argument("--split", default="train")
    association_data.add_argument("--scenes", nargs="*", default=None)
    association_data.add_argument("--max-frames-per-scene", type=int, default=None)
    association_data.add_argument("--frame-stride", type=int, default=1)
    association_data.add_argument("--positive-steps", nargs="+", type=int, default=[1, 5, 15, 30])
    association_data.add_argument("--negative-frame-tolerance", type=int, default=2)
    association_data.add_argument("--negatives-per-positive", type=int, default=2)
    association_data.add_argument("--max-samples-per-class-label", type=int, default=50000)
    association_data.add_argument("--seed", type=int, default=2026)
    association_data.add_argument("--out", required=True)
    association_data.set_defaults(func=cmd_build_association_dataset)

    association_train = sub.add_parser(
        "train-association-model",
        help="Train a lightweight logistic pairwise association scorer",
    )
    association_train.add_argument("--dataset", required=True)
    association_train.add_argument("--epochs", type=int, default=80)
    association_train.add_argument("--batch-size", type=int, default=4096)
    association_train.add_argument("--learning-rate", type=float, default=0.05)
    association_train.add_argument("--weight-decay", type=float, default=1e-4)
    association_train.add_argument("--val-fraction", type=float, default=0.2)
    association_train.add_argument("--seed", type=int, default=2026)
    association_train.add_argument("--out", required=True)
    association_train.set_defaults(func=cmd_train_association_model)

    gt2d = sub.add_parser("export-gt-2d", help="Export oracle 2D detections from visible GT boxes")
    gt2d.add_argument("--data-root", required=True)
    gt2d.add_argument("--year", type=int, default=2026)
    gt2d.add_argument("--split", default="val")
    gt2d.add_argument("--scenes", nargs="*", default=None)
    gt2d.add_argument("--frame-stride", type=int, default=1)
    gt2d.add_argument("--max-frames-per-scene", type=int, default=None)
    gt2d.add_argument("--min-box-area", type=float, default=16.0)
    gt2d.add_argument("--out", required=True)
    gt2d.set_defaults(func=cmd_export_gt_2d)

    yolo_predictions = sub.add_parser(
        "import-yolo-predictions",
        help="Convert Ultralytics save_txt predictions to the common 2D TSV",
    )
    yolo_predictions.add_argument("--manifest", required=True)
    yolo_predictions.add_argument("--labels-dir", required=True)
    yolo_predictions.add_argument("--frame-width", type=int, default=1920)
    yolo_predictions.add_argument("--frame-height", type=int, default=1080)
    yolo_predictions.add_argument("--min-score", type=float, default=0.01)
    yolo_predictions.add_argument("--nms-iou", type=float, default=0.70)
    yolo_predictions.add_argument("--scenes", nargs="*", default=None)
    yolo_predictions.add_argument("--max-frame-id", type=int, default=None)
    yolo_predictions.add_argument("--out", required=True)
    yolo_predictions.set_defaults(func=cmd_import_yolo_predictions)

    coco_predictions = sub.add_parser(
        "import-coco-predictions",
        help="Convert COCO result JSON predictions to the common 2D TSV",
    )
    coco_predictions.add_argument("--predictions", required=True)
    coco_predictions.add_argument("--annotations", required=True)
    coco_predictions.add_argument("--manifest", required=True)
    coco_predictions.add_argument("--category-offset", type=int, default=1)
    coco_predictions.add_argument("--frame-width", type=int, default=1920)
    coco_predictions.add_argument("--frame-height", type=int, default=1080)
    coco_predictions.add_argument("--min-score", type=float, default=0.01)
    coco_predictions.add_argument("--nms-iou", type=float, default=0.70)
    coco_predictions.add_argument("--scenes", nargs="*", default=None)
    coco_predictions.add_argument("--max-frame-id", type=int, default=None)
    coco_predictions.add_argument("--out", required=True)
    coco_predictions.set_defaults(func=cmd_import_coco_predictions)

    ensemble = sub.add_parser(
        "ensemble-detections",
        help="Fuse multiple 2D detection TSVs with weighted box fusion",
    )
    ensemble.add_argument("--inputs", nargs="+", required=True)
    ensemble.add_argument("--weights", nargs="+", type=float, default=None)
    ensemble.add_argument("--wbf-iou", type=float, default=0.65)
    ensemble.add_argument("--final-nms-iou", type=float, default=0.80)
    ensemble.add_argument("--min-score", type=float, default=0.01)
    ensemble.add_argument("--score-mode", choices=["max", "mean", "noisy_or"], default="noisy_or")
    ensemble.add_argument("--max-per-frame-class", type=int, default=None)
    ensemble.add_argument("--out", required=True)
    ensemble.set_defaults(func=cmd_ensemble_detections)

    filter_dets = sub.add_parser(
        "filter-detections",
        help="Drop tiny 2D detections before fusion or 3D lifting",
    )
    filter_dets.add_argument("--input", required=True)
    filter_dets.add_argument("--out", required=True)
    filter_dets.add_argument("--min-area", type=float, default=0.0)
    filter_dets.add_argument("--min-width", type=float, default=0.0)
    filter_dets.add_argument("--min-height", type=float, default=0.0)
    filter_dets.add_argument("--class-min-area", default=None)
    filter_dets.add_argument("--class-min-width", default=None)
    filter_dets.add_argument("--class-min-height", default=None)
    filter_dets.set_defaults(func=cmd_filter_detections)

    stabilize = sub.add_parser(
        "stabilize-detections",
        help="Apply per-camera online 2D tracklet smoothing and optional gap bridging to detection TSVs",
    )
    stabilize.add_argument("--input", required=True)
    stabilize.add_argument("--out", required=True)
    stabilize.add_argument("--min-iou", type=float, default=0.20)
    stabilize.add_argument("--center-gate", type=float, default=0.85)
    stabilize.add_argument("--max-gap-frames", type=int, default=45)
    stabilize.add_argument("--smoothing-alpha", type=float, default=0.80)
    stabilize.add_argument("--velocity-alpha", type=float, default=0.55)
    stabilize.add_argument("--min-hits-for-smoothing", type=int, default=2)
    stabilize.add_argument("--bridge-max-gap-frames", type=int, default=0)
    stabilize.add_argument("--bridge-min-score", type=float, default=0.35)
    stabilize.add_argument("--bridge-score-decay", type=float, default=0.85)
    stabilize.add_argument("--frame-step", type=int, default=1)
    stabilize.add_argument("--final-nms-iou", type=float, default=0.80)
    stabilize.add_argument("--frame-width", type=int, default=1920)
    stabilize.add_argument("--frame-height", type=int, default=1080)
    stabilize.set_defaults(func=cmd_stabilize_detections)

    yolo_infer = sub.add_parser(
        "infer-yolo-manifest",
        help="Run chunked Ultralytics inference on selected manifest frames",
    )
    yolo_infer.add_argument("--model", required=True)
    yolo_infer.add_argument("--manifest", required=True)
    yolo_infer.add_argument("--images-root", required=True)
    yolo_infer.add_argument("--scenes", nargs="*", default=None)
    yolo_infer.add_argument("--max-frame-id", type=int, default=None)
    yolo_infer.add_argument("--imgsz", type=int, default=1536)
    yolo_infer.add_argument("--confidence", type=float, default=0.01)
    yolo_infer.add_argument("--model-iou", type=float, default=0.70)
    yolo_infer.add_argument("--post-nms-iou", type=float, default=0.70)
    yolo_infer.add_argument("--max-det", type=int, default=500)
    yolo_infer.add_argument("--batch-size", type=int, default=8)
    yolo_infer.add_argument("--device", default="0")
    yolo_infer.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    yolo_infer.add_argument("--frame-width", type=int, default=1920)
    yolo_infer.add_argument("--frame-height", type=int, default=1080)
    yolo_infer.add_argument("--tile-size", type=int, default=0)
    yolo_infer.add_argument("--tile-overlap", type=float, default=0.20)
    yolo_infer.add_argument("--tile-full-image", action=argparse.BooleanOptionalAction, default=True)
    yolo_infer.add_argument("--out", required=True)
    yolo_infer.set_defaults(func=cmd_infer_yolo_manifest)

    dfine_infer = sub.add_parser(
        "infer-dfine-manifest",
        help="Run batched D-FINE inference on selected manifest frames",
    )
    dfine_infer.add_argument("--dfine-root", required=True)
    dfine_infer.add_argument("--config", required=True)
    dfine_infer.add_argument("--checkpoint", required=True)
    dfine_infer.add_argument("--manifest", required=True)
    dfine_infer.add_argument("--images-root", required=True)
    dfine_infer.add_argument("--scenes", nargs="*", default=None)
    dfine_infer.add_argument("--max-frame-id", type=int, default=None)
    dfine_infer.add_argument("--input-size", type=int, default=960)
    dfine_infer.add_argument("--confidence", type=float, default=0.01)
    dfine_infer.add_argument("--nms-iou", type=float, default=0.70)
    dfine_infer.add_argument("--batch-size", type=int, default=8)
    dfine_infer.add_argument("--device", default="cuda:0")
    dfine_infer.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    dfine_infer.add_argument("--frame-width", type=int, default=1920)
    dfine_infer.add_argument("--frame-height", type=int, default=1080)
    dfine_infer.add_argument("--tile-size", type=int, default=0)
    dfine_infer.add_argument("--tile-overlap", type=float, default=0.20)
    dfine_infer.add_argument("--tile-full-image", action=argparse.BooleanOptionalAction, default=True)
    dfine_infer.add_argument("--out", required=True)
    dfine_infer.set_defaults(func=cmd_infer_dfine_manifest)

    lift = sub.add_parser("lift-2d", help="Lift 2D detections into 3D candidates using calibration and priors")
    lift.add_argument("--data-root", required=True)
    lift.add_argument("--year", type=int, default=2026)
    lift.add_argument("--split", default="val")
    lift.add_argument("--detections", required=True)
    lift.add_argument("--priors", default=None)
    lift.add_argument("--residual-model", default=None)
    lift.add_argument("--residual-scale", type=float, default=1.0)
    lift.add_argument("--max-residual-uncertainty", type=float, default=None)
    lift.add_argument("--use-depth", action="store_true")
    lift.add_argument("--depth-scale", type=float, default=0.001)
    lift.add_argument("--depth-root", default=None)
    lift.add_argument("--depth-lift-mode", choices=["none", "backproject", "blend", "foreground-pointcloud"], default="none")
    lift.add_argument("--depth-blend-alpha", type=float, default=0.65)
    lift.add_argument("--depth-percentile", type=float, default=35.0)
    lift.add_argument("--depth-max-reprojection-px", type=float, default=80.0)
    lift.add_argument("--static-depth-root", default=None)
    lift.add_argument("--use-oracle-ids", action="store_true")
    lift.add_argument("--out", required=True)
    lift.set_defaults(func=cmd_lift_2d)

    fuse = sub.add_parser("fuse-3d", help="Fuse lifted 3D candidates into frame-level detections")
    fuse.add_argument("--lifted", required=True)
    fuse.add_argument("--distance-m", type=float, default=1.5)
    fuse.add_argument(
        "--class-distance-m",
        default=None,
        help="Optional class-specific radii such as 0:1.0,1:2.0,6:2.2",
    )
    fuse.add_argument("--min-sources", type=int, default=1)
    fuse.add_argument("--merge-iou", type=float, default=0.08)
    fuse.add_argument("--nms-iou", type=float, default=0.35)
    fuse.add_argument("--nms-distance-m", type=float, default=0.25)
    fuse.add_argument("--single-camera-score-factor", type=float, default=0.92)
    fuse.add_argument(
        "--streaming-sorted",
        action="store_true",
        help="Fuse a lifted TSV already sorted by scene_id, frame_id, class_id without materializing all groups",
    )
    fuse.add_argument("--out", required=True)
    fuse.set_defaults(func=cmd_fuse_3d)

    track = sub.add_parser("track-online", help="Assign online Track 1 IDs to fused detections")
    track.add_argument("--fused", required=True)
    track.add_argument("--max-distance-m", type=float, default=2.5)
    track.add_argument("--max-age", type=int, default=45)
    track.add_argument("--min-score", type=float, default=0.0)
    track.add_argument(
        "--class-min-scores",
        default=None,
        help="Optional class thresholds such as 0:0.6,1:0.4,6:0.5",
    )
    track.add_argument(
        "--class-max-distances-m",
        default=None,
        help="Optional class association gates in meters such as 0:1.4,1:2.4,6:2.2",
    )
    track.add_argument(
        "--class-max-costs",
        default=None,
        help="Optional class association cost limits such as 0:1.05,1:1.25,6:1.20",
    )
    track.add_argument(
        "--class-max-ages",
        default=None,
        help="Optional class track ages in frames such as 0:45,1:75,6:70",
    )
    track.add_argument(
        "--class-confirmation-hits",
        default=None,
        help="Optional class minimum birth hits such as 0:2,1:1,6:1",
    )
    track.add_argument(
        "--class-duplicate-birth-distances-m",
        default=None,
        help="Optional class duplicate-birth suppression radii in meters such as 0:0.5,1:0.9,6:0.9",
    )
    track.add_argument(
        "--class-immediate-birth-scores",
        default=None,
        help="Optional class scores that bypass delayed confirmation, such as 0:0.90",
    )
    track.add_argument(
        "--adaptive-calibration",
        action="store_true",
        help="Adapt per-scene/class thresholds from early unlabeled fused detections",
    )
    track.add_argument("--adaptive-warmup-frames", type=int, default=180)
    track.add_argument("--adaptive-strength", type=float, default=0.18)
    track.add_argument("--adaptive-report", default=None)
    track.add_argument("--decimals", type=int, default=2)
    track.add_argument("--max-cost", type=float, default=1.35)
    track.add_argument("--distance-weight", type=float, default=1.0)
    track.add_argument("--iou-weight", type=float, default=0.35)
    track.add_argument("--yaw-weight", type=float, default=0.08)
    track.add_argument("--score-weight", type=float, default=0.18)
    track.add_argument("--source-weight", type=float, default=0.12)
    track.add_argument("--position-alpha", type=float, default=0.85)
    track.add_argument("--velocity-alpha", type=float, default=0.70)
    track.add_argument(
        "--association-model",
        default=None,
        help="Optional learned pairwise association model JSON from train-association-model",
    )
    track.add_argument(
        "--association-weight",
        type=float,
        default=0.0,
        help="Blend weight for learned association cost; 0 keeps heuristic behavior",
    )
    track.add_argument(
        "--association-min-probability",
        type=float,
        default=None,
        help="Optional hard gate on learned same-object probability",
    )
    track.add_argument(
        "--confirmation-hits",
        type=int,
        default=1,
        help="Minimum detections before a new online track is emitted",
    )
    track.add_argument(
        "--confirmation-mode",
        choices=["immediate", "confirmed_only", "backfill"],
        default="immediate",
        help="How to emit tracks before they reach confirmation hits",
    )
    track.add_argument(
        "--duplicate-birth-distance-m",
        type=float,
        default=0.0,
        help="Suppress unmatched detections near a track already updated in the same frame; 0 disables",
    )
    track.add_argument(
        "--duplicate-birth-iou",
        type=float,
        default=0.0,
        help="Suppress unmatched detections with high 3D IoU to a same-frame updated track; 0 disables",
    )
    track.add_argument(
        "--immediate-birth-score",
        type=float,
        default=None,
        help="Allow detections at or above this score to bypass delayed confirmation",
    )
    track.add_argument(
        "--immediate-birth-min-sources",
        type=int,
        default=None,
        help="Allow detections supported by this many cameras to bypass delayed confirmation",
    )
    track.add_argument(
        "--min-track-confidence",
        type=float,
        default=0.0,
        help="Drop online tracks whose causal confidence falls below this value",
    )
    track.add_argument(
        "--adaptive-birth-score",
        type=float,
        default=None,
        help="Only create a new track immediately when unmatched detection score reaches this value",
    )
    track.add_argument(
        "--adaptive-birth-min-sources",
        type=int,
        default=None,
        help="Only create a new track immediately when unmatched detection has this many fused sources",
    )
    track.add_argument("--out", required=True)
    track.set_defaults(func=cmd_track_online)

    relink = sub.add_parser(
        "relink-tracklets",
        help="Merge fragmented Track 1 IDs using a SUSHI-lite directed tracklet graph",
    )
    relink.add_argument("--input", required=True)
    relink.add_argument("--out", required=True)
    relink.add_argument("--decimals", type=int, default=2)
    relink.add_argument("--max-gap-frames", type=int, default=90)
    relink.add_argument("--max-distance-m", type=float, default=2.0)
    relink.add_argument("--max-cost", type=float, default=1.25)
    relink.add_argument("--frame-step", type=int, default=1)
    relink.add_argument("--min-tracklet-hits", type=int, default=2)
    relink.add_argument("--distance-weight", type=float, default=1.0)
    relink.add_argument("--velocity-weight", type=float, default=0.20)
    relink.add_argument("--iou-weight", type=float, default=0.25)
    relink.add_argument("--yaw-weight", type=float, default=0.08)
    relink.add_argument("--size-weight", type=float, default=0.25)
    relink.add_argument(
        "--association-model",
        default=None,
        help="Optional learned pairwise association model JSON from train-association-model",
    )
    relink.add_argument("--association-weight", type=float, default=0.0)
    relink.add_argument("--association-min-probability", type=float, default=None)
    relink.set_defaults(func=cmd_relink_tracklets)

    post3d = sub.add_parser(
        "postprocess-3d",
        help="Apply class geometry priors, yaw smoothing, and optional short-gap interpolation to Track 1 output",
    )
    post3d.add_argument("--input", required=True)
    post3d.add_argument("--out", required=True)
    post3d.add_argument("--priors", default=None)
    post3d.add_argument("--size-strength", type=float, default=0.35)
    post3d.add_argument("--z-strength", type=float, default=0.50)
    post3d.add_argument(
        "--class-size-strengths",
        default=None,
        help="Optional class-specific size blend strengths, e.g. 0:0.15,1:0.50,6:0.45",
    )
    post3d.add_argument(
        "--class-z-strengths",
        default=None,
        help="Optional class-specific z blend strengths, e.g. 0:0.65,1:0.45,6:0.50",
    )
    post3d.add_argument("--yaw-alpha", type=float, default=0.75)
    post3d.add_argument("--velocity-yaw-alpha", type=float, default=0.35)
    post3d.add_argument("--velocity-min-speed-mpf", type=float, default=0.03)
    post3d.add_argument("--interpolate-max-gap-frames", type=int, default=0)
    post3d.add_argument("--interpolate-min-track-length", type=int, default=3)
    post3d.add_argument("--interpolate-max-step-distance-m", type=float, default=0.75)
    post3d.add_argument("--decimals", type=int, default=2)
    post3d.set_defaults(func=cmd_postprocess_3d)

    sweep = sub.add_parser(
        "sweep-tracker",
        help="Sweep tracker confidence and association settings on cached fused detections",
    )
    sweep.add_argument("--fused", required=True)
    sweep.add_argument("--data-root", required=True)
    sweep.add_argument("--year", type=int, default=2026)
    sweep.add_argument("--split", default="val")
    sweep.add_argument("--scenes", nargs="+", required=True)
    sweep.add_argument("--max-frames-per-scene", type=int, default=None)
    sweep.add_argument("--frame-stride", type=int, default=1)
    sweep.add_argument(
        "--min-scores",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60],
    )
    sweep.add_argument("--max-costs", nargs="+", type=float, default=[1.20, 1.35])
    sweep.add_argument("--max-distances-m", nargs="+", type=float, default=[1.8, 2.2])
    sweep.add_argument("--max-age", type=int, default=45)
    sweep.add_argument("--decimals", type=int, default=6)
    sweep.add_argument("--out-dir", required=True)
    sweep.set_defaults(func=cmd_sweep_tracker)

    class_sweep = sub.add_parser(
        "sweep-class-thresholds",
        help="Select a separate confidence threshold for each class",
    )
    class_sweep.add_argument("--fused", required=True)
    class_sweep.add_argument("--data-root", required=True)
    class_sweep.add_argument("--year", type=int, default=2026)
    class_sweep.add_argument("--split", default="val")
    class_sweep.add_argument("--scenes", nargs="+", required=True)
    class_sweep.add_argument("--max-frames-per-scene", type=int, default=None)
    class_sweep.add_argument("--frame-stride", type=int, default=1)
    class_sweep.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            0.01,
            0.02,
            0.03,
            0.05,
            0.08,
            0.10,
            0.15,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
            1.01,
        ],
    )
    class_sweep.add_argument("--max-costs", nargs="+", type=float, default=[1.20, 1.35])
    class_sweep.add_argument("--max-distances-m", nargs="+", type=float, default=[1.8, 2.2])
    class_sweep.add_argument("--max-age", type=int, default=45)
    class_sweep.add_argument("--decimals", type=int, default=6)
    class_sweep.add_argument("--out-dir", required=True)
    class_sweep.set_defaults(func=cmd_sweep_class_thresholds)

    fusion_sweep = sub.add_parser(
        "sweep-class-fusion",
        help="Select class-specific multi-view fusion radii and source constraints",
    )
    fusion_sweep.add_argument("--lifted", required=True)
    fusion_sweep.add_argument("--data-root", required=True)
    fusion_sweep.add_argument("--year", type=int, default=2026)
    fusion_sweep.add_argument("--split", default="val")
    fusion_sweep.add_argument("--scenes", nargs="+", required=True)
    fusion_sweep.add_argument("--max-frames-per-scene", type=int, default=None)
    fusion_sweep.add_argument("--frame-stride", type=int, default=1)
    fusion_sweep.add_argument(
        "--distances-m",
        nargs="+",
        type=float,
        default=[0.8, 1.2, 1.6, 2.0, 2.5],
    )
    fusion_sweep.add_argument("--min-sources", nargs="+", type=int, default=[1, 2])
    fusion_sweep.add_argument(
        "--nms-distances-m",
        nargs="+",
        type=float,
        default=[0.25, 0.50],
    )
    fusion_sweep.add_argument("--merge-iou", type=float, default=0.08)
    fusion_sweep.add_argument("--nms-iou", type=float, default=0.35)
    fusion_sweep.add_argument("--single-camera-score-factor", type=float, default=0.92)
    fusion_sweep.add_argument("--tracker-min-score", type=float, default=0.0)
    fusion_sweep.add_argument("--tracker-max-cost", type=float, default=1.20)
    fusion_sweep.add_argument("--tracker-max-distance-m", type=float, default=1.8)
    fusion_sweep.add_argument("--max-age", type=int, default=45)
    fusion_sweep.add_argument("--decimals", type=int, default=6)
    fusion_sweep.add_argument("--out-dir", required=True)
    fusion_sweep.set_defaults(func=cmd_sweep_class_fusion)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
