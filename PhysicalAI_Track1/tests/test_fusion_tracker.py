from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

from physicalai_track1.association_model import AssociationScorer, FEATURE_NAMES, association_features
from physicalai_track1.dataset import TrackBox, iter_gt_boxes
from physicalai_track1.calibration import CameraCalibration
from physicalai_track1.detection_ensemble import ensemble_detections
from physicalai_track1.detector_adapters import import_coco_predictions, import_yolo_predictions
from physicalai_track1.detections import Detection2D, read_detections, write_detections
from physicalai_track1.fusion import FusedDetection3D, fuse_candidates, iter_fused_sorted_candidate_groups
from physicalai_track1.geometry_residual import GeometryResidualPredictor, geometry_features
from physicalai_track1.geometry import angle_distance, circular_mean
from physicalai_track1.lifting import LiftedCandidate, lift_detection
from physicalai_track1.tracklet_graph import relink_tracklets
from physicalai_track1.tracklet_stabilization import stabilize_detection_list
from physicalai_track1.tracker import online_track


def make_box(
    frame_id: int,
    x: float,
    y: float,
    *,
    class_id: int = 0,
    score: float = 0.9,
    yaw: float = 0.0,
    width: float = 0.8,
    length: float = 1.2,
) -> TrackBox:
    return TrackBox(
        scene_id=1,
        class_id=class_id,
        object_id=-1,
        frame_id=frame_id,
        x=x,
        y=y,
        z=0.9,
        width=width,
        length=length,
        height=1.8,
        yaw=yaw,
        score=score,
    )


class GeometryTest(unittest.TestCase):
    def test_circular_mean_handles_pi_wrap(self) -> None:
        mean = circular_mean([math.pi - 0.1, -math.pi + 0.1])
        self.assertLess(angle_distance(mean, math.pi), 1e-6)

    def test_numpy_geometry_residual_predictor(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed in the local test interpreter")
        camera = CameraCalibration(
            scene_name="Warehouse_020",
            camera_id="Camera_0000",
            frame_width=1920,
            frame_height=1080,
            homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            inv_homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            camera_matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 1.0),
            ),
            intrinsic_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            extrinsic_matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 1.0),
            ),
        )
        det = Detection2D("Warehouse_020", "Camera_0000", 0, 0, 1.0, 10.0, 20.0, 30.0, 60.0)
        prior = {"width": 1.0, "length": 1.0, "height": 2.0, "z": 1.0}
        feature, _ = geometry_features(det, camera, prior)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.npz"
            np.savez_compressed(
                model_path,
                feature_mean=np.zeros_like(feature),
                feature_std=np.ones_like(feature),
                target_mean=np.asarray([1.0, -2.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                target_std=np.ones(6, dtype=np.float32),
                weight_0=np.zeros((4, len(feature)), dtype=np.float32),
                bias_0=np.zeros(4, dtype=np.float32),
                weight_1=np.zeros((4, 4), dtype=np.float32),
                bias_1=np.zeros(4, dtype=np.float32),
                weight_2=np.zeros((4, 4), dtype=np.float32),
                bias_2=np.zeros(4, dtype=np.float32),
                weight_3=np.zeros((14, 4), dtype=np.float32),
                bias_3=np.zeros(14, dtype=np.float32),
                metadata=np.asarray("{}"),
            )
            prediction = GeometryResidualPredictor(model_path).predict(feature)

        self.assertEqual(prediction["dx"], 1.0)
        self.assertEqual(prediction["dy"], -2.0)
        self.assertAlmostEqual(prediction["dz"], 0.1, places=5)

    def test_scales_and_gates_geometry_residuals(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is not installed in the local test interpreter")

        class StubPredictor:
            def predict(self, feature):
                return {
                    "dx": 4.0,
                    "dy": -2.0,
                    "dz": 1.0,
                    "dlog_width": math.log(2.0),
                    "dlog_length": math.log(2.0),
                    "dlog_height": math.log(2.0),
                    "yaw": math.pi / 2.0,
                    "center_uncertainty": 4.0,
                }

        camera = CameraCalibration(
            scene_name="Warehouse_020",
            camera_id="Camera_0000",
            frame_width=1920,
            frame_height=1080,
            homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            inv_homography=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            camera_matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 1.0),
            ),
            intrinsic_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            extrinsic_matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 1.0),
            ),
        )
        det = Detection2D("Warehouse_020", "Camera_0000", 0, 0, 1.0, 10.0, 20.0, 30.0, 60.0)
        prior = {"width": 1.0, "length": 1.0, "height": 2.0, "z": 1.0}

        scaled = lift_detection(
            det,
            camera,
            prior,
            residual_predictor=StubPredictor(),
            residual_scale=0.25,
        )
        gated = lift_detection(
            det,
            camera,
            prior,
            residual_predictor=StubPredictor(),
            max_residual_uncertainty=3.0,
        )

        self.assertAlmostEqual(scaled.box.x, 21.0)
        self.assertAlmostEqual(scaled.box.y, 59.5)
        self.assertAlmostEqual(scaled.box.yaw, math.pi / 8.0)
        self.assertAlmostEqual(scaled.geometry_uncertainty, 1.0)
        self.assertAlmostEqual(gated.box.x, 20.0)
        self.assertEqual(gated.geometry_uncertainty, 0.0)


class FusionTest(unittest.TestCase):
    def test_deduplicates_camera_before_weighted_fusion(self) -> None:
        candidates = [
            LiftedCandidate(
                make_box(0, 0.0, 0.0, score=0.95),
                "Camera_1",
                0.95,
                reprojection_error=1.0,
            ),
            LiftedCandidate(
                make_box(0, 0.15, 0.05, score=0.90),
                "Camera_2",
                0.90,
                reprojection_error=2.0,
            ),
            LiftedCandidate(
                make_box(0, 0.5, 0.4, score=0.30),
                "Camera_1",
                0.30,
                reprojection_error=8.0,
            ),
        ]

        fused = fuse_candidates(candidates, distance_m=1.0)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].cameras, ("Camera_1", "Camera_2"))
        self.assertEqual(fused[0].source_count, 3)
        self.assertLess(fused[0].box.x, 0.10)

    def test_keeps_distinct_nearby_objects(self) -> None:
        candidates = [
            LiftedCandidate(make_box(0, 0.0, 0.0, width=0.4, length=0.5), "Camera_1", 0.9),
            LiftedCandidate(make_box(0, 0.1, 0.0, width=0.4, length=0.5), "Camera_2", 0.9),
            LiftedCandidate(make_box(0, 1.3, 0.0, width=0.4, length=0.5), "Camera_1", 0.9),
            LiftedCandidate(make_box(0, 1.4, 0.0, width=0.4, length=0.5), "Camera_2", 0.9),
        ]

        fused = fuse_candidates(candidates, distance_m=0.5, nms_distance_m=0.25)

        self.assertEqual(len(fused), 2)

    def test_applies_class_specific_fusion_radius(self) -> None:
        candidates = [
            LiftedCandidate(make_box(0, 0.0, 0.0), "Camera_1", 0.9),
            LiftedCandidate(make_box(0, 1.4, 0.0), "Camera_2", 0.9),
        ]

        globally_fused = fuse_candidates(candidates, distance_m=0.5)
        class_fused = fuse_candidates(
            candidates,
            distance_m=0.5,
            class_distance_m={0: 1.5},
        )

        self.assertEqual(len(globally_fused), 2)
        self.assertEqual(len(class_fused), 1)

    def test_streaming_sorted_fusion_matches_batch_fusion(self) -> None:
        candidates = [
            LiftedCandidate(make_box(0, 0.0, 0.0), "Camera_1", 0.9),
            LiftedCandidate(make_box(0, 0.1, 0.0), "Camera_2", 0.8),
            LiftedCandidate(make_box(0, 3.0, 0.0), "Camera_1", 0.7),
            LiftedCandidate(make_box(1, 0.0, 0.0), "Camera_1", 0.9),
            LiftedCandidate(make_box(1, 0.1, 0.0), "Camera_2", 0.8),
        ]

        batch = fuse_candidates(candidates, distance_m=0.5, nms_distance_m=0.25)
        streaming = list(
            iter_fused_sorted_candidate_groups(
                sorted(candidates, key=lambda c: (c.box.scene_id, c.box.frame_id, c.box.class_id)),
                distance_m=0.5,
                nms_distance_m=0.25,
            )
        )

        self.assertEqual(len(streaming), len(batch))
        self.assertEqual(
            [(d.box.scene_id, d.box.frame_id, d.box.class_id) for d in streaming],
            [(d.box.scene_id, d.box.frame_id, d.box.class_id) for d in batch],
        )


class TrackerTest(unittest.TestCase):
    def test_preserves_ids_through_crossing(self) -> None:
        detections = []
        for frame_id in range(7):
            detections.append(
                FusedDetection3D(
                    make_box(frame_id, -3.0 + frame_id, -0.15, score=0.95),
                    0.95,
                    ("Camera_1", "Camera_2"),
                    2,
                )
            )
            detections.append(
                FusedDetection3D(
                    make_box(frame_id, 3.0 - frame_id, 0.15, score=0.90),
                    0.90,
                    ("Camera_2", "Camera_3"),
                    2,
                )
            )

        outputs = online_track(
            detections,
            max_distance_m=1.8,
            position_alpha=1.0,
            velocity_alpha=0.3,
        )
        trajectories: dict[int, list[TrackBox]] = {}
        for box in outputs:
            trajectories.setdefault(box.object_id, []).append(box)

        self.assertEqual(len(trajectories), 2)
        self.assertTrue(all(len(boxes) == 7 for boxes in trajectories.values()))
        directions = [boxes[-1].x - boxes[0].x for boxes in trajectories.values()]
        self.assertTrue(any(delta > 0.0 for delta in directions))
        self.assertTrue(any(delta < 0.0 for delta in directions))

    def test_applies_class_specific_score_thresholds(self) -> None:
        person = FusedDetection3D(
            make_box(0, 0.0, 0.0, score=0.5),
            0.5,
            ("Camera_1",),
            1,
        )
        forklift_box = TrackBox(
            scene_id=1,
            class_id=1,
            object_id=-1,
            frame_id=0,
            x=2.0,
            y=0.0,
            z=1.0,
            width=1.2,
            length=2.0,
            height=2.0,
            yaw=0.0,
            score=0.2,
        )
        forklift = FusedDetection3D(forklift_box, 0.2, ("Camera_2",), 1)

        outputs = online_track(
            [person, forklift],
            min_score=0.0,
            class_min_scores={0: 0.6, 1: 0.1},
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].class_id, 1)

    def test_applies_class_specific_association_distance(self) -> None:
        detections = [
            FusedDetection3D(
                make_box(0, 0.0, 0.0, class_id=1, width=1.2, length=2.0),
                0.95,
                ("Camera_1", "Camera_2"),
                2,
            ),
            FusedDetection3D(
                make_box(2, 3.8, 0.0, class_id=1, width=1.2, length=2.0),
                0.95,
                ("Camera_1", "Camera_2"),
                2,
            ),
        ]

        global_outputs = online_track(detections, max_distance_m=1.0)
        class_outputs = online_track(
            detections,
            max_distance_m=1.0,
            class_max_distances_m={1: 3.0},
        )

        self.assertEqual(len({box.object_id for box in global_outputs}), 2)
        self.assertEqual(len({box.object_id for box in class_outputs}), 1)

    def test_applies_class_specific_max_age(self) -> None:
        detections = [
            FusedDetection3D(make_box(0, 0.0, 0.0, class_id=3), 0.95, ("Camera_1",), 1),
            FusedDetection3D(make_box(3, 0.2, 0.0, class_id=3), 0.95, ("Camera_1",), 1),
        ]

        global_outputs = online_track(detections, max_distance_m=1.0, max_age=2)
        class_outputs = online_track(
            detections,
            max_distance_m=1.0,
            max_age=2,
            class_max_ages={3: 3},
        )

        self.assertEqual(len({box.object_id for box in global_outputs}), 2)
        self.assertEqual(len({box.object_id for box in class_outputs}), 1)

    def test_delayed_confirmation_filters_single_frame_births(self) -> None:
        detections = [
            FusedDetection3D(make_box(0, 0.0, 0.0, score=0.95), 0.95, ("Camera_1",), 1),
            FusedDetection3D(make_box(1, 0.2, 0.0, score=0.95), 0.95, ("Camera_1",), 1),
            FusedDetection3D(make_box(0, 8.0, 0.0, score=0.90), 0.90, ("Camera_2",), 1),
        ]

        outputs = online_track(
            detections,
            max_distance_m=1.0,
            position_alpha=1.0,
            confirmation_hits=2,
            confirmation_mode="confirmed_only",
        )

        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0].x, 0.2)
        self.assertEqual(len({box.object_id for box in outputs}), 1)

    def test_backfill_confirmation_restores_confirmed_history(self) -> None:
        detections = [
            FusedDetection3D(make_box(0, 0.0, 0.0, score=0.95), 0.95, ("Camera_1",), 1),
            FusedDetection3D(make_box(1, 0.2, 0.0, score=0.95), 0.95, ("Camera_1",), 1),
        ]

        outputs = online_track(
            detections,
            max_distance_m=1.0,
            confirmation_hits=2,
            confirmation_mode="backfill",
        )

        self.assertEqual([box.frame_id for box in outputs], [0, 1])
        self.assertEqual(len({box.object_id for box in outputs}), 1)

    def test_duplicate_birth_suppression_skips_nearby_unmatched_detection(self) -> None:
        detections = [
            FusedDetection3D(make_box(0, 0.0, 0.0, score=0.95), 0.95, ("Camera_1",), 1),
            FusedDetection3D(make_box(1, 0.2, 0.0, score=0.95), 0.95, ("Camera_1",), 1),
            FusedDetection3D(make_box(1, 0.25, 0.05, score=0.70), 0.70, ("Camera_2",), 1),
        ]

        without_suppression = online_track(detections, max_distance_m=1.0)
        with_suppression = online_track(
            detections,
            max_distance_m=1.0,
            duplicate_birth_distance_m=0.5,
        )

        self.assertEqual(len({box.object_id for box in without_suppression}), 2)
        self.assertEqual(len({box.object_id for box in with_suppression}), 1)

    def test_duplicate_birth_suppression_applies_to_first_frame_births(self) -> None:
        detections = [
            FusedDetection3D(make_box(0, 0.0, 0.0, score=0.95), 0.95, ("Camera_1",), 1),
            FusedDetection3D(make_box(0, 0.1, 0.05, score=0.70), 0.70, ("Camera_2",), 1),
        ]

        outputs = online_track(
            detections,
            duplicate_birth_distance_m=0.5,
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len({box.object_id for box in outputs}), 1)

    def test_adaptive_confirmation_emits_high_confidence_birth_immediately(self) -> None:
        low = FusedDetection3D(make_box(0, 0.0, 0.0, score=0.80), 0.80, ("Camera_1",), 1)
        high = FusedDetection3D(make_box(0, 5.0, 0.0, score=0.95), 0.95, ("Camera_2",), 1)

        outputs = online_track(
            [low, high],
            confirmation_hits=2,
            confirmation_mode="confirmed_only",
            immediate_birth_score=0.90,
        )

        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0].x, 5.0)

    def test_adaptive_confirmation_emits_multicamera_birth_immediately(self) -> None:
        single_camera = FusedDetection3D(make_box(0, 0.0, 0.0, score=0.80), 0.80, ("Camera_1",), 1)
        multi_camera = FusedDetection3D(
            make_box(0, 5.0, 0.0, score=0.80),
            0.80,
            ("Camera_2", "Camera_3"),
            2,
        )

        outputs = online_track(
            [single_camera, multi_camera],
            confirmation_hits=2,
            confirmation_mode="confirmed_only",
            immediate_birth_min_sources=2,
        )

        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0].x, 5.0)


class AssociationModelTest(unittest.TestCase):
    def test_feature_vector_matches_schema(self) -> None:
        a = make_box(0, 0.0, 0.0, class_id=1, width=1.2, length=2.0)
        b = make_box(5, 1.0, 0.0, class_id=1, width=1.2, length=2.0)

        features = association_features(a, b)

        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(features[FEATURE_NAMES.index("dt")], 5.0)
        self.assertEqual(features[FEATURE_NAMES.index("class_1")], 1.0)
        self.assertEqual(features[FEATURE_NAMES.index("class_0")], 0.0)

    def test_association_scorer_prefers_close_same_object_geometry(self) -> None:
        close = make_box(1, 0.1, 0.0, class_id=0)
        far = make_box(1, 8.0, 0.0, class_id=0)
        anchor = make_box(0, 0.0, 0.0, class_id=0)
        weights = [0.0 for _ in FEATURE_NAMES]
        weights[FEATURE_NAMES.index("dist_xy")] = -4.0
        scorer = AssociationScorer(
            feature_names=list(FEATURE_NAMES),
            mean=[0.0 for _ in FEATURE_NAMES],
            std=[1.0 for _ in FEATURE_NAMES],
            weights=weights,
            bias=2.0,
        )

        self.assertGreater(scorer.predict_proba(anchor, close), scorer.predict_proba(anchor, far))


class DetectorAdapterTest(unittest.TestCase):
    def _write_manifest(self, root: Path) -> Path:
        manifest = root / "val_frames.tsv"
        manifest.write_text(
            "video_path\tframe_id\timage_path\tlabel_path\n"
            "/data/val/Warehouse_020/videos/Camera_0000.mp4\t30\t"
            "/images/Warehouse_020_Camera_0000_000030.jpg\t/labels/unused.txt\n",
            encoding="utf-8",
        )
        return manifest

    def test_imports_yolo_predictions_and_applies_nms(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_manifest(root)
            labels = root / "labels"
            labels.mkdir()
            (labels / "Warehouse_020_Camera_0000_000030.txt").write_text(
                "0 0.5 0.5 0.2 0.4 0.9\n"
                "0 0.5 0.5 0.2 0.4 0.5\n",
                encoding="utf-8",
            )
            output = root / "detections.tsv"

            result = import_yolo_predictions(manifest, labels, output, nms_iou=0.7)
            detections = list(read_detections(output))

            self.assertEqual(result["raw_detections"], 2)
            self.assertEqual(result["detections"], 1)
            self.assertEqual(detections[0].scene_name, "Warehouse_020")
            self.assertEqual(detections[0].camera_id, "Camera_0000")
            self.assertAlmostEqual(detections[0].x1, 768.0)
            self.assertAlmostEqual(detections[0].y2, 756.0)

    def test_imports_coco_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_manifest(root)
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 7,
                                "file_name": "Warehouse_020_Camera_0000_000030.jpg",
                                "width": 1920,
                                "height": 1080,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps(
                    [
                        {
                            "image_id": 7,
                            "category_id": 2,
                            "bbox": [100.0, 200.0, 300.0, 400.0],
                            "score": 0.8,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "detections.tsv"

            result = import_coco_predictions(predictions, annotations, manifest, output)
            detections = list(read_detections(output))

            self.assertEqual(result["detections"], 1)
            self.assertEqual(detections[0].class_id, 1)
            self.assertEqual((detections[0].x1, detections[0].y2), (100.0, 600.0))


class DetectionEnsembleTest(unittest.TestCase):
    def test_weighted_box_fusion_merges_overlapping_detector_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            yolo_path = root / "yolo.tsv"
            dfine_path = root / "dfine.tsv"
            out_path = root / "ensemble.tsv"
            write_detections(
                [
                    Detection2D("Warehouse_020", "Camera_0000", 0, 0, 0.80, 100.0, 100.0, 200.0, 300.0),
                    Detection2D("Warehouse_020", "Camera_0000", 0, 0, 0.70, 500.0, 100.0, 620.0, 300.0),
                ],
                yolo_path,
            )
            write_detections(
                [
                    Detection2D("Warehouse_020", "Camera_0000", 0, 0, 0.90, 110.0, 100.0, 210.0, 300.0),
                ],
                dfine_path,
            )

            result = ensemble_detections(
                [yolo_path, dfine_path],
                out_path,
                weights=[1.0, 1.0],
                wbf_iou=0.60,
                final_nms_iou=0.80,
                score_mode="noisy_or",
            )
            detections = list(read_detections(out_path))

        self.assertEqual(result["detections"], 2)
        self.assertEqual(len(detections), 2)
        merged = min(detections, key=lambda det: det.x1)
        self.assertGreater(merged.x1, 100.0)
        self.assertLess(merged.x1, 110.0)
        self.assertGreater(merged.score, 0.90)

    def test_ensemble_keeps_classes_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a_path = root / "a.tsv"
            b_path = root / "b.tsv"
            out_path = root / "ensemble.tsv"
            write_detections(
                [Detection2D("Warehouse_020", "Camera_0000", 0, 0, 0.90, 100.0, 100.0, 200.0, 300.0)],
                a_path,
            )
            write_detections(
                [Detection2D("Warehouse_020", "Camera_0000", 0, 1, 0.90, 100.0, 100.0, 200.0, 300.0)],
                b_path,
            )

            ensemble_detections([a_path, b_path], out_path, wbf_iou=0.60)
            detections = list(read_detections(out_path))

        self.assertEqual({det.class_id for det in detections}, {0, 1})
        self.assertEqual(len(detections), 2)


class TrackletStabilizationTest(unittest.TestCase):
    def test_smooths_matched_jitter_after_track_is_established(self) -> None:
        detections = [
            Detection2D("Warehouse_020", "Camera_0000", 0, 0, 0.9, 0.0, 0.0, 10.0, 10.0),
            Detection2D("Warehouse_020", "Camera_0000", 1, 0, 0.9, 2.0, 0.0, 12.0, 10.0),
            Detection2D("Warehouse_020", "Camera_0000", 2, 0, 0.9, 5.0, 0.0, 15.0, 10.0),
        ]

        stabilized, stats = stabilize_detection_list(
            detections,
            min_iou=0.10,
            smoothing_alpha=0.50,
            velocity_alpha=1.0,
            min_hits_for_smoothing=2,
            final_nms_iou=0.0,
        )

        self.assertEqual(len(stabilized), 3)
        self.assertEqual(stats["smoothed_detections"], 1)
        by_frame = {det.frame_id: det for det in stabilized}
        self.assertLess(by_frame[2].x1, 5.0)
        self.assertGreater(by_frame[2].x1, 3.0)

    def test_bridges_short_missing_frame_between_matched_detections(self) -> None:
        detections = [
            Detection2D("Warehouse_020", "Camera_0000", 0, 0, 0.9, 0.0, 0.0, 10.0, 10.0),
            Detection2D("Warehouse_020", "Camera_0000", 2, 0, 0.8, 2.0, 0.0, 12.0, 10.0),
        ]

        stabilized, stats = stabilize_detection_list(
            detections,
            min_iou=0.10,
            center_gate=2.0,
            max_gap_frames=2,
            bridge_max_gap_frames=2,
            bridge_min_score=0.1,
            bridge_score_decay=0.5,
            frame_step=1,
            final_nms_iou=0.0,
        )

        self.assertEqual(stats["bridged_detections"], 1)
        self.assertEqual([det.frame_id for det in stabilized], [0, 1, 2])
        bridge = stabilized[1]
        self.assertAlmostEqual(bridge.x1, 1.0)
        self.assertAlmostEqual(bridge.score, 0.4)


class TrackletGraphRelinkTest(unittest.TestCase):
    def test_merges_non_overlapping_fragmented_tracklets(self) -> None:
        boxes = [
            TrackBox(1, 0, 1, 0, 0.0, 0.0, 0.9, 0.8, 1.0, 1.8, 0.0),
            TrackBox(1, 0, 1, 1, 1.0, 0.0, 0.9, 0.8, 1.0, 1.8, 0.0),
            TrackBox(1, 0, 2, 3, 3.0, 0.0, 0.9, 0.8, 1.0, 1.8, 0.0),
            TrackBox(1, 0, 2, 4, 4.0, 0.0, 0.9, 0.8, 1.0, 1.8, 0.0),
        ]

        relinked, stats = relink_tracklets(
            boxes,
            max_gap_frames=3,
            max_distance_m=2.0,
            max_cost=2.0,
            frame_step=1,
        )

        self.assertEqual(stats["accepted_edges"], 1)
        self.assertEqual(stats["objects_after"], 1)
        self.assertEqual({box.object_id for box in relinked}, {1})

    def test_does_not_merge_temporally_overlapping_tracklets(self) -> None:
        boxes = [
            TrackBox(1, 0, 1, 0, 0.0, 0.0, 0.9, 0.8, 1.0, 1.8, 0.0),
            TrackBox(1, 0, 1, 2, 2.0, 0.0, 0.9, 0.8, 1.0, 1.8, 0.0),
            TrackBox(1, 0, 2, 1, 1.0, 0.1, 0.9, 0.8, 1.0, 1.8, 0.0),
            TrackBox(1, 0, 2, 3, 3.0, 0.1, 0.9, 0.8, 1.0, 1.8, 0.0),
        ]

        _relinked, stats = relink_tracklets(
            boxes,
            max_gap_frames=3,
            max_distance_m=2.0,
            max_cost=2.0,
            frame_step=1,
        )

        self.assertEqual(stats["accepted_edges"], 0)
        self.assertEqual(stats["objects_after"], 2)


class DatasetStrideTest(unittest.TestCase):
    def test_iter_gt_boxes_filters_frames_by_stride(self) -> None:
        item = {
            "object_type": "Person",
            "object_id": 1,
            "3d_location": [1.0, 2.0, 0.9],
            "3d_bounding_box_scale": [0.8, 1.0, 1.8],
            "3d_bounding_box_rotation": [0.0, 0.0, 0.0],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir) / "Warehouse_020"
            scene.mkdir()
            (scene / "ground_truth.json").write_text(
                json.dumps({"0": [item], "1": [item], "2": [item]}),
                encoding="utf-8",
            )

            boxes = list(iter_gt_boxes(scene, frame_stride=2))

            self.assertEqual([box.frame_id for box in boxes], [0, 2])


if __name__ == "__main__":
    unittest.main()
