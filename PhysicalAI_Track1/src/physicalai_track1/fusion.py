from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Tuple

from .dataset import TrackBox
from .geometry import box3d_iou, circular_mean, distance_xy
from .lifting import LiftedCandidate, read_lifted_candidates


@dataclass(frozen=True)
class FusedDetection3D:
    box: TrackBox
    score: float
    cameras: Tuple[str, ...]
    source_count: int
    cluster_spread_m: float = 0.0
    mean_reprojection_error: float = 0.0


def _cluster_radius(box: TrackBox, base: float) -> float:
    return max(base, 0.65 * max(box.width, box.length))


def _candidate_weight(cand: LiftedCandidate) -> float:
    score = max(1e-4, min(1.0, cand.score))
    reproj_weight = 1.0 / (1.0 + (max(0.0, cand.reprojection_error) / 12.0) ** 2)
    uncertainty_weight = 1.0 / (1.0 + max(0.0, cand.geometry_uncertainty) ** 2)
    return score * score * reproj_weight * uncertainty_weight


def _compatible(a: LiftedCandidate, b: LiftedCandidate, distance_m: float, merge_iou: float) -> bool:
    dist = distance_xy(a.box, b.box)
    radius = max(_cluster_radius(a.box, distance_m), _cluster_radius(b.box, distance_m))
    if dist <= radius:
        return True
    return box3d_iou(a.box, b.box) >= merge_iou


def _dedupe_by_camera(cluster: List[LiftedCandidate]) -> List[LiftedCandidate]:
    best: Dict[str, LiftedCandidate] = {}
    for cand in cluster:
        prev = best.get(cand.camera_id)
        if prev is None or _candidate_weight(cand) > _candidate_weight(prev):
            best[cand.camera_id] = cand
    return list(best.values())


def _weighted_center(cluster: List[LiftedCandidate]) -> Tuple[float, float, float, float, float, float, float, float]:
    weights = [_candidate_weight(c) for c in cluster]
    weight_sum = sum(weights) or 1.0
    x = sum(c.box.x * w for c, w in zip(cluster, weights)) / weight_sum
    y = sum(c.box.y * w for c, w in zip(cluster, weights)) / weight_sum
    z = sum(c.box.z * w for c, w in zip(cluster, weights)) / weight_sum
    width = sum(c.box.width * w for c, w in zip(cluster, weights)) / weight_sum
    length = sum(c.box.length * w for c, w in zip(cluster, weights)) / weight_sum
    height = sum(c.box.height * w for c, w in zip(cluster, weights)) / weight_sum
    yaw = circular_mean([c.box.yaw for c in cluster], weights)
    spread = sum((((c.box.x - x) ** 2 + (c.box.y - y) ** 2) ** 0.5) * w for c, w in zip(cluster, weights)) / weight_sum
    return x, y, z, width, length, height, yaw, spread


def _suppress_duplicates(
    detections: List[FusedDetection3D],
    nms_iou: float,
    nms_distance_m: float,
) -> List[FusedDetection3D]:
    by_key: Dict[Tuple[int, int, int], List[FusedDetection3D]] = defaultdict(list)
    for det in detections:
        b = det.box
        by_key[(b.scene_id, b.frame_id, b.class_id)].append(det)

    kept: List[FusedDetection3D] = []
    for key in sorted(by_key):
        selected: List[FusedDetection3D] = []
        ranked = sorted(
            by_key[key],
            key=lambda d: (len(d.cameras), d.score, -d.cluster_spread_m),
            reverse=True,
        )
        for det in ranked:
            duplicate = False
            for prev in selected:
                if box3d_iou(det.box, prev.box) >= nms_iou:
                    duplicate = True
                    break
                if nms_distance_m > 0.0 and distance_xy(det.box, prev.box) <= nms_distance_m:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(det)
        kept.extend(selected)
    return kept


def _fuse_candidate_group(
    candidates: List[LiftedCandidate],
    distance_m: float,
    class_distance_m: Mapping[int, float] | None,
    min_sources: int,
    merge_iou: float,
    single_camera_score_factor: float,
) -> List[FusedDetection3D]:
    if not candidates:
        return []
    class_id = candidates[0].box.class_id
    group_distance_m = (
        float(class_distance_m.get(class_id, distance_m))
        if class_distance_m is not None
        else distance_m
    )
    remaining = sorted(candidates, key=lambda c: c.score, reverse=True)
    clusters: List[List[LiftedCandidate]] = []
    for cand in remaining:
        placed = False
        for cluster in clusters:
            if any(_compatible(cand, other, group_distance_m, merge_iou) for other in cluster):
                cluster.append(cand)
                placed = True
                break
        if not placed:
            clusters.append([cand])

    fused: List[FusedDetection3D] = []
    for cluster in clusters:
        unique_cluster = _dedupe_by_camera(cluster)
        cameras = tuple(sorted({c.camera_id for c in unique_cluster}))
        if len(unique_cluster) < min_sources:
            continue
        ref = unique_cluster[0].box
        x, y, z, width, length, height, yaw, spread = _weighted_center(unique_cluster)
        max_score = max(c.score for c in unique_cluster)
        camera_bonus = min(1.20, 0.90 + 0.12 * len(cameras))
        spread_penalty = 1.0 / (1.0 + spread / max(0.25, group_distance_m))
        single_factor = single_camera_score_factor if len(cameras) == 1 else 1.0
        score = min(1.0, max_score * camera_bonus * spread_penalty * single_factor)
        mean_reproj = sum(c.reprojection_error for c in unique_cluster) / len(unique_cluster)
        fused.append(
            FusedDetection3D(
                box=TrackBox(
                    scene_id=ref.scene_id,
                    class_id=ref.class_id,
                    object_id=-1,
                    frame_id=ref.frame_id,
                    x=x,
                    y=y,
                    z=z,
                    width=width,
                    length=length,
                    height=height,
                    yaw=yaw,
                    score=score,
                ),
                score=score,
                cameras=cameras,
                source_count=len(cluster),
                cluster_spread_m=spread,
                mean_reprojection_error=mean_reproj,
            )
        )
    return fused


def _candidate_key(cand: LiftedCandidate) -> Tuple[int, int, int]:
    b = cand.box
    return b.scene_id, b.frame_id, b.class_id


def fuse_candidates(
    candidates: Iterable[LiftedCandidate],
    distance_m: float = 1.5,
    class_distance_m: Mapping[int, float] | None = None,
    min_sources: int = 1,
    merge_iou: float = 0.08,
    nms_iou: float = 0.35,
    nms_distance_m: float = 0.25,
    single_camera_score_factor: float = 0.92,
) -> List[FusedDetection3D]:
    groups = defaultdict(list)
    for cand in candidates:
        b = cand.box
        groups[(b.scene_id, b.frame_id, b.class_id)].append(cand)

    fused: List[FusedDetection3D] = []
    for key in sorted(groups):
        fused.extend(
            _fuse_candidate_group(
                groups[key],
                distance_m=distance_m,
                class_distance_m=class_distance_m,
                min_sources=min_sources,
                merge_iou=merge_iou,
                single_camera_score_factor=single_camera_score_factor,
            )
        )
    return _suppress_duplicates(fused, nms_iou=nms_iou, nms_distance_m=nms_distance_m)


def iter_fused_sorted_candidate_groups(
    candidates: Iterable[LiftedCandidate],
    distance_m: float = 1.5,
    class_distance_m: Mapping[int, float] | None = None,
    min_sources: int = 1,
    merge_iou: float = 0.08,
    nms_iou: float = 0.35,
    nms_distance_m: float = 0.25,
    single_camera_score_factor: float = 0.92,
) -> Iterator[FusedDetection3D]:
    previous_key: Tuple[int, int, int] | None = None
    group: List[LiftedCandidate] = []
    for cand in candidates:
        key = _candidate_key(cand)
        if previous_key is not None and key < previous_key:
            raise ValueError(
                "Streaming fusion requires candidates sorted by scene_id, frame_id, class_id"
            )
        if previous_key is not None and key != previous_key:
            fused = _fuse_candidate_group(
                group,
                distance_m=distance_m,
                class_distance_m=class_distance_m,
                min_sources=min_sources,
                merge_iou=merge_iou,
                single_camera_score_factor=single_camera_score_factor,
            )
            yield from _suppress_duplicates(fused, nms_iou=nms_iou, nms_distance_m=nms_distance_m)
            group = []
        previous_key = key
        group.append(cand)

    if group:
        fused = _fuse_candidate_group(
            group,
            distance_m=distance_m,
            class_distance_m=class_distance_m,
            min_sources=min_sources,
            merge_iou=merge_iou,
            single_camera_score_factor=single_camera_score_factor,
        )
        yield from _suppress_duplicates(fused, nms_iou=nms_iou, nms_distance_m=nms_distance_m)


def write_fused_detections(detections: Iterable[FusedDetection3D], out_path: Path | str) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        f.write(
            "scene_id\tframe_id\tclass_id\tx\ty\tz\twidth\tlength\theight\tyaw\tscore\tcameras\tsource_count\tcluster_spread_m\tmean_reprojection_error\n"
        )
        for det in detections:
            b = det.box
            f.write(
                "\t".join(
                    [
                        str(b.scene_id),
                        str(b.frame_id),
                        str(b.class_id),
                        f"{b.x:.6f}",
                        f"{b.y:.6f}",
                        f"{b.z:.6f}",
                        f"{b.width:.6f}",
                        f"{b.length:.6f}",
                        f"{b.height:.6f}",
                        f"{b.yaw:.6f}",
                        f"{det.score:.6f}",
                        ",".join(det.cameras),
                        str(det.source_count),
                        f"{det.cluster_spread_m:.6f}",
                        f"{det.mean_reprojection_error:.6f}",
                    ]
                )
            )
            f.write("\n")
            count += 1
    return count


def read_fused_detections(path: Path | str) -> Iterator[FusedDetection3D]:
    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline()
        for line_no, line in enumerate(f, start=2):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 11:
                raise ValueError(f"{path}:{line_no}: expected fused detection TSV")
            box = TrackBox(
                scene_id=int(parts[0]),
                frame_id=int(parts[1]),
                class_id=int(parts[2]),
                object_id=-1,
                x=float(parts[3]),
                y=float(parts[4]),
                z=float(parts[5]),
                width=float(parts[6]),
                length=float(parts[7]),
                height=float(parts[8]),
                yaw=float(parts[9]),
                score=float(parts[10]),
            )
            cameras = tuple(parts[11].split(",")) if len(parts) > 11 and parts[11] else ()
            source_count = int(parts[12]) if len(parts) > 12 and parts[12] else len(cameras)
            cluster_spread_m = float(parts[13]) if len(parts) > 13 and parts[13] else 0.0
            mean_reprojection_error = float(parts[14]) if len(parts) > 14 and parts[14] else 0.0
            yield FusedDetection3D(
                box=box,
                score=float(parts[10]),
                cameras=cameras,
                source_count=source_count,
                cluster_spread_m=cluster_spread_m,
                mean_reprojection_error=mean_reprojection_error,
            )


def fuse_lifted_file(
    lifted_path: Path | str,
    out_path: Path | str,
    distance_m: float = 1.5,
    class_distance_m: Mapping[int, float] | None = None,
    min_sources: int = 1,
    merge_iou: float = 0.08,
    nms_iou: float = 0.35,
    nms_distance_m: float = 0.25,
    single_camera_score_factor: float = 0.92,
) -> dict:
    detections = fuse_candidates(
        read_lifted_candidates(lifted_path),
        distance_m=distance_m,
        class_distance_m=class_distance_m,
        min_sources=min_sources,
        merge_iou=merge_iou,
        nms_iou=nms_iou,
        nms_distance_m=nms_distance_m,
        single_camera_score_factor=single_camera_score_factor,
    )
    count = write_fused_detections(detections, out_path)
    return {
        "output": str(out_path),
        "detections": count,
        "distance_m": distance_m,
        "class_distance_m": dict(class_distance_m or {}),
        "min_sources": min_sources,
        "merge_iou": merge_iou,
        "nms_iou": nms_iou,
        "nms_distance_m": nms_distance_m,
        "single_camera_score_factor": single_camera_score_factor,
    }


def fuse_lifted_file_streaming_sorted(
    lifted_path: Path | str,
    out_path: Path | str,
    distance_m: float = 1.5,
    class_distance_m: Mapping[int, float] | None = None,
    min_sources: int = 1,
    merge_iou: float = 0.08,
    nms_iou: float = 0.35,
    nms_distance_m: float = 0.25,
    single_camera_score_factor: float = 0.92,
) -> dict:
    count = write_fused_detections(
        iter_fused_sorted_candidate_groups(
            read_lifted_candidates(lifted_path),
            distance_m=distance_m,
            class_distance_m=class_distance_m,
            min_sources=min_sources,
            merge_iou=merge_iou,
            nms_iou=nms_iou,
            nms_distance_m=nms_distance_m,
            single_camera_score_factor=single_camera_score_factor,
        ),
        out_path,
    )
    return {
        "output": str(out_path),
        "detections": count,
        "distance_m": distance_m,
        "class_distance_m": dict(class_distance_m or {}),
        "min_sources": min_sources,
        "merge_iou": merge_iou,
        "nms_iou": nms_iou,
        "nms_distance_m": nms_distance_m,
        "single_camera_score_factor": single_camera_score_factor,
        "streaming_sorted": True,
    }
