from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from .dataset import iter_scene_dirs


def _frame_image_name(scene: str, camera: str, frame_id: int, extension: str) -> str:
    return f"{scene}_{camera}_{frame_id:06d}.{extension.lstrip('.')}"


def _video_frame_count(video_path: Path) -> int:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - preprocessing environment only
        raise RuntimeError("OpenCV (cv2) is required to build frame manifests from video files") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return max(0, frame_count)


def export_frame_manifest(
    data_root: Path | str,
    year: int,
    split: str,
    output_dir: Path | str,
    scenes: Optional[Sequence[str]] = None,
    frame_stride: int = 1,
    max_frames_per_scene: Optional[int] = None,
    image_extension: str = "jpg",
) -> dict:
    """Write a video-frame manifest without requiring ground-truth labels.

    This is the test-set counterpart to ``export-yolo``. It creates the same
    manifest schema used by the detector inference and frame extraction code,
    but it does not create label files.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if max_frames_per_scene is not None and max_frames_per_scene < 1:
        raise ValueError("max_frames_per_scene must be >= 1 when provided")

    output = Path(output_dir)
    images_root = output / "images" / split
    labels_root = output / "labels" / split
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    scene_filter = set(scenes or [])
    rows: list[str] = []
    frames_by_scene: Counter[str] = Counter()
    videos_seen = 0
    videos_with_frames = 0
    failed_videos: list[str] = []

    for scene_dir in iter_scene_dirs(data_root, year, split):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        videos_dir = scene_dir / "videos"
        if not videos_dir.exists():
            failed_videos.append(str(videos_dir))
            continue
        for video_path in sorted(videos_dir.glob("*.mp4")):
            videos_seen += 1
            camera_id = video_path.stem
            frame_count = _video_frame_count(video_path)
            if frame_count <= 0:
                failed_videos.append(str(video_path))
                continue
            videos_with_frames += 1
            frame_limit = frame_count
            if max_frames_per_scene is not None:
                frame_limit = min(frame_limit, max_frames_per_scene)
            for frame_id in range(0, frame_limit, frame_stride):
                image_name = _frame_image_name(scene_dir.name, camera_id, frame_id, image_extension)
                image_path = images_root / image_name
                label_path = labels_root / f"{Path(image_name).stem}.txt"
                rows.append(f"{video_path}\t{frame_id}\t{image_path}\t{label_path}")
                frames_by_scene[scene_dir.name] += 1

    manifest = output / f"{split}_frames.tsv"
    manifest.write_text(
        "video_path\tframe_id\timage_path\tlabel_path\n" + "\n".join(rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output),
        "split": split,
        "manifest": str(manifest),
        "images_root": str(images_root),
        "frame_stride": frame_stride,
        "max_frames_per_scene": max_frames_per_scene,
        "videos_seen": videos_seen,
        "videos_with_frames": videos_with_frames,
        "frames_referenced": len(rows),
        "frames_by_scene": dict(sorted(frames_by_scene.items())),
        "failed_videos": failed_videos,
        "note": "Frames are referenced only; run extract-frames on the manifest to materialize images.",
    }
