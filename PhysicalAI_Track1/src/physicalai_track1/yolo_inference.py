from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .constants import ID_TO_CLASS
from .detections import Detection2D, write_detections
from .detector_adapters import classwise_nms, read_frame_manifest


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _tile_windows(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        return [(0, 0, width, height)]
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    windows: list[tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        x = 0
        y2 = min(height, y + tile_size)
        y1 = max(0, y2 - tile_size)
        while x < width:
            x2 = min(width, x + tile_size)
            x1 = max(0, x2 - tile_size)
            window = (x1, y1, x2, y2)
            if window not in windows:
                windows.append(window)
            if x2 >= width:
                break
            x += step
        if y2 >= height:
            break
        y += step
    return windows


def infer_yolo_manifest(
    model_path: Path | str,
    manifest_path: Path | str,
    images_root: Path | str,
    out_path: Path | str,
    scenes: Sequence[str] | None = None,
    max_frame_id: int | None = None,
    imgsz: int = 1536,
    confidence: float = 0.01,
    model_iou: float = 0.70,
    post_nms_iou: float = 0.70,
    max_det: int = 500,
    batch_size: int = 8,
    device: str = "0",
    half: bool = True,
    frame_width: int = 1920,
    frame_height: int = 1080,
    tile_size: int = 0,
    tile_overlap: float = 0.20,
    tile_full_image: bool = True,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - detector environment only
        raise RuntimeError("Ultralytics is required for infer-yolo-manifest") from exc

    frames = read_frame_manifest(manifest_path, frame_width=frame_width, frame_height=frame_height)
    scene_filter = set(scenes or [])
    selected = [
        metadata
        for metadata in frames.values()
        if (not scene_filter or metadata.scene_name in scene_filter)
        and (max_frame_id is None or metadata.frame_id <= max_frame_id)
    ]
    selected.sort(key=lambda item: (item.scene_name, item.camera_id, item.frame_id))

    root = Path(images_root)
    metadata_by_name = {item.image_name: item for item in selected}
    image_paths: list[str] = []
    missing_images = 0
    for metadata in selected:
        image_path = root / metadata.image_name
        if image_path.exists():
            image_paths.append(str(image_path))
        else:
            missing_images += 1

    model = YOLO(str(model_path))
    raw: list[Detection2D] = []
    processed_images = 0
    if tile_size <= 0:
        for batch_paths in _chunks(image_paths, max(1, batch_size)):
            results = model.predict(
                source=batch_paths,
                imgsz=imgsz,
                conf=confidence,
                iou=model_iou,
                max_det=max_det,
                device=device,
                half=half,
                batch=len(batch_paths),
                stream=True,
                verbose=False,
            )
            for original_path, result in zip(batch_paths, results):
                processed_images += 1
                metadata = metadata_by_name.get(Path(original_path).name)
                if metadata is None or result.boxes is None:
                    continue
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                scores = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                for coords, score, class_value in zip(xyxy, scores, classes):
                    class_id = int(class_value)
                    if class_id not in ID_TO_CLASS or float(score) < confidence:
                        continue
                    x1, y1, x2, y2 = map(float, coords)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    raw.append(
                        Detection2D(
                            scene_name=metadata.scene_name,
                            camera_id=metadata.camera_id,
                            frame_id=metadata.frame_id,
                            class_id=class_id,
                            score=float(score),
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )
    else:
        from PIL import Image

        windows = _tile_windows(frame_width, frame_height, tile_size, tile_overlap)
        for image_path_str in image_paths:
            image_path = Path(image_path_str)
            metadata = metadata_by_name.get(image_path.name)
            if metadata is None:
                continue
            processed_images += 1
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tile_payload = []
                tile_offsets: list[tuple[int, int]] = []
                if tile_full_image:
                    tile_payload.append(image)
                    tile_offsets.append((0, 0))
                for x1, y1, x2, y2 in windows:
                    tile_payload.append(image.crop((x1, y1, x2, y2)))
                    tile_offsets.append((x1, y1))

                for batch_indices in _chunks(list(range(len(tile_payload))), max(1, batch_size)):
                    batch_images = [tile_payload[index] for index in batch_indices]
                    results = model.predict(
                        source=batch_images,
                        imgsz=imgsz,
                        conf=confidence,
                        iou=model_iou,
                        max_det=max_det,
                        device=device,
                        half=half,
                        batch=len(batch_images),
                        stream=True,
                        verbose=False,
                    )
                    for index, result in zip(batch_indices, results):
                        if result.boxes is None:
                            continue
                        offset_x, offset_y = tile_offsets[index]
                        xyxy = result.boxes.xyxy.detach().cpu().tolist()
                        scores = result.boxes.conf.detach().cpu().tolist()
                        classes = result.boxes.cls.detach().cpu().tolist()
                        for coords, score, class_value in zip(xyxy, scores, classes):
                            class_id = int(class_value)
                            if class_id not in ID_TO_CLASS or float(score) < confidence:
                                continue
                            x1, y1, x2, y2 = map(float, coords)
                            x1 = min(max(x1 + offset_x, 0.0), float(frame_width))
                            x2 = min(max(x2 + offset_x, 0.0), float(frame_width))
                            y1 = min(max(y1 + offset_y, 0.0), float(frame_height))
                            y2 = min(max(y2 + offset_y, 0.0), float(frame_height))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            raw.append(
                                Detection2D(
                                    scene_name=metadata.scene_name,
                                    camera_id=metadata.camera_id,
                                    frame_id=metadata.frame_id,
                                    class_id=class_id,
                                    score=float(score),
                                    x1=x1,
                                    y1=y1,
                                    x2=x2,
                                    y2=y2,
                                )
                            )

    kept = classwise_nms(raw, post_nms_iou)
    count = write_detections(kept, out_path)
    return {
        "model": str(model_path),
        "manifest_frames": len(frames),
        "selected_images": len(selected),
        "processed_images": processed_images,
        "missing_images": missing_images,
        "raw_detections": len(raw),
        "detections": count,
        "imgsz": imgsz,
        "confidence": confidence,
        "model_iou": model_iou,
        "post_nms_iou": post_nms_iou,
        "batch_size": batch_size,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_full_image": tile_full_image,
        "output": str(out_path),
    }
