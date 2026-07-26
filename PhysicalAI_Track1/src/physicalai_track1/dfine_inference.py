from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from .constants import ID_TO_CLASS
from .detections import Detection2D, write_detections
from .detector_adapters import classwise_nms, read_frame_manifest


def _chunks(items: list[tuple[str, object]], size: int):
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


def infer_dfine_manifest(
    dfine_root: Path | str,
    config_path: Path | str,
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    images_root: Path | str,
    out_path: Path | str,
    scenes: Sequence[str] | None = None,
    max_frame_id: int | None = None,
    input_size: int = 960,
    confidence: float = 0.01,
    nms_iou: float = 0.70,
    batch_size: int = 8,
    device: str = "cuda:0",
    amp: bool = True,
    frame_width: int = 1920,
    frame_height: int = 1080,
    tile_size: int = 0,
    tile_overlap: float = 0.20,
    tile_full_image: bool = True,
) -> dict:
    try:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - D-FINE environment only
        raise RuntimeError("PyTorch, torchvision, and Pillow are required for D-FINE inference") from exc

    root = Path(dfine_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.core import YAMLConfig
    except ImportError as exc:  # pragma: no cover - D-FINE environment only
        raise RuntimeError(f"Cannot import D-FINE from {root}") from exc

    config = Path(config_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    cfg = YAMLConfig(str(config), resume=str(checkpoint_file))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model = cfg.model
    model.load_state_dict(state)
    model = model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )
    frames = read_frame_manifest(manifest_path, frame_width=frame_width, frame_height=frame_height)
    scene_filter = set(scenes or [])
    selected = [
        metadata
        for metadata in frames.values()
        if (not scene_filter or metadata.scene_name in scene_filter)
        and (max_frame_id is None or metadata.frame_id <= max_frame_id)
    ]
    selected.sort(key=lambda item: (item.scene_name, item.camera_id, item.frame_id))

    image_root = Path(images_root)
    inputs: list[tuple[str, object]] = []
    missing_images = 0
    for metadata in selected:
        image_path = image_root / metadata.image_name
        if image_path.exists():
            inputs.append((str(image_path), metadata))
        else:
            missing_images += 1

    raw: list[Detection2D] = []
    processed_images = 0
    amp_enabled = amp and str(device).startswith("cuda")
    if tile_size <= 0:
        for batch in _chunks(inputs, max(1, batch_size)):
            tensors = []
            original_sizes = []
            metadata_rows = []
            for image_path, metadata in batch:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    tensors.append(preprocess(image))
                original_sizes.append([width, height])
                metadata_rows.append(metadata)

            image_tensor = torch.stack(tensors).to(device)
            size_tensor = torch.tensor(original_sizes, dtype=torch.float32, device=device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    outputs = model(image_tensor)
                    labels, boxes, scores = postprocessor(outputs, size_tensor)

            for metadata, item_labels, item_boxes, item_scores in zip(
                metadata_rows,
                labels.detach().cpu(),
                boxes.detach().cpu(),
                scores.detach().cpu(),
            ):
                processed_images += 1
                for class_value, coords, score_value in zip(item_labels, item_boxes, item_scores):
                    score = float(score_value)
                    if score < confidence:
                        continue
                    class_id = int(class_value)
                    if class_id not in ID_TO_CLASS:
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
                            score=score,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )
    else:
        windows = _tile_windows(frame_width, frame_height, tile_size, tile_overlap)
        for image_path, metadata in inputs:
            processed_images += 1
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tile_payload = []
                original_sizes = []
                offsets: list[tuple[int, int]] = []
                if tile_full_image:
                    tile_payload.append(preprocess(image))
                    original_sizes.append([image.width, image.height])
                    offsets.append((0, 0))
                for x1, y1, x2, y2 in windows:
                    tile = image.crop((x1, y1, x2, y2))
                    tile_payload.append(preprocess(tile))
                    original_sizes.append([tile.width, tile.height])
                    offsets.append((x1, y1))

            for start in range(0, len(tile_payload), max(1, batch_size)):
                end = min(len(tile_payload), start + max(1, batch_size))
                image_tensor = torch.stack(tile_payload[start:end]).to(device)
                size_tensor = torch.tensor(original_sizes[start:end], dtype=torch.float32, device=device)
                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        outputs = model(image_tensor)
                        labels, boxes, scores = postprocessor(outputs, size_tensor)
                for offset, item_labels, item_boxes, item_scores in zip(
                    offsets[start:end],
                    labels.detach().cpu(),
                    boxes.detach().cpu(),
                    scores.detach().cpu(),
                ):
                    offset_x, offset_y = offset
                    for class_value, coords, score_value in zip(item_labels, item_boxes, item_scores):
                        score = float(score_value)
                        if score < confidence:
                            continue
                        class_id = int(class_value)
                        if class_id not in ID_TO_CLASS:
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
                                score=score,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                            )
                        )

    kept = classwise_nms(raw, nms_iou)
    count = write_detections(kept, out_path)
    return {
        "dfine_root": str(root),
        "config": str(config),
        "checkpoint": str(checkpoint_file),
        "manifest_frames": len(frames),
        "selected_images": len(selected),
        "processed_images": processed_images,
        "missing_images": missing_images,
        "raw_detections": len(raw),
        "detections": count,
        "input_size": input_size,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "batch_size": batch_size,
        "amp": amp_enabled,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "tile_full_image": tile_full_image,
        "output": str(out_path),
    }
