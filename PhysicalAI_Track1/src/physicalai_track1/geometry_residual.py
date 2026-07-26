from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional until residual commands are used
    np = None

try:
    import h5py
except ImportError:  # pragma: no cover - optional until depth residual commands are used
    h5py = None

from .calibration import CameraCalibration, load_scene_cameras
from .constants import CLASS_TO_ID, GT_KEY_ALIASES, ID_TO_CLASS
from .dataset import iter_gt_frames, iter_scene_dirs
from .detections import Detection2D
from .priors import load_priors, prior_for_class


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("NumPy is required for geometry residual dataset, training, and inference")


def _get_any(record: Mapping, logical_key: str):
    for key in GT_KEY_ALIASES[logical_key]:
        if key in record:
            return record[key]
    raise KeyError(f"Missing {logical_key}; tried {GT_KEY_ALIASES[logical_key]}")


def geometry_features(
    det: Detection2D,
    camera: CameraCalibration,
    class_prior: Mapping[str, float],
    depth_features: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    _require_numpy()
    frame_w = float(camera.frame_width)
    frame_h = float(camera.frame_height)
    u, v = det.bottom_center
    baseline_x, baseline_y = camera.image_to_ground(u, v)
    box_w = max(1.0, det.x2 - det.x1)
    box_h = max(1.0, det.y2 - det.y1)
    direction = math.radians(camera.direction_deg)
    class_one_hot = [1.0 if det.class_id == class_id else 0.0 for class_id in sorted(ID_TO_CLASS)]
    geometry = [
        det.x1 / frame_w,
        det.y1 / frame_h,
        det.x2 / frame_w,
        det.y2 / frame_h,
        u / frame_w,
        v / frame_h,
        box_w / frame_w,
        box_h / frame_h,
        (box_w * box_h) / (frame_w * frame_h),
        math.log(box_w / box_h),
        baseline_x,
        baseline_y,
        math.sin(direction),
        math.cos(direction),
        float(class_prior["width"]),
        float(class_prior["length"]),
        float(class_prior["height"]),
        float(class_prior["z"]),
    ]
    calibration = [value for row in camera.homography for value in row]
    calibration.extend(value for row in camera.camera_matrix for value in row)
    return np.asarray(class_one_hot + geometry + calibration + list(depth_features or []), dtype=np.float32), (
        baseline_x,
        baseline_y,
    )


def _camera_name(camera_id: str) -> str:
    camera_name = str(camera_id)
    if not camera_name.startswith("Camera_"):
        try:
            camera_name = f"Camera_{int(camera_name):04d}"
        except ValueError:
            pass
    return camera_name


def _depth_file(scene_dir: Path, camera_id: str) -> Path:
    return scene_dir / "depth_maps" / f"{_camera_name(camera_id)}.h5"


def _depth_npz_file(depth_root: Path | str, split: str, scene_name: str, camera_id: str, frame_id: int) -> Path:
    return Path(depth_root) / split / scene_name / _camera_name(camera_id) / f"{frame_id:06d}.npz"


def _load_depth_npz(path: Path | str):
    _require_numpy()
    with np.load(path) as data:
        return np.asarray(data["depth_m"], dtype=np.float32)


def _frame_depth_key(frame_id: int) -> str:
    return f"distance_to_image_plane_{frame_id:05d}.png"


def _empty_depth_features() -> list[float]:
    return [0.0] * 32


def _safe_depth_stats(values: np.ndarray, depth_scale: float) -> list[float]:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return _empty_depth_features()
    meters = finite.astype(np.float32) * depth_scale
    p10, p25, p50, p75, p90 = np.percentile(meters, [10, 25, 50, 75, 90])
    mean = float(meters.mean())
    std = float(meters.std())
    valid_ratio = float(finite.size / max(1, values.size))
    return [
        float(p10),
        float(p25),
        float(p50),
        float(p75),
        float(p90),
        mean,
        std,
        valid_ratio,
        math.log1p(float(p50)),
        math.log1p(float(p90)),
    ]


def depth_crop_features(
    depth_map,
    xyxy: Sequence[float],
    depth_scale: float = 0.001,
    source_width: float | None = None,
    source_height: float | None = None,
) -> list[float]:
    _require_numpy()
    height, width = depth_map.shape
    x1, y1, x2, y2 = map(float, xyxy)
    if source_width is not None and source_height is not None:
        sx = width / max(1.0, float(source_width))
        sy = height / max(1.0, float(source_height))
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy
    ix1 = max(0, min(width - 1, int(math.floor(x1))))
    iy1 = max(0, min(height - 1, int(math.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return _empty_depth_features()

    crop = depth_map[iy1:iy2, ix1:ix2]
    crop_stats = _safe_depth_stats(crop, depth_scale)

    box_w = max(1, ix2 - ix1)
    box_h = max(1, iy2 - iy1)
    cx = ix1 + box_w // 2
    cy = iy1 + box_h // 2
    patch_r = max(2, min(box_w, box_h) // 10)
    center = depth_map[max(0, cy - patch_r) : min(height, cy + patch_r + 1), max(0, cx - patch_r) : min(width, cx + patch_r + 1)]

    by1 = iy1 + int(0.65 * box_h)
    bottom = depth_map[by1:iy2, ix1:ix2]
    center_median = _safe_depth_stats(center, depth_scale)[2]
    bottom_median = _safe_depth_stats(bottom, depth_scale)[2]
    depth_span = crop_stats[4] - crop_stats[0]
    bottom_minus_center = bottom_median - center_median
    return crop_stats + [center_median, bottom_median, depth_span, bottom_minus_center]


def _region_mask_for_class(class_id: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.ones((height, width), dtype=bool)
    yy = np.linspace(0.0, 1.0, height, endpoint=False)[:, None]
    xx = np.linspace(0.0, 1.0, width, endpoint=False)[None, :]
    if class_id == 0:  # person: torso/lower-body, avoid head/background.
        mask &= (yy >= 0.25) & (yy <= 0.95) & (xx >= 0.20) & (xx <= 0.80)
    elif class_id in {1, 6}:  # forklift / pallet truck: lower visible body.
        mask &= (yy >= 0.35) & (yy <= 0.98) & (xx >= 0.08) & (xx <= 0.92)
    elif class_id in {2, 3}:  # compact robots.
        mask &= (yy >= 0.18) & (yy <= 0.92) & (xx >= 0.12) & (xx <= 0.88)
    elif class_id in {4, 5}:  # humanoid robots.
        mask &= (yy >= 0.20) & (yy <= 0.96) & (xx >= 0.18) & (xx <= 0.82)
    return mask


def _static_depth_path(static_depth_root: Path | str | None, scene_name: str, camera_id: str) -> Path | None:
    if static_depth_root is None:
        return None
    camera_name = str(camera_id)
    if not camera_name.startswith("Camera_"):
        try:
            camera_name = f"Camera_{int(camera_name):04d}"
        except ValueError:
            pass
    return Path(static_depth_root) / scene_name / f"{camera_name}.npy"


def object_aware_depth_features(
    depth_map,
    xyxy: Sequence[float],
    class_id: int,
    camera: CameraCalibration,
    baseline_xy: tuple[float, float],
    scene_name: str | None = None,
    camera_id: str | None = None,
    static_depth_root: Path | str | None = None,
    background_depth=None,
    depth_scale: float = 0.001,
    foreground_delta_m: float = 0.20,
) -> list[float]:
    """Depth features for residual learning.

    These features do not directly replace the BEV estimate. They expose
    object-aware visible-surface depth, static-background agreement, and
    backprojected offset so the MLP can learn centroid correction.
    """
    _require_numpy()
    crop_feats = depth_crop_features(
        depth_map,
        xyxy,
        depth_scale=depth_scale,
        source_width=camera.frame_width,
        source_height=camera.frame_height,
    )
    height, width = depth_map.shape
    sx = width / max(1.0, float(camera.frame_width))
    sy = height / max(1.0, float(camera.frame_height))
    x1, y1, x2, y2 = map(float, xyxy)
    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy
    ix1 = max(0, min(width - 1, int(math.floor(x1))))
    iy1 = max(0, min(height - 1, int(math.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return _empty_depth_features()

    crop = np.asarray(depth_map[iy1:iy2, ix1:ix2], dtype=np.float32)
    valid = np.isfinite(crop) & (crop > 0)
    region = _region_mask_for_class(class_id, crop.shape)
    selected = valid & region

    bg = None
    if background_depth is not None and getattr(background_depth, "shape", None) == depth_map.shape:
        bg = np.asarray(background_depth[iy1:iy2, ix1:ix2], dtype=np.float32)
    elif scene_name is not None and camera_id is not None:
        path = _static_depth_path(static_depth_root, scene_name, camera_id)
        if path is not None and path.exists():
            loaded = np.load(path)
            if loaded.shape == depth_map.shape:
                bg = loaded[iy1:iy2, ix1:ix2].astype(np.float32)
    if bg is not None:
        dynamic = valid & (np.abs(crop - bg) * depth_scale >= foreground_delta_m)
        if int((dynamic & region).sum()) >= 8:
            selected = dynamic & region

    if int(selected.sum()) < 8:
        selected = valid & region
    if int(selected.sum()) < 8:
        return crop_feats + [0.0] * (32 - len(crop_feats))

    ys, xs = np.nonzero(selected)
    depths_m = crop[ys, xs] * depth_scale
    finite = np.isfinite(depths_m) & (depths_m > 0)
    if int(finite.sum()) < 8:
        return crop_feats + [0.0] * (32 - len(crop_feats))
    xs = xs[finite] + ix1
    ys = ys[finite] + iy1
    depths_m = depths_m[finite]

    p10, p50, p90 = np.percentile(depths_m, [10, 50, 90])
    if class_id in {1, 6}:
        ref_depth = float(np.percentile(depths_m, 35))
        ref_y = float(np.percentile(ys, 72))
    elif class_id == 0:
        ref_depth = float(np.percentile(depths_m, 45))
        ref_y = float(np.percentile(ys, 68))
    else:
        ref_depth = float(np.percentile(depths_m, 50))
        ref_y = float(np.percentile(ys, 60))
    ref_x = float(np.median(xs))
    ref_img_x = ref_x / max(1e-6, sx)
    ref_img_y = ref_y / max(1e-6, sy)
    point = camera.image_depth_to_world(ref_img_x, ref_img_y, ref_depth)
    dx = dy = dz = reproj = 0.0
    has_point = 0.0
    if point is not None:
        projected = camera.ground_to_image(point[0], point[1])
        reproj = float(((projected[0] - ref_img_x) ** 2 + (projected[1] - ref_img_y) ** 2) ** 0.5)
        if np.isfinite(reproj):
            dx = float(point[0] - baseline_xy[0])
            dy = float(point[1] - baseline_xy[1])
            dz = float(point[2])
            has_point = 1.0

    area = max(1, crop.size)
    selected_ratio = float(selected.sum() / area)
    valid_ratio = float(valid.sum() / area)
    dynamic_ratio = float(((np.abs(crop - bg) * depth_scale >= foreground_delta_m) & valid).sum() / area) if bg is not None else 0.0
    spread = float(p90 - p10)
    extras = [
        float(p10),
        float(p50),
        float(p90),
        spread,
        valid_ratio,
        selected_ratio,
        dynamic_ratio,
        dx,
        dy,
        dz,
        reproj,
        has_point,
        math.log1p(max(0.0, spread)),
        math.log1p(max(0.0, abs(dx))),
        math.log1p(max(0.0, abs(dy))),
        math.log1p(max(0.0, reproj)),
        float(ref_img_x / max(1, camera.frame_width)),
        float(ref_img_y / max(1, camera.frame_height)),
    ]
    return (crop_feats + extras)[:32]


def build_geometry_residual_dataset(
    data_root: Path | str,
    year: int,
    split: str,
    out_path: Path | str,
    priors_path: Path | str | None = None,
    scenes: Sequence[str] | None = None,
    frame_stride: int = 30,
    max_frames_per_scene: int | None = None,
    min_box_area: float = 16.0,
    use_depth: bool = False,
    depth_scale: float = 0.001,
    depth_root: Path | str | None = None,
    static_depth_root: Path | str | None = None,
) -> dict:
    _require_numpy()
    if use_depth and depth_root is None and h5py is None:
        raise RuntimeError("h5py is required to build a depth-aware geometry residual dataset")
    priors = load_priors(priors_path)
    scene_filter = set(scenes or [])
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    class_ids: list[int] = []
    scene_ids: list[int] = []
    depth_samples = 0
    missing_depth_samples = 0

    for scene_index, scene_dir in enumerate(iter_scene_dirs(data_root, year, split)):
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        cameras = load_scene_cameras(scene_dir)
        depth_handles = {}
        if use_depth and depth_root is None:
            for camera_id in cameras:
                path = _depth_file(scene_dir, camera_id)
                if path.exists():
                    depth_handles[str(camera_id)] = h5py.File(path, "r")
        for frame_key, frame_items in iter_gt_frames(scene_dir, max_frames=max_frames_per_scene):
            frame_id = int(frame_key)
            if frame_stride > 1 and frame_id % frame_stride != 0:
                continue
            depth_key = _frame_depth_key(frame_id)
            frame_npz_depth: dict[str, np.ndarray | None] = {}
            for item in frame_items:
                class_id = CLASS_TO_ID[_get_any(item, "object_type")]
                location = list(map(float, _get_any(item, "location")))
                scale = list(map(float, _get_any(item, "scale")))
                rotation = list(map(float, _get_any(item, "rotation")))
                prior = prior_for_class(priors, class_id)
                for camera_id, xyxy in _get_any(item, "bbox2d").items():
                    camera = cameras.get(str(camera_id))
                    if camera is None:
                        continue
                    x1, y1, x2, y2 = map(float, xyxy)
                    if max(0.0, x2 - x1) * max(0.0, y2 - y1) < min_box_area:
                        continue
                    det = Detection2D(
                        scene_name=scene_dir.name,
                        camera_id=str(camera_id),
                        frame_id=frame_id,
                        class_id=class_id,
                        score=1.0,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                    depth_feature = None
                    if use_depth:
                        if depth_root is not None:
                            camera_key = str(camera_id)
                            if camera_key not in frame_npz_depth:
                                path = _depth_npz_file(depth_root, split, scene_dir.name, camera_key, frame_id)
                                frame_npz_depth[camera_key] = _load_depth_npz(path) if path.exists() else None
                            depth_map = frame_npz_depth[camera_key]
                            if depth_map is not None:
                                depth_feature = object_aware_depth_features(
                                    depth_map,
                                    xyxy,
                                    class_id,
                                    camera,
                                    camera.image_to_ground(*det.bottom_center),
                                    scene_name=scene_dir.name,
                                    camera_id=str(camera_id),
                                    static_depth_root=static_depth_root,
                                    depth_scale=1.0,
                                )
                                depth_samples += 1
                            else:
                                depth_feature = _empty_depth_features()
                                missing_depth_samples += 1
                        else:
                            handle = depth_handles.get(str(camera_id))
                            if handle is not None and depth_key in handle:
                                depth_feature = object_aware_depth_features(
                                    handle[depth_key],
                                    xyxy,
                                    class_id,
                                    camera,
                                    camera.image_to_ground(*det.bottom_center),
                                    scene_name=scene_dir.name,
                                    camera_id=str(camera_id),
                                    static_depth_root=static_depth_root,
                                    depth_scale=depth_scale,
                                )
                                depth_samples += 1
                            else:
                                depth_feature = _empty_depth_features()
                                missing_depth_samples += 1
                    feature, (baseline_x, baseline_y) = geometry_features(det, camera, prior, depth_feature)
                    target = np.asarray(
                        [
                            location[0] - baseline_x,
                            location[1] - baseline_y,
                            location[2] - float(prior["z"]),
                            math.log(max(1e-4, scale[0]) / float(prior["width"])),
                            math.log(max(1e-4, scale[1]) / float(prior["length"])),
                            math.log(max(1e-4, scale[2]) / float(prior["height"])),
                            math.sin(rotation[2]),
                            math.cos(rotation[2]),
                        ],
                        dtype=np.float32,
                    )
                    if np.all(np.isfinite(feature)) and np.all(np.isfinite(target)):
                        features.append(feature)
                        targets.append(target)
                        class_ids.append(class_id)
                        scene_ids.append(scene_index)
        for handle in depth_handles.values():
            handle.close()

    if not features:
        raise ValueError("No geometry residual samples were generated")
    x = np.stack(features)
    y = np.stack(targets)
    metadata = {
        "year": year,
        "split": split,
        "frame_stride": frame_stride,
        "samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "classes": ID_TO_CLASS,
        "use_depth": bool(use_depth),
        "depth_scale": depth_scale,
        "depth_root": None if depth_root is None else str(depth_root),
        "depth_samples": depth_samples,
        "missing_depth_samples": missing_depth_samples,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=x,
        targets=y,
        class_ids=np.asarray(class_ids, dtype=np.int16),
        scene_ids=np.asarray(scene_ids, dtype=np.int16),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return {
        "output": str(out),
        "samples": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "frame_stride": frame_stride,
        "use_depth": bool(use_depth),
        "depth_samples": depth_samples,
        "missing_depth_samples": missing_depth_samples,
    }


def train_geometry_residual_model(
    dataset_path: Path | str,
    out_path: Path | str,
    epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    seed: int = 2026,
    device: str = "cuda",
    patience: int = 8,
    min_delta: float = 1e-4,
) -> dict:
    _require_numpy()
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to train the geometry residual model") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    payload = np.load(dataset_path, allow_pickle=False)
    x = payload["features"].astype(np.float32)
    y = payload["targets"].astype(np.float32)
    scene_ids = payload["scene_ids"].astype(np.int64)
    unique_scenes = sorted(set(scene_ids.tolist()))
    if len(unique_scenes) < 2:
        raise ValueError("Geometry residual training requires samples from at least two scenes")
    random.Random(seed).shuffle(unique_scenes)
    val_count = max(1, int(round(len(unique_scenes) * val_fraction)))
    val_scenes = set(unique_scenes[-val_count:])
    val_mask = np.asarray([scene in val_scenes for scene in scene_ids])
    train_mask = ~val_mask

    feature_mean = x[train_mask].mean(axis=0)
    feature_std = x[train_mask].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    target_mean = y[train_mask, :6].mean(axis=0)
    target_std = y[train_mask, :6].std(axis=0)
    target_std[target_std < 1e-6] = 1.0
    x_norm = (x - feature_mean) / feature_std
    y_reg = (y[:, :6] - target_mean) / target_std
    y_full = np.concatenate([y_reg, y[:, 6:8]], axis=1).astype(np.float32)
    train_data = TensorDataset(torch.from_numpy(x_norm[train_mask]), torch.from_numpy(y_full[train_mask]))
    val_data = TensorDataset(torch.from_numpy(x_norm[val_mask]), torch.from_numpy(y_full[val_mask]))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(x.shape[1], 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 128),
        nn.SiLU(),
        nn.Linear(128, 14),
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    target_mean_tensor = torch.from_numpy(target_mean).to(resolved_device)
    target_std_tensor = torch.from_numpy(target_std).to(resolved_device)

    def loss_fn(output, target):
        means = output[:, :6]
        log_vars = output[:, 6:12].clamp(-5.0, 3.0)
        regression = (torch.exp(-log_vars) * (means - target[:, :6]).square() + log_vars).mean()
        yaw_pred = torch.nn.functional.normalize(output[:, 12:14], dim=1)
        yaw_target = torch.nn.functional.normalize(target[:, 6:8], dim=1)
        yaw = (1.0 - (yaw_pred * yaw_target).sum(dim=1)).mean()
        return regression + 0.5 * yaw

    best_state = None
    best_val = float("inf")
    best_selection_score = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch_x)
            train_count += len(batch_x)
        model.eval()
        val_sum = 0.0
        val_count_samples = 0
        center_error_sum = 0.0
        z_error_sum = 0.0
        dimension_log_error_sum = 0.0
        yaw_error_sum = 0.0
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(resolved_device)
                batch_y = batch_y.to(resolved_device)
                output = model(batch_x)
                loss = loss_fn(output, batch_y)
                val_sum += float(loss) * len(batch_x)
                val_count_samples += len(batch_x)
                predicted_regression = output[:, :6] * target_std_tensor + target_mean_tensor
                target_regression = batch_y[:, :6] * target_std_tensor + target_mean_tensor
                center_error_sum += float(
                    torch.linalg.vector_norm(
                        predicted_regression[:, :2] - target_regression[:, :2],
                        dim=1,
                    ).sum()
                )
                z_error_sum += float(
                    (predicted_regression[:, 2] - target_regression[:, 2]).abs().sum()
                )
                dimension_log_error_sum += float(
                    (predicted_regression[:, 3:6] - target_regression[:, 3:6])
                    .abs()
                    .mean(dim=1)
                    .sum()
                )
                yaw_pred = torch.nn.functional.normalize(output[:, 12:14], dim=1)
                yaw_target = torch.nn.functional.normalize(batch_y[:, 6:8], dim=1)
                yaw_error_sum += float(
                    torch.acos((yaw_pred * yaw_target).sum(dim=1).clamp(-1.0, 1.0)).sum()
                )
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(1, train_count),
            "val_loss": val_sum / max(1, val_count_samples),
            "val_center_error_m": center_error_sum / max(1, val_count_samples),
            "val_z_error_m": z_error_sum / max(1, val_count_samples),
            "val_dimension_log_error": dimension_log_error_sum / max(1, val_count_samples),
            "val_yaw_error_deg": math.degrees(yaw_error_sum / max(1, val_count_samples)),
        }
        row["selection_score"] = (
            row["val_center_error_m"]
            + 0.2 * row["val_dimension_log_error"]
            + 0.002 * row["val_yaw_error_deg"]
            + 0.5 * row["val_z_error_m"]
        )
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["selection_score"] < best_selection_score - min_delta:
            best_val = row["val_loss"]
            best_selection_score = row["selection_score"]
            best_state = {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    json.dumps(
                        {
                            "early_stop": True,
                            "epoch": epoch,
                            "best_val_loss": best_val,
                            "best_selection_score": best_selection_score,
                            "patience": patience,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    exported = {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "metadata": np.asarray(
            json.dumps(
                {
                    "dataset": str(dataset_path),
                    "train_samples": int(train_mask.sum()),
                    "val_samples": int(val_mask.sum()),
                    "val_scenes": sorted(map(int, val_scenes)),
                    "best_val_loss": best_val,
                    "best_selection_score": best_selection_score,
                    "patience": patience,
                    "min_delta": min_delta,
                    "history": history,
                }
            )
        ),
    }
    for export_index, module_index in enumerate((0, 2, 4, 6)):
        exported[f"weight_{export_index}"] = best_state[f"{module_index}.weight"].astype(np.float32)
        exported[f"bias_{export_index}"] = best_state[f"{module_index}.bias"].astype(np.float32)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **exported)
    return {
        "output": str(out),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "best_val_loss": best_val,
        "best_selection_score": best_selection_score,
        "epochs_ran": len(history),
        "device": str(resolved_device),
    }


class GeometryResidualPredictor:
    def __init__(self, path: Path | str):
        _require_numpy()
        payload = np.load(path, allow_pickle=False)
        self.feature_mean = payload["feature_mean"]
        self.feature_std = payload["feature_std"]
        self.target_mean = payload["target_mean"]
        self.target_std = payload["target_std"]
        self.weights = [payload[f"weight_{index}"] for index in range(4)]
        self.biases = [payload[f"bias_{index}"] for index in range(4)]

    def predict(self, feature: np.ndarray) -> dict:
        value = (feature.astype(np.float32) - self.feature_mean) / self.feature_std
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = weight @ value + bias
            if index < len(self.weights) - 1:
                value = value / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
        means = value[:6] * self.target_std + self.target_mean
        sigma = np.exp(0.5 * np.clip(value[6:12], -5.0, 3.0)) * self.target_std
        return {
            "dx": float(np.clip(means[0], -5.0, 5.0)),
            "dy": float(np.clip(means[1], -5.0, 5.0)),
            "dz": float(np.clip(means[2], -2.0, 2.0)),
            "dlog_width": float(np.clip(means[3], -math.log(3.0), math.log(3.0))),
            "dlog_length": float(np.clip(means[4], -math.log(3.0), math.log(3.0))),
            "dlog_height": float(np.clip(means[5], -math.log(3.0), math.log(3.0))),
            "yaw": math.atan2(float(value[12]), float(value[13])),
            "center_uncertainty": float(min(10.0, math.hypot(float(sigma[0]), float(sigma[1])))),
        }
