#!/usr/bin/env python3
"""Create multi-camera RGB + BEV qualitative figures for Track 1 submissions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {
    0: "Person",
    1: "Forklift",
    2: "NovaCarter",
    3: "Transporter",
    4: "FourierGR1T2",
    5: "AgilityDigit",
    6: "PalletTruck",
}

CLASS_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
}


@dataclass(frozen=True)
class Box3D:
    scene_id: int
    class_id: int
    object_id: int
    frame_id: int
    x: float
    y: float
    z: float
    width: float
    length: float
    height: float
    yaw: float


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def lanczos_filter() -> int:
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, "LANCZOS", Image.BICUBIC)


def read_submission(path: Path) -> list[Box3D]:
    rows = path.read_text(encoding="utf-8").splitlines()
    boxes: list[Box3D] = []
    for row in rows:
        parts = row.strip().replace(",", " ").split()
        if len(parts) < 11:
            continue
        boxes.append(
            Box3D(
                scene_id=int(float(parts[0])),
                class_id=int(float(parts[1])),
                object_id=int(float(parts[2])),
                frame_id=int(float(parts[3])),
                x=float(parts[4]),
                y=float(parts[5]),
                z=float(parts[6]),
                width=float(parts[7]),
                length=float(parts[8]),
                height=float(parts[9]),
                yaw=float(parts[10]),
            )
        )
    return boxes


def camera_by_id(calibration_path: Path) -> dict[str, dict]:
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    return {str(sensor["id"]): sensor for sensor in data.get("sensors", []) if sensor.get("type") == "camera"}


def project(camera_matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float] | None:
    x, y, z = point
    u = camera_matrix[0][0] * x + camera_matrix[0][1] * y + camera_matrix[0][2] * z + camera_matrix[0][3]
    v = camera_matrix[1][0] * x + camera_matrix[1][1] * y + camera_matrix[1][2] * z + camera_matrix[1][3]
    w = camera_matrix[2][0] * x + camera_matrix[2][1] * y + camera_matrix[2][2] * z + camera_matrix[2][3]
    if abs(w) < 1e-8:
        return None
    return u / w, v / w


def box_corners(box: Box3D) -> list[tuple[float, float, float]]:
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    hw = box.width / 2.0
    hl = box.length / 2.0
    hh = box.height / 2.0
    corners = []
    for dx, dy, dz in [
        (-hw, -hl, -hh),
        (hw, -hl, -hh),
        (hw, hl, -hh),
        (-hw, hl, -hh),
        (-hw, -hl, hh),
        (hw, -hl, hh),
        (hw, hl, hh),
        (-hw, hl, hh),
    ]:
        wx = box.x + dx * c - dy * s
        wy = box.y + dx * s + dy * c
        wz = box.z + dz
        corners.append((wx, wy, wz))
    return corners


def visible_projection(camera: dict, box: Box3D, image_size: tuple[int, int]) -> list[tuple[float, float]] | None:
    matrix = camera["cameraMatrix"]
    points = [project(matrix, corner) for corner in box_corners(box)]
    if any(point is None for point in points):
        return None
    pts = [(float(u), float(v)) for u, v in points if u is not None and v is not None]
    width, height = image_size
    xs = [u for u, _ in pts]
    ys = [v for _, v in pts]
    if max(xs) < 0 or min(xs) > width or max(ys) < 0 or min(ys) > height:
        return None
    if max(xs) - min(xs) < 2 or max(ys) - min(ys) < 2:
        return None
    return pts


def draw_projected_box(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color: tuple[int, int, int], scale: float) -> None:
    p = [(int(round(u * scale)), int(round(v * scale))) for u, v in pts]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        draw.line([p[a], p[b]], fill=color, width=3)


def annotate_camera_panel(image_path: Path, camera_id: str, boxes: list[Box3D], camera: dict, panel_size: tuple[int, int]) -> tuple[Image.Image, int]:
    img = Image.open(image_path).convert("RGB")
    raw_w, raw_h = img.size
    scale = min(panel_size[0] / raw_w, panel_size[1] / raw_h)
    new_size = (int(raw_w * scale), int(raw_h * scale))
    img = img.resize(new_size, lanczos_filter())
    panel = Image.new("RGB", panel_size, (16, 18, 22))
    offset = ((panel_size[0] - new_size[0]) // 2, 34)
    panel.paste(img, offset)
    draw = ImageDraw.Draw(panel)
    title_font = font(20, bold=True)
    label_font = font(14, bold=True)
    draw.text((10, 6), camera_id, fill=(255, 255, 255), font=title_font)
    visible = 0
    for box in boxes:
        pts = visible_projection(camera, box, (raw_w, raw_h))
        if pts is None:
            continue
        visible += 1
        shifted = [((u * scale + offset[0]) / scale, (v * scale + offset[1]) / scale) for u, v in pts]
        color = rgb(CLASS_COLORS.get(box.class_id, "#ffffff"))
        draw_projected_box(draw, shifted, color, scale)
        cx = sum(u for u, _ in pts) / len(pts) * scale + offset[0]
        cy = min(v for _, v in pts) * scale + offset[1]
        label = f"{CLASS_NAMES.get(box.class_id, box.class_id)} {box.object_id}"
        tw = int(draw.textlength(label, font=label_font))
        draw.rectangle((cx, cy - 18, cx + tw + 6, cy), fill=(0, 0, 0))
        draw.text((cx + 3, cy - 18), label, fill=color, font=label_font)
    draw.text((panel_size[0] - 112, 8), f"{visible} boxes", fill=(220, 220, 220), font=font(16))
    return panel, visible


def bev_bounds(boxes: list[Box3D]) -> tuple[float, float, float, float]:
    xs = sorted(b.x for b in boxes)
    ys = sorted(b.y for b in boxes)
    n = len(boxes)
    lo = max(0, int(n * 0.03))
    hi = min(n - 1, int(n * 0.97))
    xmin, xmax = xs[lo], xs[hi]
    ymin, ymax = ys[lo], ys[hi]
    span = max(xmax - xmin, ymax - ymin, 10.0)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    pad = max(6.0, 0.08 * span)
    return cx - span / 2.0 - pad, cx + span / 2.0 + pad, cy - span / 2.0 - pad, cy + span / 2.0 + pad


def draw_bev(boxes: list[Box3D], size: tuple[int, int], scene_id: int, frame_id: int) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    title = font(28, bold=True)
    small = font(17)
    draw.text((28, 18), f"Global BEV, scene {scene_id}, frame {frame_id}", fill=(20, 20, 20), font=title)
    legend_classes = sorted({b.class_id for b in boxes})
    reserve_legend = size[0] >= 850 and len(legend_classes) > 0
    right_margin = 300 if reserve_legend else 45
    rect = (55, 80, size[0] - right_margin, size[1] - 95)
    xmin, xmax, ymin, ymax = bev_bounds(boxes)

    def p(x: float, y: float) -> tuple[int, int]:
        px = rect[0] + int((x - xmin) / (xmax - xmin) * (rect[2] - rect[0]))
        py = rect[3] - int((y - ymin) / (ymax - ymin) * (rect[3] - rect[1]))
        return px, py

    for i in range(6):
        x = rect[0] + int((rect[2] - rect[0]) * i / 5)
        y = rect[1] + int((rect[3] - rect[1]) * i / 5)
        draw.line((x, rect[1], x, rect[3]), fill=(220, 225, 230), width=1)
        draw.line((rect[0], y, rect[2], y), fill=(220, 225, 230), width=1)
    draw.rectangle(rect, outline=(40, 40, 40), width=2)

    for box in boxes:
        color = rgb(CLASS_COLORS.get(box.class_id, "#333333"))
        pts2d = [(x, y) for x, y, _ in box_corners(box)[:4]]
        poly = [p(x, y) for x, y in pts2d]
        draw.line(poly + [poly[0]], fill=color, width=3)
        cx, cy = p(box.x, box.y)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
        draw.text((cx + 5, cy - 8), str(box.object_id), fill=color, font=font(13, bold=True))

    draw.text((55, size[1] - 72), f"x: {xmin:.1f} to {xmax:.1f} m", fill=(80, 80, 80), font=small)
    draw.text((55, size[1] - 45), f"y: {ymin:.1f} to {ymax:.1f} m", fill=(80, 80, 80), font=small)
    legend_x = size[0] - 260 if reserve_legend else max(rect[0] + 10, rect[2] - 230)
    legend_y = 100 if reserve_legend else rect[1] + 15
    legend_h = 34 * len(legend_classes) + 18
    draw.rectangle(
        (legend_x - 12, legend_y - 10, size[0] - 35, legend_y + legend_h),
        fill=(255, 255, 255),
        outline=(210, 215, 220),
    )
    for idx, class_id in enumerate(legend_classes):
        yy = legend_y + idx * 28
        color = rgb(CLASS_COLORS.get(class_id, "#333333"))
        draw.rectangle((legend_x, yy + 4, legend_x + 20, yy + 22), fill=color, outline=(30, 30, 30))
        draw.text((legend_x + 28, yy), CLASS_NAMES.get(class_id, str(class_id)), fill=(40, 40, 40), font=small)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cameras", type=int, default=6)
    args = parser.parse_args()

    scene_name = f"Warehouse_{args.scene_id:03d}"
    scene_dir = args.dataset_root / "test" / scene_name
    cameras = camera_by_id(scene_dir / "calibration.json")
    boxes = [b for b in read_submission(args.submission) if b.scene_id == args.scene_id and b.frame_id == args.frame_id]
    if not boxes:
        raise SystemExit(f"No boxes for scene {args.scene_id}, frame {args.frame_id}")

    panel_size = (560, 340)
    scored: list[tuple[int, str, Path]] = []
    for camera_id, camera in sorted(cameras.items()):
        image_path = args.frames_root / f"{scene_name}_{camera_id}_{args.frame_id:06d}.jpg"
        if not image_path.exists():
            continue
        with Image.open(image_path) as img:
            image_size = img.size
        visible = sum(visible_projection(camera, box, image_size) is not None for box in boxes)
        scored.append((visible, camera_id, image_path))
    scored.sort(reverse=True)
    selected = scored[: args.max_cameras]
    if not selected:
        raise SystemExit("No camera images found")

    canvas = Image.new("RGB", (2320, 1040), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 22), "Multi-camera STAR-3D predictions on a real Track 1 test scene",
              fill=(20, 20, 20), font=font(38, bold=True))
    draw.text((34, 72), "Projected 3D boxes share the same global identities as the BEV panel.",
              fill=(70, 70, 70), font=font(22))

    for idx, (_, camera_id, image_path) in enumerate(selected):
        panel, _ = annotate_camera_panel(image_path, camera_id, boxes, cameras[camera_id], panel_size)
        x = 30 + (idx % 3) * (panel_size[0] + 18)
        y = 125 + (idx // 3) * (panel_size[1] + 22)
        canvas.paste(panel, (x, y))

    bev = draw_bev(boxes, (560, 702), args.scene_id, args.frame_id)
    canvas.paste(bev, (1740, 125))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out)
    print("Selected cameras:", ", ".join(c for _, c, _ in selected))
    print("Frame boxes:", len(boxes))


if __name__ == "__main__":
    main()
