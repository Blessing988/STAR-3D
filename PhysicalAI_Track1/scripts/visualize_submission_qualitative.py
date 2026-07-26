#!/usr/bin/env python3
"""Generate paper-ready qualitative PNG figures from Track 1 submission files."""

from __future__ import annotations

import argparse
import math
import zipfile
from collections import Counter, defaultdict
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
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_submission(path: Path) -> list[Box3D]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            target = "track1.txt" if "track1.txt" in names else names[0]
            rows = zf.read(target).decode("utf-8").splitlines()
    else:
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


def choose_scenes(boxes: list[Box3D], count: int) -> list[int]:
    by_scene: dict[int, list[Box3D]] = defaultdict(list)
    for box in boxes:
        by_scene[box.scene_id].append(box)

    scored = []
    for scene_id, scene_boxes in by_scene.items():
        class_count = len({b.class_id for b in scene_boxes})
        object_count = len({(b.class_id, b.object_id) for b in scene_boxes})
        frame_count = len({b.frame_id for b in scene_boxes})
        scored.append((class_count, object_count, frame_count, len(scene_boxes), scene_id))
    scored.sort(reverse=True)
    return [item[-1] for item in scored[:count]]


def bounds_for(boxes: list[Box3D], pad: float = 8.0) -> tuple[float, float, float, float]:
    xs = [b.x for b in boxes]
    ys = [b.y for b in boxes]
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    span = max(xmax - xmin, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return cx - span / 2.0, cx + span / 2.0, cy - span / 2.0, cy + span / 2.0


def make_projector(
    boxes: list[Box3D],
    rect: tuple[int, int, int, int],
) -> tuple[callable, tuple[float, float, float, float]]:
    left, top, right, bottom = rect
    xmin, xmax, ymin, ymax = bounds_for(boxes)

    def project(x: float, y: float) -> tuple[int, int]:
        px = left + int(round((x - xmin) / (xmax - xmin) * (right - left)))
        py = bottom - int(round((y - ymin) / (ymax - ymin) * (bottom - top)))
        return px, py

    return project, (xmin, xmax, ymin, ymax)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    label_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    xmin, xmax, ymin, ymax = bounds
    draw.rectangle(rect, outline=(45, 45, 45), width=2)
    grid_color = (218, 222, 227)
    text_color = (80, 80, 80)
    for i in range(1, 5):
        x = left + int((right - left) * i / 5)
        y = top + int((bottom - top) * i / 5)
        draw.line([(x, top), (x, bottom)], fill=grid_color, width=1)
        draw.line([(left, y), (right, y)], fill=grid_color, width=1)
    draw.text((left, bottom + 10), f"x: {xmin:.0f} to {xmax:.0f} m", fill=text_color, font=label_font)
    draw.text((left, bottom + 36), f"y: {ymin:.0f} to {ymax:.0f} m", fill=text_color, font=label_font)


def oriented_box_xy(box: Box3D) -> list[tuple[float, float]]:
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    half_w = box.width / 2.0
    half_l = box.length / 2.0
    corners = [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l)]
    return [(box.x + dx * c - dy * s, box.y + dx * s + dy * c) for dx, dy in corners]


def draw_legend(draw: ImageDraw.ImageDraw, classes: list[int], x: int, y: int) -> None:
    small = font(23)
    for idx, class_id in enumerate(classes):
        yy = y + idx * 32
        color = rgb(CLASS_COLORS.get(class_id, "#333333"))
        draw.rectangle((x, yy + 4, x + 22, yy + 22), fill=color, outline=(30, 30, 30))
        draw.text((x + 32, yy), CLASS_NAMES.get(class_id, str(class_id)), fill=(30, 30, 30), font=small)


def plot_scene_tracks(boxes: list[Box3D], scene_id: int, out_path: Path, max_tracks: int) -> None:
    scene = [b for b in boxes if b.scene_id == scene_id]
    tracks: dict[tuple[int, int], list[Box3D]] = defaultdict(list)
    for box in scene:
        tracks[(box.class_id, box.object_id)].append(box)
    ranked = sorted(tracks.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_tracks]

    img = Image.new("RGB", (1800, 1500), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(42, bold=True)
    label_font = font(22)
    draw.text((70, 35), f"Predicted BEV trajectories, scene {scene_id}", fill=(20, 20, 20), font=title_font)

    rect = (90, 120, 1510, 1340)
    project, bounds = make_projector(scene, rect)
    draw_axes(draw, rect, bounds, label_font)

    for (class_id, object_id), tb in ranked:
        tb = sorted(tb, key=lambda b: b.frame_id)
        color = rgb(CLASS_COLORS.get(class_id, "#333333"))
        points = [project(b.x, b.y) for b in tb]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3, joint="curve")
        sx, sy = points[0]
        ex, ey = points[-1]
        draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=color, outline="white", width=2)
        draw.polygon([(ex + 9, ey), (ex - 7, ey - 7), (ex - 7, ey + 7)], fill=color, outline="white")
        if len(points) >= 12:
            mx, my = points[len(points) // 2]
            draw.text((mx + 4, my + 4), str(object_id), fill=color, font=font(15))

    draw_legend(draw, sorted({b.class_id for b in scene}), 1540, 140)
    draw.text((70, 1410), "Circles mark track starts; arrows mark most recent positions.",
              fill=(80, 80, 80), font=label_font)
    img.save(out_path)


def plot_scene_snapshots(boxes: list[Box3D], scene_id: int, out_path: Path, max_boxes_per_frame: int) -> None:
    scene = [b for b in boxes if b.scene_id == scene_id]
    frames = sorted({b.frame_id for b in scene})
    picks = [frames[0], frames[len(frames) // 3], frames[(2 * len(frames)) // 3], frames[-1]]

    img = Image.new("RGB", (2400, 700), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, bold=True)
    label_font = font(18)
    draw.text((60, 28), f"Online 3D boxes in global BEV, scene {scene_id}", fill=(20, 20, 20), font=title_font)

    panel_w = 540
    for idx, frame_id in enumerate(picks):
        left = 55 + idx * 580
        top = 105
        rect = (left, top, left + panel_w, top + 510)
        project, bounds = make_projector(scene, rect)
        draw_axes(draw, rect, bounds, label_font)
        draw.text((left, top - 34), f"Frame {frame_id}", fill=(30, 30, 30), font=font(24, bold=True))
        frame_boxes = [b for b in scene if b.frame_id == frame_id]
        frame_boxes = sorted(frame_boxes, key=lambda b: (b.class_id, b.object_id))[:max_boxes_per_frame]
        for box in frame_boxes:
            color = rgb(CLASS_COLORS.get(box.class_id, "#333333"))
            pts = [project(x, y) for x, y in oriented_box_xy(box)]
            draw.line(pts + [pts[0]], fill=color, width=2)
            cx, cy = project(box.x, box.y)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)
    img.save(out_path)


def plot_class_counts(boxes: list[Box3D], out_path: Path) -> None:
    counts = Counter(b.class_id for b in boxes)
    classes = sorted(counts)
    img = Image.new("RGB", (1800, 840), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(42, bold=True)
    label_font = font(22)
    draw.text((70, 35), "Class distribution in the best official submission", fill=(20, 20, 20), font=title_font)

    left, top, right, bottom = 110, 135, 1710, 650
    max_value = max(counts.values())
    draw.line((left, bottom, right, bottom), fill=(40, 40, 40), width=2)
    draw.line((left, top, left, bottom), fill=(40, 40, 40), width=2)
    bar_w = int((right - left) / max(len(classes), 1) * 0.65)
    step = int((right - left) / max(len(classes), 1))
    for idx, class_id in enumerate(classes):
        value = counts[class_id]
        color = rgb(CLASS_COLORS.get(class_id, "#333333"))
        x0 = left + idx * step + (step - bar_w) // 2
        x1 = x0 + bar_w
        y0 = bottom - int((value / max_value) * (bottom - top))
        draw.rectangle((x0, y0, x1, bottom), fill=color, outline=(25, 25, 25))
        draw.text((x0, y0 - 30), f"{value:,}", fill=(30, 30, 30), font=font(17), anchor=None)
        label = CLASS_NAMES.get(class_id, str(class_id))
        draw.text((x0 - 8, bottom + 18), label, fill=(30, 30, 30), font=font(18))
    draw.text((75, 380), "Predicted boxes", fill=(50, 50, 50), font=label_font)
    img.save(out_path)


def track_lengths(boxes: list[Box3D]) -> list[int]:
    tracks: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for box in boxes:
        tracks[(box.scene_id, box.class_id, box.object_id)].add(box.frame_id)
    return sorted(len(frames) for frames in tracks.values())


def plot_continuity_comparison(base: list[Box3D], refined: list[Box3D], out_path: Path) -> None:
    img = Image.new("RGB", (1500, 950), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(38, bold=True)
    label_font = font(22)
    draw.text((70, 35), "Track continuity before and after BEV refinement", fill=(20, 20, 20), font=title_font)

    left, top, right, bottom = 120, 130, 1360, 780
    draw.rectangle((left, top, right, bottom), outline=(45, 45, 45), width=2)
    for i in range(1, 5):
        x = left + int((right - left) * i / 5)
        y = top + int((bottom - top) * i / 5)
        draw.line((x, top, x, bottom), fill=(218, 222, 227), width=1)
        draw.line((left, y, right, y), fill=(218, 222, 227), width=1)

    series = [
        (track_lengths(base), "Adaptive online", (108, 117, 125)),
        (track_lengths(refined), "BEV refinement", rgb(CLASS_COLORS[0])),
    ]
    max_len = max(max(s[0]) for s in series if s[0])
    min_len = 1

    def project(length: int, frac: float) -> tuple[int, int]:
        x_norm = (math.log10(max(length, 1)) - math.log10(min_len)) / (math.log10(max_len) - math.log10(min_len))
        x = left + int(x_norm * (right - left))
        y = bottom - int(frac * (bottom - top))
        return x, y

    legend_x, legend_y = 965, 160
    for legend_idx, (lengths, label, color) in enumerate(series):
        points = [project(length, (idx + 1) / len(lengths)) for idx, length in enumerate(lengths)]
        draw.line(points, fill=color, width=5)
        yy = legend_y + legend_idx * 40
        draw.line((legend_x, yy + 14, legend_x + 58, yy + 14), fill=color, width=6)
        draw.text((legend_x + 72, yy), label, fill=(35, 35, 35), font=font(23, bold=True))

    draw.text((left, bottom + 45), "Track length in frames (log scale)", fill=(50, 50, 50), font=label_font)
    draw.text((35, 430), "Cumulative fraction", fill=(50, 50, 50), font=label_font)
    for value in [1, 10, 100, 1000, max_len]:
        if value <= max_len:
            x, _ = project(value, 0)
            draw.text((x - 15, bottom + 12), str(value), fill=(70, 70, 70), font=font(17))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, y = project(1, frac)
        draw.text((left - 55, y - 10), f"{frac:.2f}", fill=(70, 70, 70), font=font(17))
    img.save(out_path)


def write_tex_snippet(out_dir: Path, scene_ids: list[int]) -> None:
    first_scene = scene_ids[0]
    snippet = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=0.49\linewidth]{{figures/qual_bev_scene_{first_scene}_tracks.png}}
\includegraphics[width=0.49\linewidth]{{figures/qual_bev_scene_{first_scene}_snapshots.png}}
\caption{{Qualitative predictions from STAR-3D on a real test scene. Left:
predicted object trajectories in the global bird's-eye-view coordinate system.
Right: online 3D boxes at representative frames. Colors denote object classes,
with the legend shown in the trajectory view.}}
\label{{fig:qualitative_bev}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.58\linewidth]{{figures/qual_track_continuity_comparison.png}}
\caption{{Effect of conservative BEV tracklet refinement on predicted track
continuity. The refinement shifts tracks toward longer temporal support while
keeping the online detector and lifting stages fixed.}}
\label{{fig:qualitative_continuity}}
\end{{figure}}
"""
    (out_dir / "qualitative_figures_snippet.tex").write_text(snippet, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, required=True, help="Best submission zip or track1.txt.")
    parser.add_argument("--baseline", type=Path, help="Baseline submission zip for continuity comparison.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene", type=int, action="append", help="Scene id to visualize. Can be repeated.")
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--max-tracks", type=int, default=80)
    parser.add_argument("--max-boxes-per-frame", type=int, default=140)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    boxes = read_submission(args.best)
    scene_ids = args.scene or choose_scenes(boxes, args.num_scenes)

    plot_class_counts(boxes, args.out_dir / "qual_class_distribution.png")
    for scene_id in scene_ids:
        plot_scene_tracks(boxes, scene_id, args.out_dir / f"qual_bev_scene_{scene_id}_tracks.png", args.max_tracks)
        plot_scene_snapshots(
            boxes,
            scene_id,
            args.out_dir / f"qual_bev_scene_{scene_id}_snapshots.png",
            args.max_boxes_per_frame,
        )

    if args.baseline:
        baseline_boxes = read_submission(args.baseline)
        plot_continuity_comparison(baseline_boxes, boxes, args.out_dir / "qual_track_continuity_comparison.png")

    write_tex_snippet(args.out_dir, scene_ids)
    print(f"Wrote qualitative figures to {args.out_dir}")
    print("Scenes:", ", ".join(str(s) for s in scene_ids))


if __name__ == "__main__":
    main()
