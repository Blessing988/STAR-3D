from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: missing PLY end_header")
            header.append(line.decode("ascii", errors="replace").strip())
            if header[-1] == "end_header":
                break
        count = 0
        for line in header:
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
                break
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        data = np.fromfile(handle, dtype=dtype, count=count)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    return points, colors


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    payload = np.empty(len(points), dtype=dtype)
    payload["x"] = points[:, 0]
    payload["y"] = points[:, 1]
    payload["z"] = points[:, 2]
    payload["red"] = colors[:, 0]
    payload["green"] = colors[:, 1]
    payload["blue"] = colors[:, 2]
    with path.open("wb") as handle:
        handle.write(header)
        payload.tofile(handle)


def slice_starts(low: float, high: float, size: float, overlap: float) -> list[float]:
    if high <= low:
        return [low]
    step = max(1e-3, size * (1.0 - overlap))
    starts = list(np.arange(low, high - size + 1e-3, step))
    if not starts:
        starts = [low]
    if starts[-1] + size < high:
        starts.append(high - size)
    return [float(x) for x in starts]


def process_file(ply_path: Path, gt_path: Path | None, out_split_dir: Path, voxel_size: float, overlap: float, min_points: int) -> int:
    points, colors = read_binary_ply(ply_path)
    if len(points) < min_points:
        return 0
    gt_df = None
    if gt_path is not None and gt_path.exists():
        gt_df = pd.read_csv(
            gt_path,
            sep=" ",
            header=None,
            names=["label", "id", "x", "y", "z", "dx", "dy", "dz", "rx", "ry", "rz"],
        )
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    x_starts = slice_starts(float(min_bound[0]), float(max_bound[0]), voxel_size, overlap)
    y_starts = slice_starts(float(min_bound[1]), float(max_bound[1]), voxel_size, overlap)
    written = 0
    stem = ply_path.stem
    for xi, x0 in enumerate(x_starts):
        for yi, y0 in enumerate(y_starts):
            x1 = x0 + voxel_size
            y1 = y0 + voxel_size
            mask = (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
            if int(mask.sum()) < min_points:
                continue
            gt_local = None
            if gt_df is not None:
                gt_mask = (gt_df["x"] >= x0) & (gt_df["x"] < x1) & (gt_df["y"] >= y0) & (gt_df["y"] < y1)
                gt_local = gt_df[gt_mask].copy()
                if gt_local.empty:
                    continue
            out_name = f"{stem}_{xi:04d}_{yi:04d}"
            write_binary_ply(out_split_dir / "pcd" / f"{out_name}.ply", points[mask], colors[mask])
            if gt_local is not None:
                gt_out = out_split_dir / "gt" / f"{out_name}.txt"
                gt_out.parent.mkdir(parents=True, exist_ok=True)
                gt_local.to_csv(gt_out, sep=" ", header=False, index=False)
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice ZIO-style fused PCDs into overlapping 20m chunks.")
    parser.add_argument("--pcd-data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--voxel-size", type=float, default=20.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--min-points", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    for split in args.splits:
        in_pcd = args.pcd_data_root / split / "pcd"
        in_gt = args.pcd_data_root / split / "gt"
        out_split = args.out_dir / split
        split_written = 0
        for ply_path in sorted(in_pcd.glob("*.ply")):
            gt_path = in_gt / f"{ply_path.stem}.txt"
            written = process_file(ply_path, gt_path if in_gt.exists() else None, out_split, args.voxel_size, args.overlap, args.min_points)
            split_written += written
            print(f"{ply_path.name}: slices={written}", flush=True)
        total += split_written
        print(f"SPLIT {split}: slices={split_written}", flush=True)
    print(f"TOTAL slices={total}", flush=True)


if __name__ == "__main__":
    main()
