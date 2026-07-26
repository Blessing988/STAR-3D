from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def export_one(npz_path: Path, in_root: Path, out_root: Path, make_color: bool) -> tuple[Path, Path | None]:
    rel = npz_path.relative_to(in_root).with_suffix(".png")
    metric_path = out_root / "metric_uint16_mm" / rel
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    depth_m = data["depth_m"].astype(np.float32)
    depth_mm = np.clip(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0, 0, 65535)
    cv2.imwrite(str(metric_path), depth_mm.astype(np.uint16))

    color_path = None
    if make_color:
        color_path = out_root / "preview_turbo" / rel
        color_path.parent.mkdir(parents=True, exist_ok=True)
        valid = depth_m[depth_m > 0]
        if valid.size:
            lo, hi = np.percentile(valid, [2, 98])
            norm = np.clip((depth_m - lo) / (hi - lo + 1e-6), 0, 1)
        else:
            norm = np.zeros_like(depth_m, dtype=np.float32)
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(color_path), color)

    return metric_path, color_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Depth Pro npz depth maps to PNG images.")
    parser.add_argument("--in-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--make-color", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    count = 0
    for split in args.splits:
        split_root = args.in_root / split
        if not split_root.exists():
            print(f"skip_missing_split {split_root}", flush=True)
            continue
        for npz_path in sorted(split_root.rglob("*.npz")):
            export_one(npz_path, args.in_root, args.out_root, args.make_color)
            count += 1
            if count % 500 == 0:
                print(f"exported {count}", flush=True)
            if args.limit is not None and count >= args.limit:
                print(f"done {count}", flush=True)
                return
    print(f"done {count}", flush=True)


if __name__ == "__main__":
    main()
