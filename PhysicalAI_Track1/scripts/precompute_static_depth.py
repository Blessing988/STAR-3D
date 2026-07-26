from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from physicalai_track1.dataset import year_dir


def compute_static_depth(
    depth_file: Path,
    out_file: Path,
    max_samples: int,
    sample_stride: int,
    method: str,
    overwrite: bool = False,
) -> bool:
    if out_file.exists() and not overwrite:
        return False
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(depth_file, "r") as handle:
        keys = sorted(handle.keys())
        sampled = keys[:: max(1, sample_stride)][: max(1, max_samples)]
        if not sampled:
            raise ValueError(f"{depth_file}: no frames")
        stack = np.stack([np.asarray(handle[key], dtype=np.uint16) for key in sampled], axis=0)
    if method == "mode":
        # Fast approximate mode for uint16 depth: exact mode is expensive at 1080p.
        # Quantize to 10 mm bins, mode bins, then use median values from selected bin.
        bins = stack // 10
        flat = bins.reshape(bins.shape[0], -1)
        out = np.empty(flat.shape[1], dtype=np.uint16)
        for idx in range(flat.shape[1]):
            vals, counts = np.unique(flat[:, idx], return_counts=True)
            out[idx] = vals[np.argmax(counts)] * 10
        background = out.reshape(stack.shape[1:]).astype(np.uint16)
    else:
        background = np.median(stack, axis=0).astype(np.uint16)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    np.save(tmp, background)
    tmp_npy = tmp if tmp.suffix == ".npy" else Path(str(tmp) + ".npy")
    if tmp_npy != tmp:
        tmp_npy.replace(out_file)
    else:
        tmp.replace(out_file)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--sample-stride", type=int, default=30)
    parser.add_argument("--method", choices=["median", "mode"], default="median")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = year_dir(args.data_root, args.year) / args.split
    scene_filter = set(args.scenes or [])
    written = 0
    skipped = 0
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir() or (scene_filter and scene_dir.name not in scene_filter):
            continue
        depth_dir = scene_dir / "depth_maps"
        for depth_file in sorted(depth_dir.glob("*.h5")):
            out_file = args.out_root / scene_dir.name / f"{depth_file.stem}.npy"
            did_write = compute_static_depth(
                depth_file,
                out_file,
                max_samples=args.max_samples,
                sample_stride=args.sample_stride,
                method=args.method,
                overwrite=args.overwrite,
            )
            if did_write:
                written += 1
                print(f"WROTE {out_file}", flush=True)
            else:
                skipped += 1
    print({"written": written, "skipped": skipped, "out_root": str(args.out_root)})


if __name__ == "__main__":
    main()
