from __future__ import annotations

import argparse
import os
from pathlib import Path


def link_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small symlink subset of sliced PCD/GT data.")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()

    src_split = args.src_root / args.split
    dst_split = args.dst_root / args.split
    count = 0
    for ply in sorted((src_split / "pcd").glob("*.ply")):
        gt = src_split / "gt" / f"{ply.stem}.txt"
        if not gt.exists():
            continue
        link_or_replace(ply, dst_split / "pcd" / ply.name)
        link_or_replace(gt, dst_split / "gt" / gt.name)
        count += 1
        if count >= args.limit:
            break
    print({"linked": count, "src": str(args.src_root), "dst": str(args.dst_root)})


if __name__ == "__main__":
    main()
