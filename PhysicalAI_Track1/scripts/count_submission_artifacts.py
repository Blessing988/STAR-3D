#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def count_entries(path: Path) -> dict[str, int | bool]:
    counts: dict[str, int | bool] = {
        "exists": path.exists(),
        "dirs": 0,
        "files": 0,
        "txt": 0,
        "jpg": 0,
        "png": 0,
        "other": 0,
    }
    if not path.is_dir():
        return counts

    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            counts["dirs"] = int(counts["dirs"]) + 1
        elif entry.is_file(follow_symlinks=False):
            counts["files"] = int(counts["files"]) + 1
            ext = Path(entry.name).suffix.lower().lstrip(".")
            key = ext if ext in {"txt", "jpg", "png"} else "other"
            counts[key] = int(counts[key]) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    result: dict[str, object] = {
        "run_dir": str(run_dir),
        "directories": {
            rel: count_entries(run_dir / rel)
            for rel in ["predictions", "predictions/labels", "star"]
        },
        "files": {},
    }
    for name in [
        "detections.tsv",
        "lifted.tsv",
        "track1_test_submission.txt",
        "track1.txt",
        "track1.zip",
    ]:
        path = run_dir / name
        result["files"][name] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        }

    if args.manifest:
        manifest = Path(args.manifest)
        rows = max(0, sum(1 for _ in manifest.open("r", encoding="utf-8")) - 1)
        result["manifest_rows"] = rows
        prediction_files = result["directories"]["predictions"]["files"]  # type: ignore[index]
        result["prediction_files_remaining"] = rows - int(prediction_files)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
