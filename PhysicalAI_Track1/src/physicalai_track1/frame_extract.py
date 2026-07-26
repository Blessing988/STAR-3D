from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def extract_frames_from_manifest(manifest_path: Path | str, overwrite: bool = False) -> dict:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for frame extraction") from exc

    manifest = Path(manifest_path)
    rows_by_video: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    with open(manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows_by_video[row["video_path"]].append((int(row["frame_id"]), Path(row["image_path"])))

    extracted = 0
    skipped = 0
    failed = 0
    for video_path, requests in sorted(rows_by_video.items()):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            failed += len(requests)
            continue
        for frame_id, out_path in sorted(requests):
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                failed += 1
                continue
            if not cv2.imwrite(str(out_path), frame):
                failed += 1
                continue
            extracted += 1
        cap.release()

    return {
        "manifest": str(manifest),
        "videos": len(rows_by_video),
        "extracted": extracted,
        "skipped": skipped,
        "failed": failed,
    }

