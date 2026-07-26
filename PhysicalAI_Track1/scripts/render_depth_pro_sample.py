from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VIDEO = Path("/path/to/PhysicalAI-SmartSpaces/MTMC_Tracking_2026/train/Warehouse_000/videos/Camera_0000.mp4")
DEPTH = Path("/path/to/scratch/PhysicalAI_Track1/depth_estimates/depth_pro_2026_smoke_2f_1536_gpu0063/train/Warehouse_000/Camera_0000/000000.npz")
OUT = Path("/path/to/scratch/PhysicalAI_Track1/depth_estimates/depth_pro_sample_train_W000_C0000_f000000.png")


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {VIDEO}")

    data = np.load(DEPTH)
    depth = data["depth_m"].astype(np.float32)
    h, w = depth.shape
    rgb = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    valid = depth[depth > 0]
    lo, hi = np.percentile(valid, [2, 98])
    norm = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    canvas = np.concatenate([rgb, color], axis=1)
    cv2.imwrite(str(OUT), canvas)
    print(OUT)
    print({
        "depth_shape": depth.shape,
        "depth_min": float(np.nanmin(depth)),
        "depth_median": float(np.nanmedian(valid)),
        "depth_max": float(np.nanmax(depth)),
        "focal_px": float(data["focal_px"]),
    })


if __name__ == "__main__":
    main()
