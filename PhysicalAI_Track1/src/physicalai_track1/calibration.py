from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .dataset import load_calibration


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def _mat3_inverse(m: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float, float], ...]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("Singular 3x3 matrix")
    inv_det = 1.0 / det
    return (
        ((e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det),
        ((f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det),
        ((d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det),
    )


def _homogeneous_3x3(m: Sequence[Sequence[float]], x: float, y: float) -> Point2D:
    u = m[0][0] * x + m[0][1] * y + m[0][2]
    v = m[1][0] * x + m[1][1] * y + m[1][2]
    w = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(w) < 1e-12:
        raise ValueError("Homogeneous projection has near-zero scale")
    return u / w, v / w


def _project_3x4(m: Sequence[Sequence[float]], x: float, y: float, z: float) -> Optional[Point2D]:
    u = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3]
    v = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3]
    w = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3]
    if abs(w) < 1e-12:
        return None
    return u / w, v / w


def _rq_3x3(m: Sequence[Sequence[float]]) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Tuple[float, ...], ...]]:
    import numpy as np

    a = np.asarray(m, dtype=float)
    q, r = np.linalg.qr(np.flipud(a).T)
    r = np.flipud(r.T)
    q = q.T[:, ::-1]
    r = r[:, ::-1]
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    t = np.diag(signs)
    r = r @ t
    q = t @ q
    if np.linalg.det(q) < 0.0:
        r[:, 2] *= -1.0
        q[2, :] *= -1.0
    return tuple(tuple(float(v) for v in row) for row in r), tuple(tuple(float(v) for v in row) for row in q)


def _attrs(sensor: Mapping) -> Dict[str, str]:
    return {str(a.get("name")): str(a.get("value")) for a in sensor.get("attributes", [])}


def _float_attr(attrs: Mapping[str, str], name: str, default: float) -> float:
    value = attrs.get(name)
    if value is None:
        return default
    value = value.strip()
    if not value or value.lower() in {"none", "null", "nan"}:
        return default
    return float(value)


@dataclass(frozen=True)
class CameraCalibration:
    scene_name: str
    camera_id: str
    frame_width: int
    frame_height: int
    homography: Tuple[Tuple[float, float, float], ...]
    inv_homography: Tuple[Tuple[float, float, float], ...]
    camera_matrix: Tuple[Tuple[float, float, float, float], ...]
    intrinsic_matrix: Tuple[Tuple[float, float, float], ...]
    extrinsic_matrix: Tuple[Tuple[float, float, float, float], ...]
    direction_deg: float = 0.0

    def decompose_projection(self) -> Tuple[object, object, object]:
        import numpy as np

        p = np.asarray(self.camera_matrix, dtype=float)
        m = p[:, :3]
        k, r = _rq_3x3(m)
        k_np = np.asarray(k, dtype=float)
        r_np = np.asarray(r, dtype=float)
        if abs(k_np[2, 2]) > 1e-12:
            k_np = k_np / k_np[2, 2]
        t_np = np.linalg.solve(k_np, p[:, 3])
        return k_np, r_np, t_np

    def image_depth_to_world(self, u: float, v: float, depth_to_image_plane: float) -> Optional[Point3D]:
        """Backproject pixel plus camera-plane depth to world coordinates.

        The 2026 depth maps are stored as `distance_to_image_plane`, so the
        value is interpreted as camera-frame z after applying `depth_scale`.
        """
        import numpy as np

        if depth_to_image_plane <= 0.0:
            return None
        try:
            k = np.asarray(self.intrinsic_matrix, dtype=float)
            e = np.asarray(self.extrinsic_matrix, dtype=float)
            r = e[:, :3]
            t = e[:, 3]
            ray = np.linalg.solve(k, np.asarray([float(u), float(v), 1.0], dtype=float))
            camera_point = ray * float(depth_to_image_plane)
            world_point = np.linalg.inv(r) @ (camera_point - t)
        except Exception:
            return None
        if not np.all(np.isfinite(world_point)):
            return None
        return float(world_point[0]), float(world_point[1]), float(world_point[2])

    def image_to_ground(self, u: float, v: float) -> Point2D:
        return _homogeneous_3x3(self.inv_homography, u, v)

    def ground_to_image(self, x: float, y: float) -> Point2D:
        return _homogeneous_3x3(self.homography, x, y)

    def world_to_image(self, x: float, y: float, z: float) -> Optional[Point2D]:
        return _project_3x4(self.camera_matrix, x, y, z)


def load_scene_cameras(scene_dir: Path | str) -> Dict[str, CameraCalibration]:
    scene_path = Path(scene_dir)
    cal = load_calibration(scene_path)
    cameras: Dict[str, CameraCalibration] = {}
    for sensor in cal.get("sensors", []):
        if sensor.get("type") != "camera":
            continue
        attrs = _attrs(sensor)
        frame_width = int(_float_attr(attrs, "frameWidth", 1920.0))
        frame_height = int(_float_attr(attrs, "frameHeight", 1080.0))
        homography = tuple(tuple(float(v) for v in row) for row in sensor["homography"])
        camera_matrix = tuple(tuple(float(v) for v in row) for row in sensor["cameraMatrix"])
        intrinsic_matrix = tuple(tuple(float(v) for v in row) for row in sensor["intrinsicMatrix"])
        extrinsic_matrix = tuple(tuple(float(v) for v in row) for row in sensor["extrinsicMatrix"])
        cameras[str(sensor["id"])] = CameraCalibration(
            scene_name=scene_path.name,
            camera_id=str(sensor["id"]),
            frame_width=frame_width,
            frame_height=frame_height,
            homography=homography,
            inv_homography=_mat3_inverse(homography),
            camera_matrix=camera_matrix,
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            direction_deg=_float_attr(attrs, "direction", 0.0),
        )
    return cameras
