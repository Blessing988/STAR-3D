from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from .dataset import TrackBox

Point = Tuple[float, float]


def angle_wrap(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_distance(a: float, b: float) -> float:
    return abs(angle_wrap(a - b))


def circular_mean(angles: Iterable[float], weights: Iterable[float] | None = None) -> float:
    angle_list = list(angles)
    if weights is None:
        weight_list = [1.0 for _ in angle_list]
    else:
        weight_list = list(weights)
    sin_sum = 0.0
    cos_sum = 0.0
    for angle, weight in zip(angle_list, weight_list):
        sin_sum += math.sin(angle) * weight
        cos_sum += math.cos(angle) * weight
    if not angle_list or (abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12):
        return 0.0
    return math.atan2(sin_sum, cos_sum)


def distance_xy(a: TrackBox, b: TrackBox) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def rotated_rect_corners_xy(box: TrackBox) -> List[Point]:
    """Return BEV rectangle corners in counter-clockwise order."""
    hw = box.width * 0.5
    hl = box.length * 0.5
    local = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    return [
        (box.x + c * px - s * py, box.y + s * px + c * py)
        for px, py in local
    ]


def polygon_area(poly: Sequence[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _inside(point: Point, edge_start: Point, edge_end: Point) -> bool:
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end
    return _cross(bx - ax, by - ay, px - ax, py - ay) >= -1e-9


def _line_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> Point:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def convex_polygon_intersection(subject: Sequence[Point], clip: Sequence[Point]) -> List[Point]:
    """Sutherland-Hodgman clipping for convex polygons."""
    output = list(subject)
    for i, edge_start in enumerate(clip):
        edge_end = clip[(i + 1) % len(clip)]
        input_poly = output
        output = []
        if not input_poly:
            break
        prev = input_poly[-1]
        for curr in input_poly:
            curr_inside = _inside(curr, edge_start, edge_end)
            prev_inside = _inside(prev, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(_line_intersection(prev, curr, edge_start, edge_end))
            prev = curr
    return output


def bev_intersection_area(a: TrackBox, b: TrackBox) -> float:
    pa = rotated_rect_corners_xy(a)
    pb = rotated_rect_corners_xy(b)
    return polygon_area(convex_polygon_intersection(pa, pb))


def box3d_volume(box: TrackBox) -> float:
    return max(0.0, box.width) * max(0.0, box.length) * max(0.0, box.height)


def box3d_iou(a: TrackBox, b: TrackBox) -> float:
    if a.class_id != b.class_id:
        return 0.0
    inter_bev = bev_intersection_area(a, b)
    if inter_bev <= 0.0:
        return 0.0
    a_min_z = a.z - a.height * 0.5
    a_max_z = a.z + a.height * 0.5
    b_min_z = b.z - b.height * 0.5
    b_max_z = b.z + b.height * 0.5
    inter_h = max(0.0, min(a_max_z, b_max_z) - max(a_min_z, b_min_z))
    if inter_h <= 0.0:
        return 0.0
    inter = inter_bev * inter_h
    union = box3d_volume(a) + box3d_volume(b) - inter
    return inter / union if union > 0.0 else 0.0
