"""根据前视 RGB 图像，将示范中的圆环定位到当前帧。"""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


def detect_rings(image: np.ndarray) -> np.ndarray:
    """返回可见紫色目标环的中心 (x, y)，输入为 256×256 RGB。"""
    if image.shape != (256, 256, 3):
        raise ValueError("需要 256×256×3 RGB 图像")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (135, 34, 70), (168, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    return centers[(area >= 55) & (area <= 650) & (width >= 8) & (height >= 6)]


@dataclass(frozen=True)
class _Track:
    position: np.ndarray
    velocity: np.ndarray
    last_seen: int
    cost: float = 0.0
    missed: int = 0
    ambiguity: tuple[int, ...] = ()
    gaps: tuple[tuple[int, int], ...] = ()


def associate(
    frames: Sequence[np.ndarray],
    source_index: int,
    source_xy: Sequence[float],
    beam_width: int = 4,
) -> dict:
    """用同一段 RGB 比较静态近邻、单假设和保留多个假设。

    source_xy 是调用方指定的历史证据点。最后一帧是当前观察，成本越低越优先。
    """
    if not 0 <= source_index < len(frames) or beam_width < 1:
        raise ValueError("source_index 或 beam_width 无效")
    source_xy = np.asarray(source_xy, dtype=float)
    if source_xy.shape != (2,) or not np.isfinite(source_xy).all():
        raise ValueError("source_xy 需提供有限的 (x, y) 坐标")

    detections = [detect_rings(frame) for frame in frames[source_index:]]
    source, current = detections[0], detections[-1]
    origin = source_xy
    if len(source):
        distances = np.linalg.norm(source - source_xy, axis=1)
        if distances.min() <= 10:
            origin = source[distances.argmin()]

    # 静态对照保留历史坐标，直接在当前帧选最近的圆环。
    distances = np.linalg.norm(current - origin, axis=1)
    static = [
        {"xy": current[index].tolist(), "cost": float(distances[index])}
        for index in np.argsort(distances)
    ]
    initial = _Track(origin, np.zeros(2), last_seen=source_index)
    return {
        "source_index": source_index,
        "source_xy": source_xy.tolist(),
        "source_center": origin.tolist(),
        "frame_count": len(detections),
        "current_ring_count": len(current),
        "static": static,
        "motion": _follow(detections, initial, 1),
        "beam": _follow(detections, initial, beam_width),
    }


def _follow(detections: Sequence[np.ndarray], initial: _Track, width: int) -> list[dict]:
    tracks = [initial]
    final_index = initial.last_seen + len(detections) - 1
    for frame_index, points in enumerate(detections[1:], start=initial.last_seen + 1):
        candidates = []
        for track in tracks:
            elapsed = frame_index - track.last_seen
            gaps = track.gaps
            if elapsed > 1:
                gaps = (*gaps, (track.last_seen + 1, frame_index - 1))

            # 速度用近期证据估计。遮挡时间变长时放宽匹配门限，并停止长期外推。
            expected = track.position + track.velocity * min(elapsed, 6)
            residual = np.linalg.norm(points - expected, axis=1)
            gate = min(9 + 2 * (elapsed - 1), 28)
            indices = np.flatnonzero(residual <= gate)
            indices = indices[np.argsort(residual[indices])]
            if len(indices):
                indices = indices[residual[indices] <= residual[indices[0]] + 4]
            ambiguous = len(indices) > 1
            notes = track.ambiguity
            if ambiguous and (not notes or frame_index - notes[-1] > 3):
                notes = (*notes, frame_index)

            for index in indices[:width]:
                position = points[index]
                displacement = (position - track.position) / elapsed
                velocity = 0.6 * displacement + 0.4 * track.velocity
                scale = min(3 + (elapsed - 1), 9)
                candidates.append(_Track(position, velocity, frame_index,
                                         track.cost + (residual[index] / scale) ** 2,
                                         track.missed, notes, gaps))

            # 遮挡或接近交叉时保留未观察分支，稳定观测下继续跟踪。
            uncertain = ambiguous or bool(len(indices) and residual[indices[0]] > 4)
            if not len(indices) or (width > 1 and uncertain):
                candidates.append(_Track(track.position, track.velocity, track.last_seen,
                                         track.cost + 2.5, track.missed + 1, notes, track.gaps))

        tracks = []
        signatures = set()
        for track in sorted(candidates, key=lambda item: item.cost):
            # 同一观测只保留最低代价轨迹，把名额留给其他可能的圆环。
            signature = (track.last_seen, *np.round(track.position, 1))
            if signature in signatures:
                continue
            signatures.add(signature)
            tracks.append(track)
            if len(tracks) == width:
                break

    result = []
    unique_centers = set()
    for track in tracks:
        visible = track.last_seen == final_index
        point = tuple(track.position) if visible else None
        if point in unique_centers:
            continue
        unique_centers.add(point)
        gaps = track.gaps if visible else (*track.gaps, (track.last_seen + 1, final_index))
        result.append({
            "xy": list(point) if visible else None,
            "last_seen_xy": track.position.tolist(),
            "last_seen_frame": track.last_seen,
            "cost": float(track.cost),
            "missing_frames": track.missed,
            "gap_count": len(gaps),
            "longest_gaps": sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:5],
            "ambiguity_frames": track.ambiguity,
        })
    return result
