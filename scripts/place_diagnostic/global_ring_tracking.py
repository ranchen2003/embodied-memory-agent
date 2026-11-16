"""用一对一匹配为前视 RGB 中的所有目标环维护身份。"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ring_association import detect_rings


@dataclass
class _Ring:
    center: np.ndarray
    velocity: np.ndarray
    last_seen: int
    gaps: list[tuple[int, int]] = field(default_factory=list)
    cost: float = 0.0


def global_associate(
    frames: Sequence[np.ndarray],
    source_index: int,
    source_xy: Sequence[float],
    max_missed: int = 120,
) -> dict:
    """将历史证据点关联到当前帧。frames[0] 应能看到全部目标环。"""
    if not 0 <= source_index < len(frames) or max_missed < 1:
        raise ValueError("source_index 或 max_missed 无效")
    source_xy = np.asarray(source_xy, dtype=float)
    if source_xy.shape != (2,) or not np.isfinite(source_xy).all():
        raise ValueError("source_xy 需提供有限的 (x, y) 坐标")

    rings = [_Ring(center, np.zeros(2), 0) for center in detect_rings(frames[0])]
    selected_id = None
    source_center = None
    source_matched_frame = None
    initial_ids = list(range(len(rings)))
    out_of_gate_frames = 0
    contested_frames = 0
    for frame_index, frame in enumerate(frames):
        if frame_index:
            points = detect_rings(frame)
            # 只允许在有限遮挡窗口内恢复既有身份。
            active_ids = [i for i in initial_ids if frame_index - rings[i].last_seen <= max_missed + 1]
            if active_ids:
                assignments, contested = _match(rings, active_ids, points, frame_index)
                contested_frames += int(contested)
                out_of_gate_frames += int(len(assignments) < len(active_ids))
                for ring_id, point_index, cost in assignments:
                    ring = rings[ring_id]
                    elapsed = frame_index - ring.last_seen
                    if elapsed > 1:
                        ring.gaps.append((ring.last_seen + 1, frame_index - 1))
                    point = points[point_index]
                    displacement = (point - ring.center) / elapsed
                    ring.velocity = 0.6 * displacement + 0.4 * ring.velocity
                    ring.center = point
                    ring.last_seen = frame_index
                    ring.cost += cost

        if frame_index == source_index:
            # 历史标注可位于圆环的局部遮挡帧，优先使用此时的证据。
            source_candidates = [(i, np.linalg.norm(ring.center - source_xy)) for i, ring in enumerate(rings)
                                 if source_index - ring.last_seen <= 8]
            if source_candidates:
                ring_id, distance = min(source_candidates, key=lambda item: item[1])
                if distance <= 10:
                    selected_id = ring_id
                    source_center = rings[ring_id].center.tolist()
                    source_matched_frame = rings[ring_id].last_seen

    end = len(frames) - 1
    candidates = [
        {"track_id": i, "xy": ring.center.tolist(), "cost": ring.cost}
        for i, ring in enumerate(rings) if ring.last_seen == end
    ]
    result = {
        "source_index": source_index,
        "source_xy": source_xy.tolist(),
        "source_center": source_center,
        "source_matched_frame": source_matched_frame,
        "source_track_id": selected_id,
        "initial_ring_count": len(rings),
        "current_visible_tracks": candidates,
        "incomplete_assignment_frames": out_of_gate_frames,
        "contested_frames": contested_frames,
        "xy": None,
    }
    if selected_id is not None:
        ring = rings[selected_id]
        gaps = ring.gaps[:]
        if ring.last_seen < end:
            gaps.append((ring.last_seen + 1, end))
        result.update({
            "xy": ring.center.tolist() if ring.last_seen == end else None,
            "last_seen_frame": ring.last_seen,
            "last_seen_xy": ring.center.tolist(),
            "cost": ring.cost,
            "gap_count": len(gaps),
            "longest_gaps": sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:5],
        })
    return result


def _match(rings, active_ids, points, frame_index):
    elapsed = np.array([frame_index - rings[i].last_seen for i in active_ids])
    centers = np.array([rings[i].center for i in active_ids])
    velocity = np.array([rings[i].velocity for i in active_ids])
    expected = centers + velocity * np.minimum(elapsed[:, None], 6)
    distance = np.linalg.norm(expected[:, None, :] - points[None, :, :], axis=2)
    gate = np.minimum(9 + 2 * (elapsed - 1), 28)
    scale = np.minimum(3 + elapsed - 1, 9)
    # 给漏检保留独立空位。每个真实观测只能匹配给一个身份。
    cost = (distance / scale[:, None]) ** 2 + np.minimum(0.1 * (elapsed[:, None] - 1), 3)
    cost[distance > gate[:, None]] = np.inf
    unmatched = np.full((len(active_ids), len(active_ids)), np.inf)
    np.fill_diagonal(unmatched, 9.0)
    rows, columns = linear_sum_assignment(np.hstack((cost, unmatched)))
    assignments = [(active_ids[row], col, float(cost[row, col]))
                   for row, col in zip(rows, columns) if col < len(points)]
    contested = bool(len(points) and ((cost < 9).sum(axis=0) > 1).any())
    return assignments, contested
