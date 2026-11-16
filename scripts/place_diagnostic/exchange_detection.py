"""用固定视角中的圆环槽位及局部交换轨迹关联历史目标。"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class MotionSegment:
    start: int
    stop: int


def find_motion_segments(detections, slots) -> list[MotionSegment]:
    """按绝对帧号分组。出现离开所有静止槽的圆环才产生运动候选。"""
    sampled = sorted(detections)
    # 稀疏输入的相邻可观测帧不能被当成两次独立运动。
    cadence = int(np.median(np.diff(sampled))) if len(sampled) > 1 else 1
    max_gap = max(8, cadence)
    indices = []
    for frame in sampled:
        centers = detections[frame]
        distance = np.linalg.norm(centers[:, None, :] - slots[None, :, :], axis=2)
        if (distance.min(axis=1) > 6).any():
            indices.append(frame)
    if not indices:
        return []
    blocks = np.split(indices, np.flatnonzero(np.diff(indices) > max_gap) + 1)
    blocks = [block for block in blocks if block[-1] - block[0] >= 6]
    segments = []
    end = max(detections)
    for i, block in enumerate(blocks):
        left_limit = (int(blocks[i - 1][-1]) + int(block[0])) // 2 + 1 if i else 0
        right_limit = (int(block[-1]) + int(blocks[i + 1][0])) // 2 if i + 1 < len(blocks) else end
        segments.append(MotionSegment(max(left_limit, int(block[0]) - 5), min(right_limit, int(block[-1]) + 5)))
    return segments


def _arc(a, b, phase, bulge):
    midpoint = (a + b) / 2
    axis = (a - b) / 2
    perpendicular = np.array([-axis[1], axis[0]])
    angle = np.pi * np.asarray(phase)[..., None]
    return midpoint + np.cos(angle) * axis + bulge * np.sin(angle) * perpendicular


def _embedding(observations):
    width = max(map(len, observations), default=0)
    points = np.full((len(observations), max(1, width), 2), np.inf)
    for index, row in enumerate(observations):
        points[index, :len(row)] = row
    return points


def _distance_cost(predicted, observations):
    distances = np.linalg.norm(predicted[:, :, None, :] - observations[:, None, :, :], axis=-1)
    # 缺失或偏移较大的检测按截断距离计入，不视作身份已确定。
    return np.minimum(distances.min(axis=2), 15) / 15


def score_segment(segment: MotionSegment, detections, slots, hypotheses: int = 4) -> list[dict]:
    """对同一片段给出不移动或若干组互斥的交换。补看时可独立重算。"""
    subset = [(frame, points) for frame, points in sorted(detections.items())
              if segment.start <= frame <= segment.stop]
    if not subset:
        return [{"perm": list(range(len(slots))), "cost": 0.0, "weight": 1.0, "pairs": []}]
    frames, observations = zip(*subset)
    observations = _embedding(observations)
    phase = (np.asarray(frames) - segment.start) / max(1, segment.stop - segment.start)
    static = _distance_cost(np.broadcast_to(slots, (len(frames), *slots.shape)), observations)
    coefficients = (-1.0, -0.65, -0.35, 0.35, 0.65, 1.0)
    pair_fits = []
    for i, j in combinations(range(len(slots)), 2):
        static_cost = static[:, [i, j]].sum(axis=1).mean()
        shapes = np.stack([np.stack([_arc(slots[i], slots[j], phase, h),
                                     _arc(slots[j], slots[i], phase, h)], axis=1) for h in coefficients])
        costs = [_distance_cost(shape, observations).sum(axis=1).mean() for shape in shapes]
        index = int(np.argmin(costs))
        # 换位需要在同一段内得到足够的运动证据。值接近时同时保留静止与换位。
        pair_fits.append((float(costs[index] - static_cost + 0.10), i, j, coefficients[index]))

    candidates = [(0.0, ())]
    for cost, i, j, bulge in pair_fits:
        candidates.append((cost, ((i, j, bulge),)))
    for (ci, i, j, hi), (ck, k, l, hk) in combinations(pair_fits, 2):
        if len({i, j, k, l}) == 4:
            candidates.append((ci + ck, ((i, j, hi), (k, l, hk))))
    candidates = sorted(candidates, key=lambda item: item[0])[:hypotheses]
    weights = np.exp(-(np.array([cost for cost, _ in candidates]) - candidates[0][0]) / 0.15)
    weights /= weights.sum()
    result = []
    for (cost, pairs), weight in zip(candidates, weights):
        permutation = list(range(len(slots)))
        for i, j, _ in pairs:
            permutation[i], permutation[j] = j, i
        result.append({"perm": permutation, "cost": cost, "weight": float(weight), "pairs": pairs})
    return result


def origin_at(segment, hypothesis, slots, frame, xy):
    """历史证据正处在换位中时，按该候选的轨迹寻找其出发槽位。"""
    geometry = slots.copy()
    phase = (frame - segment.start) / (segment.stop - segment.start)
    for i, j, bulge in hypothesis["pairs"]:
        geometry[i] = _arc(slots[i], slots[j], phase, bulge)
        geometry[j] = _arc(slots[j], slots[i], phase, bulge)
    return int(np.argmin(np.linalg.norm(geometry - xy, axis=1)))

def permutations_for(segment, rows, slots, anchor):
    """按历史观测所在时刻修正换位候选的出发槽位。"""
    index, xy, source_slot = anchor
    weights = {}
    for row in rows:
        permutation = list(row["perm"])
        if segment.start < index < segment.stop:
            origin = origin_at(segment, row, slots, index, xy)
            permutation[source_slot], permutation[origin] = permutation[origin], permutation[source_slot]
        key = tuple(permutation)
        weights[key] = weights.get(key, 0.0) + row["weight"]
    return weights


def segment_evidence(segment: MotionSegment, detections, slots, option) -> dict:
    """量化首选假设无法解释的离槽观测，独立于候选之间的相对差异。"""
    outside_count = 0
    unexplained = 0
    for frame, points in detections.items():
        if not segment.start <= frame <= segment.stop or not len(points):
            continue
        departure = np.linalg.norm(points[:, None, :] - slots[None, :, :], axis=2).min(axis=1) > 6
        moving = points[departure]
        if not len(moving):
            continue
        phase = (frame - segment.start) / max(1, segment.stop - segment.start)
        geometry = slots.copy()
        for i, j, bulge in option["pairs"]:
            geometry[i] = _arc(slots[i], slots[j], phase, bulge)
            geometry[j] = _arc(slots[j], slots[i], phase, bulge)
        distance = np.linalg.norm(moving[:, None, :] - geometry[None, :, :], axis=2)
        outside_count += len(moving)
        unexplained += int((distance.min(axis=1) > 8).sum())
    return {"moving_observations": outside_count, "unexplained": unexplained,
            "unexplained_fraction": unexplained / outside_count if outside_count else 0.0}
