"""表示换位对应关系，并按其对最终目标的影响分配回看预算。"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Exchange:
    """一次静止槽位之间的换位，候选排列以旧槽位为下标。"""

    start: int
    end: int
    permutations: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...]
    extra_frames: tuple[int, ...] = ()
    residual: float = 0.0

    def __post_init__(self):
        if self.start >= self.end or not self.permutations or len(self.permutations) != len(self.weights):
            raise ValueError("换位片段或排列分数无效")
        if len(set(self.permutations)) != len(self.permutations):
            raise ValueError("同一片段的候选排列不能重复")
        slots = tuple(range(len(self.permutations[0])))
        if not slots or any(tuple(sorted(row)) != slots for row in self.permutations):
            raise ValueError("每个候选必须是一一对应的槽位置换")
        if not np.isfinite(self.weights).all() or min(self.weights) < 0 or sum(self.weights) <= 0:
            raise ValueError("候选权重必须有限、非负且总和为正")
        if not 0 <= self.residual <= 1:
            raise ValueError("无法解释的观测比例必须在 0 和 1 之间")
        if tuple(sorted(set(self.extra_frames))) != self.extra_frames or any(
            not self.start <= frame <= self.end for frame in self.extra_frames
        ):
            raise ValueError("待回看的帧号必须有序、唯一且在片段内")

    def branches(self):
        """返回候选排列和归一化权重。"""
        total = sum(self.weights)
        return zip(self.permutations, (weight / total for weight in self.weights))

    def transition(self):
        """将所有互斥排列转换成条件转移矩阵。"""
        slots = len(self.permutations[0])
        matrix = np.zeros((slots, slots))
        for permutation, weight in self.branches():
            matrix[np.arange(slots), permutation] += weight
        return matrix


def target_distribution(slot_count: int, source_slot: int, exchanges: Sequence[Exchange], commit=False):
    """沿时间传播指定圆环所在槽位的权重。"""
    state = np.zeros(slot_count)
    if not 0 <= source_slot < slot_count:
        raise ValueError("历史位置不在槽位集合中")
    state[source_slot] = 1
    for exchange in exchanges:
        if len(exchange.permutations[0]) != slot_count:
            raise ValueError("换位记录的槽位数不一致")
        if commit:
            permutation = max(exchange.branches(), key=lambda item: item[1])[0]
            state = apply_permutation(state, permutation)
        else:
            state = state @ exchange.transition()
    return state


def apply_permutation(prior, permutation):
    posterior = np.empty_like(prior)
    posterior[list(permutation)] = prior
    return posterior


def review_values(slot_count, source_slot, exchanges):
    """估计澄清各次换位能为最终目标减少多少误选风险。"""
    front = [target_distribution(slot_count, source_slot, ())]
    for exchange in exchanges:
        front.append(front[-1] @ exchange.transition())
    tail = [np.eye(slot_count)]
    for exchange in reversed(exchanges):
        tail.append(exchange.transition() @ tail[-1])
    tail.reverse()

    baseline = float(front[-1].max())
    values = []
    for index, exchange in enumerate(exchanges):
        after = tail[index + 1]
        outcomes = [(apply_permutation(front[index], permutation) @ after, weight)
                    for permutation, weight in exchange.branches()]
        expected = sum(weight * float(posterior.max()) for posterior, weight in outcomes)
        local_uncertainty = 1 - max(weight for _, weight in exchange.branches())
        # 相对权重可能在整体拟合不充分时仍偏向某一解。用落点分布分歧补足该信号。
        disagreement = max(float(np.abs(left - right).sum() / 2) for left, _ in outcomes for right, _ in outcomes)
        values.append((max(0., expected - baseline), local_uncertainty, exchange.residual * disagreement))
    return values


def read_with_budget(slot_count, source_slot, exchanges, frame_budget, refine: Callable[[Exchange], Exchange],
                     priority="target"):
    """每次回看后更新共同的记忆，所有策略执行相同的帧数限制。"""
    if frame_budget < 0 or priority not in {"target", "gain", "local", "residual", "recent"}:
        raise ValueError("回看预算或优先规则无效")
    current = list(exchanges)
    initial = target_distribution(slot_count, source_slot, current)
    reviewed = []
    read_frames = set()
    remaining = set(range(len(current)))
    while remaining:
        values = review_values(slot_count, source_slot, current)
        options = []
        for index in sorted(remaining):
            exchange = current[index]
            frames = set(exchange.extra_frames) - read_frames
            if not frames or len(read_frames) + len(frames) > frame_budget:
                continue
            gain, uncertainty, structural = values[index]
            score = {"target": (gain + structural) / len(frames), "gain": gain / len(frames),
                     "local": uncertainty / len(frames), "residual": exchange.residual / len(frames),
                     "recent": exchange.end}[priority]
            if score > 0:
                options.append((score, exchange.end, index, frames))
        if not options:
            break
        _, _, index, frames = max(options)
        previous = current[index]
        updated = refine(previous)
        if (previous.start, previous.end) != (updated.start, updated.end):
            raise ValueError("回看不得移动原有片段的时间边界")
        current[index] = updated
        reviewed.append({"start": previous.start, "end": previous.end, "frames": sorted(frames),
                         "estimated_gain": values[index][0], "residual_times_disagreement": values[index][2]})
        read_frames.update(frames)
        remaining.remove(index)
    final = target_distribution(slot_count, source_slot, current)
    return {"initial_weights": initial.tolist(), "final_weights": final.tolist(),
            "reviewed": reviewed, "read_frames": sorted(read_frames)}
