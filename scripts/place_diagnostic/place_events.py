"""从公开相机观测和夹爪状态构建放置候选及前后证据。"""

import re
from dataclasses import dataclass

import cv2
import numpy as np


COLORS = {
    "red": ((0, 7, 180, 85), (173, 179, 120, 85)),
    "green": ((46, 85, 165, 45),),
    "blue": ((96, 130, 130, 45),),
}
ORDINALS = {word: index for index, word in enumerate(("first", "second", "third", "fourth", "fifth"), 1)}


@dataclass(frozen=True)
class Placement:
    """每条放置候选都可回到原始视频中的释放前后画面。"""

    release_frame: int
    before_frame: int
    after_frame: int
    color: str
    ring_xy: tuple[float, float]
    displacement: float


def release_candidates(aperture):
    """只用观测开度判断一次关闭到张开的转换。"""
    closed = None
    for index, width in enumerate(aperture):
        if closed is None and width < 0.025:
            closed = index
        if closed is not None and width >= 0.032:
            yield closed, index
            closed = None


def empty_closure_anchor(aperture, closing_opening, fps):
    """在这类示范中，以唯一的持续空夹爪闭合为按键代理。"""
    minimum_frames = max(1, round(0.25 * fps))
    candidates = []
    for start, stop in closing_opening:
        below = aperture[start:stop] < 0.004
        boundaries = np.flatnonzero(np.diff(np.r_[False, below, False])).reshape(-1, 2)
        if any(end - begin >= minimum_frames for begin, end in boundaries):
            candidates.append((start, stop))
    return candidates[0] if len(candidates) == 1 else None


def components(mask, minimum, maximum):
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [(int(stats[index, cv2.CC_STAT_AREA]), centers[index]) for index in range(1, count)
            if minimum <= stats[index, cv2.CC_STAT_AREA] <= maximum]


def cube_centers(hsv):
    centers = {}
    for color, ranges in COLORS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for hue_min, hue_max, saturation, brightness in ranges:
            mask |= cv2.inRange(hsv, (hue_min, saturation, brightness), (hue_max, 255, 255))
        blobs = components(mask, 21, 260)
        if blobs:
            centers[color] = max(blobs, key=lambda item: item[0])[1]
    return centers


def candidate_frame_indices(close, release, decision_frame):
    """给出一次开闭转换的观测范围，保证不跨决策截点。"""
    if release < 40 or release + 18 >= decision_frame:
        return (), ()
    # 夹爪遮住方块时，还可使用夹取前的可见位置判断物体是否已被移动。
    before_frames = sorted({max(0, close - 20), max(0, close - 8),
                            release - 40, release - 25, release - 12})
    after_frames = (release + 6, release + 12, release + 18)
    return before_frames, after_frames


def detect_placement(frame_at, close, release, decision_frame):
    """用方块位移和释放后圆环的可见证据确认候选。"""
    before_frames, after_frames = candidate_frame_indices(close, release, decision_frame)
    before = [cube_centers(cv2.cvtColor(frame_at(index), cv2.COLOR_BGR2HSV)) for index in before_frames]
    after = []
    for index in after_frames:
        hsv = cv2.cvtColor(frame_at(index), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (130, 52, 82), (171, 255, 255))
        rings = components(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), 20, 650)
        after.append((index, cube_centers(hsv), rings))

    candidates = []
    for index, cubes, rings in after:
        for color, point in cubes.items():
            movements = [(float(np.linalg.norm(point - past[color])), prior)
                         for prior, past in zip(before_frames, before) if color in past]
            if not movements or not rings:
                continue
            motion, previous = max(movements)
            distance, ring = min(((float(np.linalg.norm(point - center)), center) for _, center in rings),
                                 key=lambda item: item[0])
            if motion >= 8.0 and distance <= 13.0:
                result = Placement(release, previous, index, color, tuple(float(x) for x in ring), motion)
                candidates.append((motion / (1 + distance / 8), result))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def goal_conditions(goal):
    color_match = re.search(r"place the (red|green|blue) cube", goal.lower())
    ordinal_match = re.search(r"\b(first|second|third|fourth|fifth|last)\b", goal.lower())
    if not color_match or not ordinal_match:
        raise ValueError(f"当前诊断不支持这条指令：{goal}")
    relation = next((word for word in ("before", "after") if f" {word} the button" in goal.lower()), None)
    return color_match[1], ordinal_match[1], relation


def read_placement(events, goal, anchor_interval, match_color=True):
    """在给定记忆中按同一时间条件读取，可关掉颜色约束作对照。"""
    color, ordinal, relation = goal_conditions(goal)
    eligible = sorted((event for event in events if not match_color or color == event.color),
                      key=lambda event: event.release_frame)
    if relation:
        if anchor_interval is None:
            return None
        start, stop = anchor_interval
        eligible = [event for event in eligible if (event.release_frame < start if relation == "before"
                                                      else event.release_frame >= stop)]
    rank = len(eligible) if ordinal == "last" else ORDINALS[ordinal]
    return eligible[rank - 1] if 0 < rank <= len(eligible) else None


def read_relevant_recent(events, goal):
    """相关性对照按目标颜色筛选，只用最近记录打破并列。"""
    color, _, _ = goal_conditions(goal)
    return max((event for event in events if event.color == color),
               key=lambda event: event.release_frame, default=None)


def retain_near_events(events, goal, length, budget):
    """将每次释放的两侧画面成对保留，其余容量覆盖剩余时段。"""
    color, _, _ = goal_conditions(goal)
    ordered = sorted(events, key=lambda event: (event.color != color, event.release_frame))
    selected = set()
    for event in ordered:
        pair = {event.before_frame, event.after_frame}
        if len(selected | pair) <= budget:
            selected.update(pair)
    return fill_uniformly(selected, length, budget)


def retain_release_pairs(closing_opening, length, budget):
    """较强的等预算对照，围绕夹爪张开保存对称帧。"""
    releases = [release for _, release in closing_opening if 12 <= release < length - 12]
    count = min(len(releases), budget // 2)
    chosen = np.linspace(0, len(releases) - 1, count, dtype=int)
    selected = {int(releases[index] + offset) for index in chosen for offset in (-12, 12)}
    return fill_uniformly(selected, length, budget)


def fill_uniformly(selected, length, budget):
    available = np.array(sorted(set(range(length)) - selected))
    count = min(budget - len(selected), len(available))
    if count:
        selected.update(available[np.linspace(0, len(available) - 1, count, dtype=int)].tolist())
    return sorted(selected)
