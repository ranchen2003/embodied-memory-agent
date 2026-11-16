"""在公开 VideoUnmaskSwap 训练回合上诊断遮挡后的目标身份关联。"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from place_diagnostic.exchange_detection import find_motion_segments, score_segment
from place_diagnostic.permutation_memory import Exchange, target_distribution
from place_diagnostic.place_events import COLORS, cube_centers
from place_diagnostic.ring_association import associate_detections

TASK = "VideoUnmaskSwap"
REVISION = "ea22d395232719c13df778a2c2219a1ebdfe367e"
EPISODES = (15, 16, 17)
STRIDE = 2
TOLERANCE = 5.0
METHODS = ("static", "single_motion", "multi_motion", "fixed_slot")


@dataclass(frozen=True)
class Binding:
    """记录方块最后可见的证据，以及它与此后一个容器的初始对应。"""

    cube_frame: int
    cube_xy: np.ndarray
    cover_frame: int
    cover_index: int


def first_requested_color(instruction):
    match = re.search(r"\bthe (red|green|blue) cube\b", instruction.casefold())
    if match is None:
        raise ValueError(f"无法从公开指令解析第一项颜色：{instruction}")
    return match[1]


def video_prefix(path, cutoff):
    """仅读取首次执行画面之前的示范及当前帧。"""
    capture = cv2.VideoCapture(str(path))
    try:
        for index in range(cutoff + 1):
            ok, frame = capture.read()
            if not ok or frame.shape != (256, 256, 3):
                raise ValueError(f"{path.name} 第 {index} 帧不可读或尺寸不符")
            if index % STRIDE == 0 or index == cutoff:
                yield index, frame
    finally:
        capture.release()


def cover_centers(hsv, frame, background):
    # 灰白容器进入画面时相对于初始桌面有明显变化，固定隔板不随之移动。
    changed = cv2.absdiff(frame, background).max(axis=2) >= 25
    mask = (changed & (hsv[:, :, 1] <= 48) & (hsv[:, :, 2] >= 154)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    return centers[(area >= 35) & (area <= 420)
                   & (width >= 8) & (width <= 28)
                   & (height >= 7) & (height <= 27)]


def color_evidence(path, cutoff, color):
    """只诊断冻结颜色检测器的面积限制，不将第二视角写入预测。"""
    largest_area = 0
    largest_frame = None
    detected, oversize = [], []
    for index, frame in video_prefix(path, cutoff):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hue_min, hue_max, saturation, brightness in COLORS[color]:
            mask |= cv2.inRange(hsv, (hue_min, saturation, brightness), (hue_max, 255, 255))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas = stats[1:, cv2.CC_STAT_AREA]
        area = int(areas.max(initial=0))
        if area > largest_area:
            largest_area, largest_frame = area, index
        if (area > 260):
            oversize.append(index)
        if ((areas >= 21) & (areas <= 260)).any():
            detected.append(index)
    return {"largest_color_region_pixels": largest_area, "largest_region_frame": largest_frame,
            "frames_over_area_limit": oversize, "frames_accepted_by_front_detector": detected}


def bind_cube_to_cover(colors, covers):
    """检测至少三帧的初始露出，继而要求空窗，并写入第一条空间对应。"""
    samples = sorted(colors)
    visible = [index for index, frame in enumerate(samples) if colors[frame] is not None]
    for run in np.split(visible, np.flatnonzero(np.diff(visible) > 1) + 1):
        if len(run) < 3:
            continue
        end = int(run[-1])
        # 连续四次采样未见颜色才确认该次露出结束。
        if end + 4 >= len(samples) or any(colors[frame] is not None for frame in samples[end + 1:end + 5]):
            continue
        cube_frame = samples[end]
        cube_xy = colors[cube_frame]
        for frame in samples[end + 1:end + 9]:
            points = covers[frame]
            if not len(points):
                continue
            distance = np.linalg.norm(points - cube_xy, axis=1)
            nearest = int(distance.argmin())
            if distance[nearest] <= 18:
                return Binding(cube_frame, cube_xy, frame, nearest)
        # 不使用后续孤立的色斑来替代第一次已经结束的露出。
        return None
    return None


def endpoint_support(slots, current):
    """用相同的 6 像素离槽条件检查静态槽位模型适用性。"""
    if not len(current):
        return 0.0
    distance = np.linalg.norm(slots[:, None] - current[None, :], axis=2)
    # 双向覆盖率避免大量消失的旧槽位或新增容器被隐藏。
    return float(min((distance.min(axis=0) <= 6).mean(),
                     (distance.min(axis=1) <= 6).mean()))


def fixed_slot_prediction(covers, source_index, final_support):
    """仅当首尾槽位基本一致时复用圆环实验中的分段和槽位置换。"""
    if final_support < 0.8:
        return None, []
    slots = next(iter(covers.values()))
    segments = find_motion_segments(covers, slots)
    movements = []
    for segment in segments:
        options = score_segment(segment, covers, slots)
        movements.append(Exchange(segment.start, segment.stop,
                                  tuple(tuple(option["perm"]) for option in options),
                                  tuple(option["weight"] for option in options)))
    probabilities = target_distribution(len(slots), source_index, movements, commit=True)
    return slots[int(probabilities.argmax())].tolist(), [[segment.start, segment.stop] for segment in segments]


def predict(episode, video):
    """输入为公开指令和前视录像，评分文件在此函数返回后才会读取。"""
    color = first_requested_color(episode["instruction"])
    cutoff = episode["first_execution_frame"]
    colors, covers = {}, {}
    for index, frame in video_prefix(video, cutoff):
        if index == 0:
            background = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colors[index] = cube_centers(hsv).get(color)
        covers[index] = cover_centers(hsv, frame, background)
    result = {
        "episode_index": episode["episode_index"], "difficulty": episode["difficulty"],
        "target_color": color, "cutoff": cutoff,
        "decoded_frames": cutoff + 1, "processed_frames": len(covers),
        "visible_cube_frames": [frame for frame in colors if colors[frame] is not None],
        "final_cover_centers": covers[cutoff].tolist(),
        "binding": None, "predictions": {name: None for name in METHODS},
    }
    binding = bind_cube_to_cover(colors, covers)
    if binding is None:
        result["failure"] = "未从可见颜色与后续容器形成初始对应"
        return result
    samples = {index: points for index, points in covers.items() if index >= binding.cover_frame}
    result["tracking_sample_frames"] = list(samples)
    slots = samples[binding.cover_frame]
    cover_xy = slots[binding.cover_index]
    result["binding"] = {"cube_frame": binding.cube_frame, "cube_xy": binding.cube_xy.tolist(),
                         "cover_frame": binding.cover_frame, "cover_xy": cover_xy.tolist()}
    matches = associate_detections(list(samples.values()), 0, cover_xy)
    support = endpoint_support(slots, covers[cutoff])
    fixed_slot, segments = fixed_slot_prediction(samples, binding.cover_index, support)
    result["end_slot_support"] = support
    result["fixed_slot_segments"] = segments
    result["predictions"] = {"static": matches["static"][0]["xy"] if matches["static"] else None,
                             "single_motion": matches["motion"][0]["xy"],
                             "multi_motion": matches["beam"][0]["xy"],
                             "fixed_slot": fixed_slot}
    result["motion_trace"] = {name: matches[name] for name in ("motion", "beam")}
    return result


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score(predictions, reference_path):
    references = load_json(reference_path)
    if references["revision"] != REVISION:
        raise ValueError("评分文件与公开轨迹版本不一致")
    by_episode = {row["episode_index"]: row["checkpoint_reference_labels"]
                  for row in references["episodes"]}
    for result in predictions:
        labels = by_episode[result["episode_index"]]
        label = next(item for item in labels if item["frame"] == result["cutoff"])
        goal = label["online_grounded_subtask_label"]
        match = re.search(r"at <(\d+),\s*(\d+)> that hides the (red|green|blue) cube", goal or "")
        if match is None or match[3] != result["target_color"]:
            raise ValueError(f"第 {result['episode_index']} 条坐标标签与公开任务不一致")
        target = np.array([int(match[2]), int(match[1])], dtype=float)
        result["evaluation_reference"] = {"label": goal, "xy": target.tolist()}
        current = np.asarray(result["final_cover_centers"]).reshape(-1, 2)
        error = float(np.linalg.norm(current - target, axis=1).min()) if len(current) else None
        result["best_possible_detection_error_pixels"] = error
        result["scores"] = {}
        for method, predicted in result["predictions"].items():
            error = float(np.linalg.norm(np.asarray(predicted) - target)) if predicted is not None else None
            result["scores"][method] = {"error_pixels": error, "hit": error is not None and error <= TOLERANCE}
    return {method: {"hits": sum(item["scores"][method]["hit"] for item in predictions),
                     "predictions": sum(item["predictions"][method] is not None for item in predictions),
                     "episodes": len(predictions)}
            for method in METHODS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True, help="数据清单 coverage.json")
    parser.add_argument("--video-dir", type=Path, required=True, help="已核验的前视 MP4 所在目录")
    parser.add_argument("--wrist-dir", type=Path, help="用于独立诊断颜色证据的腕部 MP4 目录")
    parser.add_argument("--references", type=Path, required=True, help="独立的评分标签")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON")
    args = parser.parse_args()
    coverage = load_json(args.coverage)
    if coverage["revision"] != REVISION:
        raise ValueError("训练清单版本与预先选定的数据版本不一致")
    episodes = sorted((row for row in coverage["episodes"] if row["task"] == TASK),
                      key=lambda item: item["episode_index"])
    if tuple(row["episode_index"] for row in episodes) != EPISODES:
        raise ValueError("必须包括全部三条 VideoUnmaskSwap 公开训练回合")
    predictions = [predict(row, args.video_dir / f"file-{row['episode_index']:03d}.mp4") for row in episodes]
    if args.wrist_dir is not None:
        for result in predictions:
            filename = f"file-{result['episode_index']:03d}.mp4"
            result["color_evidence_diagnostic"] = {
                camera: color_evidence(directory / filename, result["cutoff"], result["target_color"])
                for camera, directory in (("front", args.video_dir), ("wrist", args.wrist_dir))
            }
    summary = {"revision": REVISION, "task": TASK, "scope": "公开训练回合的离线首次决策定位",
               "observation": "公开任务指令及前视 RGB", "stride": STRIDE,
               "match_tolerance_pixels": TOLERANCE, "results": predictions,
               "summary": score(predictions, args.references)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
