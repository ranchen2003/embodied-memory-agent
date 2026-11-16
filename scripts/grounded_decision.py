"""从放置任务的可见历史生成带定位的操作子目标。"""

import json
from bisect import bisect_left

import cv2
import numpy as np

from place_diagnostic.exchange_detection import find_motion_segments, permutations_for, score_segment
from place_diagnostic.permutation_memory import exchange_record, target_distribution
from place_diagnostic.place_events import (
    candidate_frame_indices, cube_centers, detect_placement, empty_closure_anchor, goal_conditions,
    read_placement, read_relevant_recent, release_candidates,
)
from place_diagnostic.ring_association import associate, detect_rings


def choose_with_model(client, goal, anchor, events, frames):
    """让模型从同一份事件记忆选择记录，使用图中的像素位置仍由定位算法处理。"""
    user_input = {"goal": goal, "estimated_button_interval": anchor,
                  "event_candidates": [{"id": index, "release_frame": event.release_frame,
                                        "observed_color": event.color,
                                        "image_order": [event.before_frame, event.after_frame]}
                                       for index, event in enumerate(events)]}
    image_indices = [len(frames) - 1]
    for event in events:
        image_indices.extend([event.before_frame, event.after_frame])
    inputs = [cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB) for index in image_indices]
    system = ('Choose the recorded placement required by the goal. The first image is the current view; '
              'the remaining images are before/after pairs in candidate order. Return a JSON object only, '
              'for example {"event_id": 0}. Use {"event_id": null} if the evidence is insufficient.')
    result = client.ask(system, json.dumps(user_input), inputs)
    event_id = result.get("event_id")
    if event_id is None and "event_id" in result:
        return None
    if type(event_id) is not int or not 0 <= event_id < len(events):
        raise ValueError("模型返回了记忆候选范围以外的事件编号")
    return events[event_id]


def ring_localizers(frames, stride, *, sampled_indices=None):
    """同一张图像和相同采样间隔，分别按逐帧和换位片段追踪。"""
    cutoff = len(frames) - 1
    ids = [*range(0, cutoff, stride), cutoff] if sampled_indices is None else sampled_indices
    rgb = {index: cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB) for index in ids}
    rings = {index: detect_rings(frame) for index, frame in rgb.items()}
    slots = rings[0]
    sections = find_motion_segments(rings, slots) if len(slots) else []
    hypotheses = [(section, score_segment(section, rings, slots)) for section in sections]

    def locate(event):
        if event is None or not len(slots) or not len(rings[cutoff]):
            return {"framewise": None, "exchange": None}
        index, xy = event.after_frame, np.asarray(event.ring_xy)
        observation_ids = sorted(set(ids) | {index})
        if index not in rgb:
            rgb[index] = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
        observed = [rgb[key] for key in observation_ids]
        tracked = associate(observed, observation_ids.index(index), xy)["motion"]
        distances = np.linalg.norm(slots - xy, axis=1)
        source_slot = int(distances.argmin())
        destination = None
        if distances[source_slot] <= 12 or any(item.start < index < item.stop for item in sections):
            anchor = (index, xy, source_slot)
            exchanges = [exchange_record(section, permutations_for(section, options, slots, anchor))
                         for section, options in hypotheses if index < section.stop]
            weights = target_distribution(len(slots), source_slot, exchanges, commit=True)
            inferred = slots[int(weights.argmax())]
            current_distances = np.linalg.norm(rings[cutoff] - inferred, axis=1)
            if current_distances.min() <= 12:
                destination = rings[cutoff][current_distances.argmin()].tolist()
        return {"framewise": tracked[0]["xy"] if tracked else None, "exchange": destination}

    return locate, len(ids)


def nearby_evidence(indices, expected, *, lower, upper):
    """在已保留的帧中找最近证据，最大允许两帧差，并保持释放两侧互斥。"""
    retained = set()
    for frame in expected:
        at = bisect_left(indices, frame)
        nearby = [i for i in indices[max(0, at - 1):at + 1] if lower <= i < upper and abs(i - frame) <= 2]
        if nearby:
            retained.add(min(nearby, key=lambda i: (abs(i - frame), i)))
    return sorted(retained)


def grounded_place(frames, aperture, goal, fps, stride, model=None, *, model_reader="deepseek", sampled_indices=None):
    """只使用已看到的相机和夹爪观测，分别返回两个读法和两个定位法的结果。"""
    cutoff = len(frames) - 1
    if cutoff < 1 or len(aperture) != cutoff or stride < 1:
        raise ValueError("画面须读到决策帧，夹爪历史须在该帧之前结束")
    color, _, relation = goal_conditions(goal)
    releases = list(release_candidates(aperture))
    evidence = None
    if sampled_indices is not None:
        if (not len(sampled_indices) or sampled_indices[0] != 0 or sampled_indices[-1] != cutoff or
                any(a >= b for a, b in zip(sampled_indices, sampled_indices[1:]))):
            raise ValueError("保留帧须严格递增并覆盖首帧与决策帧")
    events = []
    for close, release in releases:
        if sampled_indices is not None:
            before, after = candidate_frame_indices(close, release, cutoff)
            evidence = (nearby_evidence(sampled_indices, before, lower=0, upper=release),
                        nearby_evidence(sampled_indices, after, lower=release + 6, upper=release + 19))
        event = detect_placement(lambda t: frames[t], close, release, cutoff, evidence_frames=evidence)
        if event is not None:
            events.append(event)
    anchor = empty_closure_anchor(aperture, releases, fps) if relation else None
    readers = {"latest_object": read_relevant_recent(events, goal),
               "object_and_time": read_placement(events, goal, anchor)}
    model_error = None
    if model is not None:
        readers[model_reader] = None
        if events:
            try:
                readers[model_reader] = choose_with_model(model, goal, anchor, events, frames)
            except (RuntimeError, ValueError) as error:
                model_error = {"type": type(error).__name__, "message": str(error)}
    locate, frame_count = ring_localizers(frames, stride, sampled_indices=sampled_indices)
    current_colors = cube_centers(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2HSV))
    source = current_colors[color].tolist() if color in current_colors else None
    results = {}
    locations = {}
    for reader, event in readers.items():
        key = event.after_frame if event else None
        if key not in locations:
            locations[key] = locate(event)
        for localizer, target in locations[key].items():
            results[f"{reader}+{localizer}"] = {
                "operation": "place", "object": f"{color} cube", "destination": "ring",
                "camera": "front", "frame": cutoff,
                "object_pixel_xy": source, "target_pixel_xy": target,
                "release_frame": event.release_frame if event else None,
                "evidence_frames": [event.before_frame, event.after_frame] if event else [],
            }
    return {"candidate_events": len(events), "tracking_frames": frame_count,
            "button_interval": anchor, "model_error": model_error, "decision": results}
