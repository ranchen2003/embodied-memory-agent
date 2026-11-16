"""把历史放置识别、指令读取和当前落点对应拆开测量。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import requests

from offline_pilot import REVISION, ROOT, cache_file, select_frames
from place_diagnostic.global_ring_tracking import global_associate
from place_diagnostic.place_events import (
    candidate_frame_indices, detect_placement, empty_closure_anchor, goal_conditions, read_placement,
    read_relevant_recent, release_candidates,
    retain_near_events, retain_release_pairs,
)
from place_diagnostic.ring_association import associate


FRAME_BUDGETS = (8, 24)
PIXEL_TOLERANCE = 5.0


def read_frames(video, limit):
    """读到第一帧执行观测。后续实际执行画面不会进入预测。"""
    capture = cv2.VideoCapture(str(video))
    try:
        for index in range(limit + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"无法读取 {video.name} 的第 {index} 帧")
            yield frame
    finally:
        capture.release()


def support_windows(placement):
    start, end = placement["release_interval"]
    release = placement["release_proxy_frame"]
    # 前段允许保留夹取阶段，后段只统计释放后短窗口内的画面。
    return (max(0, start - 35), release - 7), (release + 4, min(end, release + 25))


def pair_covered(indices, placement):
    return all(any(start <= index < end for index in indices)
               for start, end in support_windows(placement))


def score_events(events, references):
    matched = set()
    incorrect = 0
    for event in events:
        candidate = next((row for row in references if row["color"] == event.color
                          and row["release_interval"][0] <= event.release_frame < row["release_interval"][1]
                          and row["placement_index"] not in matched), None)
        if candidate is None:
            incorrect += 1
        else:
            matched.add(candidate["placement_index"])
    return {"matched": len(matched), "false_detections": incorrect, "reference_total": len(references)}


def distances(candidates, target):
    return [None if item["xy"] is None else round(float(np.linalg.norm(np.asarray(item["xy"]) - target)), 2)
            for item in candidates]


def score_rings(associations, label):
    outputs = {}
    for method, rows in associations.items():
        errors = distances(rows, label)
        rank = next((i + 1 for i, error in enumerate(errors)
                     if error is not None and error <= PIXEL_TOLERANCE), None)
        outputs[method] = {"candidates": rows, "distance_px": errors, "correct_rank": rank}
    return outputs


def alternate_read(events, goal, anchor, reference):
    """同一视频换一个确有不同答案的指令，仅核验历史事件。"""
    targets = [row for row in reference["reference_placements"] if row["color"] == reference["task_relevant_color"]]
    if reference["task"] == "VideoPlaceButton":
        annotated_anchor = reference["press_intervals"][0]
        eligible = [row for row in targets if row["release_interval"][1] <= annotated_anchor[0]]
        if not eligible:
            return None
        desired = eligible[-1]
        changed = goal.replace("first placed after", "last placed before")
    else:
        ordinal = ("first", "second", "third", "fourth", "fifth")
        correct = next(row for row in targets if row["placement_index"] == reference["correct_placement_index"])
        rank = correct["color_ordinal"]
        if rank < 2 or rank > len(ordinal):
            return None
        desired = targets[rank - 2]
        changed = goal.replace(f"the {ordinal[rank - 1]} target", f"the {ordinal[rank - 2]} target")
    if changed == goal:
        raise ValueError("未生成预期的替换指令")
    chosen = read_placement(events, changed, anchor)
    baseline = read_relevant_recent(events, changed)
    start, stop = desired["release_interval"]
    return {"goal": changed, "expected_interval": [start, stop],
            "selected": asdict(chosen) if chosen else None,
            "correct": bool(chosen and start <= chosen.release_frame < stop),
            "relevant_recent_correct": bool(baseline and start <= baseline.release_frame < stop)}


def run_episode(session, work_dir, episode, reference, stride, task_goals):
    number = episode["episode_index"]
    data = cache_file(session, work_dir, f"data/chunk-000/file-{number:03d}.parquet")
    video = cache_file(session, work_dir, f"videos/observation.images.front/chunk-000/file-{number:03d}.mp4")
    table = pq.read_table(data, columns=["observation.state", "is_video_demo", "task_index"])
    demo = table["is_video_demo"].to_numpy()
    stop = int(np.count_nonzero(demo))
    if not 0 < stop < len(demo) or not demo[:stop].all() or stop != reference["decision_frame"]:
        raise ValueError(f"第 {number} 条的示范截点不一致")
    states = np.asarray(table["observation.state"].slice(0, stop).to_pylist())
    goal = task_goals[table["task_index"][0].as_py()]
    if goal != reference["official_query_text"]:
        raise ValueError(f"第 {number} 条的任务指令与标注不一致")
    closing_opening = list(release_candidates(states[:, 7]))

    # 一次解码，只保留放置前后的必需帧和固定间隔的目标跟踪帧。
    event_indices = {index for close, release in closing_opening
                     for portion in candidate_frame_indices(close, release, stop) for index in portion}
    history_index = reference["historical_xy_frame"]
    track_indices = [*range(0, stop, stride), stop]
    needed = event_indices | set(track_indices) | {history_index}
    rgb = {index: frame for index, frame in enumerate(read_frames(video, stop)) if index in needed}
    events = [event for close, release in closing_opening
              if (event := detect_placement(rgb.__getitem__, close, release, stop)) is not None]

    expected = next(item for item in reference["reference_placements"]
                    if item["placement_index"] == reference["correct_placement_index"])
    retention = {}
    for budget in FRAME_BUDGETS:
        kept = select_frames(states, budget, episode["control_freq"])
        kept["release_paired"] = retain_release_pairs(closing_opening, stop, budget)
        kept["paired"] = retain_near_events(events, goal, stop, budget)
        retention[str(budget)] = {
            method: {"frames": frames,
                     "official_target_pair_covered": pair_covered(frames, expected),
                     "all_release_pairs_covered": sum(pair_covered(frames, item) for item in reference["reference_placements"]),
                     "all_release_pairs": len(reference["reference_placements"])} for method, frames in kept.items()}

    needs_anchor = goal_conditions(goal)[2] is not None
    anchor = empty_closure_anchor(states[:, 7], closing_opening, episode["control_freq"]) if needs_anchor else None
    annotated_anchor = reference["press_intervals"][0] if needs_anchor else None
    selected = {"relevant_recent": read_relevant_recent(events, goal),
                "time_only": read_placement(events, goal, anchor, match_color=False),
                "color_and_time": read_placement(events, goal, anchor),
                "known_button_anchor": read_placement(events, goal, annotated_anchor) if needs_anchor else None}
    start, end = reference["correct_release_interval"]
    reading = {method: {"selected": asdict(event) if event else None,
                        "event_correct": bool(event and start <= event.release_frame < end)}
               for method, event in selected.items()}

    def locate(index, xy):
        indices = sorted({index, *track_indices})
        sequence = [cv2.cvtColor(rgb[i], cv2.COLOR_BGR2RGB) for i in indices]
        source_index = indices.index(index)
        associations = associate(sequence, source_index, xy)
        global_result = global_associate(sequence, source_index, xy, max_missed=max(1, 120 // stride))
        # 静态近邻只提交一个目标，其余环不计作已覆盖候选。
        predictions = {"static": associations["static"][:1], "motion": associations["motion"],
                       "beam": associations["beam"], "global_motion": [global_result]}
        return {"frames": indices, "results": score_rings(predictions, reference["current_xy"])}

    chosen = selected["color_and_time"]
    current = {"known_history_ring": locate(history_index, reference["historical_xy"]),
               "inferred_history_ring": locate(chosen.after_frame, chosen.ring_xy) if chosen else None}
    return {"episode_index": number, "task": episode["task"], "seed": episode["seed"], "goal": goal,
            "first_execution_frame": stop, "reference": reference,
            "button_anchor": {"interval": anchor, "reference_interval": annotated_anchor,
                              "source": "持续空夹爪闭合代理",
                              "reference_start_match": bool(anchor and annotated_anchor[0] <= anchor[0] < annotated_anchor[1])}
            if needs_anchor else None,
            "release_candidates": closing_opening, "observed_placements": [asdict(event) for event in events],
            "placement_detection": score_events(events, reference["reference_placements"]),
            "placement_rgb_frames_scored": len(event_indices), "decoded_frames": stop + 1,
            "retention": retention, "reading": reading, "alternate": alternate_read(events, goal, anchor, reference),
            "current_localization": current}


def summarize(rows):
    memory = {str(budget): {
        method: {"target_pairs": sum(row["retention"][str(budget)][method]["official_target_pair_covered"] for row in rows),
                 "all_target_pairs": len(rows),
                 "all_pairs_covered": sum(row["retention"][str(budget)][method]["all_release_pairs_covered"] for row in rows),
                 "all_pairs": sum(row["retention"][str(budget)][method]["all_release_pairs"] for row in rows)}
        for method in rows[0]["retention"][str(budget)]} for budget in FRAME_BUDGETS}
    reader = {method: sum(row["reading"][method]["event_correct"] for row in rows)
              for method in ("relevant_recent", "time_only", "color_and_time")}
    targets = {}
    for source in ("known_history_ring", "inferred_history_ring"):
        targets[source] = {}
        for method in rows[0]["current_localization"]["known_history_ring"]["results"]:
            ranks = [row["current_localization"][source]["results"][method]["correct_rank"]
                     if row["current_localization"][source] else None for row in rows]
            targets[source][method] = {"top1": ranks.count(1), "in_candidates": sum(x is not None for x in ranks)}
    paired = [row for row in rows if row["alternate"]]
    button_rows = [row for row in rows if row["button_anchor"]]
    return {"episodes": len(rows), "retention": memory, "event_correct": reader,
            "button_proxy": {"correct_start": sum(row["button_anchor"]["reference_start_match"] for row in button_rows),
                             "reference_total": len(button_rows),
                             "event_correct_with_known_anchor": sum(row["reading"]["known_button_anchor"]["event_correct"] for row in button_rows)},
            "placement_detection": {key: sum(row["placement_detection"][key] for row in rows)
                                    for key in ("matched", "false_detections", "reference_total")},
            "placement_rgb_frames_scored": sum(row["placement_rgb_frames_scored"] for row in rows),
            "paired_relevant_recent_correct": sum(row["reading"]["relevant_recent"]["event_correct"] and
                                                  row["alternate"]["relevant_recent_correct"] for row in paired),
            "paired_event_correct": sum(row["reading"]["color_and_time"]["event_correct"] and
                                        row["alternate"]["correct"] for row in paired), "paired_total": len(paired),
            "localization_within_5px": targets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".local", help="缓存和结果目录")
    parser.add_argument("--reference", type=Path, default=ROOT / "configs" / "reference_places.json", help="开发样例评分标注")
    parser.add_argument("--stride", type=int, default=1, help="目标跟踪的固定采样间隔")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride 必须为正数")
    reference_file = json.loads(args.reference.read_text(encoding="utf-8"))
    if reference_file["revision"] != REVISION:
        raise ValueError("标注与轨迹版本不一致")
    references = {item["episode_index"]: item for item in reference_file["episodes"]}
    with requests.Session() as session:
        manifest = json.loads(cache_file(session, args.work_dir, "meta/source_episodes.json").read_text(encoding="utf-8"))
        tasks = pq.read_table(cache_file(session, args.work_dir, "meta/tasks.parquet")).to_pylist()
        task_goals = {row["task_index"]: row["task"] for row in tasks}
        rows = [run_episode(session, args.work_dir, episode, references[episode["episode_index"]], args.stride, task_goals)
                for episode in manifest if episode["episode_index"] in references]
    if len(rows) != len(references):
        raise ValueError("部分标注轨迹未出现在数据中")
    output = args.work_dir / "runs" / "place-diagnostic" / f"stride_{args.stride}"
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    report = {"revision": REVISION, "split": "六条公开开发轨迹", "tracking_stride": args.stride,
              "score_pixel_tolerance": PIXEL_TOLERANCE, "summary": summary, "episodes": rows}
    (output / "trace.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
