"""用公开开发轨迹检验换位记忆及同预算回看。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from offline_pilot import REVISION, ROOT
from place_diagnostic.exchange_detection import find_motion_segments, origin_at, score_segment, segment_evidence
from place_diagnostic.global_ring_tracking import global_associate
from place_diagnostic.permutation_memory import Exchange, read_with_budget, target_distribution
from place_diagnostic.ring_association import associate, detect_rings


BUDGETS = (32, 72)
PRIORITIES = ("target", "gain", "local", "residual", "recent")
PIXEL_TOLERANCE = 5.0


def read_prefix(path, cutoff):
    """解码到第一帧执行观测。后续影像不会进入候选算法。"""
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        for index in range(cutoff + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"无法读取 {path.name} 的第 {index} 帧")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def permutations_for(segment, rows, slots, anchor):
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


def exchange_record(segment, alternatives, extra_frames=(), residual=0.0):
    return Exchange(segment.start, segment.stop, tuple(alternatives), tuple(alternatives.values()),
                    tuple(extra_frames), residual)


def compose_alternatives(before, after):
    """局部重分段后合成各段的置换，保留最多四个不同对应结果。"""
    merged = {}
    for left, weight_left in before.items():
        for right, weight_right in after.items():
            route = tuple(right[index] for index in left)
            merged[route] = merged.get(route, 0.0) + weight_left * weight_right
    return dict(sorted(merged.items(), key=lambda item: -item[1])[:4])


def memory_size(records):
    """比较实际置换记录的紧凑 JSON 字节，不含原视频或软件对象开销。"""
    single = []
    multiple = []
    for segment, (options, _, evidence) in records.items():
        common = {"frames": [segment.start, segment.stop], "residual": round(evidence["unexplained_fraction"], 6)}
        single.append({**common, "permutations": [options[0]["perm"]], "weights": [1.0]})
        multiple.append({**common, "permutations": [option["perm"] for option in options],
                         "weights": [round(option["weight"], 6) for option in options]})
    return {"single_bytes": len(json.dumps(single, separators=(",", ":")).encode("utf-8")),
            "multiple_bytes": len(json.dumps(multiple, separators=(",", ":")).encode("utf-8")),
            "segments": len(records)}


def score_weights(weights, slots, target):
    weights = np.asarray(weights, dtype=float)
    ranked = np.argsort(-weights, kind="stable")
    candidates = [{"xy": slots[index].tolist(), "weight": round(float(weights[index]), 6),
                   "error_pixels": round(float(np.linalg.norm(slots[index] - target)), 3)}
                  for index in ranked if weights[index] > 1e-9]
    return {"candidates": candidates, "correct": bool(candidates and candidates[0]["error_pixels"] <= PIXEL_TOLERANCE)}


def score_point(point, target):
    if point is None:
        return {"candidates": [], "correct": False}
    error = float(np.linalg.norm(np.asarray(point) - target))
    return {"candidates": [{"xy": point, "error_pixels": round(error, 3)}], "correct": error <= PIXEL_TOLERANCE}


def run_case(video, row, stride):
    reference = row["reference"]
    cutoff = row["first_execution_frame"]
    if cutoff != reference["decision_frame"]:
        raise ValueError("回放与标注的决策帧不一致")
    frames = read_prefix(video, cutoff)
    image_cache = {}

    def get(index):
        if index not in image_cache:
            image_cache[index] = detect_rings(cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB))
        return image_cache[index]

    coarse = {index: get(index) for index in (*range(0, cutoff, stride), cutoff)}
    slots = coarse[0]
    if not len(slots):
        raise ValueError("第一帧没有可用于定位的静止圆环")
    sections = find_motion_segments(coarse, slots)
    sections_by_range = {(segment.start, segment.stop): segment for segment in sections}
    records = {}
    for section in sections:
        more = [index for index in range(section.start, section.stop + 1) if index not in coarse]
        alternatives = score_segment(section, coarse, slots)
        evidence = segment_evidence(section, coarse, slots, alternatives[0])
        records[section] = (alternatives, more, evidence)
    dense_cache = {}

    inferred = row["reading"]["color_and_time"]["selected"]
    versions = {"known_history_ring": (reference["historical_xy_frame"], reference["historical_xy"]),
                "inferred_history_ring": (inferred["after_frame"], inferred["ring_xy"]) if inferred else None}
    scores = {}
    for name, source_point in versions.items():
        if source_point is None:
            scores[name] = {"failure": "未找到满足指令的历史事件"}
            continue
        index, xy = source_point
        target = reference["current_xy"]
        indices = sorted({index, *coarse})
        baseline_frames = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in indices]
        source_index = indices.index(index)
        track = associate(baseline_frames, source_index, xy)
        joint_track = global_associate(baseline_frames, source_index, xy, max_missed=max(1, 120 // stride))
        result = {"source_frame": index, "source_xy": xy,
                  "static": score_point(track["static"][0]["xy"] if track["static"] else None, target),
                  "motion": score_point(track["motion"][0]["xy"] if track["motion"] else None, target),
                  "global_motion": score_point(joint_track["xy"], target)}
        distance = np.linalg.norm(slots - xy, axis=1)
        source_slot = int(distance.argmin())
        if distance[source_slot] > 12 and not any(section.start < index < section.stop for section in sections):
            scores[name] = {**result, "failure": "历史目标未落在可靠槽位内"}
            continue
        anchor = (index, xy, source_slot)
        future = [exchange_record(section, permutations_for(section, rows, slots, anchor), frames,
                                  evidence["unexplained_fraction"])
                  for section, (rows, frames, evidence) in records.items() if section.stop > index]

        def refine(exchange):
            segment = sections_by_range[(exchange.start, exchange.end)]
            if segment not in dense_cache:
                detections = {i: get(i) for i in range(exchange.start, exchange.end + 1)}
                subparts = find_motion_segments(detections, slots) or [segment]
                dense_cache[segment] = [(part, score_segment(part, detections, slots)) for part in subparts]
            alternatives = {tuple(range(len(slots))): 1.0}
            for part, candidates in dense_cache[segment]:
                if part.stop > index:
                    alternatives = compose_alternatives(alternatives, permutations_for(part, candidates, slots, anchor))
            return exchange_record(segment, alternatives)

        args = (len(slots), source_slot)
        result.update(source_slot=source_slot,
                      commit=score_weights(target_distribution(*args, future, commit=True), slots, target),
                      preserve=score_weights(target_distribution(*args, future), slots, target))
        for budget in BUDGETS:
            for priority in PRIORITIES:
                retrieval = read_with_budget(*args, future, budget, refine, priority)
                result[f"{priority}_{budget}"] = {**score_weights(retrieval["final_weights"], slots, target), **retrieval}
        dense = [refine(exchange) for exchange in future]
        result["dense"] = {**score_weights(target_distribution(*args, dense), slots, target),
                           "read_frames": sorted({index for exchange in future for index in exchange.extra_frames})}
        scores[name] = result
    return {"episode_index": row["episode_index"], "slots": slots.tolist(), "cutoff": cutoff,
            "coarse_frame_count": len(coarse), "decoded_frame_count": len(frames),
            "permutation_record_size": memory_size(records),
            "segmented_exchanges": [{"start": section.start, "end": section.stop, "alternatives": rows,
                                      "extra_frames": more, "evidence": evidence}
                                     for section, (rows, more, evidence) in records.items()], "scores": scores}


def summarize(cases):
    comparison = {}
    for origin in ("known_history_ring", "inferred_history_ring"):
        if not any(origin in case["scores"] for case in cases):
            continue
        comparison[origin] = {}
        for method in ("static", "motion", "global_motion", "commit", "preserve",
                       *(f"{p}_{b}" for b in BUDGETS for p in PRIORITIES), "dense"):
            results = [case["scores"][origin].get(method, {}) for case in cases]
            comparison[origin][method] = {"correct": sum(r.get("correct", False) for r in results), "total": len(cases),
                                          "extra_frames": sum(len(r.get("read_frames", ())) for r in results),
                                          "localized": sum(bool(r.get("candidates")) for r in results)}
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".local", help="公开数据和运行目录")
    parser.add_argument("--stride", type=int, default=4, help="候选写入的帧间隔")
    parser.add_argument("--original-manifest", type=Path, help="原版定向核验清单，按已知历史锚点评分")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("帧间隔必须为正数")
    if args.original_manifest:
        trace = json.loads(args.original_manifest.read_text(encoding="utf-8"))
        cases_to_run = []
        for reference in trace["episodes"]:
            name = Path(reference["local_file"])
            if name.name != str(name):
                raise ValueError("原版回放文件必须与清单同目录")
            cutoff = reference["first_execution_frame"]
            row = {"episode_index": reference["episode_index"], "first_execution_frame": cutoff,
                   "reference": {**reference, "decision_frame": cutoff},
                   "reading": {"color_and_time": {"selected": None}}}
            cases_to_run.append((args.original_manifest.parent / name, row))
        scope = "original_displaced"
    else:
        source_trace = args.work_dir / "runs" / "place-diagnostic" / "stride_1" / "trace.json"
        trace = json.loads(source_trace.read_text(encoding="utf-8"))
        if trace["revision"] != REVISION:
            raise ValueError("放置诊断与当前数据版本不一致")
        front = args.work_dir / "data" / "videos" / "observation.images.front" / "chunk-000"
        cases_to_run = [(front / f"file-{row['episode_index']:03d}.mp4", row) for row in trace["episodes"]]
        scope = "plus_train"

    cases = []
    for video, row in cases_to_run:
        started = perf_counter()
        case = run_case(video, row, args.stride)
        if args.original_manifest:
            case["scores"].pop("inferred_history_ring")
        case["seconds"] = round(perf_counter() - started, 3)
        cases.append(case)
    output = args.work_dir / "runs" / "exchange-memory" / scope / f"stride_{args.stride}"
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(cases)
    (output / "trace.json").write_text(json.dumps({"revision": trace["revision"], "scope": scope,
                                                  "stride": args.stride, "budgets": BUDGETS,
                                                  "summary": summary, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
