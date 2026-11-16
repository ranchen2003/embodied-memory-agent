"""在公开示范上交叉验证事件读取和历史对象定位，并输出可接执行器的子目标。"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq

from grounded_decision import grounded_place
from offline_pilot import REVISION, ROOT
from run_exchange_memory import read_prefix
from vision_api import VisionAPI


TASKS = {"VideoPlaceButton", "VideoPlaceOrder"}


def run_episode(work, entry, stride, model):
    index = entry["episode_index"]
    cutoff = entry["first_execution_frame"]
    relative = f"chunk-000/file-{index:03d}"
    data = work / "data" / f"{relative}.parquet"
    video = work / "data" / "videos/observation.images.front" / f"{relative}.mp4"
    table = pq.read_table(data, columns=["observation.state", "is_video_demo"])
    demo = table["is_video_demo"].to_numpy()
    if not 0 < cutoff < len(demo) or not demo[:cutoff].all() or demo[cutoff]:
        raise ValueError(f"第 {index} 条的示范与决策截点未对齐")
    states = np.asarray(table["observation.state"].slice(0, cutoff).to_pylist())
    frames = read_prefix(video, cutoff)
    return grounded_place(frames, states[:, 7], entry["instruction"], entry["control_freq"], stride, model)


def score(predictions, reference):
    """仅用于离线评分，决策器不接收历史事件和当前圆环的标注。"""
    expected_start, expected_end = reference["correct_release_interval"]
    for result in predictions.values():
        release = result["release_frame"]
        xy = result["target_pixel_xy"]
        error = float(np.linalg.norm(np.asarray(xy) - reference["current_xy"])) if xy is not None else None
        result["event_correct"] = release is not None and expected_start <= release < expected_end
        result["pixel_error"] = round(error, 3) if error is not None else None
        result["target_correct"] = error is not None and error <= 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="实验数据根目录")
    parser.add_argument("--output", type=Path, required=True, help="新建的决策记录目录")
    parser.add_argument("--stride", type=int, default=2, help="共同使用的跟踪采样间隔")
    parser.add_argument("--deepseek", action="store_true", help="显式启用 DeepSeek 图文读取对照")
    parser.add_argument("--reference", type=Path, default=ROOT / "configs/reference_places.json", help="仅用于评分的标签")
    args = parser.parse_args()
    key = os.getenv("DEEPSEEK_API_KEY") if args.deepseek else None
    if args.deepseek and not key:
        parser.error("启用 DeepSeek 时需要本机环境变量 DEEPSEEK_API_KEY")
    if args.stride < 1 or args.output.exists():
        parser.error("已存在的输出目录须保留，帧间隔须为正数")
    manifest = json.loads((args.work_dir / "data/plus-agent-coverage/coverage.json").read_text(encoding="utf-8"))
    labels = json.loads(args.reference.read_text(encoding="utf-8"))
    if manifest["revision"] != REVISION or labels["revision"] != REVISION:
        parser.error("数据、标签和代码使用的公开版本不一致")
    episodes = [row for row in manifest["episodes"] if row["task"] in TASKS]
    references = {row["episode_index"]: row for row in labels["episodes"]}
    if {row["episode_index"] for row in episodes} != references.keys() or len(episodes) != 6:
        parser.error("需要两个任务的全部 6 条公开训练回合及评分标签")
    args.output.mkdir(parents=True)
    client = VisionAPI("https://api.deepseek.com/chat/completions", "deepseek-flash", key,
                       max_calls=len(episodes), max_tokens=128, thinking=False) if args.deepseek else None
    results = []
    totals = defaultdict(lambda: {"episodes": 0, "event_correct": 0, "target_correct": 0, "both_correct": 0})
    try:
        for row in episodes:
            started = perf_counter()
            calls_before = client.calls if client is not None else 0
            result = run_episode(args.work_dir, row, args.stride, client)
            score(result["decision"], references[row["episode_index"]])
            result.update(episode_index=row["episode_index"], task=row["task"],
                          seed=row["seed"], elapsed_seconds=round(perf_counter() - started, 3))
            if client is not None and client.calls > calls_before:
                result["api"] = {"request": client.last_call.copy(), "usage": client.last_usage.copy()}
            for name, item in result["decision"].items():
                totals[name]["episodes"] += 1
                totals[name]["event_correct"] += int(item["event_correct"])
                totals[name]["target_correct"] += int(item["target_correct"])
                totals[name]["both_correct"] += int(item["event_correct"] and item["target_correct"])
            results.append(result)
            (args.output / f"episode_{row['episode_index']:03d}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        if client is not None:
            client.close()
    summary = {"revision": REVISION, "scope": "公开训练回合的固定轨迹诊断", "stride": args.stride,
               "episodes": [item["episode_index"] for item in results], "target_tolerance_pixels": 5,
               "deepseek": "deepseek-flash" if args.deepseek else None,
               "results": totals}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
