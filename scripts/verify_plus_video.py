"""核验 RoboMME-Plus 全部公开训练轨迹的前视视频与可复用决策检查点。"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from time import sleep

import cv2
import numpy as np
import pyarrow.parquet as pq
import requests


REPOSITORY = "Yinpei/robomme_plus_data_lerobot"
REVISION = "ea22d395232719c13df778a2c2219a1ebdfe367e"
SOURCE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
VIDEO_DIRECTORY = Path("videos/observation.images.front/chunk-000")
EPISODE_COUNT = 50
FRAME_COUNT = 56_462
TIME_TOLERANCE_SECONDS = 0.0001

# 每类任务按同一规则选取第一个匹配决策点。规则仅用于离线挑选评价位置。
CHECKPOINT_RULES = {
    "BinFill": (r"^press the button", "按颜色核对已累计入筐的数量"),
    "PickXtimes": (r"^press the button", "核对已经完成的抓放次数"),
    "SwingXtimes": (r"^put (?:down the cube|the \w+ cube on the table)", "核对摆动方向与完成次数"),
    "StopCube": (r"^press the button", "核对物体经过目标的累计次数"),
    "VideoUnmask": (r"^pick up .*container.* hid", "示范中方块颜色与遮盖容器的对应"),
    "VideoUnmaskSwap": (r"^pick up .*container.* hid", "方块与容器关联及换位后的身份"),
    "ButtonUnmask": (r"^pick up .*container.* hid", "按键后短暂可见的方块与容器关联"),
    "ButtonUnmaskSwap": (r"^pick up .*container.* hid", "方块与容器关联及换位后的身份"),
    "PickHighlight": (r"^pick up the first highlighted cube", "先前高亮出现时的对象身份"),
    "VideoRepick": (r"^pick up the correct object", "示范中被抓取的对象及其当前身份"),
    "VideoPlaceButton": (r"^place the cube onto the correct target", "按键时间、放置对象和历史圆环身份"),
    "VideoPlaceOrder": (r"^place the cube onto the correct target", "指定序次的放置事件及历史圆环身份"),
    "MoveCube": (r"^move the cube to the target using the demonstrated method", "示范动作的操作方式"),
    "InsertPeg": (r"^pick up the peg by grasping", "示范中的抓取端与后续插入侧"),
    "PatternLock": (r"^move to grid", "示范绘制轨迹的起始位置"),
    "RouteStick": (r"by circling around the board", "示范路线首段的方向和绕行方式"),
}


@dataclass(frozen=True)
class ReleaseFile:
    path: str
    bytes: int
    sha256: str

    def verify(self, content: bytes) -> bool:
        return len(content) == self.bytes and hashlib.sha256(content).hexdigest() == self.sha256


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path: Path, content):
    data = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_bytes(path, data.encode("utf-8"))


def video_path(index: int) -> str:
    return (VIDEO_DIRECTORY / f"file-{index:03d}.mp4").as_posix()


def cache_video(item: ReleaseFile, work_dir: Path, reuse_dir: Path | None) -> str:
    destination = work_dir / item.path
    if destination.is_file() and item.verify(destination.read_bytes()):
        return "已校验缓存"
    if reuse_dir is not None:
        existing = reuse_dir / item.path
        if existing.is_file():
            content = existing.read_bytes()
            if item.verify(content):
                write_bytes(destination, content)
                return "复用现有视频"

    for attempt in range(3):
        try:
            response = requests.get(
                f"{SOURCE}/{item.path}", params={"download": 1}, timeout=(10, 35)
            )
            response.raise_for_status()
            content = response.content
            if not item.verify(content):
                raise ValueError(f"发行文件大小或 SHA-256 校验不符：{item.path}")
        except (requests.RequestException, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f"无法下载并校验：{item.path}（{type(error).__name__}）") from error
            sleep(attempt + 1)
        else:
            write_bytes(destination, content)
            return "已下载并校验"


def load_release(data_dir: Path) -> tuple[dict, Path, dict[str, ReleaseFile]]:
    coverage = load_json(data_dir / "coverage.json")
    if (coverage["source"], coverage["revision"]) != (REPOSITORY, REVISION):
        raise ValueError("覆盖清单的来源或固定版本不符")
    if sorted(row["episode_index"] for row in coverage["episodes"]) != list(range(EPISODE_COUNT)):
        raise ValueError("覆盖清单应完整保留 0 到 49 号公开轨迹")
    release_dir = data_dir / "evaluation_only" / "pinned_release" / REVISION
    manifest = load_json(release_dir / "manifest.json")
    if (manifest["episodes"], manifest["frames"]) != (EPISODE_COUNT, FRAME_COUNT):
        raise ValueError("官方清单的轨迹数或总帧数不符")
    files = {row["path"]: ReleaseFile(**row) for row in manifest["files"]}
    return coverage, release_dir, files


def stage_videos(files, work_dir, reuse_dir, workers):
    counts = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(cache_video, files[video_path(index)], work_dir, reuse_dir): index
            for index in range(EPISODE_COUNT)
        }
        for count, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            counts[future.result()] += 1
            if count % 10 == 0:
                print(f"已校验 {count}/{EPISODE_COUNT} 个前视视频，最新轨迹为 {index:03d}", flush=True)
    return dict(counts)


def checked_parquet(release_dir, files, path, columns):
    relative = path.as_posix()
    local = release_dir / relative
    if not local.is_file() or not files[relative].verify(local.read_bytes()):
        raise ValueError(f"官方 Parquet 未通过校验：{relative}")
    return pq.read_table(local, columns=columns).to_pydict()


def reference_subtasks(release_dir, files):
    catalogue = checked_parquet(
        release_dir, files, Path("meta/subtasks.parquet"), ["subtask_index", "subtask"]
    )
    return dict(zip(catalogue["subtask_index"], catalogue["subtask"], strict=True))


def verify_parquet(episode, release_dir, files):
    columns = [
        "episode_index", "frame_index", "native_frame_index", "timestamp", "is_video_demo",
        "next.done", "subtask_online_index", "subtask_grounded_online_index",
    ]
    observed = checked_parquet(release_dir, files, Path(episode["source_parquet"]), columns)
    count = episode["recorded_frames"]
    start = episode["first_execution_frame"]
    if observed["frame_index"] != list(range(count)) or set(observed["episode_index"]) != {episode["episode_index"]}:
        raise ValueError("官方 Parquet 的回合编号或帧数与清单不符")
    if observed["is_video_demo"] != [True] * start + [False] * (count - start):
        raise ValueError("示范边界与覆盖清单不符")
    terminal = next((index for index, done in enumerate(observed["next.done"]) if done), None)
    if terminal != episode["first_terminal_frame"]:
        raise ValueError("终止位置与覆盖清单不符")
    return observed


def evaluation_candidate(episode, table, labels):
    rule, evidence = CHECKPOINT_RULES[episode["task"]]
    start = episode["first_execution_frame"]
    terminal = episode["first_terminal_frame"]
    stop = terminal if terminal is not None else episode["recorded_frames"] - 1
    actions = table["subtask_online_index"]
    frames = (frame for frame in range(start, stop) if frame == start or actions[frame] != actions[frame - 1])
    frame = next((frame for frame in frames if re.search(rule, labels.get(actions[frame]) or "", re.I)), None)
    row = {"episode_index": episode["episode_index"], "task": episode["task"],
           "history_evidence_to_review": evidence, "visual_review": "待人工核查"}
    if frame is None:
        return {**row, "status": "没有匹配的非终止在线子任务"}
    action = labels[actions[frame]]
    grounded_action = labels.get(table["subtask_grounded_online_index"][frame])
    return {
        **row,
        "status": "已定位官方专家子任务起始帧",
        "official_online_subtask": action,
        "official_online_grounded_subtask": grounded_action,
        "matches_online_annotation": True,
        "has_2d_pixel_reference": bool(re.search(r"<\s*\d+\s*,\s*\d+\s*>", grounded_action or "")),
        "decision_frame": frame,
        "decision_native_frame": table["native_frame_index"][frame],
        "decision_timestamp_seconds": round(table["timestamp"][frame], 6),
        "history_end_exclusive": frame,
    }


def verify_timeline(video, episode, table, selected):
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    first_execution = episode["first_execution_frame"]
    expected_count = episode["recorded_frames"]
    fps = episode["control_freq"]
    markers = {first_execution}
    if first_execution > 0:
        markers.add(first_execution - 1)
    if selected is not None:
        markers.add(selected)
    observed_markers = {}
    count, max_error = 0, 0.0
    try:
        if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), fps, atol=1e-6, rtol=0):
            raise ValueError("OpenCV 无法打开视频或视频帧率与官方观测不符")
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_count:
            raise ValueError("视频流帧数与 Parquet 记录不一致")
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if count >= expected_count or bgr.shape != (256, 256, 3):
                raise ValueError("视频实际解码超过 Parquet 帧数或画面尺寸不符")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            error = abs(timestamp - table["timestamp"][count])
            if not np.isfinite(error) or error > TIME_TOLERANCE_SECONDS:
                raise ValueError(f"视频时间戳与第 {count} 行 Parquet 不一致")
            max_error = max(max_error, error)
            if count in markers:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                observed_markers[count] = {
                    "frame": count, "native_frame": table["native_frame_index"][count],
                    "timestamp_seconds": round(timestamp, 6),
                    "phase": "示范" if count < first_execution else "执行",
                    "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                }
            count += 1
    finally:
        cap.release()
    if count != expected_count or observed_markers.keys() != markers:
        raise ValueError("未解码全部帧或关键边界图片缺失")
    return count, max_error, observed_markers


def analyse(coverage, release_dir, files, work_dir):
    source_labels = reference_subtasks(release_dir, files)
    video_rows, candidate_rows = [], []
    for episode in coverage["episodes"]:
        index = episode["episode_index"]
        path = video_path(index)
        try:
            table = verify_parquet(episode, release_dir, files)
            candidate = evaluation_candidate(episode, table, source_labels)
            selected = candidate.get("decision_frame")
            count, max_error, markers = verify_timeline(work_dir / path, episode, table, selected)
        except ValueError as error:
            raise ValueError(f"回合 {index:03d} 核验失败：{error}") from error
        first_execution = episode["first_execution_frame"]
        video_rows.append({
            "episode_index": index, "task": episode["task"], "suite": episode["suite"],
            "seed": episode["seed"], "difficulty": episode["difficulty"],
            "instruction": episode["instruction"],
            "front_video": path, "release_sha256": files[path].sha256, "bytes": files[path].bytes,
            "decoded_frame_count": count, "max_timestamp_error_seconds": round(max_error, 8),
            "first_execution_frame": markers[first_execution],
            "last_demonstration_frame": markers.get(first_execution - 1),
        })
        if selected is not None:
            candidate["decoded_candidate_frame"] = markers[selected]
        candidate_rows.append(candidate)

    summary = {"source": REPOSITORY, "revision": REVISION, "split": coverage["split"],
               "episodes_verified": len(video_rows), "tasks": dict(Counter(row["task"] for row in video_rows)),
               "suites": dict(Counter(row["suite"] for row in video_rows)),
               "front_bytes_verified": sum(row["bytes"] for row in video_rows),
               "front_frames_decoded": sum(row["decoded_frame_count"] for row in video_rows),
               "max_timestamp_error_seconds": max(row["max_timestamp_error_seconds"] for row in video_rows),
               "first_execution_frames_verified": len(video_rows), "episodes_with_demonstration":
                   sum(row["last_demonstration_frame"] is not None for row in video_rows)}
    if summary["front_frames_decoded"] != FRAME_COUNT or len(summary["tasks"]) != len(CHECKPOINT_RULES):
        raise ValueError("核验结果未覆盖完整公开训练分段")
    write_json(work_dir / "front_video_index.json", {**summary, "episodes": video_rows})
    evaluation = {
        "source": REPOSITORY, "revision": REVISION, "split": coverage["split"],
        "use": "仅离线评价，专家标签、候选点选择规则及未来数据均不参与在线决策",
        "selection": "每项任务用统一的规则寻找首个匹配的非终止在线子任务起始帧，历史仅可用该帧之前的观测",
        "label_limit": "验证了专家记录的动作及对应视频帧，历史证据的必要性、可见性和正确答案集合均待人工核查",
        "rules": {task: {"online_subtask_regex": rule, "history_evidence_to_review": evidence}
                  for task, (rule, evidence) in CHECKPOINT_RULES.items()},
        "candidate_status": dict(Counter(row["status"] for row in candidate_rows)),
        "candidate_frames_with_2d_pixel_reference": sum(bool(row.get("has_2d_pixel_reference")) for row in candidate_rows),
        "human_verified": 0,
        "episodes": candidate_rows,
    }
    write_json(work_dir / "evaluation_only" / "candidate_reference_labels.json", evaluation)
    print(json.dumps({**summary, "candidate_status": evaluation["candidate_status"],
                      "with_official_2d_pixel_reference": evaluation["candidate_frames_with_2d_pixel_reference"],
                      "human_verified": 0}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="官方 Parquet 核验脚本的输出目录")
    parser.add_argument("--work-dir", required=True, type=Path, help="视频缓存及检查记录的输出目录")
    parser.add_argument("--reuse-video-dir", type=Path, help="可复用的视频根目录，含 videos 子目录")
    parser.add_argument("--workers", type=int, default=8, help="并行下载任务数")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("并行任务数至少为 1")
    coverage, release_dir, files = load_release(args.data_dir)
    downloads = stage_videos(files, args.work_dir, args.reuse_video_dir, args.workers)
    print(downloads, flush=True)
    analyse(coverage, release_dir, files, args.work_dir)


if __name__ == "__main__":
    main()
