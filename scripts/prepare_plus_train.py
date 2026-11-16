"""Prepare the complete public RoboMME-Plus training schedule without videos."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from itertools import groupby
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import requests


REVISION = "ea22d395232719c13df778a2c2219a1ebdfe367e"
REPOSITORY = "Yinpei/robomme_plus_data_lerobot"
SOURCE = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
TASK_SUITES = {
    "Counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "Permanence": (
        "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap",
    ),
    "Reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "Imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
OBSERVATION_COLUMNS = (
    "observation.state", "is_video_demo", "timestamp", "frame_index", "native_frame_index",
)
ANNOTATION_COLUMNS = (
    "subtask_index", "subtask_grounded_index",
    "subtask_online_index", "subtask_grounded_online_index",
)
CHECK_COLUMNS = ("episode_index", "task_index", "control_freq", "next.done")


def write_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path, value):
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    write_bytes(path, content.encode("utf-8"))


def request_file(relative_path):
    for attempt in range(3):
        try:
            response = requests.get(f"{SOURCE}/{relative_path}", timeout=(10, 40))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def verify_file(content, expected):
    return (
        len(content) == expected["bytes"]
        and hashlib.sha256(content).hexdigest() == expected["sha256"]
    )


def cache_release_manifest(work_dir):
    path = work_dir / "manifest.json"
    if not path.is_file():
        write_bytes(path, request_file("manifest.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def cache_file(cache_root, expected):
    relative_path = expected["path"]
    path = cache_root / relative_path
    if path.is_file() and verify_file(path.read_bytes(), expected):
        return path, False
    content = request_file(relative_path)
    if not verify_file(content, expected):
        raise ValueError(f"Release checksum failed: {relative_path}")
    write_bytes(path, content)
    return path, True


def episode_path(source):
    return f"data/chunk-000/file-{source['episode_index']:03}.parquet"


def load_metadata(cache_root, manifest, pool):
    files = {file["path"]: file for file in manifest["files"]}
    metadata_paths = (
        "meta/source_episodes.json", "meta/tasks.parquet",
        "meta/subtasks.parquet", "meta/info.json",
    )
    metadata = list(pool.map(lambda name: cache_file(cache_root, files[name]), metadata_paths))
    sources = json.loads(metadata[0][0].read_text(encoding="utf-8"))
    sources = sorted(sources, key=lambda source: source["episode_index"])
    tasks = read_dictionary(metadata[1][0], "task_index", "task")
    annotations = read_dictionary(metadata[2][0], "subtask_index", "subtask")
    info = json.loads(metadata[3][0].read_text(encoding="utf-8"))

    episodes = list(
        pool.map(lambda source: cache_file(cache_root, files[episode_path(source)]), sources)
    )
    downloads = sum(path.stat().st_size for path, downloaded in metadata + episodes if downloaded)
    return sources, tasks, annotations, info, episodes, downloads


def read_dictionary(path, index, value):
    table = pq.read_table(path, columns=[index, value])
    return {row[index]: row[value] for row in table.to_pylist()}


def verify_dataset(sources, info, manifest):
    suite_by_task = {
        task: suite for suite, tasks in TASK_SUITES.items() for task in tasks
    }
    expected = {task: 5 if task == "MoveCube" else 3 for task in suite_by_task}
    actual = Counter(source["task"] for source in sources)
    if actual != expected:
        raise ValueError(f"Unexpected official training task coverage: {actual}")
    if [source["episode_index"] for source in sources] != list(range(50)):
        raise ValueError("Official episodes must be indexed 0 through 49 exactly once.")
    if len({(source["task"], source["task_episode"]) for source in sources}) != 50:
        raise ValueError("Duplicate task-local episode identifier.")
    if manifest["episodes"] != 50 or manifest["frames"] != 56_462:
        raise ValueError("Reference release manifest totals changed.")
    if info["total_episodes"] != 50 or info["fps"] != 20:
        raise ValueError("Release metadata differs from 50 training episodes at 20 Hz.")
    return suite_by_task


def observation_columns(path):
    """Read only the observable time/state data, with no action or label columns."""
    return pq.read_table(path, columns=list(OBSERVATION_COLUMNS))


def episode_schedule(source, path, tasks, suite, stride):
    observed = observation_columns(path)
    check = pq.read_table(path, columns=list(CHECK_COLUMNS))
    episode = source["episode_index"]
    total = observed.num_rows
    frame = observed["frame_index"].to_numpy()
    native = observed["native_frame_index"].to_numpy()
    ts = observed["timestamp"].to_numpy()
    flags = observed["is_video_demo"].to_numpy()
    fps = source["control_freq"]
    indices = np.arange(total)

    if total != source["frames"] or not np.array_equal(frame, indices):
        raise ValueError(f"Invalid frame count or indexing in episode {episode}")
    if np.any(np.diff(native) <= 0) or not np.allclose(ts, indices / fps, atol=1e-4):
        raise ValueError(f"Invalid native frame/time sequence in episode {episode}")
    if set(check["episode_index"].to_numpy()) != {episode}:
        raise ValueError(f"Parquet contains a different episode: {episode}")
    if set(check["control_freq"].to_numpy()) != {fps} or fps != 20:
        raise ValueError(f"Unexpected episode frequency: {episode}")
    task_ids = set(check["task_index"].to_numpy())
    if len(task_ids) != 1:
        raise ValueError(f"Changing task text within episode {episode}")
    task_text = tasks[int(next(iter(task_ids)))]
    state = np.asarray(observed["observation.state"].to_pylist(), dtype=np.float64)
    if state.shape != (total, 8) or not np.isfinite(state).all():
        raise ValueError(f"Invalid observation.state in episode {episode}")

    first_execution = int(np.count_nonzero(flags))
    if not np.array_equal(flags, indices < first_execution):
        raise ValueError(f"Demonstration phase is not a contiguous prefix: {episode}")
    done = np.flatnonzero(check["next.done"].to_numpy())
    last_eligible_exclusive = int(done[0]) if done.size else total - 1
    if first_execution >= last_eligible_exclusive:
        raise ValueError(f"No nonterminal execution observations in episode {episode}")

    checkpoints = []
    for f in range(first_execution, last_eligible_exclusive, stride):
        phase = "online"
        if f == first_execution:
            phase = "demonstration_end" if first_execution else "execution_start"
        checkpoints.append({
            "frame": f,
            "native_frame": int(native[f]),
            "time_seconds": round(f / fps, 3),
            "phase": phase,
            "history_end_exclusive": f,
        })

    return {
        "episode_index": episode,
        "suite": suite,
        "task": source["task"],
        "task_episode": source["task_episode"],
        "seed": source["seed"],
        "difficulty": source["difficulty"],
        "variant": source["variant"],
        "instruction": task_text,
        "control_freq": fps,
        "source_parquet": episode_path(source),
        "recorded_frames": total,
        "demonstration_end_exclusive": first_execution,
        "first_execution_frame": first_execution,
        "first_terminal_frame": int(done[0]) if done.size else None,
        "checkpoints": checkpoints,
    }


def reference_annotations(path, schedule, annotations):
    label_table = pq.read_table(path, columns=list(ANNOTATION_COLUMNS))
    data = label_table.to_pydict()
    start = 0
    spans = []
    for annotation, rows in groupby(data["subtask_index"]):
        end = start + len(list(rows))
        if start < schedule["demonstration_end_exclusive"]:
            spans.append({
                "start": start,
                "end_exclusive": min(end, schedule["demonstration_end_exclusive"]),
                "recorded_subtask": annotations.get(annotation),
            })
        start = end
    targets = []
    for item in schedule["checkpoints"]:
        frame = item["frame"]
        labels = {name: annotations.get(data[name][frame]) for name in ANNOTATION_COLUMNS}
        targets.append({
            "frame": frame,
            "recorded_subtask": labels["subtask_index"],
            "recorded_grounded_subtask": labels["subtask_grounded_index"],
            "online_subtask_label": labels["subtask_online_index"],
            "online_grounded_subtask_label": labels["subtask_grounded_online_index"],
        })
    return {
        "episode_index": schedule["episode_index"],
        "demonstration_label_spans": spans,
        "checkpoint_reference_labels": targets,
    }


def summarize(schedules):
    totals = {}
    for name, tasks in TASK_SUITES.items():
        totals[name] = {}
        for task in tasks:
            subset = [episode for episode in schedules if episode["task"] == task]
            totals[name][task] = {
                "episodes": len(subset),
                "recorded_frames": sum(episode["recorded_frames"] for episode in subset),
                "checkpoints": sum(len(episode["checkpoints"]) for episode in subset),
            }
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16, help="Execution frames per online checkpoint.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent small Parquet downloads.")
    args = parser.parse_args()
    if args.stride < 1 or args.workers < 1:
        parser.error("--stride and --workers must be positive")

    work_dir = args.work_dir.resolve()
    cache_root = work_dir / "evaluation_only" / "pinned_release" / REVISION
    manifest = cache_release_manifest(cache_root)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sources, tasks, annotations, info, files, downloads = load_metadata(
            cache_root, manifest, pool
        )

    by_task = verify_dataset(sources, info, manifest)
    schedules = []
    targets = []
    for source, (path, _) in zip(sources, files, strict=True):
        schedule = episode_schedule(source, path, tasks, by_task[source["task"]], args.stride)
        schedules.append(schedule)
        targets.append(reference_annotations(path, schedule, annotations))
    total_frames = sum(schedule["recorded_frames"] for schedule in schedules)
    if total_frames != manifest["frames"]:
        raise ValueError(f"Verified frame count differs from release: {total_frames}")
    total_checkpoints = sum(len(schedule["checkpoints"]) for schedule in schedules)

    common = {
        "benchmark": "RoboMME-Plus",
        "source": REPOSITORY,
        "revision": REVISION,
        "split": "public_train",
        "checkpoint_stride_frames": args.stride,
    }
    write_json(work_dir / "evaluation_only" / "checkpoint_targets.json", {
        **common,
        "use": "Offline annotation audit and scoring only. Never supply to the agent.",
        "meaning": "Recorded expert subgoal labels are references rather than all acceptable actions.",
        "episodes": targets,
    })
    write_json(work_dir / "coverage.json", {
        **common,
        "episodes_count": len(schedules),
        "tasks_count": len(by_task),
        "suites_count": len(TASK_SUITES),
        "recorded_frames": total_frames,
        "checkpoints_count": total_checkpoints,
        "phase_counts": dict(Counter(
            checkpoint["phase"] for schedule in schedules for checkpoint in schedule["checkpoints"]
        )),
        "checkpoint_rule": (
            "Use the first execution frame and a fixed frame stride thereafter. "
            "Stop before the first next.done frame, or before the last recording frame if absent. "
            "Subgoal annotations are never used to choose checkpoints."
        ),
        "history_rule": (
            "At frame f, history includes only local frames less than f, and the current "
            "observation is frame f. This file is a schedule for the evaluator, not a planner prompt."
        ),
        "observation_columns": list(OBSERVATION_COLUMNS),
        "by_suite": summarize(schedules),
        "episodes": schedules,
    })
    print(
        f"Verified {len(schedules)} episodes / {len(by_task)} tasks / {len(TASK_SUITES)} suites, "
        f"{total_frames} frames, {total_checkpoints} checkpoints; downloaded {downloads:,} data bytes."
    )
    print(work_dir / "coverage.json")


if __name__ == "__main__":
    main()
