"""Inspect frame retention on public RoboMME-Plus demonstration histories."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[1]
REVISION = "ea22d395232719c13df778a2c2219a1ebdfe367e"
SOURCE = f"https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot/resolve/{REVISION}"
TASKS = ("VideoPlaceButton", "VideoPlaceOrder")
LABELS = {"press the button", "drop the cube onto target"}
METHODS = ("recent", "uniform", "state_change")


def cache_file(session, name):
    path = ROOT / "data" / name
    if not path.exists():
        response = session.get(f"{SOURCE}/{name}", timeout=(10, 30))
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return path


def demonstration(table):
    demo = table["is_video_demo"].to_numpy()
    stop = int(np.count_nonzero(demo))
    if not 0 < stop < len(demo) or not demo[:stop].all():
        raise ValueError("Expected a contiguous demonstration followed by execution")
    prefix = table.slice(0, stop)
    states = np.asarray(prefix["observation.state"].to_pylist(), dtype=float)
    return states, prefix["subtask_index"].to_numpy()


def normalized_change(values):
    scale = max(float(np.quantile(values, 0.95)), 1e-6)
    return np.minimum(values / scale, 4.0)


def select_frames(states, budget, fps):
    length = len(states)
    budget = min(budget, length)
    recent = np.arange(length - budget, length).tolist()
    uniform = np.linspace(0, length - 1, budget, dtype=int).tolist()

    delta = np.diff(states, axis=0, prepend=states[:1])
    motion = np.linalg.norm(delta[:, :7], axis=1)
    gripper = np.abs(delta[:, 7])
    score = normalized_change(motion) + normalized_change(gripper)
    candidates = np.argsort(-score, kind="stable").tolist()
    selected = []
    suppressed = np.zeros(length, dtype=bool)
    radius = max(1, min(fps // 2, length // (2 * budget)))
    for index in candidates:
        if not suppressed[index]:
            selected.append(index)
            suppressed[max(0, index - radius):index + radius + 1] = True
            if len(selected) == budget:
                break
    selected_indices = set(selected)
    remaining = [index for index in candidates if index not in selected_indices]
    selected.extend(remaining[:budget - len(selected)])
    return {"recent": recent, "uniform": uniform, "state_change": sorted(selected)}


def reference_intervals(label_indices, names):
    boundaries = [0, *(np.flatnonzero(label_indices[1:] != label_indices[:-1]) + 1), len(label_indices)]
    references = []
    for start, stop in zip(boundaries, boundaries[1:]):
        label = names[int(label_indices[start])].strip().lower()
        if label in LABELS:
            references.append({"label": label, "start": int(start), "end_exclusive": int(stop)})
    return references


def summarize(episodes, budget):
    rows = []
    for episode in episodes:
        choices = select_frames(episode["states"], budget, episode["control_freq"])
        for name, indices in choices.items():
            for interval in episode["references"]:
                hit = any(interval["start"] <= index < interval["end_exclusive"] for index in indices)
                interval.setdefault("covered", {})[name] = hit
            live = {key: value for key, value in episode.items() if key not in ("states", "references")}
            rows.append({**live, "method": name, "frames_kept": len(indices),
                         "reference_intervals": len(episode["references"]),
                         "covered_intervals": sum(x["covered"][name] for x in episode["references"]),
                         "frame_indices": indices})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=24, help="Number of frames retained before decision")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")

    with requests.Session() as session:
        manifest = json.loads(cache_file(session, "meta/source_episodes.json").read_text(encoding="utf-8"))
        labels = pq.read_table(cache_file(session, "meta/subtasks.parquet")).to_pylist()
        names = {row["subtask_index"]: row["subtask"] for row in labels}
        episodes = []
        for row in manifest:
            if row["task"] not in TASKS:
                continue
            path = cache_file(session, f"data/chunk-000/file-{row['episode_index']:03d}.parquet")
            table = pq.read_table(path, columns=["observation.state", "is_video_demo", "subtask_index"])
            states, label_indices = demonstration(table)
            references = reference_intervals(label_indices, names)
            if not references:
                raise ValueError(f"No reference intervals for episode {row['episode_index']}")
            episodes.append({**row, "decision_frame": len(states), "states": states, "references": references})

    if len(episodes) != 6:
        raise ValueError(f"Expected six public development episodes, found {len(episodes)}")
    rows = summarize(episodes, args.budget)
    output = ROOT / "runs" / "pilot" / f"budget_{args.budget}"
    output.mkdir(parents=True, exist_ok=True)
    fields = ("task", "episode_index", "seed", "difficulty", "decision_frame", "method",
              "frames_kept", "reference_intervals", "covered_intervals")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    trace = {"dataset_revision": REVISION, "frame_budget": args.budget, "rows": rows,
             "reference_intervals": {str(ep["episode_index"]): ep["references"] for ep in episodes}}
    (output / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        hits = sum(row["covered_intervals"] for row in group)
        total = sum(row["reference_intervals"] for row in group)
        print(f"{method:12s} {hits}/{total} labelled intervals ({hits / total:.1%})")
    print(output)


if __name__ == "__main__":
    main()
