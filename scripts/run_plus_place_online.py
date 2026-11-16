"""在两个官方圆环任务的固定训练回合中比较图像记忆。"""

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from grounded_decision import grounded_place
from place_diagnostic.place_events import (
    candidate_frame_indices, cube_centers, fill_uniformly, goal_conditions, release_candidates,
)
from place_diagnostic.ring_association import detect_rings


TRAIN_SEEDS = {
    "VideoPlaceButton": (1155046303, 1911834806, 1073749011),
    "VideoPlaceOrder": (1834444142, 1775545447, 2124331336),
}
MODES = ("current", "uniform", "complete")


def retain_history(aperture, budget, mode):
    """事件保留选择依赖公开夹爪时序，两个记忆方式都遵守同一 RGB 预算。"""
    horizon = len(aperture)
    if horizon < 1 or budget < 1 or mode not in MODES[1:]:
        raise ValueError("记忆策略需要示范历史、正数预算和有效模式")
    if mode == "uniform":
        return np.linspace(0, horizon - 1, num=min(horizon, budget), dtype=int).tolist()
    chosen = {0}
    for close, release in release_candidates(aperture):
        before, after = candidate_frame_indices(close, release, horizon)
        if before and after:
            around_event = chosen.union(before, after)
            if len(around_event) <= budget:
                chosen = around_event
    return [int(index) for index in fill_uniformly(chosen, horizon, budget)]


def infer_subgoal(observation, goal, mode, *, history_frames, fps):
    """只从对外 RGB 和夹爪批次读取预测所需字段。"""
    rgb = observation["front_rgb_list"]
    current = len(rgb) - 1
    if current < 0:
        raise ValueError("环境没有给出当前前视图")
    color, _, _ = goal_conditions(goal)
    current_rgb = rgb[-1]
    if mode == "current":
        colors = cube_centers(cv2.cvtColor(current_rgb, cv2.COLOR_RGB2HSV))
        source = colors.get(color)
        rings = detect_rings(current_rgb)
        target = (rings[np.linalg.norm(rings - source, axis=1).argmin()]
                  if source is not None and len(rings) else None)
        return {"operation": "place", "object": f"{color} cube", "destination": "ring", "camera": "front",
                "frame": current, "object_pixel_xy": source.tolist() if source is not None else None,
                "target_pixel_xy": target.tolist() if target is not None else None,
                "release_frame": None, "evidence_frames": [], "candidate_events": 0,
                "retained_demo_rgb": [], "demo_gripper_samples": 0}
    gripper = np.asarray(observation["gripper_state_list"], dtype=float)
    if gripper.shape != (len(rgb), 2) or not np.isfinite(gripper).all():
        raise ValueError("公开夹爪状态与前视观测未对齐")
    aperture = gripper[:-1].mean(axis=1)
    selected = retain_history(aperture, history_frames, mode)
    indices = [*selected, current]

    # 未保留的图像用空位占住绝对帧号，决策函数无法读取原始全量图像。
    frames = [None] * len(rgb)
    for index in indices:
        frames[index] = cv2.cvtColor(rgb[index], cv2.COLOR_RGB2BGR)
    decisions = grounded_place(frames, aperture, goal, fps, stride=1, sampled_indices=indices)
    result = decisions["decision"]["object_and_time+exchange"]
    return {**result, "candidate_events": decisions["candidate_events"], "retained_demo_rgb": selected,
            "demo_gripper_samples": len(aperture)}


def save_front_image(directory, stem, phase, observation):
    """保存用于复核的未经标注的前视图，不改变策略可用信息。"""
    image = cv2.cvtColor(observation["front_rgb_list"][-1], cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"无法编码{phase}前视图")
    name = f"{stem}_{phase}.png"
    (directory / name).write_bytes(encoded.tobytes())
    return name


def run_one(builder, executor, task, episode, mode, *, history_frames, fps, max_waypoints,
            output_dir=None, allow_cpu_render=False):
    started = perf_counter()
    seed, difficulty = builder.resolve_episode(episode)
    if seed != TRAIN_SEEDS[task][episode]:
        raise ValueError(f"{task} 训练回合 {episode} 与预登记种子不符")
    result = {"task": task, "episode": episode, "seed": seed, "difficulty": difficulty, "mode": mode,
              "max_control_steps": builder.episode_max_steps(episode), "success": False}
    stem = f"{task}_{episode}_{mode}"
    env = None
    stage = "make_env"
    try:
        env = builder.make_env_for_episode(episode, include_front_depth=True,
                                            include_front_camera_extrinsic=True,
                                            include_front_camera_intrinsic=True)
        stage = "render_device"
        backend = env.unwrapped.backend
        renderer = backend.render_device
        gpu_render = bool(renderer is not None and renderer.is_cuda() and renderer.can_render())
        result["backend"] = {"render": backend.render_backend, "physics": backend.sim_backend,
                             "device": renderer.name if renderer is not None else None, "gpu_render": gpu_render}
        if not allow_cpu_render and not gpu_render:
            raise RuntimeError("正式评测需要通过 SAPIEN 的 GPU 渲染设备检查")
        stage = "reset"
        observation, info = env.reset()
        if info.get("status") in {"fail", "timeout", "error"}:
            raise RuntimeError(f"重置状态为 {info['status']}")
        result["reset_seconds"] = round(perf_counter() - started, 3)
        stage = "decision"
        goal = info["task_goal"]
        if not (isinstance(goal, (tuple, list)) and len(goal) == 1 and isinstance(goal[0], str)):
            raise ValueError("当前官方任务须给出一条文本指令")
        result["goal"] = goal[0]
        result["demonstration_frames"] = len(observation["front_rgb_list"]) - 1
        if output_dir is not None:
            result["decision_image"] = save_front_image(output_dir, stem, "decision", observation)
        decision_started = perf_counter()
        subgoal = infer_subgoal(observation, goal[0], mode, history_frames=history_frames, fps=fps)
        result["decision_seconds"] = round(perf_counter() - decision_started, 3)
        result["subgoal"] = subgoal
        if subgoal["object_pixel_xy"] is None or subgoal["target_pixel_xy"] is None:
            result["status"] = "no_subgoal"
            return result

        stage = "execution"
        final_obs, final_info, trace = executor(env, observation, info, pick_pixel_xy=subgoal["object_pixel_xy"],
                                                place_pixel_xy=subgoal["target_pixel_xy"], max_waypoints=max_waypoints)
        result.update(status=trace["status"], official_status=final_info.get("status"),
                      success=final_info.get("status") == "success", execution=trace)
        result["execution_total_control_steps"] = final_info.get("execution_total_control_steps")
        if output_dir is not None and final_obs.get("front_rgb_list"):
            try:
                result["final_image"] = save_front_image(output_dir, stem, "final", final_obs)
            except (cv2.error, OSError) as error:
                result["final_image_error"] = str(error)
    except (AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        result.update(status="error", error_stage=stage, error_type=type(error).__name__, error_message=str(error))
    finally:
        result["elapsed_seconds"] = round(perf_counter() - started, 3)
        if env is not None:
            try:
                env.close()
            except (OSError, RuntimeError) as error:
                result["close_error"] = str(error)
    return result


def write_json(path, value):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TRAIN_SEEDS, action="append", help="仅运行指定任务，可重复")
    parser.add_argument("--episode", choices=range(3), type=int, action="append", help="仅运行指定回合，可重复")
    parser.add_argument("--mode", choices=MODES, action="append", help="仅运行指定对照组，可重复")
    parser.add_argument("--history-frames", default=512, type=int, help="两种记忆共同使用的历史图像容量")
    parser.add_argument("--max-waypoints", default=8, type=int, help="每回合的 waypoint 命令上限")
    parser.add_argument("--allow-cpu-render-diagnostic", action="store_true", help="标记并允许本机 CPU 渲染接口诊断")
    parser.add_argument("--output", required=True, type=Path, help="新建输出目录")
    args = parser.parse_args()
    if min(args.history_frames, args.max_waypoints) < 1:
        parser.error("图像和动作预算须为正数")

    from robomme_plus import BenchmarkEnvBuilder
    from robomme_plus.evaluation.timing import CONTROL_FREQUENCY
    from plus_pick_place import execute_pick_place

    tasks = [task for task in TRAIN_SEEDS if not args.task or task in args.task]
    episodes = [index for index in range(3) if not args.episode or index in args.episode]
    modes = [mode for mode in MODES if not args.mode or mode in args.mode]
    matrix = [{"task": task, "episode": episode, "seed": TRAIN_SEEDS[task][episode], "mode": mode}
              for task, episode, mode in product(tasks, episodes, modes)]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "matrix.json", {"split": "train", "waypoint_budget": args.max_waypoints,
               "history_rgb_budget": args.history_frames, "control_hz": CONTROL_FREQUENCY,
               "cpu_render_diagnostic": args.allow_cpu_render_diagnostic, "planned": matrix})
    totals = defaultdict(lambda: {"runs": 0, "success": 0, "statuses": defaultdict(int)})

    builders = {task: BenchmarkEnvBuilder(task, dataset="train", action_space="waypoint", gui_render=False)
                for task in tasks}
    for item in matrix:
        task, episode, mode = item["task"], item["episode"], item["mode"]
        print(f"启动 {task}/{episode}/{mode}", flush=True)
        result = run_one(builders[task], execute_pick_place, task, episode, mode,
                         history_frames=args.history_frames, fps=CONTROL_FREQUENCY, max_waypoints=args.max_waypoints,
                         output_dir=args.output, allow_cpu_render=args.allow_cpu_render_diagnostic)
        write_json(args.output / f"{task}_{episode}_{mode}.json", result)
        counts = totals[mode]
        counts["runs"] += 1
        counts["success"] += int(result["success"])
        counts["statuses"][result["status"]] += 1
        write_json(args.output / "summary.json", {"planned": len(matrix), "finished": sum(x["runs"] for x in totals.values()),
                                                   "modes": totals})
        print(f"完成 {task}/{episode}/{mode}，状态 {result['status']}", flush=True)
        if result.get("error_stage") == "render_device" and not args.allow_cpu_render_diagnostic:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
