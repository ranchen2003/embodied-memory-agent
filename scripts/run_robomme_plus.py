"""通过官方 RoboMME-Plus waypoint 接口运行视觉语言智能体。"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import json
from numbers import Real
import os
from pathlib import Path
import platform
import time


TERMINAL = {"success", "fail", "timeout", "error"}
SYSTEM = """You control a single Panda arm in a simulated tabletop task.
Use only the provided goal, camera observations and robot state. Follow the goal.
Return only a JSON object with {"waypoint":[x,y,z,roll,pitch,yaw,gripper]} for one feasible motion,
or {"stop":true} if you believe no further action is needed. Waypoints are absolute world poses
in metres and extrinsic XYZ Euler radians. Positive x is forward from the robot, positive y is
to its left, positive z is up. The centre of the tabletop is the world origin with z=0.
Gripper -1 closes and +1 opens. Keep practical clearance when moving over objects.
The simulator decides success. Use the updated observation after every waypoint."""


@dataclass(frozen=True)
class Frame:
    index: int
    phase: str
    image: object


class RGBMemory:
    """仅保存可见的前置相机图像。两种读取方式共用相同的容量限制。"""

    def __init__(self, mode: str, capacity: int):
        self.mode = mode
        self.frames = deque(maxlen=capacity)
        self.observed_frames = 0
        self.bytes = 0
        self.peak_bytes = 0

    def add(self, images, phase: str):
        for image in images:
            if len(self.frames) == self.frames.maxlen:
                self.bytes -= self.frames[0].image.nbytes
            self.frames.append(Frame(self.observed_frames, phase, image))
            self.observed_frames += 1
            self.bytes += image.nbytes
            self.peak_bytes = max(self.peak_bytes, self.bytes)

    def read(self, budget: int) -> list[Frame]:
        count = min(budget, len(self.frames))
        if count == 0:
            return []
        frames = list(self.frames)
        if self.mode == "recent" or count == 1:
            return frames[-count:]
        return [frames[index * (len(frames) - 1) // (count - 1)] for index in range(count)]


def _array(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb(value):
    import numpy as np

    image = _array(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("官方 RGB 帧需要为 H×W×3 的 uint8")
    # 复制环境的缓冲区，避免下一步复用的张量污染历史。
    return image.copy(order="C")


def _state(value):
    import numpy as np

    array = _array(value).ravel()
    if not np.isfinite(array).all():
        raise ValueError("观测机器人状态含非有限数据")
    return array.tolist()


def _public_observation(obs) -> tuple[list, object, object, dict]:
    # 只读取文档明示的相机图像与本体感知，不转发原始观测字典。
    front = obs["front_rgb_list"]
    wrist = obs.get("wrist_rgb_list", [])
    if not len(front):
        raise ValueError("缺少当前前视 RGB")
    states = {}
    for public_key, label in (("eef_state_list", "ee_xyz_rpy"), ("gripper_state_list", "gripper")):
        values = obs.get(public_key, [])
        if len(values):
            states[label] = _state(values[-1])
    return [_rgb(value) for value in front[:-1]], _rgb(front[-1]), _rgb(wrist[-1]) if len(wrist) else None, states


def _control_steps(info, previous):
    # waypoint 可执行若干内部控制步，不能用 API 次数或返回帧数代替。
    if "execution_total_control_steps" in info:
        return int(info["execution_total_control_steps"])
    return previous


def _policy_prompt(goal, history, wrist, states, previous_action, feedback):
    labels = [f"frame={item.index} ({item.phase})" for item in history]
    labels.append("current front RGB")
    if wrist is not None:
        labels.append("current wrist RGB")
    return json.dumps({"task_goal": goal, "image_order": labels, "robot_state": states,
                       "last_waypoint": previous_action, "last_env_feedback": feedback}, ensure_ascii=False)


def _action(answer):
    import numpy as np

    if not isinstance(answer, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    if answer.get("stop") is True and "waypoint" not in answer:
        return None
    values = answer.get("waypoint")
    if not isinstance(values, list) or len(values) != 7 or any(
        isinstance(value, bool) or not isinstance(value, Real) for value in values
    ):
        raise ValueError("waypoint 必须是 7 个数值，或单独输出 stop=true")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError("waypoint 超出有限 float32 数值范围")
    if array[-1] not in (-1, 1):
        raise ValueError("waypoint 夹爪值必须为 -1 或 1")
    return array.astype(np.float32)


def _numeric_usage(client):
    usage = getattr(client, "last_usage", None)
    if not isinstance(usage, dict):
        return None
    values = {}
    for name, alias in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        count = usage.get(name, usage.get(alias))
        if type(count) is int and count >= 0:
            values[name] = count
    return values or None


def _append(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def run_episode(builder, client, task: str, episode: int, dataset: str, output: Path,
                memory: RGBMemory, history_frames: int, max_calls: int):
    """运行固定回合，注入兼容的 memory 或 client 可复用同一评估循环。"""
    start = time.perf_counter()
    env = None
    stage = "build"
    result = {"task": task, "episode": episode, "dataset": dataset, "status": "error",
              "end_reason": None, "max_calls": max_calls, "api_calls": 0, "api_seconds": 0.0,
              "requests_with_usage": 0, "input_tokens": 0, "output_tokens": 0,
              "api_image_count": 0, "api_raw_image_bytes": 0, "env_step_calls": 0,
              "env_step_seconds": 0.0, "control_steps": None, "action_errors": 0,
              "action_error_penalty_steps": 0}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "steps.jsonl").open("w", encoding="utf-8") as log:
        try:
            # 评估端只用官方控制步上限，回合元数据不会传入策略。
            result["max_control_steps"] = builder.episode_max_steps(episode)
            env = builder.make_env_for_episode(episode)
            stage = "reset"
            reset_start = time.perf_counter()
            obs, info = env.reset()
            result["env_reset_seconds"] = time.perf_counter() - reset_start
            if info.get("status") == "error":
                raise RuntimeError("环境重置失败")
            goal = info["task_goal"]
            if isinstance(goal, (list, tuple)):
                goal = goal[0]
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("缺少公开任务目标")
            result["task_goal"] = goal
            past, front, wrist, states = _public_observation(obs)
            result["reset_demo_frames"] = len(past)
            memory.add(past, "demonstration")
            result["control_steps"] = _control_steps(info, None)
            last_action = None
            feedback = {"status": "ongoing", "action_error": False, "trial_reset": False}
            result.update(status="timeout", end_reason="api_call_limit")
            for call in range(1, max_calls + 1):
                seen = info.get("execution_control_steps")
                if seen is not None and int(seen) >= result["max_control_steps"]:
                    result["end_reason"] = "control_step_limit"
                    break
                stage = "policy_input"
                history = memory.read(history_frames)
                images = [item.image for item in history] + [front]
                if wrist is not None:
                    images.append(wrist)
                prompt = _policy_prompt(goal, history, wrist, states, last_action, feedback)
                image_bytes = sum(image.nbytes for image in images)
                event = {"call": call, "memory_frames": [item.index for item in history],
                         "image_count": len(images), "raw_image_bytes": image_bytes,
                         "control_steps_before": result["control_steps"]}
                stage = "api"
                result["api_calls"] += 1
                result["api_image_count"] += len(images)
                result["api_raw_image_bytes"] += image_bytes
                api_start = time.perf_counter()
                try:
                    answer = client.ask(SYSTEM, prompt, images)
                finally:
                    event["api_seconds"] = time.perf_counter() - api_start
                    result["api_seconds"] += event["api_seconds"]
                usage = _numeric_usage(client)
                if usage is not None:
                    result["requests_with_usage"] += 1
                    for name, quantity in usage.items():
                        result[name] += quantity
                    event["usage"] = usage
                stage = "action_validation"
                action = _action(answer)
                if action is None:
                    _append(log, {**event, "policy_stop": True})
                    result.update(status="fail", end_reason="policy_stop_before_success")
                    break
                last_action = action.tolist()
                stage = "env_step"
                result["env_step_calls"] += 1
                step_start = time.perf_counter()
                try:
                    obs, _, terminated, truncated, info = env.step(action)
                finally:
                    event["env_step_seconds"] = time.perf_counter() - step_start
                    result["env_step_seconds"] += event["env_step_seconds"]
                result["control_steps"] = _control_steps(info, result["control_steps"])
                status = info.get("status")
                stage = "feedback"
                if status not in TERMINAL | {None, "ongoing"}:
                    raise ValueError("环境返回未知状态")
                action_error = bool(info.get("action_error", False))
                trial_reset = bool(info.get("trial_reset", False))
                result["action_errors"] += action_error
                if action_error:
                    result["action_error_penalty_steps"] += int(info.get("error_penalty_steps", 0))
                _append(log, {**event, "waypoint": last_action, "control_steps_after": result["control_steps"],
                              "env_status": status, "action_error": action_error, "trial_reset": trial_reset,
                              "terminated": bool(terminated), "truncated": bool(truncated)})
                if status in TERMINAL or bool(terminated) or bool(truncated):
                    outcome = status if status in TERMINAL else "error"
                    result.update(status=outcome, end_reason=f"environment:{outcome}")
                    break
                feedback = {"status": "ongoing", "action_error": action_error, "trial_reset": trial_reset}
                stage = "observation"
                memory.add([front], "execution")
                past, front, wrist, states = _public_observation(obs)
                memory.add(past, "execution")
        except KeyboardInterrupt:
            result.update(status="interrupted", end_reason=stage)
            raise
        except Exception as error:
            # 服务报错正文可能包含请求地址或凭证，只记实际故障阶段及类型。
            result.update(status="error", end_reason=stage, error_type=type(error).__name__)
            failed = dict(event) if stage in {"api", "action_validation", "env_step", "feedback"} else {}
            _append(log, {**failed, "error_stage": stage, "error_type": type(error).__name__})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as error:
                    result["cleanup_error_type"] = type(error).__name__
            result.update(success=result["status"] == "success", elapsed_seconds=time.perf_counter() - start,
                          history_peak_bytes=memory.peak_bytes, history_frames_observed=memory.observed_frames)
            if not result["requests_with_usage"]:
                result.update(input_tokens=None, output_tokens=None)
            (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("train", "test"), required=True)
    parser.add_argument("--task", required=True, help="官方任务 ID，如 VideoPlaceButton")
    parser.add_argument("--episodes", nargs="+", type=int, required=True, help="该任务分段内的固定回合编号")
    parser.add_argument("--memory", choices=("recent", "uniform"), default="recent")
    parser.add_argument("--history-frames", type=int, default=4, help="每次调用最多发送的历史前视帧数")
    parser.add_argument("--history-capacity", type=int, default=1024, help="前视缓存容量，单位为帧")
    parser.add_argument("--max-calls", type=int, default=20, help="整次运行的模型调用上限")
    parser.add_argument("--max-calls-per-episode", type=int, help="单回合模型调用上限，可低于整次运行上限")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="保存服务凭证的环境变量名")
    parser.add_argument("--output", type=Path, required=True, help="新输出目录")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if (args.max_calls < 1 or (args.max_calls_per_episode is not None and args.max_calls_per_episode < 1)
            or args.history_capacity < 1 or not 0 <= args.history_frames <= args.history_capacity
            or min(args.episodes) < 0 or len(set(args.episodes)) != len(args.episodes)):
        parser.error("需要有效的调用与历史预算，以及非负、无重复的回合编号")
    try:
        import robomme_plus
        from robomme_plus import BenchmarkEnvBuilder
    except ImportError:
        parser.error("需要先在运行此脚本的 Python 中安装官方 RoboMME-Plus 环境")
    from vision_api import VisionAPI

    if args.task not in BenchmarkEnvBuilder.get_task_list():
        parser.error("任务 ID 不在官方 RoboMME-Plus 列表中")
    builder = BenchmarkEnvBuilder(args.task, dataset=args.dataset, action_space="waypoint", gui_render=False)
    if max(args.episodes) >= builder.get_episode_num():
        parser.error("回合编号超出选择的分段范围")
    if args.output.exists():
        parser.error("输出目录已存在，请为本次运行指定新目录")
    client = VisionAPI(args.endpoint, args.model, api_key=os.getenv(args.api_key_env), max_calls=args.max_calls)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"benchmark": "robomme_plus", "version": getattr(robomme_plus, "__version__", None),
              "dataset": args.dataset, "task": args.task, "episodes": args.episodes, "action_space": "waypoint",
              "memory": args.memory, "history_frames": args.history_frames, "history_capacity": args.history_capacity,
              "max_calls": args.max_calls, "max_calls_per_episode": args.max_calls_per_episode,
              "model": args.model, "platform": platform.platform()}
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = []
    remaining = args.max_calls
    for episode in args.episodes:
        if remaining == 0:
            results.append({"episode": episode, "status": "not_run", "end_reason": "run_api_call_limit", "success": False})
            continue
        episode_limit = min(remaining, args.max_calls_per_episode or remaining)
        result = run_episode(builder, client, args.task, episode, args.dataset, args.output / f"episode_{episode:03d}",
                             RGBMemory(args.memory, args.history_capacity), args.history_frames, episode_limit)
        results.append(result)
        remaining -= result["api_calls"]
        print(f"{args.task} {args.dataset}/{episode}: {result['status']}，API {result['api_calls']} 次")
        if result["end_reason"] == "api" and result["status"] == "error":
            # 服务故障时停止后续调用，保留未运行回合的状态和分母。
            for pending in args.episodes[len(results):]:
                results.append({"episode": pending, "status": "not_run", "end_reason": "api_error", "success": False})
            break
    summary = {"requested": len(args.episodes), "statuses": dict(Counter(row["status"] for row in results)),
               "success_rate": sum(row["success"] for row in results) / len(args.episodes),
               "api_calls": args.max_calls - remaining, "episodes": results}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(any(row["status"] in {"error", "not_run"} for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
