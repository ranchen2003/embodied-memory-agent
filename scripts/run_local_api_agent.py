"""用真实图文 API 选择两项技能，在本机视觉桌面环境中执行。"""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import torch

from sim_visual_pick_place import CAMERA, IMAGE_SIZE, VisualPickPlace, as_array, evaluate
from vision_api import VisionAPI


SYSTEM = (
    'You control a simulated arm. Use the current photo and the observed history. '
    'Choose exactly one command: {"action":"pick"}, {"action":"place"} or {"action":"finish"}. '
    'pick grasps and lifts the red cube. place releases the held cube at the green marker. '
    'finish ends the episode. Output a JSON object only.'
)


def run_episode(controller, client, seed):
    cube, edges, goal = controller.prepare(seed, record_video=True)
    holding = None
    placed = False
    history = []
    steps = controller.record["api_steps"]
    for _ in range(client.max_calls):
        observation = controller.observation
        rgb = as_array(observation["sensor_data"][CAMERA]["rgb"])[0]
        gap = float(as_array(observation["agent"]["qpos"]).reshape(-1)[-2:].sum())
        prompt = ("Goal: move the red cube to the green marker and finish with the gripper open. "
                  f"Gripper width is {gap:.3f} m. Skill feedback: {json.dumps(history)}. What should the robot do now?")
        entry = {"joint_steps_before": len(controller.record["actions"])}
        steps.append(entry)
        try:
            decision = client.ask(SYSTEM, prompt, [rgb])
        finally:
            entry["request"] = client.last_call.copy()
            entry["usage"] = client.last_usage.copy()
        action = decision.get("action")
        entry["action"] = action if isinstance(action, str) else None
        if action == "pick" and holding is None and not placed:
            holding = controller.pick(cube, edges)
            feedback = "The pick skill reports a held cube and the gripper lifted."
        elif action == "place" and holding is not None:
            controller.place(goal, *holding)
            holding = None
            placed = True
            feedback = "The place skill released the cube and the open gripper retreated."
        elif action == "finish":
            entry["feedback"] = "模型请求结束"
            entry["joint_steps_after"] = len(controller.record["actions"])
            return
        else:
            feedback = "The command was rejected by the skill preconditions."
        history.append(feedback)
        entry["feedback"] = feedback
        entry["joint_steps_after"] = len(controller.record["actions"])
    raise RuntimeError("调用预算用完，模型没有结束回合")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI 兼容 HTTP 端点")
    parser.add_argument("--model", required=True, help="模型名")
    parser.add_argument("--seed", type=int, default=2, help="本机桌面场景种子")
    parser.add_argument("--max-calls", type=int, default=5, help="单回合调用上限")
    parser.add_argument("--output", type=Path, required=True, help="回放和运行记录目录")
    args = parser.parse_args()
    if args.seed < 0 or args.max_calls < 1:
        parser.error("种子须非负，调用预算须为正整数")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    client = VisionAPI(args.endpoint, args.model, os.getenv("VISION_API_KEY"), args.max_calls, max_tokens=96)
    torch.set_num_threads(1)
    env = gym.make("PickCube-v1", num_envs=1, sim_backend="cpu", render_backend="cpu", obs_mode="rgbd",
                   render_mode="rgb_array", control_mode="pd_joint_pos", max_episode_steps=120,
                   sensor_configs={CAMERA: {"width": IMAGE_SIZE, "height": IMAGE_SIZE}})
    record = {"environment": "本机可见目标桌面测试", "model": args.model, "seed": args.seed,
              "call_budget": args.max_calls, "passed": False, "api_steps": [], "stages": [], "actions": []}
    controller = None
    try:
        controller = VisualPickPlace(env, directory, record, native_goal=False, record_oracle=False)
        run_episode(controller, client, args.seed)
        record.update(evaluate(env))
    except (RuntimeError, ValueError) as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if controller is not None and controller.video_writer is not None:
            controller.video_writer.release()
        client.close()
        env.close()
        result = directory / "api_agent.json"
        result.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(result)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
