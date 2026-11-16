"""启动官方训练回合，核查公开 RGB-D、标定和位姿执行入口。"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

from plus_waypoint import ground_pixel, waypoint


def check_environment(task, episode, pixel, send_hold):
    from robomme_plus import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(task, dataset="train", action_space="waypoint", gui_render=False)
    return probe_episode(builder, episode, pixel, send_hold)


def probe_episode(builder, episode, pixel, send_hold):
    report = {"task": builder.env_id, "split": "train", "episode": episode,
              "episode_count": builder.get_episode_num(), "max_control_steps": builder.episode_max_steps(episode)}
    environment = None
    stage = "make_env"
    try:
        environment = builder.make_env_for_episode(
            episode,
            include_front_depth=True,
            include_front_camera_extrinsic=True,
            include_front_camera_intrinsic=True,
        )
        stage = "reset"
        observation, info = environment.reset()
        if info.get("status") in {"error", "fail"}:
            raise RuntimeError("环境初始化失败")
        stage = "observation"
        frame = observation["front_rgb_list"][-1]
        report.update(reset_status=info.get("status"), reset_frames=len(observation["front_rgb_list"]),
                      front_shape=list(frame.shape), depth_shape=list(observation["front_depth_list"][-1].shape),
                      intrinsic_shape=list(info["front_camera_intrinsic"].shape),
                      extrinsic_shape=list(observation["front_camera_extrinsic_list"][-1].shape))
        if pixel is not None:
            stage = "ground_pixel"
            report.update(pixel_xy=pixel, world_xyz_m=ground_pixel(observation, info, pixel).tolist())
        if send_hold:
            stage = "hold_waypoint"
            pose = np.asarray(observation["eef_state_list"][-1], dtype=np.float64)
            measured_opening = float(np.sum(observation["gripper_state_list"][-1]))
            grip = 1 if measured_opening >= 0.04 else -1
            action = waypoint(pose[:3], pose[3:], grip)
            _, _, terminated, truncated, feedback = environment.step(action)
            report.update(hold_waypoint=action.tolist(), waypoint_status=feedback.get("status"),
                          waypoint_action_error=bool(feedback.get("action_error", False)),
                          control_steps=int(feedback.get("execution_total_control_steps", 0)))
            if terminated or truncated or report["waypoint_action_error"]:
                raise RuntimeError("原位动作终止回合或被官方执行器判错")
        report["ok"] = True
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        report.update(ok=False, error_stage=stage, error_type=type(error).__name__, error_message=str(error))
    finally:
        if environment is not None:
            environment.close()
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="VideoPlaceButton")
    parser.add_argument("--episode", type=int, default=0, help="公开训练任务内部回合编号")
    parser.add_argument("--pixel", type=float, nargs=2, metavar=("X", "Y"), help="验证反投影的当前目标像素")
    parser.add_argument("--send-hold", action="store_true", help="向仿真发送一次原位 waypoint")
    parser.add_argument("--source", type=Path, help="预检尚未安装的官方仓库目录")
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("回合编号必须是非负数")
    if args.source is not None:
        sys.path.insert(0, str(args.source.resolve() / "src"))
    try:
        report = check_environment(args.task, args.episode, args.pixel, args.send_hold)
    except (ImportError, IndexError, KeyError, ValueError) as error:
        report = {"ok": False, "error_stage": "import_or_metadata", "error_type": type(error).__name__,
                  "error_message": str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
