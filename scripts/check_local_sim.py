"""在 Windows 上检查 ManiSkill 单环境能否渲染 RGB 并执行关节指令。"""

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

MOVEMENT_THRESHOLD_RAD = 1e-4


def as_numpy(value):
    """将单环境的张量或数组转换为 NumPy 数组。"""
    import numpy as np

    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def rgb_frame(observation):
    """读取观测中的一幅相机图像，去掉环境批次维。"""
    import numpy as np

    cameras = observation.get("sensor_data", {})
    for name, sensor in cameras.items():
        if "rgb" not in sensor:
            continue
        frame = as_numpy(sensor["rgb"])
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8:
            return name, frame
    raise RuntimeError("观测中没有符合 H×W×3、uint8 格式的 RGB 图像。")


def run_check(args, result):
    """在仿真中驱动 Panda 第一关节，并记录图像与关节实测值。"""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  注册 PickCube-v1 环境。
    import numpy as np

    started = perf_counter()
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        sim_backend="cpu",
        # Windows 上用 CPU 路径读回图像，避开 CUDA 相机数据转换。
        render_backend="cpu",
        obs_mode="rgb",
        render_mode="rgb_array",
        control_mode="pd_joint_delta_pos",
    )
    result["environment_creation_seconds"] = round(perf_counter() - started, 3)
    try:
        started = perf_counter()
        observation, _ = env.reset(seed=args.seed)
        result["reset_seconds"] = round(perf_counter() - started, 3)
        _, frame = rgb_frame(observation)
        result["initial_rgb_range"] = [int(frame.min()), int(frame.max())]
        initial_joint = float(as_numpy(observation["agent"]["qpos"]).reshape(-1)[0])

        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        # 这里只改变第一关节的归一化增量。实际转动由控制器和物理仿真决定。
        action[..., 0] = 0.35
        if not env.action_space.contains(action):
            raise RuntimeError(f"关节指令不符合动作空间 {env.action_space}。")

        started = perf_counter()
        executed_steps = 0
        for _ in range(args.steps):
            observation, _, terminated, truncated, _ = env.step(action)
            executed_steps += 1
            if as_numpy(terminated).any() or as_numpy(truncated).any():
                break

        camera, frame = rgb_frame(observation)
        final_joint = float(as_numpy(observation["agent"]["qpos"]).reshape(-1)[0])
        joint_delta = final_joint - initial_joint
        result.update(
            executed_steps=executed_steps,
            step_seconds=round(perf_counter() - started, 3),
            camera=camera,
            rgb_shape=list(frame.shape),
            rgb_dtype=str(frame.dtype),
            rgb_range=[int(frame.min()), int(frame.max())],
            rgb_spatial_stddev=round(float(frame.std(axis=(0, 1)).max()), 4),
            action=action.tolist(),
            first_joint_initial_rad=initial_joint,
            first_joint_final_rad=final_joint,
            first_joint_signed_delta_rad=joint_delta,
        )
        result["rgb_passed"] = result["rgb_spatial_stddev"] > 0
        result["joint_motion_passed"] = abs(joint_delta) >= MOVEMENT_THRESHOLD_RAD
        if args.save_image:
            from imageio.v3 import imwrite

            image_path = args.output / "pickcube_rgb.png"
            imwrite(image_path, frame)
            result["image_path"] = str(image_path)
        result["passed"] = result["rgb_passed"] and result["joint_motion_passed"]
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="保存检查结果的目录。")
    parser.add_argument("--save-image", action="store_true", help="同时保存一张 RGB 图像。")
    parser.add_argument("--seed", type=int, default=0, help="环境随机种子。")
    parser.add_argument("--steps", type=int, default=8, help="执行的仿真控制步数。")
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        parser.error("--steps 必须在 1 到 20 之间。")
    if args.seed < 0:
        parser.error("--seed 必须为非负整数。")

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "environment": "PickCube-v1",
        "simulation_backend": "cpu",
        "render_backend": "cpu",
        "control_mode": "pd_joint_delta_pos",
        "seed": args.seed,
        "requested_steps": args.steps,
        "threshold_rad": MOVEMENT_THRESHOLD_RAD,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "passed": False,
    }
    for package in ("mani-skill", "sapien", "torch", "gymnasium"):
        try:
            result["packages"][package] = version(package)
        except PackageNotFoundError:
            result["packages"][package] = None

    try:
        run_check(args, result)
    except Exception as error:
        result["passed"] = False
        result["error"] = {"type": type(error).__name__, "message": str(error)}

    result_path = args.output / "pickcube_smoke.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result_path)
    if "error" in result:
        print(result["error"]["message"], file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
