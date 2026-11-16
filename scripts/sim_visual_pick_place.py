"""用前视 RGB-D 定位桌面方块和绿色目标，再闭环执行抓取、放置和退离。"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2
import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sim_pick_place import CLOSED, OPEN, PickPlaceCheck, as_array, flag, orientation, position

CAMERA = "base_camera"
IMAGE_SIZE = 320
CUBE_HALF_SIZE = 0.020
GOAL_RADIUS = 0.025
TABLE_Z = 0.0
LIFT = np.array([0.0, 0.0, 0.09])


def configure_visible_goal(task, seed):
    """布置公开可见的桌面目标。坐标由环境种子生成，仅传入仿真环境。"""
    rng = np.random.default_rng(np.random.SeedSequence([2026, seed]))
    goal = [rng.uniform(-0.075, 0.02), rng.choice([-1, 1]) * rng.uniform(0.16, 0.175),
            TABLE_Z + CUBE_HALF_SIZE]
    task.goal_site.set_pose(Pose.create_from_pq(p=goal))
    if task.goal_site in task._hidden_objects:
        task._hidden_objects.remove(task.goal_site)
    task.goal_site.show_visual()


def colored_points(observation, color):
    """从颜色连通域读取深度，并用公开相机参数转成世界坐标。"""
    view = observation["sensor_data"][CAMERA]
    rgb = as_array(view["rgb"])[0].astype(np.float32)
    depth_mm = as_array(view["depth"])[0, :, :, 0]
    red, green, blue = np.moveaxis(rgb, -1, 0)
    if color == "red":
        mask = (red > 90) & (red > 1.9 * green) & (red > 1.9 * blue)
    else:
        mask = (green > 85) & (green > 1.65 * red) & (green > 1.65 * blue)
    mask &= depth_mm > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count == 1:
        raise RuntimeError(f"前视图中没有定位到 {color} 物体。")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    v, u = np.nonzero(labels == component)
    if len(u) < 24:
        raise RuntimeError(f"{color} 物体可用深度点太少（{len(u)}）。")

    calibration = observation["sensor_param"][CAMERA]
    intrinsics = as_array(calibration["intrinsic_cv"])[0]
    extrinsics = as_array(calibration["extrinsic_cv"])[0]
    rays = np.column_stack((u, v, np.ones_like(u))) @ np.linalg.inv(intrinsics).T
    camera_points = rays * (depth_mm[v, u, None] / 1000)
    # OpenCV 外参为世界到相机，行向量点云用右乘恢复到世界坐标。
    world_points = (camera_points - extrinsics[:3, 3]) @ extrinsics[:3, :3]
    return world_points, {"pixel_uv": [float(np.median(u)), float(np.median(v))], "point_count": len(u)}


def locate_cube(observation):
    """在已知尺寸的方块顶面拟合平面轮廓，恢复平面中心与边方向。"""
    points, report = colored_points(observation, "red")
    top_z = TABLE_Z + 2 * CUBE_HALF_SIZE
    top_face = points[np.abs(points[:, 2] - top_z) < 0.0025, :2].astype(np.float32)
    if len(top_face) < 16:
        raise RuntimeError(f"方块顶面有效深度点不足（{len(top_face)}）。")
    center, sides, angle = cv2.minAreaRect(top_face)
    if min(sides) < 0.025 or max(sides) > 0.048:
        raise RuntimeError(f"红色区域的顶面尺寸与方块不符（{sides}）。")
    theta = np.deg2rad(angle)
    first = np.array([np.cos(theta), np.sin(theta), 0.0])
    second = np.array([-np.sin(theta), np.cos(theta), 0.0])
    cube = np.array([center[0], center[1], TABLE_Z + CUBE_HALF_SIZE])
    report.update(xyz=cube.tolist(), top_face_points=len(top_face), edge_angle_deg=float(angle))
    return cube, np.array([first, second]), report


def locate_goal(observation):
    """用绿色球的部分深度点拟合球面，避免把可见球面当作目标中心。"""
    points, report = colored_points(observation, "green")
    z = TABLE_Z + CUBE_HALF_SIZE

    def sphere_residual(xy):
        return np.linalg.norm(points - [xy[0], xy[1], z], axis=1) - GOAL_RADIUS

    fit = least_squares(sphere_residual, np.median(points[:, :2], axis=0), loss="soft_l1", f_scale=0.002)
    error = float(np.median(np.abs(fit.fun)))
    if not fit.success or error > 0.004:
        raise RuntimeError(f"绿色目标球面拟合失败，误差 {error:.4f} m。")
    goal = np.r_[fit.x, z]
    report.update(xyz=goal.tolist(), sphere_fit_median_error_m=error)
    return goal, report


class VisualPickPlace(PickPlaceCheck):
    """执行器沿用关节控制和数值逆解，物体定位仅使用传感观测。"""

    video_writer = None

    def video_frame(self):
        if self.video_writer is not None:
            rgb = as_array(self.observation["sensor_data"][CAMERA]["rgb"])[0]
            self.video_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.record["video"]["frames"] += 1

    def step(self, stage, world_p, world_rotation, gripper):
        super().step(stage, world_p, world_rotation, gripper)
        self.video_frame()

    def prepare(self, seed, record_video=False):
        self.env.reset(seed=seed)
        configure_visible_goal(self.task, seed)
        self.observation = self.task.get_obs()
        self.checkpoint("00_reset")
        if record_video:
            path = self.directory / "front_camera.mp4"
            fps = int(self.task.control_freq)
            self.video_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                                (IMAGE_SIZE, IMAGE_SIZE))
            if not self.video_writer.isOpened():
                raise RuntimeError("无法写入仿真相机回放")
            self.record["video"] = {"path": path.name, "fps": fps, "frames": 0}
            self.video_frame()

        cube, edges, cube_report = locate_cube(self.observation)
        goal, goal_report = locate_goal(self.observation)
        self.record["perception"] = {"cube": cube_report, "goal": goal_report}
        return cube, edges, goal

    def pick(self, cube, edges):
        current_closing = orientation(self.tcp).as_matrix()[:, 1]
        candidates = np.concatenate((edges, -edges))
        closing = candidates[np.argmax(candidates @ current_closing)]
        grasp = self.task.agent.build_grasp_pose(np.array([0, 0, -1]), closing, cube)
        grasp_rotation = Rotation.from_matrix(grasp.to_transformation_matrix()[:3, :3])

        self.move("pregrasp", cube + LIFT, grasp_rotation, OPEN)
        self.checkpoint("01_pregrasp")
        self.move("approach", cube, grasp_rotation, OPEN)
        self.hold("close", cube, grasp_rotation, CLOSED, 7)
        self.checkpoint("02_close")
        grasp_gap = float(as_array(self.observation["agent"]["qpos"]).reshape(-1)[-2:].sum())
        self.record["gripper_gap_after_close_m"] = grasp_gap
        if not 0.025 <= grasp_gap <= 0.058:
            raise RuntimeError(f"夹爪开度 {grasp_gap:.4f} m 与持有此尺寸方块不符。")

        # 动作只使用视觉定位和机械臂本体观测。物体被遮挡时沿用抓取时的估计。
        grasp_offset = cube - position(self.tcp)
        self.move("lift", cube + LIFT, grasp_rotation, CLOSED)
        self.checkpoint("03_lift")
        return grasp_offset, grasp_rotation

    def place(self, goal, grasp_offset, grasp_rotation):
        target_tcp = goal - grasp_offset
        self.move("transfer", target_tcp + LIFT, grasp_rotation, CLOSED)
        self.checkpoint("04_above_target")
        self.move("descend", target_tcp, grasp_rotation, CLOSED)
        self.hold("release", target_tcp, grasp_rotation, OPEN, 7)
        self.checkpoint("05_release")
        self.move("retreat", target_tcp + LIFT, grasp_rotation, OPEN)
        self.hold("settle", target_tcp + LIFT, grasp_rotation, OPEN, 5)
        self.checkpoint("06_retreat")

    def run(self, seed, record_video=False):
        cube, edges, goal = self.prepare(seed, record_video)
        grasp_offset, rotation = self.pick(cube, edges)
        self.place(goal, grasp_offset, rotation)


def evaluate(env):
    """整条执行结束后由环境评估，结果不参与任何动作生成。"""
    task = env.unwrapped
    info = task.evaluate()
    cube_truth = position(task.cube.pose)
    target_truth = position(task.goal_site.pose)
    released = not flag(info["is_grasped"])
    return {
        "environment_success": flag(info["success"]),
        "released": released,
        "passed": flag(info["success"]) and released,
        "final_cube_xyz": cube_truth.tolist(),
        "true_goal_xyz": target_truth.tolist(),
        "final_goal_error_m": float(np.linalg.norm(cube_truth - target_truth)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="报告和 RGB 阶段图目录。")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="场景种子列表。")
    parser.add_argument("--record-video", action="store_true", help="逐控制步保存前视相机回放。")
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("种子需为互不重复的非负整数。")
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    result = {
        "environment": "PickCube-v1 visible tabletop goal diagnostic",
        "simulation_backend": "cpu",
        "render_backend": "cpu",
        "control_mode": "pd_joint_pos",
        "max_episode_steps": 120,
        "input": ["front RGB", "front depth", "camera calibration", "robot proprioception and URDF"],
        "camera": CAMERA,
        "image_size": IMAGE_SIZE,
        "packages": {name: version(name) for name in ("mani-skill", "sapien", "torch", "pytorch-kinematics")},
        "runs": [],
    }
    env = gym.make(
        "PickCube-v1", num_envs=1, sim_backend="cpu", render_backend="cpu", obs_mode="rgbd",
        render_mode="rgb_array", control_mode=result["control_mode"],
        max_episode_steps=result["max_episode_steps"],
        sensor_configs={CAMERA: {"width": IMAGE_SIZE, "height": IMAGE_SIZE}},
    )
    try:
        for seed in args.seeds:
            directory = out_dir / f"seed_{seed}"
            directory.mkdir(exist_ok=True)
            record = {"seed": seed, "passed": False, "stages": [], "actions": []}
            result["runs"].append(record)
            started = perf_counter()
            controller = None
            try:
                controller = VisualPickPlace(env, directory, record, native_goal=False, record_oracle=False)
                controller.run(seed, args.record_video)
                record.update(evaluate(env))
            except Exception as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
            finally:
                if controller is not None and controller.video_writer is not None:
                    controller.video_writer.release()
            record["elapsed_seconds"] = round(perf_counter() - started, 3)
    finally:
        env.close()

    result["passed"] = all(run["passed"] for run in result["runs"])
    path = out_dir / "visual_pick_place.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
