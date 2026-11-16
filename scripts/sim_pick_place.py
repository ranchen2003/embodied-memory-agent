"""在本机验证 ManiSkill 抓放控制。方块和目标真值只用于控制诊断。"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree

import gymnasium as gym
import mani_skill.envs  # 注册 PickCube-v1。
import numpy as np
import pytorch_kinematics as pk
import torch
from imageio.v3 import imwrite
from mani_skill.utils.structs.pose import Pose
from scipy.spatial.transform import Rotation

OPEN = 1.0
CLOSED = -1.0
POSITION_TOLERANCE = 0.006
ORIENTATION_TOLERANCE = np.deg2rad(8)
MAX_TRANSLATION_STEP = 0.035
MAX_ROTATION_STEP = np.deg2rad(15)


def as_array(value):
    """取得单个环境的仿真观测或状态。"""
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def flag(value):
    return bool(as_array(value).reshape(-1)[0])


def position(pose):
    return as_array(pose.p).reshape(-1, 3)[0].copy()


def orientation(pose):
    matrix = as_array(pose.to_transformation_matrix()).reshape(-1, 4, 4)[0]
    return Rotation.from_matrix(matrix[:3, :3])


def save_rgb(observation, path):
    for sensor in observation["sensor_data"].values():
        if "rgb" in sensor:
            frame = as_array(sensor["rgb"])
            frame = frame[0] if frame.ndim == 4 else frame
            imwrite(path, frame)
            return {"path": path.name, "shape": list(frame.shape)}
    raise RuntimeError("仿真观测未返回 RGB。")


class PickPlaceCheck:
    """用公开末端位姿控制器闭环执行分阶段抓放并记录动作。"""

    def __init__(self, env, directory, record, native_goal, *, record_oracle=True):
        self.env = env
        self.task = env.unwrapped
        self.directory = directory
        self.record = record
        self.native_goal = native_goal
        self.record_oracle = record_oracle
        self.observation = None
        self.info = {}
        agent = self.task.agent
        # URDF 的动力学扩展字段由 SAPIEN 使用，逆解副本只需运动学部分。
        urdf = ElementTree.parse(agent.urdf_path).getroot()
        for joint in urdf.findall("joint"):
            dynamics = joint.find("dynamics")
            if dynamics is not None:
                joint.remove(dynamics)
        self.chain = pk.build_serial_chain_from_urdf(
            ElementTree.tostring(urdf), agent.ee_link_name
        ).to(device="cpu")
        if self.chain.get_joint_parameter_names() != agent.arm_joint_names:
            raise RuntimeError("Panda URDF 的关节顺序与仿真控制器不一致。")
        self.ik = pk.PseudoInverseIK(
            self.chain,
            joint_limits=torch.tensor(self.chain.get_joint_limits()).T,
            retry_configs=torch.zeros((1, 7)),
            max_iterations=35,
            lr=0.7,
            regularlization=1e-4,
            early_stopping_any_converged=True,
        )

    @property
    def tcp(self):
        return self.task.agent.tcp.pose

    def step(self, stage, world_p, world_rotation, gripper):
        # 把子目标转到 Panda 基座坐标。数值逆解使用 URDF，不依赖 Windows 未安装的 Pinocchio。
        robot = self.task.agent.robot
        base = robot.root.pose
        inverse_rotation = orientation(base).inv()
        local_p = inverse_rotation.apply(world_p - position(base))
        local_rot = (inverse_rotation * world_rotation).as_matrix()
        target_pose = pk.Transform3d(pos=local_p, rot=local_rot)
        self.ik.initial_config = robot.get_qpos()[:, :7].clone()
        solution = self.ik.solve(target_pose)
        if not flag(solution.converged):
            position_error = float(solution.err_pos.item())
            orientation_error = float(solution.err_rot.item())
            raise RuntimeError(f"{stage} 阶段逆解未收敛，位置差 {position_error:.4f} m，姿态差 {orientation_error:.3f} rad。")
        joint_targets = as_array(solution.solutions[0, 0])
        action = np.r_[joint_targets, gripper].astype(np.float32)
        self.observation, _, terminated, truncated, self.info = self.env.step(action)
        action_record = {
            "stage": stage,
            "action": action.tolist(),
            "target_tcp_xyz": world_p.tolist(),
            "tcp_xyz": position(self.tcp).tolist(),
            "terminated": flag(terminated),
            "truncated": flag(truncated),
        }
        if self.record_oracle:
            action_record.update(
                cube_xyz=position(self.task.cube.pose).tolist(),
                grasped=flag(self.info["is_grasped"]),
                success=flag(self.info["success"]),
            )
        self.record["actions"].append(action_record)
        success = self.record_oracle and flag(self.info["success"])
        if flag(truncated) and not success:
            raise RuntimeError(f"{stage} 阶段用完允许的仿真控制步数。")
        if self.record_oracle and flag(terminated) and not success:
            raise RuntimeError(f"{stage} 阶段仿真提前终止。")

    def move(self, stage, target_p, target_rotation, gripper):
        for _ in range(30):
            current_p = position(self.tcp)
            current_rotation = orientation(self.tcp)
            position_error = target_p - current_p
            distance = np.linalg.norm(position_error)
            rotation_error = (target_rotation * current_rotation.inv()).as_rotvec()
            angle = np.linalg.norm(rotation_error)
            if distance < POSITION_TOLERANCE and angle < ORIENTATION_TOLERANCE:
                return
            next_p = current_p + position_error * min(1.0, MAX_TRANSLATION_STEP / max(distance, 1e-8))
            next_rot = Rotation.from_rotvec(
                rotation_error * min(1.0, MAX_ROTATION_STEP / max(angle, 1e-8))
            ) * current_rotation
            self.step(stage, next_p, next_rot, gripper)
        raise RuntimeError(f"{stage} 阶段未到达目标末端姿态。")

    def hold(self, stage, target_p, target_rotation, gripper, steps):
        for _ in range(steps):
            if self.native_goal and stage == "settle" and flag(self.info["success"]):
                break
            self.step(stage, target_p, target_rotation, gripper)

    def checkpoint(self, stage):
        stage_info = {
            "stage": stage,
            "step": len(self.record["actions"]),
            "tcp_xyz": position(self.tcp).tolist(),
            "rgb": save_rgb(self.observation, self.directory / f"{stage}.png"),
        }
        if self.record_oracle:
            stage_info["cube_xyz"] = position(self.task.cube.pose).tolist()
            for key in ("is_grasped", "is_obj_placed", "is_robot_static", "success"):
                if key in self.info:
                    stage_info[key] = flag(self.info[key])
        self.record["stages"].append(stage_info)

    def run(self, seed):
        self.observation, self.info = self.env.reset(seed=seed)
        cube = self.task.cube
        initial_cube = position(cube.pose)
        goal_xyz = position(self.task.goal_site.pose)
        if not self.native_goal:
            # 原任务目标可能悬空。桌面诊断保持采样平面目标，改用物理可支撑的高度。
            goal_xyz[2] = initial_cube[2]
            self.task.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))
            self.info = self.task.get_info()
            self.observation = self.task.get_obs(self.info)
        self.record["target_xyz"] = goal_xyz.tolist()
        self.record["initial_cube_xyz"] = initial_cube.tolist()
        self.checkpoint("00_reset")

        # 从方块水平边缘中选与初始夹爪最接近的闭合方向，避免对角抓取。
        cube_directions = orientation(cube.pose).as_matrix()[:, :2].T
        initial_closing = orientation(self.tcp).as_matrix()[:, 1]
        candidates = np.concatenate((cube_directions, -cube_directions))
        closing = candidates[np.argmax(candidates @ initial_closing)].copy()
        closing[2] = 0
        closing /= np.linalg.norm(closing)
        grasp_pose = self.task.agent.build_grasp_pose(np.array([0, 0, -1]), closing, initial_cube)
        grasp_rotation = Rotation.from_matrix(grasp_pose.to_transformation_matrix()[:3, :3])
        up = np.array([0, 0, 0.09])

        self.move("pregrasp", initial_cube + up, grasp_rotation, OPEN)
        self.checkpoint("01_pregrasp")
        self.move("approach", initial_cube, grasp_rotation, OPEN)
        self.hold("close", initial_cube, grasp_rotation, CLOSED, 7)
        self.checkpoint("02_close")
        if not flag(self.info["is_grasped"]):
            raise RuntimeError("闭合夹爪后未检测到抓持。")

        self.move("lift", initial_cube + up, grasp_rotation, CLOSED)
        self.checkpoint("03_lift")
        self.record["lift_height_m"] = float(position(cube.pose)[2] - initial_cube[2])
        if not flag(self.info["is_grasped"]) or self.record["lift_height_m"] < 0.045:
            raise RuntimeError("抓持后未将方块稳定抬离桌面。")

        tcp_to_cube = position(cube.pose) - position(self.tcp)
        target_tcp = goal_xyz - tcp_to_cube
        if self.native_goal:
            self.move("transfer", target_tcp, grasp_rotation, CLOSED)
            self.hold("settle", target_tcp, grasp_rotation, CLOSED, 5)
            self.checkpoint("04_native_goal")
        else:
            self.move("transfer", target_tcp + up, grasp_rotation, CLOSED)
            self.checkpoint("04_above_target")
            self.move("descend", target_tcp, grasp_rotation, CLOSED)
            self.hold("release", target_tcp, grasp_rotation, OPEN, 7)
            self.checkpoint("05_release")
            self.move("retreat", target_tcp + up, grasp_rotation, OPEN)
            self.hold("settle", target_tcp + up, grasp_rotation, OPEN, 5)
            self.checkpoint("06_retreat")

        self.record["final_goal_error_m"] = float(np.linalg.norm(position(cube.pose) - goal_xyz))
        self.record["environment_success"] = flag(self.info["success"])
        self.record["released"] = not flag(self.info["is_grasped"])
        self.record["passed"] = self.record["environment_success"] and (
            self.native_goal or self.record["released"]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="结果、动作和阶段图像输出目录。")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="环境随机种子。")
    parser.add_argument("--native-goal", action="store_true", help="保留原环境可悬空的目标。")
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("种子需为互不重复的非负整数。")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    result = {
        "environment": "PickCube-v1",
        "simulation_backend": "cpu",
        "render_backend": "cpu",
        "control_mode": "pd_joint_pos",
        "action_format": "seven_joint_targets_rad_and_normalized_gripper",
        "pose_solver": "pytorch_kinematics_cpu",
        "goal_mode": "native" if args.native_goal else "tabletop_diagnostic",
        "position_source": "simulator_ground_truth_for_control_diagnostics",
        "max_episode_steps": 50 if args.native_goal else 120,
        "packages": {
            package: version(package)
            for package in ("mani-skill", "sapien", "torch", "pytorch-kinematics")
        },
        "runs": [],
    }
    env = gym.make(
        "PickCube-v1", num_envs=1, sim_backend="cpu", render_backend="cpu",
        obs_mode="rgb", render_mode="rgb_array", control_mode=result["control_mode"],
        max_episode_steps=result["max_episode_steps"],
    )
    try:
        for seed in args.seeds:
            directory = output / f"seed_{seed}"
            directory.mkdir(exist_ok=True)
            record = {"seed": seed, "passed": False, "stages": [], "actions": []}
            result["runs"].append(record)
            started = perf_counter()
            try:
                PickPlaceCheck(env, directory, record, args.native_goal).run(seed)
            except Exception as error:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
            record["elapsed_seconds"] = round(perf_counter() - started, 3)
    finally:
        env.close()

    result["passed"] = all(run["passed"] for run in result["runs"])
    result_path = output / "pick_place.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result_path)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
