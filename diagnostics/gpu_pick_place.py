"""Known-target RoboMME-Plus executor diagnostic using public RGB-D."""
import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from robomme_plus import BenchmarkEnvBuilder


def as_array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def latest(observation, name):
    return as_array(observation[f"{name}_list"][-1])


def camera_ray(row, col, intrinsic, extrinsic):
    rotation = extrinsic[:3, :3]
    origin = -rotation.T @ extrinsic[:3, 3]
    direction = rotation.T @ np.linalg.solve(intrinsic[:3, :3], [col, row, 1.0])
    return origin, direction


def world_at_height(row, col, height, intrinsic, extrinsic):
    origin, direction = camera_ray(row, col, intrinsic, extrinsic)
    return origin + (height - origin[2]) / direction[2] * direction


def check_camera(observation, view, output):
    rgb = latest(observation, f"{view}_rgb")
    depth = latest(observation, f"{view}_depth").squeeze()
    if np.ptp(rgb) == 0 or not np.any(np.isfinite(depth) & (depth > 0)):
        raise RuntimeError(f"{view} camera returned invalid RGB-D")
    imageio.imwrite(output / f"{view}.png", rgb)
    np.save(output / f"{view}_depth.npy", depth)
    return {"rgb_shape": list(rgb.shape), "valid_depth_pixels": int(np.count_nonzero(depth > 0))}


def write_frames(writer, observation):
    for front, wrist in zip(observation["front_rgb_list"], observation["wrist_rgb_list"], strict=True):
        writer.append_data(np.concatenate([as_array(front), as_array(wrist)], axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--source", nargs=2, type=int, default=(75, 155), metavar=("ROW", "COL"))
    parser.add_argument("--target", nargs=2, type=int, default=(93, 140), metavar=("ROW", "COL"))
    parser.add_argument("--cube-height", type=float, default=0.040)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "task": "VideoPlaceButton", "split": "train", "episode": 0,
        "action_space": "waypoint", "target_source": "official public training video annotations",
        "source_pixel_rc": args.source, "target_pixel_rc": args.target,
        "cube_height_m": args.cube_height, "yaw_rad": args.yaw,
        "mode": "gpu-only" if args.gpu_only else "known-target",
        "status": "initializing", "actions": [],
    }
    started = time.monotonic()
    env = None
    try:
        builder = BenchmarkEnvBuilder("VideoPlaceButton", dataset="train", action_space="waypoint")
        result["seed"] = builder.get_episode_metadata(0)["seed"]
        env = builder.make_env_for_episode(
            0, include_front_depth=True, include_wrist_depth=True,
            include_front_camera_extrinsic=True, include_front_camera_intrinsic=True,
        )
        backend = env.unwrapped.backend
        result["render_backend"] = str(backend.render_backend)
        result["render_device"] = str(backend.render_device)
        if "cuda" not in result["render_backend"].lower():
            raise RuntimeError(f"Expected GPU rendering, got {backend.render_backend}")
        print("GPU backend:", result["render_backend"], result["render_device"], flush=True)
        obs, info = env.reset()
        result["task_goal"] = info.get("task_goal")
        result["reset_frames"] = len(obs["front_rgb_list"])
        result["front"] = check_camera(obs, "front", args.output)
        result["wrist"] = check_camera(obs, "wrist", args.output)
        if args.gpu_only:
            result["status"] = "gpu_rgbd_verified"
            return 0

        intrinsic = as_array(info["front_camera_intrinsic"]).squeeze()
        extrinsic = latest(obs, "front_camera_extrinsic").squeeze()
        # The known source pixel is on the visible cube, the ring center lies at table height.
        source = world_at_height(*args.source, args.cube_height * 0.70, intrinsic, extrinsic)
        target = world_at_height(*args.target, 0.0, intrinsic, extrinsic)
        result["source_world"] = source.tolist()
        result["target_world"] = target.tolist()
        pick_height = args.cube_height / 2
        sequence = [
            ("approach_source", source, 0.17, 1),
            ("descend", source, pick_height, 1),
            ("grasp", source, pick_height, -1),
            ("lift", source, 0.17, -1),
            ("carry", target, 0.17, -1),
            ("lower", target, pick_height + 0.018, -1),
            ("release", target, pick_height + 0.018, 1),
            ("retreat", target, 0.17, 1),
        ]
        with imageio.get_writer(args.output / "episode.mp4", fps=20, codec="libx264") as writer:
            write_frames(writer, obs)
            for label, point, height, gripper in sequence:
                action = np.array([point[0], point[1], height, np.pi, 0.0, args.yaw, gripper])
                step = {"name": label, "waypoint": action.tolist()}
                result["actions"].append(step)
                print(label, action.tolist(), flush=True)
                obs, _, terminated, truncated, info = env.step(action)
                write_frames(writer, obs)
                step["status"] = info.get("status")
                step["frames"] = len(obs["front_rgb_list"])
                if bool(terminated) or bool(truncated):
                    break
        result["status"] = info.get("status", "ongoing")
        result["failure_reason"] = info.get("failure_reason")
        return 0 if result["status"] == "success" else 1
    except Exception as exc:
        result["status"] = "error"
        result["failure_reason"] = str(exc)
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            result["elapsed_s"] = round(time.monotonic() - started, 3)
            (args.output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
