"""用公开 RGB-D 观测规划桌面方块一抓一放，调用 Plus waypoint 接口。"""

from numbers import Integral

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from plus_waypoint import ground_pixel, waypoint


WAYPOINT_COUNT = 8
TABLE_Z = 0.0  # Plus 公开世界坐标约定。


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _vector(value, size, field):
    result = _numpy(value).astype(float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{field} 的观测尺寸或取值无效")
    return result


def _cube_top(observation, info, pixel):
    depth = _numpy(observation["front_depth_list"][-1])
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("前视深度必须为非空的 H×W 或 H×W×1")
    height, width = depth.shape
    if (pixel < 0).any() or pixel[0] >= width or pixel[1] >= height:
        raise ValueError("抓取像素超出前视图")

    x, y = np.minimum(np.rint(pixel).astype(int), [width - 1, height - 1])
    x_min, y_min = max(0, x - 14), max(0, y - 14)
    columns, rows = np.meshgrid(np.arange(x_min, min(width, x + 15)),
                                np.arange(y_min, min(height, y + 15)))
    millimetres = depth[rows, columns].astype(float)
    valid = np.isfinite(millimetres) & (millimetres > 0)
    intrinsics = _numpy(info["front_camera_intrinsic"]).astype(float)
    extrinsics = _numpy(observation["front_camera_extrinsic_list"][-1]).astype(float)
    if intrinsics.shape != (3, 3) or extrinsics.shape not in {(3, 4), (4, 4)}:
        raise ValueError("相机内外参尺寸无效")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("相机内外参须为有限值")

    homogeneous = np.stack((columns, rows, np.ones_like(columns)), axis=-1)
    rays = homogeneous @ np.linalg.inv(intrinsics).T
    camera_points = rays * (np.where(valid, millimetres, 0.0)[..., None] / 1000 / rays[..., 2:])
    world_points = ((camera_points - extrinsics[:3, 3])
                    @ np.linalg.inv(extrinsics[:3, :3]).T)
    world_z = world_points[..., 2]
    floor = valid & (np.abs(world_z - TABLE_Z) <= 0.005)
    support_z = float(np.median(world_z[floor])) if floor.any() else TABLE_Z
    foreground = valid & (world_z - support_z >= 0.012) & (world_z - support_z <= 0.085)
    if not foreground.any():
        raise ValueError("抓取像素附近没有高出桌面的有效深度点")

    # 从目标像素附近的连通域取顶面，避免把另一方块或桌面算入尺寸。
    _, components = cv2.connectedComponents(foreground.astype(np.uint8))
    pixel_distances = (columns - pixel[0]) ** 2 + (rows - pixel[1]) ** 2
    nearest = np.unravel_index(np.argmin(np.where(foreground, pixel_distances, np.inf)), foreground.shape)
    if pixel_distances[nearest] > 16:
        raise ValueError("有效物体深度与抓取像素相距超过四个像素")
    points = world_points[components == components[nearest]]
    top_z = float(np.quantile(points[:, 2], 0.9))
    object_height = top_z - support_z
    points_top = points[np.abs(points[:, 2] - top_z) <= 0.004, :2]
    if len(points_top) < 12 or not 0.017 <= object_height <= 0.080:
        raise ValueError("方块顶面点数或高度不足以产生可靠抓取")

    center, dimensions, angle_deg = cv2.minAreaRect(points_top.astype(np.float32))
    thin, wide = sorted(dimensions)
    if thin < 0.014 or wide > 0.075 or wide / thin > 1.9:
        raise ValueError("目标顶面尺寸不适合当前平行夹爪")
    angle = np.deg2rad(angle_deg)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    orthogonal = np.array([-axis[1], axis[0], 0.0])
    cube = np.array([center[0], center[1], support_z + object_height / 2])
    return cube, (axis, orthogonal), object_height, (thin, wide), len(points_top)


def plan_pick_place(observation, info, *, pick_pixel_xy, place_pixel_xy):
    """由公开像素和标定给出八个绝对 7 维 waypoint，不读取任务求解信息。"""
    pick_pixel = _vector(pick_pixel_xy, 2, "抓取像素")
    place_pixel = _vector(place_pixel_xy, 2, "放置像素")
    cube, axes, height, dimensions, top_points = _cube_top(observation, info, pick_pixel)
    site = ground_pixel(observation, info, place_pixel)
    if not np.isfinite(site).all() or not TABLE_Z - 0.010 <= site[2] <= TABLE_Z + 0.030:
        raise ValueError("目标像素深度不在可放置的桌面高度范围")
    tcp = _vector(observation["eef_state_list"][-1], 6, "末端位姿")
    current_closing = Rotation.from_euler("xyz", tcp[3:]).as_matrix()[:, 1]
    candidates = np.asarray([*axes, -axes[0], -axes[1]])
    closing = candidates[np.argmax(candidates @ current_closing)]
    approaching = np.array([0.0, 0.0, -1.0])
    # Panda TCP 的第二轴是夹紧方向，第三轴是接近方向。
    orientation = np.column_stack((np.cross(closing, approaching), closing, approaching))
    rpy = Rotation.from_matrix(orientation).as_euler("xyz")

    release = site + [0.0, 0.0, height / 2 + 0.003]
    travel_height = max(0.15, cube[2] + 0.09, release[2] + 0.09)
    above_pick = np.r_[cube[:2], travel_height]
    above_site = np.r_[site[:2], travel_height]
    steps = (
        ("pregrasp", above_pick, 1), ("approach", cube, 1), ("grasp", cube, -1),
        ("lift", above_pick, -1), ("transfer", above_site, -1), ("lower", release, -1),
        ("release", release, 1), ("retreat", above_site, 1),
    )
    report = {"pick_pixel_xy": pick_pixel.tolist(), "place_pixel_xy": place_pixel.tolist(),
              "cube_xyz_m": cube.tolist(), "site_xyz_m": site.tolist(), "release_xyz_m": release.tolist(),
              "cube_height_m": height, "cube_top_size_m": list(dimensions), "top_points": top_points,
              "topdown_rpy_rad": rpy.tolist()}
    return [(stage, waypoint(position, rpy, gripper)) for stage, position, gripper in steps], report


def _flag(value):
    if value is None:
        return False
    values = _numpy(value).reshape(-1)
    return bool(values.size and values[-1])


def execute_pick_place(env, observation, info, *, pick_pixel_xy, place_pixel_xy,
                       max_waypoints=WAYPOINT_COUNT, observe_step=None):
    """执行固定计划，记录每步公开反馈，由官方回报决定任务是否成功。"""
    if isinstance(max_waypoints, bool) or not isinstance(max_waypoints, Integral) or max_waypoints < 0:
        raise ValueError("waypoint 预算须为非负整数")
    trace = {"status": "not_done", "official_status": info.get("status"),
             "waypoint_limit": int(max_waypoints), "commands": []}
    if trace["official_status"] in {"success", "fail", "timeout", "error"}:
        trace["status"] = trace["official_status"]
        return observation, info, trace
    try:
        actions, geometry = plan_pick_place(observation, info, pick_pixel_xy=pick_pixel_xy,
                                             place_pixel_xy=place_pixel_xy)
    except (KeyError, IndexError, TypeError, ValueError, np.linalg.LinAlgError, cv2.error) as error:
        trace.update(status="geometry_error", error=str(error))
        return observation, info, trace
    trace["geometry"] = geometry

    for stage, action in actions[:max_waypoints]:
        record = {"stage": stage, "waypoint": action.tolist()}
        trace["commands"].append(record)
        try:
            observation, _, terminated, truncated, info = env.step(action)
        except (RuntimeError, ValueError) as error:
            trace.update(status="error", error=str(error))
            break
        status = info.get("status")
        rejected = _flag(info.get("action_error"))
        record.update(official_status=status, terminated=_flag(terminated), truncated=_flag(truncated),
                      action_error=rejected)
        control_steps = info.get("execution_total_control_steps")
        if control_steps is not None:
            record["control_steps"] = int(_numpy(control_steps).reshape(-1)[-1])
        gripper_frames = observation.get("gripper_state_list")
        if gripper_frames is not None and len(gripper_frames):
            gap = float(_numpy(gripper_frames[-1]).sum())
            if np.isfinite(gap):
                record["gripper_gap_m"] = gap
        trace["official_status"] = status
        if observe_step is not None:
            observe_step(stage, observation, info)

        if rejected:
            trace["status"] = "action_error"
            break
        if status in {"success", "fail", "timeout", "error"}:
            trace["status"] = status
            break
        if record["terminated"] or record["truncated"]:
            trace["status"] = "terminated" if record["terminated"] else "truncated"
            break
        if stage in {"grasp", "lift"} and "gripper_gap_m" in record:
            if record["gripper_gap_m"] < 0.5 * geometry["cube_top_size_m"][0]:
                trace["status"] = "grasp_unconfirmed"
                break
    else:
        if len(trace["commands"]) < len(actions):
            trace["status"] = "waypoint_limit"
    return observation, info, trace
