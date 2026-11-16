"""由公开相机观测计算三维目标，并组装官方位姿动作。"""

from numbers import Integral

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _calibration(intrinsic_cv, extrinsic_cv):
    intrinsics = _array(intrinsic_cv).astype(float)
    extrinsics = _array(extrinsic_cv).astype(float)
    if intrinsics.shape != (3, 3) or extrinsics.shape not in {(3, 4), (4, 4)}:
        raise ValueError("相机内参应为 3×3，外参应为 3×4 或 4×4")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("相机内外参必须为有限值")
    return intrinsics, extrinsics[:3, :3], extrinsics[:3, 3]


def pixel_to_world(depth_mm, intrinsic_cv, extrinsic_cv, pixel_xy, *, radius=2):
    """将前视图中的列、行坐标反投影至世界坐标，返回以米为单位的 xyz。"""
    image = _array(depth_mm)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2 or not image.size:
        raise ValueError("深度图应为 H×W 或 H×W×1")
    point = _array(pixel_xy).astype(float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("像素坐标应为有限的 [列, 行]")
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0:
        raise ValueError("采样半径应为非负整数")
    width_height = np.asarray(image.shape[::-1])
    if (point < 0).any() or (point >= width_height).any():
        raise ValueError("目标像素超出前视图")
    x, y = np.minimum(np.rint(point).astype(int), width_height - 1)
    sample = image[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    valid = sample[np.isfinite(sample) & (sample > 0)]
    if not valid.size:
        raise ValueError("目标像素邻域没有可用深度")

    intrinsics, rotation, translation = _calibration(intrinsic_cv, extrinsic_cv)
    ray = np.linalg.solve(intrinsics, np.r_[point, 1.0])
    depth_m = float(np.median(valid)) / 1000
    camera_xyz = ray * (depth_m / ray[2])
    # OpenCV 外参为世界到相机，相机坐标转换需应用其逆变换。
    return np.linalg.solve(rotation, camera_xyz - translation)


def world_to_pixel(intrinsic_cv, extrinsic_cv, world_xyz):
    """将世界点投影回图像，供目标和标定联调核查。"""
    intrinsics, rotation, translation = _calibration(intrinsic_cv, extrinsic_cv)
    camera_xyz = rotation @ _vector(world_xyz, "世界坐标") + translation
    if camera_xyz[2] <= 0:
        raise ValueError("世界点位于相机背面")
    projection = intrinsics @ camera_xyz
    return projection[:2] / projection[2]


def ground_pixel(observation, info, pixel_xy, *, radius=2):
    """使用官方环境公开的当前深度和内外参，不读取任务真值。"""
    return pixel_to_world(
        observation["front_depth_list"][-1],
        info["front_camera_intrinsic"],
        observation["front_camera_extrinsic_list"][-1],
        pixel_xy,
        radius=radius,
    )


def _vector(values, name):
    vector = _array(values).astype(float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name}应包含三个有限值")
    return vector


def waypoint(position_xyz, rpy, gripper):
    """生成官方 7 维 waypoint，单位为米和外旋 XYZ 欧拉角弧度。"""
    if isinstance(gripper, (bool, np.bool_)) or gripper not in (-1, 1):
        raise ValueError("夹爪命令应为 -1（闭合）或 1（张开）")
    return np.r_[_vector(position_xyz, "世界坐标"), _vector(rpy, "末端姿态"), gripper]
