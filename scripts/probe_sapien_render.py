"""在指定 SAPIEN 设备上实测无窗口相机渲染。"""

from argparse import ArgumentParser
import importlib.metadata
import json
import sys

import numpy as np
import sapien


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--enumerate-only", action="store_true")
    args = parser.parse_args()
    report = {"sapien": importlib.metadata.version("sapien"), "device": args.device}
    stage = "enumerate"
    try:
        report["devices"] = sapien.render.get_device_summary()
        if args.enumerate_only:
            print(json.dumps(report, ensure_ascii=False))
            return 0
        stage = "select"
        device = sapien.Device(args.device)
        report.update(device_name=device.name, is_cuda=device.is_cuda(), can_render=device.can_render())
        stage = "renderer"
        renderer = sapien.render.RenderSystem(device)
        stage = "picture"
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), renderer])
        scene.set_ambient_light([0.7, 0.7, 0.7])
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=[1.0, 0.3, 0.2])
        builder.set_initial_pose(sapien.Pose(p=[1.0, 0.0, 0.0]))
        builder.build_kinematic("probe_cube")
        camera = scene.add_camera("probe_camera", 64, 64, np.pi / 2, 0.01, 10.0)
        scene.update_render()
        camera.take_picture()
        image = camera.get_picture("Color")
        rgb = image[..., :3]
        if image.shape[:2] != (64, 64) or not np.isfinite(rgb).all() or float(np.ptp(rgb)) <= 0:
            raise RuntimeError("未读到有效的相机图像")
        report.update(ok=True, image_shape=list(image.shape), rgb_range=[float(rgb.min()), float(rgb.max())])
    except Exception as error:
        report.update(ok=False, stage=stage, error_type=type(error).__name__, error=str(error))
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
