"""用官方训练 H5 的单帧公开 RGB-D 观测验证目标反投影。"""

from argparse import ArgumentParser
import json
from pathlib import Path

import h5py
import numpy as np

from plus_waypoint import pixel_to_world, world_to_pixel


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="本地 H5 路径或固定版本的公开 HTTPS 地址")
    parser.add_argument("--content-length", type=int, help="远程 H5 文件的确切字节数")
    parser.add_argument("--episode", required=True, type=int, help="当前任务的公开训练回合编号")
    parser.add_argument("--frame", required=True, type=int, help="H5 中的实际帧号")
    parser.add_argument("--pixel", required=True, type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--snapshot", type=Path, help="可选，写入单帧公开深度和内外参")
    args = parser.parse_args()
    if args.episode < 0 or args.frame < 0:
        parser.error("回合和帧号必须为非负数")
    if args.h5.startswith("https://"):
        if args.content_length is None or args.content_length <= 0:
            parser.error("范围读取远程 H5 时需指定确切字节数")
        import fsspec

        filesystem = fsspec.filesystem("https", client_kwargs={"trust_env": True})
        source = filesystem.open(args.h5, "rb", block_size=2**20, cache_type="blockcache", size=args.content_length)
    else:
        source = open(args.h5, "rb")

    with source, h5py.File(source, "r") as data:
        episode = data[f"episode_{args.episode}"]
        obs = episode[f"timestep_{args.frame}/obs"]
        depth = obs["front_depth"][()]
        intrinsics = episode["setup/front_camera_intrinsic"][()]
        extrinsics = obs["front_camera_extrinsic"][()]
        world_xyz = pixel_to_world(depth, intrinsics, extrinsics, args.pixel)
        reprojected_xy = world_to_pixel(intrinsics, extrinsics, world_xyz)
        if args.snapshot is not None:
            with args.snapshot.open("xb") as file:
                np.savez_compressed(file, depth_mm=depth, intrinsic_cv=intrinsics, extrinsic_cv=extrinsics,
                                    pixel_xy=np.asarray(args.pixel))
        cache = getattr(source, "cache", None)
        print(json.dumps({"episode": args.episode, "frame": args.frame, "pixel_xy": args.pixel,
                          "world_xyz_m": world_xyz.tolist(), "depth_shape": list(depth.shape),
                          "depth_dtype": str(depth.dtype), "reprojected_xy": reprojected_xy.tolist(),
                          "range_bytes": cache.total_requested_bytes if cache is not None else None},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
