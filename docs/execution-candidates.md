# Plus 执行候选与首月预检

两条执行路线接受同样的上层指令，包括操作方式、当前帧中的对象和目标像素，以及关联的历史证据。运行之前用正确的目标单独诊断执行器，随后切换为记忆决策的输出。比较记忆方法时固定执行器和预算，两条执行路线的成功率、可覆盖操作和开销分别记录。

| 候选 | 输入与动作 | 验证范围 |
| --- | --- | --- |
| 视觉定位与官方 waypoint | 公开前视深度和相机内外参将当前目标像素反投影到世界坐标。执行器还需构造抓取姿态与动作顺序，再向官方接口提交七维位姿与夹爪命令。 | 已验证单帧反投影和位姿封装。抓放动作及完整回合需在官方环境实测。 |
| 冻结 GroundSG π₀.₅ VLA | 将对象和目标转换为带像素位置的子目标，结合前视、腕视、本体状态和任务说明输入官方模型。模型输出关节目标和夹爪动作序列。 | 上游有固定权重与服务入口，训练权重归档约 11.55 GB。完整上游三层部署使用两张 GPU，OpenPI 官方单卡推理要求超过 8 GB 显存并验证 Ubuntu 22.04。本机为 RTX 5060 Laptop GPU，报告 8151 MiB。实际推理部署需另行验证。 |

官方 Plus 预检固定版本为 `777c595d9c443405fdc145452929bc5890c13e19`。Windows 使用现成 ManiSkill 环境可以导入 `BenchmarkEnvBuilder` 和读取训练回合元数据。构建 `VideoPlaceButton` 训练回合 0 时，`DemonstrationWrapper` 的运动规划依赖缺少 `mplib`，因此未进入重置或执行。此故障位于官方环境初始化，选择不同策略并不能消除该依赖。

在已有官方仿真依赖的环境直接执行以下预检。若从官方源码导入，传入 `--source /path/to/robomme_benchmark`。

```text
python probe_plus_sim.py --task VideoPlaceButton --episode 0 --send-hold
```

公开 H5 单帧验证使用 `VideoPlaceButton/episode_2/timestep_2005`，对应全局轨迹 32。预先得到的目标像素 `(117.5, 76.2037)` 反投影得到 `(-0.144224, -0.045035, 0.008311)` 米，回投影与输入像素一致。取样使用固定发布版本 `dc3b14019363baa7f869dd2f91245674922de1d8`，按需读取约 7 MiB。该结果验证官方数据的坐标与单位。

```text
python probe_plus_h5.py --h5 https://huggingface.co/datasets/Yinpei/robomme_ood_data_h5/resolve/dc3b14019363baa7f869dd2f91245674922de1d8/h5/record_dataset_VideoPlaceButton.h5 --content-length 1853592858 --episode 2 --frame 2005 --pixel 117.5 76.2037037037
```

[官方动作与观测](https://github.com/RoboMME/robomme_benchmark/blob/777c595d9c443405fdc145452929bc5890c13e19/doc/env_format.md) · [官方数据格式](https://github.com/RoboMME/robomme_benchmark/blob/777c595d9c443405fdc145452929bc5890c13e19/doc/plus_training_data_format.md) · [GroundSG 官方执行示例](https://github.com/RoboMME/robomme_benchmark/blob/777c595d9c443405fdc145452929bc5890c13e19/scripts/plus_eval/3-tier-astra/README.md) · [OpenPI 环境要求](https://github.com/Physical-Intelligence/openpi#requirements)
