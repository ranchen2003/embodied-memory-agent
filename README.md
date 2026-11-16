# 具身智能体任务记忆实验

课题为《基于记忆增强的具身智能体任务决策方法研究与实现》。当前入口用于检查机械臂示范历史的保留情况。运行只需要 Python 和 CPU。

## 从本机运行

本机的 `final_work` 是总工作区。`embodied-memory-agent/` 是独立 Git 仓库，`experiments/` 管理数据、运行产物和仿真环境。在源码仓库根目录执行以下命令。已有对应依赖时直接运行脚本。

```powershell
python -m pip install -r requirements.txt
python scripts/offline_pilot.py --work-dir ../experiments --budget 8
python scripts/offline_pilot.py --work-dir ../experiments --budget 24
```

首次运行从 Hugging Face 读取固定版本的 RoboMME-Plus 公开训练数据，以后复用指定目录下的 `data/` 缓存。每次运行在该目录的 `runs/pilot/budget_*/` 生成逐回合的 `metrics.csv` 和带采样帧、参考区间及命中情况的 `trace.json`。省略 `--work-dir` 时，脚本使用仓库内已忽略的 `.local/` 目录。[公开数据](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot) · [数据说明](https://github.com/RoboMME/robomme_benchmark/blob/main/doc/plus_training_data_format.md)

| 路径 | 内容 |
|---|---|
| `scripts/offline_pilot.py` | 读取示范历史并运行等帧数对照 |
| `scripts/run_place_diagnostic.py` | 分开评价证据保留、历史事件读取和当前落点 |
| `docs/first-month.md` | 已有结果、实验口径和组会计划 |
| `docs/first-month-results.md` | 本机抓放、逐项对照和阶段自评 |
| `../experiments/data/` | 自动生成的公开轨迹缓存 |
| `../experiments/runs/` | 自动生成的结果和逐回合记录 |

Git 记录脚本、依赖与实验说明。数据、权重、运行产物和本机密钥通过 `.gitignore` 保留在本机。正式评测计划覆盖 RoboMME-Plus 的全部 16 个任务与 800 个固定测试回合。[基准协议](https://robomme.github.io/plus.html)

## 历史放置和当前圆环

在同六条公开开发轨迹上复跑历史放置诊断。脚本缓存前视视频和任务指令，预测读取到第一帧执行画面为止。开发标注的角色和负结果见[本阶段结果](docs/first-month-results.md)。

```powershell
python -m pip install -r requirements-vision.txt
python scripts/run_place_diagnostic.py --work-dir ../experiments --stride 1
python scripts/run_place_diagnostic.py --work-dir ../experiments --stride 2
```

结果保存到 `../experiments/runs/place-diagnostic/stride_*/trace.json`，逐例记录保留帧、事件、时间条件、落点、候选排序和误差。

在生成 `stride_1/trace.json` 后，可进一步比较单轨关联、换位片段关联和同预算回看。

```powershell
python scripts/run_exchange_memory.py --work-dir ../experiments --stride 2
python scripts/run_exchange_memory.py --work-dir ../experiments --stride 4
python scripts/run_exchange_memory.py --work-dir ../experiments --stride 12
python scripts/run_exchange_memory.py --work-dir ../experiments --stride 2 --original-manifest ../experiments/data/original-displaced/manifest.json
```

原版定向核验需要按 `configs/original_displaced_manifest.json` 提取的四段公开前视视频。本机将清单和视频保存在 `../experiments/data/original-displaced/`。清单记录源视频、起始时间、帧数和解码哈希。Plus 开发诊断和原版定向压力核验分别输出到 `../experiments/runs/exchange-memory/`，评价口径见[本阶段结果](docs/first-month-results.md)。

## Windows 单环境仿真验证

`scripts/check_local_sim.py` 在 ManiSkill 的 `PickCube-v1` 上重置 Panda 环境，发送关节指令并检查新观测和实际关节位移。验证配置使用 ManiSkill 3.0.1、SAPIEN 3.0.3、CPU 物理仿真和 CPU 图像读回。实验环境位于 `../experiments/envs/maniskill-win`，命令从源码仓库根目录运行。

```powershell
..\experiments\envs\maniskill-win\Scripts\python.exe scripts/check_local_sim.py --output ../experiments/runs/local-sim --save-image
```

运行会记录版本、关节位移、RGB 形状和耗时。该检查覆盖本机的观测与动作接口，官方 Plus 全量和任务完成率由独立闭环评测验证。[ManiSkill 平台支持](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support)

以下两个命令检查完整抓放和原版 PickCube 目标。两者均使用仿真状态辅助控制，结果可用于定位执行器的问题。

```powershell
..\experiments\envs\maniskill-win\Scripts\python.exe scripts/sim_pick_place.py --output ../experiments/runs/local-pick-place
..\experiments\envs\maniskill-win\Scripts\python.exe scripts/sim_pick_place.py --native-goal --output ../experiments/runs/local-pick-place/native
```

视觉桌面测试从可见红方块和绿球的 RGB-D 观测生成抓取和放置坐标。可直接保存两个调试场景和两个固定配置后的验证场景的回放。

```powershell
..\experiments\envs\maniskill-win\Scripts\python.exe scripts/sim_visual_pick_place.py --seeds 0 1 2 3 --record-video --output ../experiments/runs/visual-pick-place
```
