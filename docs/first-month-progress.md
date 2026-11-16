# 首月实验进展

研究内容分别检验任务记忆的形成、已有记忆的读取，以及目标选择能否转化为仿真动作。原始数据和运行产物保存在仓库同级 `experiments/` 中。

## 数据核验

对固定版本全部 50 条公开训练轨迹校验前视视频与 Parquet。视频完整解码共 56,462 帧，逐帧时间戳最大差异为 6.1 微秒。每条轨迹有一个按固定任务规则定位的官方专家子任务候选点，50 个候选中有 34 个附二维位置。候选标签独立存放，其历史证据充分性仍需核查。[Plus 数据和评测协议](https://robomme.github.io/plus.html)

```powershell
python scripts/verify_plus_video.py --data-dir ../experiments/data/plus-agent-coverage --work-dir ../experiments/data/plus-video-verification --reuse-video-dir ../experiments/data
```

## 目标选择与定位的交叉对照

`VideoPlaceButton` 和 `VideoPlaceOrder` 的全部 6 条公开训练轨迹在同一决策截点截断输入。两个读取方式共用同一份事件记忆，两个定位方式共用相同的跟踪采样间隔。程序生成操作、物体及目标像素位置，可供不同执行器读取。历史事件和当前目标的评分在决策生成后完成。

| 事件读取 | 定位方式 | 历史事件正确 | 目标正确 | 两项均正确 |
|---|---|---:|---:|---:|
| 相关对象、最近优先 | 逐帧关联 | 3/6 | 2/6 | 1/6 |
| 相关对象、最近优先 | 换位片段 | 3/6 | 3/6 | 3/6 |
| 对象与时间条件 | 逐帧关联 | 6/6 | 1/6 | 1/6 |
| 对象与时间条件 | 换位片段 | 6/6 | 6/6 | 6/6 |

两个分组同属已见开发轨迹。结果衡量固定轨迹的目标判断，完整任务仍需在线仿真。方法在三条新遮挡换位任务的前视输入上均未命中，逐回合定位了当前候选不足、初始身份绑定错误和遮挡前证据不足的问题。[独立任务迁移诊断](unmask-transfer.md)

```powershell
python scripts/run_grounded_place_replay.py --work-dir ../experiments --output ../experiments/runs/place-ablation/replay-01
```

增加 `--deepseek` 可用同一事件候选和事件前后画面对照视觉模型读取。调用使用 `deepseek-flash`，当前运行环境未配置 `DEEPSEEK_API_KEY`，因此本表只包含规则读取与视觉关联的实测结果。配置后使用新的输出目录，逐回合记录 API 使用量和失败状态。[DeepSeek 图文接口](https://api-docs.deepseek.com/guides/vision/)

## 仿真和两种执行候选

共用子目标包含操作、物体、当前相机的目标位置及对应的历史证据。基准执行可分别试用公开 RGB-D 与校准生成位姿的路线，或复用 GroundSG 接收定位子目标并输出关节动作的路线。执行器选择以同一已知目标的完成情况、覆盖和资源开销为依据。对第 32 条公开视频已根据官方 H5 验证像素可反投影，尚需在在线回合核验动作结果。[执行器部署与预检](execution-candidates.md)


## 本机 GPU 推理与官方仿真预检

Windows 原生 `llama.cpp` 指定 `Vulkan1`，设备为 RTX 5060 Laptop。加载 Qwen2.5-VL-3B Q4 和视觉投影后，NVIDIA 显存从 12 MiB 变为 3,618 MiB。相同六条开发轨迹均完成本机图文请求，模型读取与换位定位组合的历史事件及目标同时正确为 2/6。同条件规则读取与换位定位为 6/6。两组结果用于分析模型证据读取，测量对象是固定轨迹，数据位置为 `../experiments/runs/place-ablation/qwen-vulkan-s2/`。

运行前用 `llama-server --list-devices` 确认 NVIDIA 对应编号，并在本机模型服务启动参数中显式指定设备。例如本机运行时使用 `--device Vulkan1 --gpu-layers 99 --ctx-size 8192`。模型服务监听 `127.0.0.1:17864`，模型别名为 `qwen25vl3b`。然后在仓库根目录执行如下命令。

```powershell
python scripts/run_grounded_place_replay.py --work-dir ../experiments --output ../experiments/runs/place-ablation/qwen-replay-01 --local-vision
```

官方 Plus 固定提交 `777c595d9c443405fdc145452929bc5890c13e19` 在 Ubuntu 22.04 WSL2 中实际重置了 `VideoPlaceButton` 的公开训练回合 2。使用本机副本的 CPU 渲染配置获取 2,006 帧，读取前视 RGB-D 并把目标反投影到世界坐标，发送原位 waypoint 后环境仍处于 `ongoing`、未报告动作错误。该动作占 7 个控制步。日志在 `../experiments/runs/plus-preflight/wsl-video-place-button-2-cpu-render.log`。

原生 Windows 的 SAPIEN 3.0.3 可在 RTX 5060 上渲染，官方任务还依赖无 Windows 轮子的 MPLib 0.1.1。WSL2 的 CUDA 可见，但当前 SAPIEN 3.0.2 只枚举到软件 Vulkan 渲染设备。正式 GPU 批跑应在支持该渲染栈的环境上复核。WSL2 的回合结果属于本机联调。[ManiSkill 平台支持](https://maniskill.readthedocs.io/en/v3.0.0b20/user_guide/getting_started/installation.html#system-support) · [MPLib 0.1.1 发行文件](https://pypi.org/project/mplib/0.1.1/#files)
