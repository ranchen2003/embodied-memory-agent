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
