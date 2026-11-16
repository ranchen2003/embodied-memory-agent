# 圆环任务的在线对照

使用 RoboMME-Plus 的 `VideoPlaceButton` 和 `VideoPlaceOrder`。每个任务采用公开训练分段的全部三个官方种子，默认运行每个种子的三个记忆配置，共 18 回合。执行入口核对官方任务索引与预登记种子。该组用于开发闭环诊断，正式比较仍采用独立测试分段。[官方评估与观测字段](https://github.com/RoboMME/robomme_benchmark/blob/777c595d9c443405fdc145452929bc5890c13e19/doc/env_format.md)

| 配置 | 预测可读的观测 | 目标决策 |
| --- | --- | --- |
| `current` | 当前公开前视图 | 根据任务颜色定位方块，选择像素距离最近的可见圆环。 |
| `uniform` | 当前前视图、公开示范夹爪均值、均匀保留的历史前视图 | 在保留画面中核对释放事件，按颜色和时间条件读取，利用换位片段定位目标。 |
| `complete` | 与 `uniform` 同样类型的观测和等量帧预算 | 优先保留释放前后的证据帧，余量均匀填充，复用相同的事件读取和换位定位。 |

两个历史配置默认各保留至多 512 帧历史 RGB，当前帧另计。二者使用相同长度的公开夹爪宽度序列。记录从官方重置返回的最后一帧识别当前场景，使用前面的帧作为示范。动作计划共用同一 RGB-D 几何执行器，受官方控制步预算和八条 waypoint 命令上限约束。离线标注可用于独立复核，在线入口不加载标注。当前帧对照使用独立的无历史几何规则，`uniform` 对 `complete` 才用于比较帧保留机制。

## 运行

正式运行使用原生 Linux 的 NVIDIA GPU 和官方固定版本 `777c595d9c443405fdc145452929bc5890c13e19`，保留官方默认渲染配置。依次通过实际相机出图、官方回合原位动作和完整抓放三个检查，再启动整批对照。以下命令在已安装官方环境的 Linux Python 中从本项目根目录执行。输出路径使用独立的新目录。

```bash
python scripts/probe_sapien_render.py --device cuda:0
python scripts/probe_plus_sim.py --task VideoPlaceButton --episode 2 --pixel 117.5 76.2037 --send-hold
python scripts/run_plus_place_online.py \
  --task VideoPlaceButton --episode 2 --mode complete \
  --output ../experiments/runs/place-online/gpu-32
```

省略任务、回合和模式过滤器可运行完整矩阵。日志记录运行矩阵、每条回合的图像预算、证据帧号、候选点、完成状态和实际控制步。分别保存首次决策时和执行结束时的前视图。场景渲染、执行超时、动作错误和未给出子目标均保留在统计中。任务成功使用官方环境反馈。运行期间逐条写出结果，运行中断时已有轨迹可用于诊断。

入口在重置前检查环境实际使用的 SAPIEN 渲染设备，报告设备名称、渲染后端和物理后端。GPU 判定依赖渲染器的 CUDA 设备和 `can_render()`，不能用 `torch.cuda.is_available()` 代替。`--allow-cpu-render-diagnostic` 可标记本机 CPU 渲染联调，运行记录应独立保存。原生 Linux 的 GPU 支持与 WSL2 的渲染限制见[ManiSkill 支持矩阵](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support)。

在六条真实公开存档回放中，相同入口的目标像素命中为 `current` 2/6、`uniform` 6/6、`complete` 6/6。两种有记忆的配置都读对了六次历史事件，所引用的图像均在各自 512 帧预算内。该结果支持核查历史观测是否帮助目标选择，当前容量下还未观察到事件优先保留优于均匀保留。以上是目标选择的存档回放结果。执行效果以真实在线回合的独立结果为准。


## 图像预算的离线检查

沿用上述接口，在同六条公开训练存档上固定指令、截点、夹爪数据和定位逻辑，仅比较图像保留方式。统计为历史事件正确、当前目标正确，分母均为 6。命中阈值为 5 像素，评分标签在决策后使用。

| 历史前视帧预算 | 均匀保留，事件／目标 | 事件优先保留，事件／目标 |
| ---: | ---: | ---: |
| 16 | 0／0 | 0／0 |
| 32 | 0／0 | 4／0 |
| 64 | 1／0 | 6／1 |
| 128 | 3／2 | 6／2 |
| 512 | 6／6 | 6／6 |

在 64 帧时，事件优先组已定位全部历史事件，当前目标只命中 1 条。128 帧也有类似差距。前后证据的保留和跨帧身份关联在低预算下需要分别检查。逐条结果见 `../experiments/runs/place-ablation/history-budget-check.json`。
