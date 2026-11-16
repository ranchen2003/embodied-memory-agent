# 视觉语言模型与机械臂闭环

高层 Agent 接收任务、环境观测和历史记录，向视觉语言模型发出图文请求。模型返回结构化决策，执行器将决策转换为仿真控制命令。新的相机观测和动作结果回到下一次决策，日志记录模型调用次数、耗时、选择的操作与环境结果。

## 本机验证

模型可由本机 `llama-server` 提供图文推理。接口监听 `127.0.0.1` 时，请求的图像和文本在本机处理。脚本同样可连接具有相同图文接口的服务，具体模型和凭证由运行配置指定。凭证不写入代码、运行日志或版本库。[llama.cpp 多模态文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) · [Qwen2.5-VL-3B 量化权重](https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/tree/5037fcf163dd95d1e41d1974465f0898ed108ca2)

```powershell
..\experiments\envs\llama-b12428\llama-server.exe --model ../experiments/models/qwen25vl3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf --mmproj ../experiments/models/qwen25vl3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf --host 127.0.0.1 --port 17864 --alias qwen25vl3b --n-gpu-layers 99 --ctx-size 4096 --parallel 1
```

服务启动后，在另一个终端执行以下命令。

```powershell
..\experiments\envs\maniskill-win\Scripts\python.exe scripts/run_local_api_agent.py --endpoint http://127.0.0.1:17864 --model qwen25vl3b --seed 2 --output ../experiments/runs/local-api-agent/qwen25vl3b
```

Windows 桌面测试使用当前相机图像、夹爪读数和技能反馈，由模型选择抓取、放置或结束，底层利用 RGB-D 定位并控制仿真机械臂。该测试验证图文请求、动作选择和再观测接口。历史依赖任务上的方法收益由独立实验评估。

一次本机运行中，`seed=2` 的模拟回合发出 4 次图文请求。模型依次返回抓取、放置、再次放置和结束。执行器拒绝再次放置，实际执行共 71 个关节控制步，环境评估抓放通过。逐步记录和相机回放保存于 `../experiments/runs/local-api-agent/qwen25vl3b/`。

## RoboMME-Plus 数据和官方环境

```powershell
python scripts/prepare_plus_train.py --work-dir ../experiments/data/plus-agent-coverage
python scripts/run_robomme_plus.py --help
```

数据准备覆盖公开训练分段全部 50 条轨迹，涉及 4 类记忆需求和 16 个任务。固定版本的 Parquet 记录经校验后生成 2,137 个检查点，其中离线评价标签保存到独立目录。该步骤支持后续固定轨迹诊断。主实验范围为官方测试分段 16 个任务、每个 50 回合。[公开训练数据](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot) · [评测协议](https://robomme.github.io/plus.html)

`run_robomme_plus.py` 使用 `BenchmarkEnvBuilder` 的 `waypoint` 接口，输入包括任务文本、保留的前视历史、当前前视和腕部画面及本体状态。模型输出世界坐标系下六维位姿和夹爪指令，官方执行器处理该指令。成功由环境状态判定。运行需先安装官方环境。当前入口支持开发回合、近期帧与均匀帧读取、调用预算、数据字段过滤和逐步日志。真实官方环境运行和研究方法接入仍待验证。[官方环境接口](https://github.com/RoboMME/robomme_benchmark/blob/main/doc/env_format.md) · [官方评估器](https://github.com/RoboMME/robomme_benchmark/tree/main/scripts/plus_eval)

## 对照方法

| 方法 | 用法 | 已核实的状态 |
|---|---|---|
| MemER | 参考任务相关关键帧的选择和高层决策、底层执行的接口设计 | 已有原版 RoboMME 适配说明。Plus 闭环需另行验证。 |
| Embodied Astra | 同为官方 waypoint 执行接口，提供外部报告的完整任务表现 | 官方 Plus 800 回合报告为 181 个成功。模型、预算和运行条件按原文记录。 |
| FrameSamp+Modul | 参考作者发布的均匀帧采样和策略对照 | 官方 Plus 800 回合报告为 22 个成功。权重、代码和输入预算需核实后复跑。 |
| 近期帧、均匀帧和任务记忆 | 固定模型、观测权限、帧数及执行器后比较记忆建模和读取方法 | 本项目提供前两种读取入口。事件记忆在此前固定轨迹诊断中已有局部实现。 |

官方结果属于论文和项目报告。项目的同条件对照将记录关键信息保留、证据选择、决策正确率、完整任务成功率和资源开销，并报告各配置的独立结果。[MemER 原版适配](https://github.com/RoboMME/robomme_policy_learning/blob/ac846a9a4aec841e483b1dc5d620dfe426bde853/docs/manual_evaluation.md#memer) · [Plus 官方实验与成本](https://robomme.github.io/blog/)
