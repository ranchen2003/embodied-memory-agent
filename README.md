# 具身智能体任务记忆实验

课题为《基于记忆增强的具身智能体任务决策方法研究与实现》。当前入口用于检查机械臂示范历史的保留情况。运行只需要 Python 和 CPU。

## 从本机运行

在项目根目录执行。已有对应依赖时直接运行脚本。

```powershell
python -m pip install -r requirements.txt
python scripts/offline_pilot.py --budget 8
python scripts/offline_pilot.py --budget 24
```

首次运行从 Hugging Face 读取固定版本的 RoboMME-Plus 公开训练数据，以后复用 `data/` 中的缓存。每次运行在 `runs/pilot/budget_*/` 生成逐回合的 `metrics.csv` 和带采样帧、参考区间及命中情况的 `trace.json`。[公开数据](https://huggingface.co/datasets/Yinpei/robomme_plus_data_lerobot) · [数据说明](https://github.com/RoboMME/robomme_benchmark/blob/main/doc/plus_training_data_format.md)

| 路径 | 内容 |
|---|---|
| `scripts/offline_pilot.py` | 读取示范历史并运行等帧数对照 |
| `docs/first-month.md` | 已有结果、实验口径和组会计划 |
| `data/` | 自动生成的公开轨迹缓存 |
| `runs/` | 自动生成的结果和逐回合记录 |

Git 记录脚本、依赖与实验说明。数据、权重、运行产物和本机密钥通过 `.gitignore` 保留在本机。正式评测计划覆盖 RoboMME-Plus 的全部 16 个任务与 800 个固定测试回合。[基准协议](https://robomme.github.io/plus.html)
