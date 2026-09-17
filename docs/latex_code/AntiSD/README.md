# AntiSD 上手材料

对照仓库：`/home/user/Documents/agentic-rl-lab/algorithm/AntiSD`  
论文：arXiv:2605.11609（工作区未找到 `text_xx` latex，公式对照用 HTML v1）

| 文件 | 内容 |
|---|---|
| [01_快速开始.md](01_快速开始.md) | 目录、环境、数据、推理、训练命令 |
| [02_模型与训练需求.md](02_模型与训练需求.md) | 架构、HF 权重、数据集、GPU/时间、Table 1 |
| [03_论文与代码对应关系.md](03_论文与代码对应关系.md) | 公式↔文件、PlantUML 流程图、实现偏差 |
| [conda_env_setup.sh](conda_env_setup.sh) | Python 3.12 + torch 2.5.1/cu124 + requirements + vLLM |
| [inference_math.py](inference_math.py) | 单题 / parquet 最小推理 |
| [run_inference.sh](run_inference.sh) | 推理包装脚本 |

```bash
bash /home/user/Documents/agentic-rl-lab/docs/AntiSD/conda_env_setup.sh
conda activate antisd
bash /home/user/Documents/agentic-rl-lab/docs/AntiSD/run_inference.sh
```
