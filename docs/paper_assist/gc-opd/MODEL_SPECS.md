# MODEL_SPECS — GC-OPD

## 4.1 架构概览

学生 Qwen3-4B/8B no-thinking + 教师 Qwen3-30B-A3B-Thinking-2507 logprob + 任务验证器。  
训练信号在 `gc_opd.py`：$A'=A+\beta c\rho$。

## 4.2 仅推理所需

- 学生 HF 权重（`config.json` + safetensors）
- 不需要教师、GoLongRL、验证器
- 完整五基准需要评测数据 + judge 模型 Qwen3-30B-A3B-Instruct-2507
- 软件：transformers / vLLM；YaRN scale 4 用于长上下文评测

## 4.3 推理成本（强制）

评测协议：输入 ≤120K，生成 ≤8192，serving context 131072，YaRN factor 4。

| 模型 | 最低【估计】 | 推荐 | 说明 |
|---|---|---|---|
| Qwen3-4B 短上下文冒烟 | 1×24GB | 1×40GB | 与论文评测不同 |
| Qwen3-4B 论文 120K | 1–2×80GB | 2×80GB | 长上下文 KV 主导 |
| Qwen3-8B 论文 120K | 2×80GB | 2–4×80GB | |
| 教师（仅训练） | 8×80GB 集群内 TP=8 | 同左 | 训练不是推理 |

论文未给 ms/token。flash-attn 建议开启。无额外 grounding 进程。  
磁盘：4B ~8GB，8B ~16GB，30B-A3B 教师 ~60GB 级【估计】。

## 4.4 权重与数据

Hugging Face 上述模型；`Kwai-Klear/GoLongRL`。

## 4.5 训练成本

8×80GB H800/H100，100 step，32 prompt × 8 rollout，prompt 32K + response 10K。相对 OPD 无额外前向。

## 4.6 本地缺口

需自备 8 卡与 30B 教师才能按论文入口训练。
