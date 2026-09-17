# MODEL_SPECS — RLCSD

## 4.1 架构概览

单模型自蒸馏：学生 decode；教师是同一架构的 snapshot/GT-conditioned forward。  
损失：`core_algos.py` 中 `rlcsd`。

## 4.2 仅推理所需

- 学生权重
- 不需要特权 hint、负样本或教师副本
- 复现表需要 AMC/AIME 与 KK 评测 parquet
- torch + transformers / vLLM；thinking 模板

## 4.3 推理成本（强制）

评测：max completion 38912，温度 0.6，数学 12 样本。

| 模型 | 最低【估计】 | 推荐 | 延迟 |
|---|---|---|---|
| 1.7B 冒烟 2K | 1×24GB | 1×40GB | 论文无 latency 表 |
| 4B eval 38K | 1×80GB | 1–2×80GB | KV 随 38K 线性增 |
| 8B eval 38K | 1–2×80GB | 2×80GB | mean@12 再 ×12 |
| Olmo-3-7B-Think | 1–2×80GB | 2×80GB | |

flash-attn 强烈建议。训练才需要 vLLM rollout 与教师前向。  
磁盘：1.7B ~4GB，8B ~16GB【估计】。

## 4.4 权重与数据

公开 HF 模型；`Leyiii/RLCSD`。

## 4.5 训练成本

8×H20 全参。4B 数学 Table 9：RLCSD **891.55 s/step**（gen 500.92，教师 logprob 14.21）。曲线约数学 500 step、逻辑 200 step。

相对 one-sided OPSD，多 $K=4$ 条负 hint 教师 forward，仍快于 dense 全词表蒸馏。

## 4.6 本地缺口

论文级需 8 卡 H20。YAML 与 PDF 超参可能不一致，先对配置。
