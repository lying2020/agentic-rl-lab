# MODEL_SPECS — TGOPD

## 4.1 架构概览

无新网络。学生 $\pi_\theta$ + 冻结域教师 $\pi_T$ + 规则验证器。创新在 prompt 级路由。

| 组件 | 论文设定 |
|---|---|
| 学生 | Qwen3.5-4B；Qwen3.6-35B-A3B（3B active） |
| 教师 | 同基座 GRPO 域专家，冻结 |
| 框架 | slime 异步；SGLang |
| 门 | $K_T=3$，$\tau=2/3$ |

## 4.2 仅推理所需

评测学生 checkpoint 的标准 generate。不需要教师、探针或门。

## 4.3 推理成本（强制）

| 设定 | 最低 GPU【估计】 | 备注 |
|---|---|---|
| 4B 评测 max 16K | 1×80GB | AIME avg@64 要多次采样 |
| 35B-A3B 评测 | 1–2×80GB MoE | 论文 serving 细节未给 |
| flash-attn | 建议 | 非方法必须 |
| 额外服务 | 无 | 训练才需要教师节点 |

论文延迟数字针对 **训练利用率**，不是推理 latency。

## 4.4 权重与数据

学生公开 Qwen 权重；教师与过滤 IF 需自备。DAPO-Math-17K、CodeI/O 公开。

## 4.5 训练成本（后置）

- 4B：40 GPU（2 train + 2 rollout + 1 teacher 节点）
- 35B：56 GPU
- 35B CodeIO 相对 Vanilla OPD step time **+5.9%**
- 4B SOPD 教师利用率 $9.8\%\to78.9\%$

## 4.6 本地缺口

无代码、无教师 ckpt。本仓库仅有 PDF `docs/7_TGOPD.pdf`。
