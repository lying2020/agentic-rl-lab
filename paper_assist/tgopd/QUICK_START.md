# Quick Start — TGOPD

方法是 **训练期门控**，不是新推理架构。推理 = 标准学生 generate。官方训练代码未开源。

## 1. 环境（论文底座）

```bash
# 论文使用 slime 异步栈 + SGLang
# https://github.com/THUDM/slime
```

## 2. 权重

- 学生：Qwen3.5-4B 或 Qwen3.6-35B-A3B
- 教师：需在同一基座上按域 GRPO 后冻结（权重未随论文释放）

## 3. 数据

- 数学：DAPO-Math-17K
- 代码：CodeI/O 的 input–output prediction
- IF：Nemotron-Cascade 2 过滤（过滤脚本未给）

## 4. 推理冒烟

用学生 HF 权重标准 chat generate 即可，与 TGOPD 算法无关。

## 5. 训练（论文路径，代码缺失）

实现 Algorithm 1：学生 decode ∥ 教师 $K_T=3$ 探针 → $q_T$ → 硬门选 $A^{\mathrm{OPD}}$ 或 $A^{\mathrm{GRPO}}$ → PPO-clip。  
IcePop/TIS 是异步共享修正，不要算进方法。

最小算法冒烟（单机）：数学、短回复、$G=4$、$K_T=3$、$\tau=2/3$，先做 mask-only 关门。

## 常见缺口

- 无官方仓、无教师权重、无 IF 过滤脚本
- 完整 4B SOPD 为 5×8 GPU
