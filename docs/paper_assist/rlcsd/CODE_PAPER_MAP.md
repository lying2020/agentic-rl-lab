# CODE_PAPER_MAP — RLCSD

论文：arXiv:2606.11709。代码：https://github.com/THU-BPM/RLCSD  
本地 PDF：`docs/9.4_RLCSD.pdf`。

## 论文模块 ↔ 代码

| 论文 | Eq. | 代码 |
|---|---|---|
| 特权模板 | §3.3 | `src/opsd_format.py` |
| 正/负 hint 数据路径 | Eq.7 | `src/self_distill_main.py` |
| $e_{\mathrm{ctr}}$ 与 two-path 损失 | Eq.7, 9–17 | `third_party/verl/verl/trainer/ppo/core_algos.py` `@register_policy_loss("rlcsd")` |
| 验证器奖励 | §3.3 | `src/verl_reward.py` |
| YAML 方法选择 | `method: rlcsd` | `configs/math_deepmath/`、`configs/logic_kk/` |
| 启动 | — | `scripts/_run_verl.sh`；`scripts/math_deepmath/run_qwen3_4b_rlcsd.sh` |
| 插件 OPSD/RLSD+contrast | Eq.18–19 | `opsd_ectr` / `rlsd_ectr` |
| 数据下载 | — | `scripts/download_data.py` → `Leyiii/RLCSD` |

## 架构图

```mermaid
flowchart TD
  X[query x] --> S["学生采样 G=8"]
  S --> Ver["规则验证器 0/1"]
  Ver --> Gp["G+ / G-"]
  Gp --> Aorm["A_ORM 组相对 Eq.8"]
  Gp --> Pos["正 hint：GT CoT 或 sibling"]
  Gp --> Neg["K 条负 sibling 排除 y"]
  Pos --> Tc["πT(· | yc*)"]
  Neg --> Tw["(1/K) Σ πT(· | yw,k*)"]
  Tc --> E["ectr = log pc − log mean pw  Eq.7"]
  Tw --> E
  E --> R["rt = λ tanh(ectr/τ)"]
  R --> M["mt = 1[|rt|>δ]"]
  Aorm --> Clamp["Ã = sign-clamp(A_ORM+rt)"]
  R --> Clamp
  M --> Loss["two-path PPO  Eq.15"]
  Clamp --> Loss
  Aorm --> Loss
```

## 训练时序

```mermaid
sequenceDiagram
  participant S as Student rollout
  participant V as Verifier
  participant T as Teacher snapshot
  participant L as rlcsd loss
  S->>S: G completions
  S->>V: extract answer
  V-->>S: G+ / G- and A_ORM
  S->>T: batch with positive hint
  S->>T: K batches with negative hints
  T-->>L: token logprobs
  L->>L: ectr, rt, mask, clamp
  L->>L: two-path clipped surrogate
```

## 调用流

```mermaid
flowchart TD
  SH[run_qwen3_4b_rlcsd.sh] --> RUN[_run_verl.sh]
  RUN --> MAIN[src/self_distill_main.py]
  MAIN --> CORE["core_algos.py  rlcsd"]
  CORE --> U["unmodulated path  A_ORM"]
  CORE --> M["modulated path  Ã"]
  U --> J["J = mean_U + η mean_M"]
  M --> J
```

## Paper ↔ Code gaps

核对 YAML，不要混用 PDF 与 README：

- 教师：PDF snapshot 每 10 step；README 写 fixed
- $\tau,\eta$：PDF $0.02,1.0$；README $1.3,0.5$
- 主表数字两套

推理：方法是训练损失；部署即原模型 generate。
