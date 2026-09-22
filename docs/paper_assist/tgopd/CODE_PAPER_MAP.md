# CODE_PAPER_MAP — TGOPD

论文：arXiv:2609.02998。本地 PDF：`docs/7_TGOPD.pdf`。  
官方实现：**未见公开仓库**。下表把论文模块映射到 **应实现的接口**，并标出可借用的开源底座。

## 论文模块 ↔ 实现落点

| 论文 | 符号 / Eq. | 建议落点 | 状态 |
|---|---|---|---|
| 学生 on-policy rollout | $G$ 条，$G=4$ | slime rollout worker | 论文用 slime；可参考 THUDM/slime |
| 教师 scoring | $\log\pi_T(y_t\mid h_t)$ | 冻结教师 forward | 无条件对所有 prompt |
| 可靠性探针 | Eq.3–4，$K_T=3$ | 教师 decode $K_T$ 条完整回答 | **本文新增** |
| 验证器 | $r\in\{0,1\}$ | math-verify / 单测 / IF 规则 | 师生共用 |
| 硬门 | Eq.5–6 | `g = (q_t >= tau)` 选优势 | **本文新增** |
| OPD 优势 | Eq.1 | `beta * (logp_T - logp_S)` | 标准 OPD |
| GRPO 优势 | Eq.2，无 std | `r - mean(r)` 广播 | 实现约定 |
| PPO-clip | Eq.7 | 标准 clipped surrogate | 共享 |
| IcePop/TIS | 附录 B | 异步 IS clip 2.0 | **不是门的一部分** |

## 架构图

```mermaid
flowchart TD
  X[prompt x] --> S[student rollout G=4]
  X --> Tprobe[teacher idle window KT=3 probes]
  Tprobe --> V1[verifier scores probes]
  V1 --> Q["qT = mean r"]
  Q --> G{"qT >= tau"}
  S --> Tscore[teacher scoring forward]
  S --> V2[verifier scores student]
  Tscore --> Aopd["A_OPD token log-ratio Eq.1"]
  V2 --> Agrpo["A_GRPO group-centered Eq.2"]
  G -->|yes| Aopd
  G -->|no| Agrpo
  Aopd --> PPO[PPO-clip Eq.7]
  Agrpo --> PPO
  PPO --> Upd[update student]
```

## 一时序：一个异步 cycle

```mermaid
sequenceDiagram
  participant R as Rollout nodes
  participant T as Teacher node
  participant V as Verifier
  participant U as Trainer
  R->>R: decode G student rollouts
  par fill teacher idle window
    T->>T: decode KT probes
  end
  T->>V: score probes
  V-->>U: qT of x
  R->>T: student tokens for scoring
  T-->>U: log piT
  R->>V: student rollouts
  V-->>U: student rewards
  U->>U: gate qT vs tau then select A
  U->>U: PPO-clip and IcePop
```

门控对应 $g(x)=1[q_T(x)\ge\tau]$，选 $A_{\mathrm{OPD}}$ 或 $A_{\mathrm{GRPO}}$。

## 调用流（应实现）

```mermaid
flowchart TD
  A[cycle start] --> B[concurrent student decode and teacher probes]
  B --> C["qT = mean verifier probes"]
  C --> D[teacher logprob on student tokens]
  D --> E[verifier on student]
  E --> F{"gate g of x"}
  F -->|1| G["A = A_OPD Eq.1"]
  F -->|0| H["A = A_GRPO Eq.2"]
  G --> I[PPO-clip Eq.7]
  H --> I
```

## Paper ↔ Code gaps

- 无官方代码；教师 GRPO 权重未释放。
- 总训练 step 除 MOPD=199 外未统一披露。
- GPU 型号未写死。
- 训练路径存在，推理不是论文贡献；评测即标准 generate + 基准脚本。
