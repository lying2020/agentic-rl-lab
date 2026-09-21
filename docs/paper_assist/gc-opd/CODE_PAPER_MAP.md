# CODE_PAPER_MAP — GC-OPD

论文：arXiv:2608.19181。代码：https://github.com/SolereZhang/GC-OPD  
本地 PDF：`docs/3_GC-OPD.pdf`。

## 论文模块 ↔ 代码

| 论文 | Eq. / Alg. | 代码 |
|---|---|---|
| Token OPD 优势 $A_t$ | Eq.1 | vanilla OPD log-ratio；`beta=0` 入口 |
| 轨迹分 $s$ | Eq.3 | `verl/verl/trainer/ppo/gc_opd.py` |
| 组 z-score 与 $\rho$ | Eq.6–7, 附录 (14) | 同上 |
| RACA $u_t,c_t$ | Eq.9–10, 附录 (15–16) | 同上 |
| $A'=A+\beta c\rho$ | Eq.11 | GC-OPD 入口 `beta=0.10` |
| 数据 32K 过滤 | §5.1 | `scripts/prepare_golongrl_32k.py` |
| 训练启动 | Table 6 | `scripts/run_gc_opd_{4b,8b}_training.sh` |
| 五基准评测 | §5.1 | `evaluation/run_main_table_evaluation.sh` |
| 冻结 rollout 诊断 | Fig.1 | `verl/examples/gc_opd/analyze_gc_opd_frozen_replay.py` |

OPD 对照：同一实现 `beta=0`。

## 架构图

```mermaid
flowchart TD
  X[long-context prompt] --> S[学生 πθold<br/>G=8 rollout]
  S --> T[教师 logprob<br/>不另生成]
  S --> V[任务验证器 R]
  T --> At["At = logπT − logπold  Eq.1"]
  At --> s["s = mean_t At  Eq.3"]
  V --> zR["R̃ = z(R)"]
  s --> zs["s̃ = z(s)"]
  zR --> rho["ρ = R̃ − s̃  Eq.7"]
  zs --> rho
  At --> raca["RACA ct  Eq.10"]
  rho --> Aprime["A' = At + β ct ρ  Eq.11"]
  raca --> Aprime
  Aprime --> PPO[原 PPO-clip]
```

## 训练时序

```mermaid
sequenceDiagram
  participant A as Actor/rollout
  participant T as Teacher
  participant V as Verifier
  participant L as gc_opd.py
  A->>A: sample G responses
  A->>T: tokens for logprob
  T-->>L: log πT
  A->>V: complete responses
  V-->>L: R
  L->>L: s, z-score, ρ, RACA, clip
  L-->>A: token advantages A'
  A->>A: PPO update
```

## 调用流

```mermaid
flowchart TD
  E[scripts/run_gc_opd_8b_training.sh] --> V[verl PPO trainer]
  V --> R[rollout + teacher logprob + verifier]
  R --> G["gc_opd.py  Eq.6–11"]
  G --> C{σR,σs > τG?}
  C -->|否| Z["ρ=0 退回 OPD"]
  C -->|是| P["A' = At + β c ρ"]
  Z --> U[clipped policy loss]
  P --> U
```

仓库无独立「论文方法推理」路径：推理即 Qwen3 generate；方法只改变训练优势。
