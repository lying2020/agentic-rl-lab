# 组会汇报：TGOPD（Verify Before You Distill）

> LaTeX 版：`BRIEFING.tex` → 运行 `bash compile_briefing.sh` 得到 `BRIEFING.pdf`（公式以 PDF 为准）

论文：AllSpark Team. *Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation*. arXiv:2609.02998v1, 2026-09-02.  
状态：Public Technical Report（2026-08-28）  
贡献者：Zhiwei Zhang, Zechen Sun, Fei Zhao, Kang Peng, Bin Liang, Huayu Deng, Yao Hu, Kam-Fai Wong, Mu Chuan  
PDF：`docs/7_TGOPD.pdf`  
框架：`slime` 异步 rollout–update（THUDM/slime）  
代码：截至组会整理时 **未见官方开源仓**；实现细节以 PDF 附录 B–D 为准。  
和组里已读材料的位置：这是 **外部冻结教师 OPD** 的 *prompt 级准入门*，不是 TrOPD 那种 token 级信任区，也不是 TOP-D 那种奖励有界化。它直接引用了 TrOPD、RG-OPD、RLSD、EOPD、REOPOLD。

文中 **【注解】** 是组会追问的展开；**【源码核验】** 表示该细节可在 PDF 正文/附录定位（本文无独立 TeX 仓）。

---

## 0. 一句话贡献

Vanilla OPD 对每个 prompt 无条件使用教师 token 级 reverse-KL；TGOPD 先用 **$K_T=3$ 次教师探针 + 验证器 pass rate** 做 prompt 级可靠性审计：通过则走稠密 OPD，否则撤回教师、改走验证器 grounding 的 GRPO。门是硬路由而不是插值。

---

## 1. 任务定义与动机剖析

### 1.1 本 paper 的任务是什么

后训练里的 **On-Policy Distillation**：学生 $\pi_\theta$ 自己采样 rollout，冻结的更强教师 $\pi_T$ 在学生轨迹上给 token 级 reverse-KL 监督。实验覆盖：

- 学生：Qwen3.5-4B（dense）与 Qwen3.6-35B-A3B（MoE，3B active）
- 三域：数学、代码、指令遵循（IF）
- 教师：对同一基座做 GRPO 得到的 **域专家冻结教师**
- 另有一条 MOPD 线：一个学生同时学三域，prompt 先路由到对应域教师，再套 TGOPD 门

一句话任务：**在承认 OPD 样本效率的前提下，先验证“这个 prompt 上教师值不值得信”，再决定放不放稠密监督。**

### 1.2 要解决的问题

Vanilla OPD 把教师监督 **均匀施加到所有 prompt**。Reverse KL 是 mode-seeking：教师一旦在某题上自信地错，学生会被强推到那个错误模式。分布侧代理（熵、师生 likelihood 一致）测的是不确定/一致性，**不是对错**。

异步部署还叠了一个系统问题：教师节点只做已生成 token 的 scoring forward，必须等学生 decode 完才能开始，因此大部分时间空转。4B 单域测量：教师节点均值利用率 **9.8%**，59% 的时间低于 5%。

### 1.3 Motivation

作者把空闲算力与可靠性缺口绑在一起：

> 教师反正在等学生 decode，不如用这段空窗生成少量教师探针，用验证器打分，得到 prompt 级可靠性 $q_T(x)$，再决定是否准入稠密 OPD。

Figure 2 是动机的硬证据（离线诊断，每 prompt 10 条教师样本，$q_T^{(10)}$ 是 pass rate，与训练时 $K_T=3$ 的在线估计量不同但测同一件事）：

- 代码域：教师自信几乎分不开高/低可靠性 prompt（AUROC **0.51**）；数学稍好（**0.73**）
- 低可靠性区（$q_T^{(10)}<0.5$）：教师最高自信样本仍错，代码 **84%**、数学 **61%**

所以“自信”不能当可靠性。

### 1.4 现有方法局限

| 方法族 | 作用位置 | 局限 |
|---|---|---|
| Vanilla OPD | 全体 prompt 的 token reverse-KL | 不检查教师对不对 |
| EOPD / TrOPD / REOPOLD | 分布侧：熵、token 信任区、likelihood clip | 不确定 ≠ 错误 |
| RG-OPD | 轨迹级：验证器与师生 gap 一致才保留蒸馏 | 仍是局部轨迹筛选，不是 prompt 级准入 |
| RLSD | 验证器定方向、蒸馏调幅度 | 对所有 prompt 都削弱教师角色 |
| SPOT | token 分支 + 学生续写验证 | 粒度更细、代价更高 |
| 多教师 MOPD | 按域路由教师 | 选到域专家后仍无条件用其稠密奖励 |

TGOPD 的切口是：**在稠密监督被admit之前**，用重复的完整教师 rollout 结果做 prompt 级硬门。

### 1.5 具象化例子

设一道代码题 $x$。教师三次探针：两次编译失败、一次通过，则

$$
q_T(x)=\frac{0+0+1}{3}=\frac13,\qquad \tau=\frac23,\qquad g(x)=\mathbf{1}[q_T\ge\tau]=0.
$$

门关闭。即使教师对某个错误 token 的 $\log\pi_T$ 很高，reverse-KL 也不会进梯度。若学生 4 条 rollout 里 1 对 3 错，GRPO 组内中心化仍给对的那条正优势、错的负优势。

对照：若三次探针 2 过 1 不过，$q_T=2/3$，门打开，该 prompt **整段**走 OPD，验证器奖励不进梯度。作者刻意不做 $0.7\cdot A^{\mathrm{OPD}}+0.3\cdot A^{\mathrm{GRPO}}$ 这种混合——两种信号密度和来源不同，插值权重任意。

---

## 2. 方法论树状拆解

**创新落点**：一级模块是 *prompt 级硬路由*；二级是无偏 binomial 可靠性估计；三级是异步空窗重叠探针。OPD/GRPO 公式本身不是新的。

```text
TGOPD
├─ 模块 A：教师可靠性探针（填教师空窗）
│   ├─ 定义：RT(x)=E_{y~πT}[r(x,y)] ∈ [0,1]
│   ├─ 估计：KT 条 i.i.d. 教师 rollout，qT=平均 pass
│   │         KT qT ~ Binomial(KT, RT)，无偏
│   ├─ 主实验：KT=3，τ=2/3（二比三多数）
│   └─ 工程：与学生 decode 并发；scoring 仍对所有 prompt 无条件做
│
├─ 模块 B：硬门 g(x)=1[qT≥τ]
│   ├─ 打开：整 prompt 用 token 级 ÂOPD（Eq.1）
│   ├─ 关闭：整 prompt 用轨迹级 ÂGRPO（Eq.2，组内均值中心化、无 std 归一）
│   └─ 若组内奖励全相同：GRPO 优势=0，该 prompt 无更新
│
└─ 模块 C：PPO-clip 更新（Eq.7）
    ├─ 优势在更新内冻结
    ├─ 异步 mismatch：IcePop/TIS（所有对照方法共享，不是门的一部分）
    └─ Vanilla OPD / 纯 GRPO 是 g≡1 / g≡0 的两端
```

模块关系：A 只产出标量 $q_T$；B 用它选优势张量；C 不改 surrogate。教师 logprob 与学生验证器奖励 **两边都算**（pipeline 规整），但梯度里只留一门。

---

## 3. 算法流程与公式细节推演

### 3.1 符号

| 符号 | 含义 |
|---|---|
| $\pi_\theta / \pi_{\mathrm{old}} / \pi_T$ | 当前学生 / 冻结 rollout 快照 / 冻结教师 |
| $x,\{y_i\}_{i=1}^G$ | prompt 与 $G$ 条学生 rollout |
| $r(x,y)\in\{0,1\}$ | 验证器：代码用单测，数学/IF 用规则裁判 |
| $K_T,q_T(x),\tau,g(x)$ | 探针数、pass rate、阈值、硬门 |

### 3.2 两条候选优势（论文 Eq.1–2）

OPD（与验证器无关）：

$$
\hat A^{\mathrm{OPD}}_{i,t}=\beta\,\mathrm{sg}\Big[\log\pi_T(y_i^t\mid x,y_i^{<t})-\log\pi_\theta(y_i^t\mid x,y_i^{<t})\Big].
$$

GRPO（无 std 归一，论文明确写成实现约定）：

$$
\hat A^{\mathrm{GRPO}}_i=r(x,y_i)-\frac1G\sum_{j=1}^G r(x,y_j),\qquad
\hat A^{\mathrm{GRPO}}_{i,t}=\hat A^{\mathrm{GRPO}}_i\ \forall t.
$$

门控选择（Eq.6，选择器不是插值）：

$$
\hat A^{\mathrm{TGOPD}}_{i,t}=g(x)\,\hat A^{\mathrm{OPD}}_{i,t}+\big(1-g(x)\big)\,\hat A^{\mathrm{GRPO}}_{i,t}.
$$

再进入标准 PPO-clip（Eq.7）。

### 3.3 手算例子（$G=4,K_T=3,\tau=2/3$）

**Prompt A（门开）**  
探针奖励 $(1,1,0)$ → $q_T=2/3$ → $g=1$。  
某 token 上 $\log\pi_T=-0.2,\log\pi_S=-1.0$，$\beta=1$，则 $\hat A^{\mathrm{OPD}}=0.8$。整条序列每个 token 都用各自的 log-ratio，验证器不进梯度。

**Prompt B（门关）**  
探针 $(0,0,1)$ → $q_T=1/3$ → $g=0$。  
学生奖励 $(1,0,0,0)$，组均值 $0.25$，则四条优势为 $+0.75,-0.25,-0.25,-0.25$，**广播到该轨迹所有 token**。

**Prompt C（门关且全错）**  
学生全 0，中心化后优势全 0，该 prompt 本 step 对参数无贡献。作者承认这是 fallback 的惰性区。

### 3.4 Algorithm 1 要点

每个 cycle：学生节点采样 $G$ 条；教师节点并发采 $K_T$ 探针；验证器打探针得 $q_T$；教师仍对学生轨迹做 scoring；再打学生奖励；按门选优势；PPO 更新。探针尽量填满教师空窗；若 $K_T$ 次 decode 在学生 batch 完成前结束，则理论上不加墙钟。实测 35B CodeIO：对齐 500 cycle，TGOPD 平均 step time **+5.9%**，decode 吞吐变化 $<0.1\%$。**【注解】** 空闲填满 ≠ 零开销。

---

## 4. 实验设定与资源开销

### 4.1 Baselines

同一学生初始化、冻结域教师、语料、slime 异步管线：

- Vanilla OPD
- TrOPD（token 信任区 RKL/FKL + 教师前缀）
- RG-OPD（验证器与师生 gap 一致才保留轨迹蒸馏）
- RLSD-style（验证器定方向，**外部冻结教师**缩放幅度；不是原 RLSD 的特权自蒸馏设定）

Teacher 行同时是“域教师自身分数”和“同架构 GRPO-only 参考”。

### 4.2 数据与评测

| 域 | 训练 | 评测（附录 Table 7） |
|---|---|---|
| 数学 | DAPO-Math-17K | AIME 2025/2026 avg@64；HMMT-Feb 2025 avg@32 |
| 代码 | CodeI/O 的 I/O 预测题 | LiveCodeBench 代码生成 pass@1 avg@6；OJBench C++/Python 总体 |
| IF | 从 Nemotron-Cascade 2 过滤 | IFBench / IFEval（各 1 run，accuracy） |

MOPD 实验固定报告 **199 step**，$K_T=3,\tau=2/3$。

### 4.3 训练超参（附录 Table 6）

| 项 | 4B / 35B 共用 |
|---|---|
| rollout batch | 64 prompt/cycle |
| $G$ | 4 |
| update batch | 128；每 cycle 2 次更新 |
| 优化器 | Adam $\beta_1=0.9,\beta_2=0.98$；lr $1\times10^{-6}$ 恒定；wd $0.1$ |
| PPO $\epsilon$ | $0.2$ |
| IcePop/TIS | ratio clip $2.0$ |
| 最大回复 | 数学 16384；代码/IF 8192 |

### 4.4 硬件与成本

- 4B：**5 节点 × 8 GPU = 40 GPU**（2 train + 2 rollout + 1 teacher）
- 35B：**7 节点 × 8 GPU = 56 GPU**（4 train + 2 rollout + 1 teacher）
- 4B 16K：SGLang static memory fraction $0.60$；max running requests 与 teacher concurrency 均 64
- GPU 型号正文未写死；【估计】按 80GB 级训练卡理解
- 墙钟：35B CodeIO 相对 Vanilla OPD **+5.9%** step time
- 利用率（Table 3）：4B SOPD 教师节点 $9.8\%\to78.9\%$，集群 $51.5\%\to69.5\%$

---

## 5. 实验结论的因果支撑

### 5.1 指标物理含义

| 指标 | 含义 | 方向 |
|---|---|---|
| Accuracy avg@$N$ | $N$ 次独立生成的平均正确率 | ↑ |
| LiveCodeBench pass@1 avg@6 | 代码生成子任务 | ↑ |
| OJBench overall | C++/Python 聚合 | ↑ |
| IFBench / IFEval | 指令遵循准确率 | ↑ |
| 七基准 Avg | MOPD 的未加权平均 | ↑ |
| Teacher-node GPU util / idle | 空窗是否被探针填满 | util↑ idle↓ |
| mean step time | 端到端是否真免费 | ↓更好 |

### 5.2 主表因果链

**实验 A → 全 6 个单域设定 TGOPD > Vanilla OPD ⇒ 模块 B 的 prompt 门有效（强支撑）**

代码增益最大（4B 平均 +3.0，35B +2.9），与 Figure 2“代码自信几乎不预测对错”一致。35B LiveCodeBench：其它蒸馏方法全部 **负迁移**（低于 base），仅 TGOPD 正迁移并超过教师（+3.0 over base，LCB +1.3 over teacher）。

**实验 B → RLSD-style 在 14 列里 8 列低于 base ⇒ “对所有 prompt 削弱教师”有害（强支撑）**

TGOPD 只在审计失败时撤教师，多数 prompt 保留完整 OPD 方向与幅度。

**实验 C → MOPD+TGOPD：4B Avg 53.40→54.54；35B 60.99→61.94 ⇒ 门可无改动接到多教师（中强支撑）**

少数列小幅回退（4B LCB −0.48 等），作者称在早期 checkpoint 方差内。

**实验 D → $\tau$ sweep（$K_T=5$，4B 数学，99 step）呈倒 U，峰在 $3/5$ ⇒ 多数票够用（中等支撑）**

过松放进错误教师，过严丢掉有用 OPD；主实验 $\tau=2/3$ 靠近该峰。

**实验 E → 关门后 Mask vs GRPO fallback：约 90% 增益来自“挡住坏教师”（强支撑）；fallback 额外一点（弱到中等）**

四组设定里 masking 相对 ungated 平均 +1.08，完整 fallback +1.20。作者选 fallback 作默认是因其一致性而非幅度。

### 5.3 不应过度解读

- 无多种子误差条；MOPD 固定 199 step 的个别回退可能是噪声
- RLSD-style 不是原论文设定，只能说明“均匀削弱外部教师”不好
- 必须有自动验证器；开放式任务未覆盖
- 门是二值的，没有不确定性感知的软门

---

## 6. 复现前置准备清单

- [ ] **数据**：DAPO-Math-17K；CodeI/O I/O 预测；Nemotron-Cascade 2 过滤 IF。IF 过滤规则正文未给脚本
- [ ] **权重**：Qwen3.5-4B / Qwen3.6-35B-A3B 学生；三域 GRPO 教师需 **自己先训再冻**（教师权重未随论文释放）
- [ ] **硬件**：最低建议复现 4B SOPD：5×8×80GB；单机 8 卡只能做大幅缩协议的算法冒烟
- [ ] **软件**：slime 异步栈 + SGLang；IcePop/TIS；PPO clip 0.2
- [ ] **代码**：无官方仓。必做核心：探针 $q_T$、硬门、OPD/GRPO 互斥、GRPO 不做 std 归一
- [ ] **算法可延后**：IcePop、多教师分区、利用率 tracing
- [ ] **Sanity**：$\tau=0$ 应退化为接近 Vanilla OPD；$g=0$ 且组内奖励常数时梯度为 0；探针必须用 **与学生同一验证器**

### 建议复现顺序

1. 先在数学 4B 跑通 Vanilla OPD（slime + 冻结 GRPO 教师 scoring）
2. 加 $K_T=3$ 探针与 $g(x)$，先实现 **mask-only** 关门（最容易验证“挡住坏信号”）
3. 再加 GRPO fallback
4. 扫 $\tau\in\{1/3,2/3,1\}$ 看倒 U
5. 最后才上代码域（最能暴露自信错误）

---

## 7. Paper ↔ Code gaps / 开放问题

- **无官方代码 / 无 TeX 仓**：超参以附录 Table 6 为准；$\beta$ 在 Eq.1 出现但主表未扫
- GPU 型号、总训练 step（除 MOPD 的 199）正文未统一披露
- IcePop 与门正交，复现时不要把异步修正算进 TGOPD 贡献
- 开放问题：软门；无验证器任务；关门策略是否应按域切换（4B MOPD 上 mask 反而更好）

**组会可追问**

1. 35B 代码超过教师，是门真挖出了学生相对优势，还是评测方差？
2. 若把 TrOPD 的 token 信任区叠到“门开”的 prompt 上，会不会双计数可靠性？
3. $K_T=3$ 的 binomial 方差在 $R_T\approx0.5$ 时最大，误开/误关有多频繁？
