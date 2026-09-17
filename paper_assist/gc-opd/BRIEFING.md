# 组会汇报：GC-OPD（Group-Calibrated On-Policy Distillation）

> LaTeX 版：`BRIEFING.tex` → 运行 `bash compile_briefing.sh` 得到 `BRIEFING.pdf`（公式以 PDF 为准）

论文：Zhang*, Wang* et al. *Beyond Teacher Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning*. arXiv:2608.19181v1, 2026-08-19.  
单位：清华大学 / 北京邮电大学 / OpenBMB  
PDF：`docs/3_GC-OPD.pdf`  
代码：https://github.com/SolereZhang/GC-OPD（verl；OPD 入口即 `beta=0`）  
数据：GoLongRL 32K 过滤子集（9527 train / 231 val）  
和组里已读材料的位置：这是 **长上下文外部教师 OPD**，用验证器做 *组内相对残差校准*，不是 TGOPD 那种“整题撤教师”，也不是 RLCSD 那种特权自蒸馏。Related work 点名 Uni-OPD、FiRe-OPD、PowerOPD、ExOPD、SG-OPD、SCOPE、MOPD。

**【源码核验】**：公式以 PDF 为准；训练入口、$\beta=0.10$、数据制备脚本与 README 对齐。

---

## 0. 一句话贡献

长上下文里教师 token 偏好会和任务验证器吵架。GC-OPD **不改教师前向**，只在每个 rollout 组内分别 z-score 验证器奖励和轨迹级 OPD 分，取其差作为有符号残差，再用 RACA 按相对 OPD 优势把残差摊到 token 上，加回原始稠密 OPD 优势。

---

## 1. 任务定义与动机剖析

### 1.1 本 paper 的任务是什么

把 Qwen3-4B / Qwen3-8B（**no-thinking**）在长上下文证据聚合任务上做 OPD 后训练。教师是 Qwen3-30B-A3B-Thinking-2507，只对学生已采样 token 提供 logprob，自己不另生成答案。

训练：GoLongRL 滤到 prompt $\le$ 32K，9527 条，九个任务族（3 个二值奖励、6 个 graded）。评测五基准未加权平均：DocMath、Frames、MRCR、CorpusQA、LBv1QA。

一句话任务：**在保留稠密教师 token 指导的同时，用组内相对的验证器信号校正“教师喜欢但任务没完成”的轨迹。**

### 1.2 要解决的问题

OPD 的 token 优势

$$
A_t^{(i)}=\log\pi_T(y_t^{(i)}\mid x,y_{<t}^{(i)})-\log\pi_{\theta_{\mathrm{old}}}(y_t^{(i)}\mid x,y_{<t}^{(i)})
$$

期望等于负 reverse-KL，测的是 **教师相对学生的偏好**，不是验证器定义的任务成功。长上下文正确性依赖散落证据和全局约束：局部通顺、漏证据的回答仍可能拿高 OPD 分。

### 1.3 Motivation

在固定的学生 8 条回答上诊断 Multi-Table Extraction（751 prompt）和 High-Recall Retrieval（2908 prompt）。随输入变长：

| 任务 | 指标 | $<$8K | 32–64K |
|---|---|---:|---:|
| MTE | pairwise disagreement | 40.6% | 64.0% |
| MTE | OPD preference gap | +0.35 | −0.37 |
| HRR | pairwise disagreement | 35.2% | 60.2% |
| HRR | OPD preference gap | +0.65 | −0.35 |

gap 由正转负：更长输入上，OPD 开始更支持 **验证器分数更低** 的回答。这就是 teacher–verifier disagreement。

### 1.4 现有方法局限

| 方法 | 接口 | 相对本文缺口 |
|---|---|---|
| ExOPD / PowerOPD / FiRe-OPD | 改教师信号几何、过滤、重加权 | 不显式对比轨迹 OPD 分与验证器 |
| SCOPE / MOPD / Reward-Weighted OPD | 按对错路由或加权轨迹 | 容易丢掉 graded 组内间距，或丢掉原始 token OPD |
| Uni-OPD | outcome-class margin | 类间平移，不是组内残差 |
| SG-OPD | token 符号门 + 教师 rollout | 门控而非残差校准 |
| 过程监督 / VinePPO | 步标签或辅助续写 | GC-OPD 两者都不要 |

作者要同时做到：保留 graded 组内差异、把轨迹级分歧变成 **token 依赖** 的校正、且 **不扔掉** 原始 $A_t$。

### 1.5 具象化例子

同一长文档、两答、$G$ 内 z-score 后：

| 回答 | 内容 | $\tilde R$ | $\tilde s$ | $\rho=\tilde R-\tilde s$ |
|---|---|---:|---:|---:|
| A | 局部通顺但漏表 | −0.8 | +1.1 | −1.9 |
| B | 抽全了表但措辞笨 | +0.8 | −0.4 | +1.2 |

Vanilla OPD 会推 A。残差给 A 负校正、B 正校正。RACA 再把 $|\rho|$ 更多摊到相对 OPD 优势高的 token 上（教师特别喜欢/不喜欢的位置），而不是均匀抹在整句。$\beta=0.10$ 保证这是校准不是另起炉灶。

---

## 2. 方法论树状拆解

**创新落点**：一级是 *组内双信号残差*；二级是 RACA 的有界正信用；教师前向零新增。

```text
GC-OPD
├─ 模块 1：组内相对评估（不改教师 pass）
│   ├─ token OPD 优势 At（与 vanilla 相同，detach）
│   ├─ 轨迹分 s = mean_t At（长度归一，避免长答放大）
│   ├─ 验证器 R（二值或 graded）
│   └─ 分别 z-score → R̃, s̃；σ 过小则整组 ρ=0，退回 vanilla
│
├─ 模块 2：有符号分歧残差
│   ├─ ρ = R̃ − s̃
│   ├─ 不为“再加一份验证器”，而是减去 OPD 已表达的相对偏好
│   └─ 若验证器偏好 i 而 OPD 偏好 j，则 ρi−ρj>0（Eq.8）
│
└─ 模块 3：RACA 信用分配 + 原 OPD 保留
    ├─ ut = (At − s) / σ_A   （组内相对，不是 token 正确性）
    ├─ ct = 1 + tanh(ut/2) ∈ (0,2)  恒正，不反转 ρ 符号
    ├─ A' = At + β ct ρ，β=0.10
    └─ clip 到 [−10,10] 后进原 PPO-clip
```

数据流：学生 rollout → 教师 logprob 与验证器并行 → 组统计 → token 校正 → 同一 actor surrogate。$\beta=0$ 精确等于 vanilla OPD（仓库入口如此实现）。

---

## 3. 算法流程与公式细节推演

### 3.1 关键公式（论文 Eq.1, 3, 6–11）

$$
s^{(i)}=\frac1{T^{(i)}}\sum_{t=1}^{T^{(i)}} A_t^{(i)},\qquad
\tilde R^{(i)}=z(R^{(i)}),\quad \tilde s^{(i)}=z(s^{(i)}),\quad
\rho^{(i)}=\tilde R^{(i)}-\tilde s^{(i)}.
$$

$$
u_t^{(i)}=\frac{A_t^{(i)}-s^{(i)}}{\sqrt{\frac1{T^{(i)}}\sum_v(A_v^{(i)}-s^{(i)})^2}+\epsilon},\qquad
c_t^{(i)}=1+\tanh(u_t^{(i)}/2).
$$

$$
A_t^{\prime(i)}=A_t^{(i)}+\beta\,c_t^{(i)}\,\rho^{(i)}.
$$

附录守卫：$\sigma_R,\sigma_s\le\tau_G=10^{-6}$ 则 $\rho=0$；token $\sigma_A\le\tau_T$ 则 $c_t=1$。

### 3.2 手算（$G=2$，两 token，演示形状）

回答 1：$A=(+1.0,-1.0)$ → $s=0$；回答 2：$A=(+0.2,+0.2)$ → $s=0.2$。  
验证器 $R=(0.2, 0.8)$。

组均值 $\mu_R=0.5,\sigma_R=0.3$；$\mu_s=0.1,\sigma_s=0.1$（示意）：

$$
\tilde R\approx(-1,+1),\quad \tilde s\approx(-1,+1)\ \text{若 OPD 与验证器同序则 }\rho\approx0.
$$

改成 $R=(0.8,0.2)$ 而 $s$ 不变：$\tilde R\approx(+1,-1)$，$\rho\approx(+2,-2)$。  
回答 1 的 token1：$u=(1-0)/\sigma_A>0$，$c>1$，负残差被放大——教师最喜欢的那个 token 被压得更狠。这正是“校正摊在高 OPD 位置”。

### 3.3 Algorithm 1 流程

冻结 $\theta_{\mathrm{old}}$ → 每 prompt 采 $G=8$ → 算 $R,A_t,s$ → 组统计与 $\rho$ → 每答算 $c_t$ → $A'=\mathrm{clip}(A+\beta c\rho)$ → 与 vanilla 相同的 token-mean PPO-clip。无额外师生前向。

---

## 4. 实验设定与资源开销

### 4.1 Baselines（共享 100 step 配方；† 为机制复现而非原论文配方）

Raw（官方 ckpt）、Vanilla OPD、ExOPD†（$\lambda=1.25$）、Uni-OPD†（$\delta=0.4$）、PowerOPD†（$\alpha=100$）、FiRe-OPD†（20th percentile 过滤）。

### 4.2 数据

GoLongRL → `scripts/prepare_golongrl_32k.py`：holdout 有序 shard 前 256 条，32K 过滤后 231 val；train 9527。最大两族 Precise Long-Range Retrieval 4693 + Evidence-Grounded Reasoning 3204 = 82.9%。长度：中位 9923，P90 26940，最大 32766。

$\beta$ 只在 231 条 High-Recall holdout 上选，**不当下游评测**。

### 4.3 训练成本（附录 Table 6 + README）

| 项 | 值 |
|---|---|
| step / batch / $G$ | 100 / 32 prompt / 8 |
| max prompt / response | 32768 / 10240 |
| lr / wd / PPO | $1\mathrm{e}{-6}$ / 0.01 / $\epsilon=0.2$，1 epoch，mini-batch 4 |
| 硬件 | 每次 run **8×80GB H800 或 H100**；TP=8，actor/ref SP=8 |
| 精度 | bfloat16；seed 42 |
| 评测上下文 | YaRN scale 4；输入≤120K，生成≤8192，serving 131072 |

相对 vanilla OPD：**零额外前向**，只多组统计和逐元素变换。【估计】墙钟应与 OPD 几乎相同；论文未给 step time 表。

---

## 5. 实验结论的因果支撑

### 5.1 指标

五基准分数均为 %；**Avg.** 为未加权均值。DocMath：四 split 上 $\max(\mathrm{rule},\mathrm{judge})$。FRAMES：$\max(\mathrm{CEM},\mathrm{judge})$。其余见评测 README。方向全部 ↑。

### 5.2 主结果（Table 2）

| 学生 | Raw | OPD | 最强非本文† | **GC-OPD** |
|---|---:|---:|---:|---:|
| Qwen3-4B | 29.08 | 39.31 | FiRe 39.50 | **40.47** |
| Qwen3-8B | 35.12 | 43.56 | FiRe 44.01 | **44.65** |

相对 OPD：+1.16 / +1.09 Avg。增益集中在 DocMath、MRCR、尤其 **CorpusQA**（4B 32.22→37.99）。Frames / LBv1QA 小或不稳定。**【因果】** 与“证据聚合需要响应级校正”动机一致（中强支撑）；单表不能隔离任务机制。

相对 Raw 的大幅提升主要来自 **整条长上下文 OPD 管线**；GC-OPD 相对 OPD 的增量才是方法贡献。

### 5.3 消融（Qwen3-8B，Table 3–4）

信号消融（都用 RACA，$\beta=0.10$）：

| 变体 | Avg | $\Delta$ vs OPD | 含义 |
|---|---:|---:|---|
| Vanilla OPD | 43.56 | — | 锚 |
| Additional OPD | 43.60 | +0.04 | 再加一份 OPD 几乎没用（强支撑：不是“多加优势”） |
| Direct reward | 44.19 | +0.63 | 组内验证器有用 |
| GC-OPD $\rho$ | 44.65 | +1.10 | 残差优于直接加奖励（中强支撑） |

Token 分配（残差固定）：

| 分配 | Avg | $\Delta$ |
|---|---:|---:|
| Absolute $\|A\|$ | 43.93 | +0.38 |
| Uniform $c=1$ | 44.28 | +0.72 |
| RACA | 44.65 | +1.10 |

均匀摊已经有用；RACA 再 +0.38。Absolute 丢掉 $A$ 的符号会伤害（支撑 RACA 保序）。

### 5.4 弱支撑 / 限制

- 无多种子；100 step 固定终点
- $\beta$ 只在单一任务 holdout 上选
- 诊断用冻结回答，不能证明“更长 → 训练增益更大”
- † baseline 是共享配方下的机制移植
- 学生 no-thinking、教师 thinking，师生模式不对称未单独消融

---

## 6. 复现前置准备清单

- [ ] **数据**：`hf download Kwai-Klear/GoLongRL`；`python scripts/prepare_golongrl_32k.py --overwrite` → 9527/231
- [ ] **权重**：`Qwen/Qwen3-4B`、`Qwen/Qwen3-8B`、`Qwen/Qwen3-30B-A3B-Thinking-2507`；评测 judge 另需 Instruct-2507
- [ ] **硬件**：论文 8×80GB；最低冒烟【估计】4B 需压短 prompt/response 才可能单卡，官方入口按 8 卡写
- [ ] **软件**：Python 3.12，CUDA 12.8，torch 2.10.0，vLLM 0.17.0，Ray 2.54，Transformers 4.57.1，FA 2.8.3
- [ ] **代码**：`bash scripts/run_gc_opd_4b_training.sh`；OPD 对照 `run_opd_*`（`beta=0`）
- [ ] **算法必做**：组 z-score 残差、RACA、$A'=A+\beta c\rho$、$\sigma$ 守卫、advantage clip
- [ ] **可延后**：五基准完整评测、frozen-rollout 分析脚本
- [ ] **Sanity**：`beta=0` 与 OPD 数值一致；组内 $R$ 全相同 → $\rho=0$；$\rho$ 符号与“验证器更高的答”一致

---

## 7. Paper ↔ Code gaps / 开放问题

- 仓库与论文主设定对齐良好：$\beta=0.10$，RACA，100 step，seed 42
- 核心张量在 `verl/verl/trainer/ppo/gc_opd.py`
- 开放：thinking 学生；$>$32K 训练；把残差与 TGOPD 门同时用（门关时无教师 $s$，残差未定义）

**组会可追问**

1. CorpusQA 大涨、LBv1QA 不涨，是验证器质量差异还是任务不需要全局聚合？
2. $\beta$ 与验证器尺度耦合，换二值/graded 混合是否该分任务设 $\beta$？
3. RACA 用 OPD 相对优势分配验证器残差，若 OPD 已系统性错，会不会把校正摊到错误 token 上？
