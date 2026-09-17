# 组会汇报：三篇 Trust-Region On-Policy Distillation 论文对照精读（TeX 源码核验版）

**材料范围**：

1. **TRB** — *Trust-Region Behavior Blending for On-Policy Distillation*（以下简称 TRB）
2. **TrOPD** — *Trust Region On-Policy Distillation*（以下简称 TrOPD）
3. **TOP-D** — *Trust Region Policy Distillation*（以下简称 TOP-D）

**核验方式**：除 PDF 正文外，逐一对照三套作者 TeX 源码（`arxiv.tex`、`main.tex`、`neurips_2026.tex`）及附录中的超参数、训练协议、限制说明和算法伪代码。本文中“源码核验”表示该细节可在所附 TeX 中定位；“分析判断”表示基于论文证据给出的研究评价，不应与论文作者主张混淆。

---

# 1. 结论先行：三篇论文在解决同一个问题，但切入层级不同

## 1.1 一句话结论

三篇都针对 **On-Policy Distillation（OPD）在师生能力差较大时的训练不稳定**，但分别干预：

- **TRB**：干预“学生从哪些前缀继续生成”——早期改变 **rollout 行为分布**；
- **TrOPD**：干预“每一个学生 token 应接受何种监督”——做 **token 级可信度分区与 KL 分流**；
- **TOP-D**：干预“奖励函数怎样避免无界”及“采样数据怎样复用”——做 **奖励有界化 + PPO 风格内部信任域迭代**。

它们可被看作同一问题的三个正交控制面：**状态/前缀分布、token 级信用分配、奖励塑形与优化器稳定性**。

## 1.2 方法地图

```text
标准 OPD：x ~ πS，教师以 log[πT(x)/πS(x)] 给 token 级密集奖励
                         │
                         ├─ 病因 A：早期学生前缀跑偏、低质量
                         │       └─ TRB：将行为策略投影至学生 KL 信任域内最接近教师的分布
                         │
                         ├─ 病因 B：某些学生 token 获得 πT≈0，K1 估计器出现离群梯度
                         │       └─ TrOPD：信任区 RKL；离群区 top-k FKL；教师前缀引导
                         │
                         └─ 病因 C：r=log(πT/πS) 在 πT→0 时趋于 -∞；严格 on-policy 不能复用样本
                                 └─ TOP-D：r~=log(αρ+1-α) 设下界；PPO clip 内部迭代复用样本
```

## 1.3 不应直接把三个结果横向比较

三篇实验协议不同：

| 项目 | TRB | TrOPD | TOP-D |
|---|---|---|---|
| 训练领域 | 数学 | 数学、代码、科学、指令 | 数学 |
| 学生/教师 | Qwen3 0.6B←4B、1.7B←8B | 1.5B/1.7B 学生，4B–7B 教师 | 1.7B/8B 学生，14B/30B 教师 |
| 训练步数 | 源码未在主表固定披露；每 20 step 测 checkpoint | 固定 200 step | 约 200–400 step |
| 主评测 | pass@1；模型配置内选择最高平均分 checkpoint | 每项 32 次评测的平均准确率 | avg@32 / avg@8 |
| 重要影响 | 结果包含 sweep 与 best-checkpoint 选择 | 统一 benchmark 比较较强 | OPD 基线与 TOP-D 的 batch/更新形态不同 |

因此可以比较“机制”，不能将表中数值当作同一赛道排名。

---

# 2. 共同背景：为什么 OPD 既必要又不稳定

给定 prompt `q`，学生策略为 `πS`，强教师为 `πT`。离线蒸馏用教师轨迹训练学生，但部署时由学生自己生成，导致前缀分布不一致（exposure bias）。OPD 改为用学生自身轨迹训练：

$$
\min_θ D_{KL}(π_S\|π_T)
= \mathbb{E}_{x\sim π_S}\left[\log\frac{π_S(x)}{π_T(x)}\right].
$$

实践中，对超长 CoT 若保存全词表概率，显存为 `O(n|V|)`，代价高；因此通常采用 token 级 `K1` / log-ratio 估计，降为 `O(n)`。但当学生采到教师几乎不会采的 token 时：

$$
r_t=\log\frac{π_T(x_t\mid h_t)}{π_S(x_t\mid h_t)}\to-\infty \quad (π_T\to0),
$$

由 score-function 乘以无界奖励形成大方差、甚至异常梯度。与此同时，早期学生本身无法写出合理推理前缀，教师即便提供局部 token 监督，也是在“错误状态”上监督。

可把三篇的共同问题概括为：

> **在不牺牲 OPD 的学生分布优势、又无法采用全词表精确 KL 的约束下，怎样使教师监督既可靠、又稳定、又可训练？**

---

# 3. 论文一：TRB（Trust-Region Behavior Blending）

## 3.1 任务、问题、动机与已有方法局限

TRB 的核心判断是：既有 OPD 改进多数修改“已经访问到的前缀上的损失/目标”，却很少控制“训练时究竟访问了哪些前缀”。

### 问题实例

学生早期在数学题上可能数步后偏离题意；教师在这个错误前缀下给 token 分数，并不能恢复到高质量推理状态。若直接改成教师生成，又会把 on-policy 蒸馏拉回 off-policy 蒸馏。

### 对比对象与局限

| 方法 | 作用位置 | 局限 |
|---|---|---|
| Vanilla OPD | 纯学生采样 | 前缀分布正确，但早期前缀质量可能极差 |
| Veto | 已访问状态的目标侧重构 | 不改变当前访问前缀的来源 |
| SKD / teacher injection | rollout 时直接替 token | 干预较硬，非显式控制分布偏移 |
| Temperature warmup | 降低学生采样温度 | 更保守但不显式追近教师 |
| SFT warmup | 先模仿教师轨迹 | 仍是离线教师分布，可能有 exposure bias |
| Fixed-ε blending | 全程保持教师行为混合 | 作者实验显示持续引导不如限时 warmup |

TRB 的动机不是“始终向教师靠近”，而是：**训练早期需要有限的教师引导；学生已进入可学习状态后，应完全回到学生分布。**

## 3.2 方法树状拆解

```text
TRB
├─ 不变：逐前缀 reverse-KL OPD 目标
│  └─ 只改采样行为策略 μ，不改学生参数的监督目标
├─ 行为策略优化：每个前缀 h 解一个约束问题
│  ├─ 目标：minμ DKL(μ || πT)
│  └─ 约束：DKL(μ || πS) ≤ ε
├─ 闭式求解：几何混合族 μβ
│  ├─ μβ ∝ πS^(1-β) πT^β
│  ├─ β*：最大可行 β
│  └─ 用二分搜索求 β*；KL 对 β 单调
└─ 时间调度：仅 warmup
   └─ εk = ε0(1-k/K)，k>K 时 ε=0，μ=πS
```

## 3.3 关键公式与直觉

对前缀 `h`：

$$
\mu^*(\cdot|h)=\arg\min_\mu D_{KL}(\mu\|π_T)
\quad s.t.\quad D_{KL}(\mu\|π_S)\le\varepsilon.
$$

闭式解：

$$
\mu_β(a|h)=\frac{π_S(a|h)^{1-β}π_T(a|h)^β}{Z_β(h)},\qquad β\in[0,1].
$$

- `β=0` 时，`μ=πS`；
- 若教师本身满足 `DKL(πT||πS)≤ε`，则直接 `μ=πT`；
- 否则将 `β` 推到恰好用尽 KL 预算的位置。

这不是概率线性平均，而是 **logit / log-probability 的线性插值后 softmax**。即：教师很不认可、学生却很高的 token 会明显被压低，但仍受学生中心 KL 约束，不会无条件复制教师。

### 小例子

令：

```text
πS=(0.50, 0.30, 0.20)
πT=(0.70, 0.20, 0.10)
```

当 `β=0.5`：

```text
μβ ∝ (sqrt(0.5×0.7), sqrt(0.3×0.2), sqrt(0.2×0.1))
   = (0.592, 0.245, 0.141)
归一化后 μ≈(0.605, 0.250, 0.145)
```

采样分布从学生向教师移动，但没有跳到教师 `(0.7,0.2,0.1)`，这就是“信任域内的温和行为校正”。

## 3.4 算法伪流程

```text
for global step k:
    ε ← ε0 × max(1-k/K, 0)
    for each rollout prefix h:
        read student distribution πS(.|h), teacher distribution πT(.|h)
        if ε = 0: μ ← πS
        else:
            find largest β∈[0,1] s.t. DKL(μβ || πS) ≤ ε  # bisection
            μ ← μβ
        sample next token from μ
    compute original reverse-KL OPD loss on generated rollouts
    update student θ
```

## 3.5 源码核验补充：真正的实验与实现细节

以下均来自 `TrustRegionBehaviorBlendingOPD/arxiv.tex`：

- 训练系统：`verl + SGLang + FSDP2`；硬件：**8× NVIDIA H100**。
- 数据：从 OpenThoughts3-1.2M 中采样 **25,600** 个训练 prompt；统一加系统提示“reason step by step，并将答案置于 `\boxed{}`”。
- 固定 OPD 项：学生 actor top-`k=16` support 上的 reverse KL；学习率 `1e-5`，AdamW，weight decay `0.01`，梯度裁剪 `1.0`，global batch `64`，每题 4 rollout，最大 prompt `1024`、response `7168`，PPO epoch `1`，rollout temperature `1.0`。
- TRB sweep：`ε0 ∈ {0.001,0.005,0.01,0.02,0.05}`，`K∈{15,25,50}`；`ε` **线性**退火。
- 一个易漏实现点：师生 Qwen3 tokenizer 的原始 EOS id 不同。论文在构建 μ 与 KL 计算前做 **EOS canonicalization**：将两侧 EOS 概率映射为共同事件，再将采样事件还原为学生 EOS。忽略此项会令停止概率被分散到不同坐标，KL 与行为混合失真。

## 3.6 实验、指标与证据强度

主表（pass@1）关键结果：

| 设置 | TRB Avg | Vanilla OPD Avg | Fixed-ε Avg | 最强其它 baseline |
|---|---:|---:|---:|---:|
| Qwen3-1.7B ← Qwen3-8B | **33.2** | 32.3 | 32.6 | Temperature warmup 32.8 |
| Qwen3-0.6B ← Qwen3-4B | **44.4** | 44.0 | 43.8 | SKD 44.2 |

作者的主要证据链：

1. **TRB 优于 Fixed-ε**：同一局部求解器，只有“限时退火”不同，说明持续教师化行为并不一定好；
2. **早期前缀 continuation probe**：固定截断长度和续写模型，仅改变前缀来源；TRB 前缀被教师或学生续写时均更易成功，支持“前缀状态变好”；
3. **教师 token 熵曲线**：TRB 仅在 warmup 内显著改变被访问前缀的教师侧熵，warmup 结束后与 OPD 接近，但最终性能差距仍保留；
4. **定性 sample**：纯学生早期样本易跑题，TRB 保持题目算术结构。但该证据只能作 sanity check，不是量化证明。

### 证据评估

优点：对“只改行为、保持损失不变”的因果切分较干净；Fixed-ε 对照很关键；源码明确承认连续引导并不更好。

局限：

- 主增益小（0.9 / 0.4 平均分）；没有多随机种子/置信区间；
- 主表对每个方法族每 20 step 评测一次，并选“**该配置平均分最高的 checkpoint**”；虽然协议对所有方法一致，但这是 peak-selection，而不是固定训练预算的稳定性比较；
- 只验证数学与两对 Qwen3 师生；
- 源码直接承认 TRB warmup 期间需要在线教师解码和师生共驻留，墙钟时间可能高于“学生 rollout 后批量教师前向”的 vanilla OPD。

---

# 4. 论文二：TrOPD（Trust Region On-Policy Distillation）

## 4.1 任务、问题与动机

TrOPD 直接针对 long-CoT OPD 的 **token-level supervision reliability**。它认为普通 OPD 的 `K1` 估计器在师生失配严重的 token 上产生不可信的 RKL 策略梯度；但将这些位置简单 mask 或 clip 又会丢失有价值的教师信息。

它提出的核心问题是：

> 一个学生生成 token 处于“教师可可靠监督的区域”还是“教师不认可、K1 估计不可靠的离群区”？两类 token 是否必须使用同一散度和同一种估计器？

## 4.2 方法树状结构

```text
TrOPD
├─ 模块 1：自适应 token 信任区判定
│  └─ Ptrust(x)=min(πT(x)/πS(x), 1)，Mx~Bernoulli(Ptrust)
├─ 模块 2：学生轨迹上的分区目标
│  ├─ Mx=1（信任区）：RKL，用 K1，O(n)
│  └─ Mx=0（离群区）：FKL，用教师 top-k，O(nk)
├─ 模块 3：教师前缀的 off-policy guidance
│  ├─ 教师生成前缀 x[:l]，学生续写 x[l:]
│  ├─ 教师段用 FKL 的 K1 形式进行 imitation
│  └─ 最大教师前缀长度余弦退火至 0
└─ 统一目标：分区 RKL/FKL + 小权重教师轨迹引导
```

## 4.3 核心公式

### 信任概率

$$
P_{trust}(x)=\min\left(\frac{π_T(x)}{π_S(x)},1\right),\qquad
M_x\sim Bernoulli(P_{trust}(x)).
$$

该形式来自 speculative decoding 的教师接受率。若教师给学生 token 的概率至少不低于学生自己，则 `P=1`；若教师远低于学生，则大概率进入 outlier。

### 分区目标

学生采样 token：

$$
\mathcal{J}^{On}_x=-M_x\log\frac{π_S}{π_T}
-(1-M_x)\sum_{v\in V_T^k}π_T(v)\log\frac{π_T(v)}{π_S(v)}.
$$

- **信任区**：RKL 的 K1 估计，保留 OPD 的 on-policy 特性与低内存；
- **离群区**：不再对“学生已采到的坏 token”使用 log ratio，而从教师 top-k token 分布出发使用近似 FKL，保留教师支持的可学习方向。

教师前缀引导：

$$
\mathcal{J}^{guide}=-β\,\mathbb{I}[x\sim π_T]\log\frac{π_T}{π_S},\qquad β=0.001.
$$

## 4.4 算法伪流程

```text
for each training batch:
    construct trajectory = teacher prefix + student continuation
    for student-generated token x:
        p = min(πT(x)/πS(x), 1)
        M ~ Bernoulli(p)
        if M == 1:
            use K1 reverse-KL term log(πT/πS)
        else:
            use teacher top-k forward-KL term
    for teacher-generated prefix token:
        use β-weighted forward-KL K1 imitation term
    update student
    cosine-anneal teacher-prefix maximum length to 0
```

## 4.5 一个数值例子

若学生对 token `a` 的概率为 0.4，而教师为 0.02：

$$P_{trust}=\min(0.02/0.4,1)=0.05.$$

该 token 以 95% 概率被视为离群。标准 OPD 对它使用 `log(0.02/0.4)≈-3.00`；TrOPD 则转而查看教师 top-64 中的高质量候选 token，并把学生分布拉向它们，而不是继续强化这个极端 RKL 信号。

## 4.6 源码核验补充

来自 `TrustRegionOPD/main.tex`：

- benchmark 训练完全统一：**200 steps**，固定 lr `5×10^-6`，prompt batch `128`，每 prompt 4 rollout，最大生成 `8096`；FKL support `k=64`；引导 `β=0.001`。
- 训练数据仅保留 OpenThoughts3 prompts：单域保留数学；多域覆盖数学、代码、科学。
- 学生/教师：
  - 单域：DeepSeek-R1-Distill-Qwen-1.5B ← Skywork-OR1-Math-7B；
  - 多域：DeepSeek-R1-Distill-Qwen-1.5B ← Skywork-OR1-7B；
  - 多域：Qwen3-SFT-1.7B ← Qwen3-Nemotron-4B。
- `Qwen3-Nemotron-4B` 不是现成教师，而是作者从 Qwen3-4B-Base 以 Nemotron SFT/RL 数据训得。其附录给出约 14M SFT 样本，RLVR 四域训练混合及 32K rollout 配置。
- 论文给出区域显存复杂度：信任区 `O(n)`；离群区 top-k FKL `O(nk)`；教师 prefix guidance 的 K1 为 `O(n)`。

## 4.7 主要结果与解释

### 单域与多域（1.5B 学生）

| 设置 | OPD Avg | REOPOLD Avg | TrOPD Avg | 相对 OPD |
|---|---:|---:|---:|---:|
| 单域 | 37.11 | 38.79 | **40.63** | +3.52 |
| 多域 | 32.99 | 35.58 | **37.61** | +4.62 |

### Qwen3-SFT-1.7B 多域

| 指标 | OPD | TrOPD | 增量 |
|---|---:|---:|---:|
| AIME24 | 48.02 | 52.08 | +4.06 |
| AIME25 | 40.72 | 44.06 | +3.34 |
| GPQA Diamond | 29.80 | 35.98 | +6.18 |
| IFBench | 37.07 | 42.18 | +5.11 |
| LiveCodeBench v6 | 32.00 | 36.00 | +4.00 |
| Overall Avg | 48.29 | **51.73** | +3.44 |

### 最关键的消融

| 方法（数学单域） | Avg |
|---|---:|
| OPD | 46.79 |
| Mask Outlier | 47.72 |
| Clip Outlier | 47.86 |
| Full top-k FKL | **1.40** |
| FKL only on Outlier | 49.00 |
| TrOPD-FKL（再加 guidance） | **49.85** |

该表最值得记住的两个结论：

1. **top-k FKL 不能作为全局独立目标**（会崩至 1.40），说明它是有偏近似；
2. **top-k FKL 只在离群区使用却有效**，表明“在何处使用”比“用什么 KL”更关键。

此外，TrOPD + AOPD 在同一表中达到 41.67，高于 TrOPD 40.63，证明 AOPD 与其机制可叠加。

## 4.8 研究评价与限制

优点：三篇中 benchmark 最完整，横向比较了 RKL/FKL/JSD、熵过滤、mask、clip、REOPOLD、AOPD；对“离群区不能只删、也不能全局改 FKL”的证据链较强。

需要谨慎的点：

- `M_x~Bernoulli(Ptrust)` 引入采样方差；没有展示 soft-gating 或确定性阈值的充分比较；
- `Ptrust` 是单 token 概率比，不一定等同于“整段 CoT 的教师可验证性”；
- 离群区 FKL 仍是 top-k 有偏近似，作者展示了它全局使用会失败，故需要进一步解释为何局部使用时稳定；
- 没有提供硬件资源与训练 wall-clock；
- 固定 200 step 且未报告多种子误差条；
- 作者明确限制在 OPD post-training，未覆盖 pretraining/mid-training，也未给实际部署研究。

---

# 5. 论文三：TOP-D（Trust Region Policy Distillation）

## 5.1 任务与动机

TOP-D 从更基础的数学形式出发：OPD 的即时奖励是

$$r_k=\log\rho_k,\qquad \rho_k=\frac{π^*(y_k|h_k)}{π_θ(y_k|h_k)}.$$

若教师对学生 token 概率趋近零，则 `r_k→-∞`。TOP-D 不首先判断 token 是否离群，而是直接改变目标教师：每一步以当前学生为锚，构造 **proximal teacher**。

同时它关注另一个 OPD 问题：严格 on-policy 采样后更新一次即丢弃轨迹，样本利用率低。

## 5.2 方法树状结构

```text
TOP-D
├─ External Proximal Teacher：奖励有界化
│  ├─ π~*=απ*+(1-α)πθ（概率空间插值）
│  ├─ r~=log(αρ+1-α)
│  └─ 对极端负奖励设置下界 log(1-α)
├─ Internal Trust Region Iterations：安全数据复用
│  ├─ 固定行为策略 πold 采 G 条 rollout
│  ├─ 计算 token return 与组内 token-level advantage
│  └─ PPO clipping 目标，对同一批数据做内部 epoch
└─ 理论闭环
   ├─ 奖励变换 → 梯度方差有界
   ├─ proximal operator → 全局收敛误差界
   └─ trust-region surrogate → 单调改进与单步误差控制
```

## 5.3 核心公式与关键设计选择

### 外部 proximal teacher

$$
\tilde π^*(y_k|h_k)=απ^*(y_k|h_k)+(1-α)π_θ(y_k|h_k),\quad α\in(0,1).
$$

得到：

$$
\tilde r_k=\log\frac{\tildeπ^*}{π_θ}
=\log(αρ_k+1-α).
$$

由于 `ρ>0`：

$$
\tilde r_k\ge\log(1-α),
$$

因此负侧不再无界。

### 为什么必须是概率空间插值

若在 log-probability 空间插值：

$$
\log\tildeπ^*=α\logπ^*+(1-α)\logπ_θ,
$$

则 `\tilde r=αr`，只是把无界奖励缩放，仍可趋于 `-∞`。TOP-D 的关键不是小奖励，而是 **非线性地压缩负侧并给下界**。

### 内部信任域/PPO 数据复用

采用 `πold` 生成数据，重要性比：

$$
p_t^i=\frac{π_θ(y_t^i|h_t^i)}{π_{old}(y_t^i|h_t^i)}.
$$

目标为 PPO clip：

$$
J(θ)=\frac{1}{\sum_i|y^i|}\sum_{i,t}
\min\left(p_t^i\hat A_t^i,
\operatorname{clip}(p_t^i,1-ε,1+ε)\hat A_t^i\right).
$$

其 return 不是简单累加，而是：

$$
\tilde R_k^i=\tilde r_k^i+
\frac{1}{|y^i|-k}\sum_{j=k+1}^{|y^i|}\tilde r_j^i,
\qquad
\hat A_k^i=\frac{\tilde R_k^i-\mu}{\sigma}.
$$

`μ,σ` 是同一 prompt 的所有 response、所有 token 上统计的 token-level 归一化。作者声称这比序列级归一化更适合 dense token reward，并缓解回答长度偏置。

## 5.4 数值直觉

取 `α=0.5`：

| `ρ=π*/πθ` | OPD `logρ` | TOP-D `log(0.5ρ+0.5)` |
|---:|---:|---:|
| 3 | 1.099 | 0.693 |
| 1 | 0 | 0 |
| 0.1 | -2.303 | -0.598 |
| `ρ→0` | `-∞` | `log(0.5)=-0.693` |

可见 TOP-D 既抑制负侧离群，也压缩正侧极端奖励；`α` 是“接近教师强度”和“方差控制强度”的耦合旋钮。

## 5.5 理论主张应如何理解

### 方差有界

在 score function 有界的假设下：

$$
Var(\tilde g_k)\le M^2|V|\max\{(\log(1-α))^2,C^*α\}.
$$

解释：`α→1` 恢复标准 OPD，负侧界发散；`α→0`，奖励及方差趋于零。

### 收敛界

把 proximal teacher 当收缩算子：

$$
T(π)=απ^*+(1-α)π.
$$

若实际更新包含优化误差 `εk`：

$$
d(π_{k+1},π^*)
\le(1-α)^{k+1}d(π_0,π^*)+
\sum_{i=0}^{k}(1-α)^{k-i}\|ε_i\|_1,
$$

故：

$$
\limsup d(π_{k+1},π^*)\le ε_∞/α.
$$

这揭示权衡：较小 `α` 更稳，但若单步优化误差不够小，最终接近教师的精度受 `ε∞/α` 限制。作者于是以 PPO 风格内部迭代降低 `εk`。

## 5.6 源码核验补充

来自 `TrustRegionPD/neurips_2026.tex`：

- 训练数据：DAPO-Math-17k；学生 Qwen3-1.7B-Base / Qwen3-8B-Base；教师 Qwen3-14B / Qwen3-30B-A3B-Instruct-2507。
- 每 prompt 采样 `G=8` 条 response；global batch 512 prompt（4096 samples），mini-batch 32 prompt（256 samples），off-policy epoch=1。
- 各方法的公共设置：AdamW，学习率 `1e-6`，最大 prompt `2048`，最大 response `16384`；采样 `temperature=1.0, top-p=1.0`；验证 `temperature=1.0, top-p=0.7`。
- TOP-D：PPO clip 上下界均 `0.2`，`α=0.1/0.2`。图中还试验了 `α=0.3`。
- 硬件：主要实验为 **4 节点×8 H200 = 32 H200**；作者声称单机 8 GPU 可复现，但未在正文给出单机实际耗时/吞吐表。
- 限制：仅学生≤8B；仅约 200–400 update step（RLVR baseline 除外）；尚未出现饱和，也未测试 >30B student。

## 5.7 主要实验

### 8B 学生、30B-A3B 教师

| 方法 | AIME24 avg@32 | AIME25 | AIME26 | AMC23 | MATH-500 avg@8 | Olympiad avg@8 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 9.38 | 8.02 | 6.15 | 53.13 | 75.23 | 40.08 |
| GRPO | 30.10 | 22.08 | 21.67 | 56.33 | 76.83 | 44.92 |
| DAPO | 32.92 | 27.81 | 32.29 | 65.39 | 81.65 | 44.23 |
| OPD | 24.58 | 23.33 | 25.42 | 76.88 | 87.98 | 59.29 |
| **TOP-D** | **50.42** | **34.06** | **44.06** | **88.13** | **91.23** | **64.67** |

对标准 OPD 的 AIME24 提升为 **+25.84 absolute points**。在 1.7B 学生上，TOP-D 对两个教师也全面优于 OPD；这与其“师生 gap 越大、无界 log ratio 越危险”的动机一致。

### 消融

- 令 `α=1.0`，即恢复无界 OPD reward：学习曲线明显不稳定；
- 去除 internal trust-region iterations（严格 on-policy）：收敛更慢，样本效率下降；
- `α∈{0.1,0.2,0.3}`：曲线差异较小，显示在该区间内的超参鲁棒性。

## 5.8 研究评价与限制

优点：机制非常简洁；外部 proximal teacher 不需要额外教师前向——在已有 logprob 的 OPD 流程中只是 reward 代数变换；理论与算法的对应关系是三篇中最完整的。

需审慎的点：

1. **“zero additional computational overhead”需精确定义**：奖励变换本身确实零额外教师前向；但 TOP-D 采用 off-policy mini-batches/PPO update，而对照 OPD 的 mini-batch=512、无 off-policy epoch。因此总训练吞吐是否零额外，仍应报告墙钟成本。
2. OPD baseline 在 8B AIME24 为 24.58，低于 DAPO 32.92；须确认标准 OPD 是否已充分调参。大幅提升的一部分可能源于“稳定化”，也可能源于对照训练配方较弱。
3. 未与 REOPOLD、EOPD、TrOPD、AOPD 进行同期横向比较。
4. 理论依赖 score-function 有界、有限 horizon、优化 surrogate 足够好的假设；方差界含 `|V|` 与未知的 `M,C*`，是定性稳定性保证，不是实际训练方差的紧界。

---

# 6. 三篇方法的严格对照

| 维度 | TRB | TrOPD | TOP-D |
|---|---|---|---|
| 最小干预单元 | 前缀/行为分布 | 学生采样 token | token reward + batch update |
| 主要风险定义 | 早期低质 prefix | 教师不认可的 outlier token | 无界 log-ratio reward |
| 核心操作 | KL 约束下的几何混合采样 | 基于 πT/πS 分区，RKL/FKL 分流 | 概率线性插值 teacher，奖励变换 |
| 教师如何介入 | 仅 warmup 时在线参与每步采样 | top-k logits；另有教师 prefix | 只需常规 OPD 已有的教师 token logprob |
| 是否改 OPD loss | 不改 per-prefix RKL | 改为 region-specific RKL/FKL | 改奖励、PPO surrogate |
| on-policy 恢复 | ε 线性退火到 0 | 教师 prefix 长度余弦退火到 0 | 用 πold/πθ 比率做有限 off-policy reuse |
| 主要超参 | ε0, K | k=64, β=0.001, prefix schedule | α, PPO ε, epoch |
| 主要工程代价 | 在线教师解码、师生共驻留 | outlier top-k FKL 的 O(nk) | reward本身低；数据复用的 PPO 管线 |
| 主要理论 | 行为分布闭式解、局部效率及序列 KL 控制 | 更偏 benchmark/经验性 | 方差、收敛、单调改进三层理论 |
| 最强证据 | 对前缀质量的 controlled continuation probe | 全面 baseline/消融 benchmark | 大幅数学提升 + 两模块学习曲线消融 |

## 6.1 表面相似、数学上不同的“混合”

| 方法 | 混合位置 | 形式 |
|---|---|---|
| TRB | **采样行为分布** | `μ∝πS^(1-β)πT^β`，几何/指数族混合 |
| TrOPD | **token 归属或损失选择** | Bernoulli gate，概率由 `min(πT/πS,1)` 决定 |
| TOP-D | **目标教师分布/奖励** | `π~=απT+(1-α)πS`，概率线性混合 |

因此不能简单说“三篇都在做 teacher-student blending”。它们虽然都使用“师生接近”的概念，但分别在 rollout policy、credit assignment、reward target 三个层面操作。

---

# 7. 可能的组合研究方向

## 7.1 正交组合假设

一个自然的组合为：

```text
Warmup 前期：TRB 改善访问状态分布
全程 token 监督：TrOPD 处理可靠区/离群区
全程 reward：TOP-D 对 RKL 信号有界化
优化器：TOP-D 的 PPO style data reuse
```

但不能直接叠加，至少有四项需要实验验证：

1. **分布语义是否冲突**：TRB 由 `μ` 采样，但 TrOPD 的 `πT/πS` gate 仍以学生 πS 解释；必须决定重要性修正或重新定义 gate。
2. **重复抑制负奖励**：TrOPD 的 mask/FKL 与 TOP-D 的 reward lower bound 都在应对极端负 RKL，可能冗余或过度削弱纠错。
3. **Teacher 前向成本**：TRB 在线教师解码 + TrOPD 离群 token top-k logits，组合时吞吐压力明显。
4. **退火时序**：TRB 的行为偏移预算、TrOPD 的教师 prefix 长度、TOP-D 的 α 都可成为“师生距离调度”。三者若各自独立退火，难以解释因果。

## 7.2 最有价值的统一实验

在同一训练框架、相同学生/教师、同一数据/步数/评测预算下，建立 2×2×2 消融：

- 行为层：TRB off/on；
- token 信用层：TrOPD off/on；
- reward/更新层：TOP-D off/on。

统一报告：固定 step 结果、多随机种子均值±置信区间、成功率/长度、梯度范数分布、teacher logprob 分布、离群 token 比例、GPU 吞吐与墙钟时间。该设计比继续堆叠 SOTA 分数更能建立机制性贡献。

---

# 8. 复现路线图

## 8.1 共用基础设施

- 训练：PyTorch 2.x、CUDA 12.x、FSDP/FSDP2；优先 `verl`；
- rollout：SGLang（TRB 源码明确使用），可比较 vLLM；
- 数学验证：`math-verify`；
- 模型：Qwen3 family、Skywork OR1、DeepSeek R1 Distill；
- 指标：固定统一为每基准固定采样次数，避免各篇的 pass@1/avg@32 差异；
- 最低建议硬件：1 节点 8×80GB GPU（H100/H200/A100），必要时降低 response length 或先用 1.7B 学生。

## 8.2 推荐顺序

### 阶段 A：先实现稳定 OPD 基线

1. 固定学生/教师，例如 Qwen3-1.7B ← Qwen3-14B 或 4B；
2. 固定数据子集与种子；
3. 同时记录 token-level teacher/student logprob、rollout 长度、梯度 norm、正确率；
4. 先复现标准 RKL/K1，确认没有 EOS/tokenizer 对齐错误。

### 阶段 B：复现 TOP-D（最简洁）

仅将 reward 替换为：

```python
rho = exp(logp_teacher - logp_behavior_student)
r_tilde = log(alpha * rho + 1 - alpha)
```

然后接入 group rollout、token-level advantage 与 PPO clip。它最适合作为稳定 OPD 的第一版，因为不依赖 top-k FKL 和动态 token gate。

### 阶段 C：复现 TrOPD benchmark

加入教师 top-64 logits、`M~Bernoulli(min(pt/ps,1))`、离群区 FKL、教师前缀长度退火。优先验证其消融次序是否重现：

```text
OPD < Mask/Clip < FKL-only-outlier < FKL-outlier + guidance
```

### 阶段 D：复现 TRB

实现 per-prefix bisection 求 `β*`、EOS canonicalization、在线教师解码与 ε warmup；最后与 Fixed-ε 严格比较。

## 8.3 必做 sanity checks

| 项目 | 检查内容 |
|---|---|
| EOS 对齐 | 同一“终止事件”在师生词表中仅占一个 KL 坐标 |
| TRB | `DKL(μβ||πS)≤ε`；β=0 回到 πS；足够大 ε 时回到 πT |
| TrOPD gate | `πT≥πS` 时 Ptrust=1；极低 πT 时 gate 应主要进入 outlier |
| TrOPD FKL | 只对 outlier 使用；确认 top-k 来自教师而非学生 |
| TOP-D reward | `α<1` 时 r~ 永远不小于 `log(1-α)`；α=1 精确退化为 OPD |
| TOP-D advantage | 按 prompt group 全部 token 而非序列级归一化 |
| 公平比较 | 相同 prompt、rollout 数、max length、总 token 预算、评测样本数与随机种子 |

---

# 9. 组会可直接讨论的问题

1. **TRB 的小增益是否超过 peak-checkpoint 选择噪声？** 应固定训练 token budget 并报告种子方差。
2. **TrOPD 的 gate 能否由 Bernoulli 改成 soft weight？** 这可能降低估计方差，也可检验“硬分区”是否必要。
3. **TOP-D 的惊人提升是否在充分调优的 OPD 上仍成立？** 需要把 OPD 的 batch、warmup、优化更新数与 TOP-D 对齐。
4. **teacher distance 应否随训练统一调度？** 可把 TRB ε、TrOPD prefix length、TOP-D α 纳入同一个可学习/自适应控制器。
5. **长 CoT 成功应按 token reward 还是 sequence outcome 定义？** 三篇均主要解决 token supervision，但最终评估是答案正确率；二者间的信用归因间隙仍未完全解决。

---

# 10. 最终判断

- **TRB** 最适合回答“OPD 一开始应从哪些状态学习”：贡献精巧、实现细节扎实（尤其 EOS 对齐），但实验增益较小且计算开销真实存在。
- **TrOPD** 最适合回答“哪些 token 的教师监督可信、离群 token 如何不浪费”：其统一 benchmark 和 outlier-only FKL 消融是三篇中最具可迁移价值的实证资产。
- **TOP-D** 最适合回答“如何从根源消除无界 log-ratio 造成的梯度风险，并提高样本利用率”：公式与工程接口最简洁、实测增益最大，但应重点复核 OPD baseline 强度、墙钟成本及与同期 OPD 方法的比较。

**建议本组的优先研究路线**：先以 TOP-D 建立可稳定运行的 OPD 训练基线，再吸收 TrOPD 的 token 可靠性分区；最后在训练早期加入 TRB 作为短期前缀质量控制。实验上必须统一协议并做多种子消融，避免把不同的 checkpoint 选择与采样指标误解为方法本身的改进。

---

# Sources（本地材料）

- `/home/user/.eagent/clipboard/15 TRB.pdf`
- `/home/user/.eagent/clipboard/14 TrOPD.pdf`
- `/home/user/.eagent/clipboard/16 TOP-D-1789305166335.pdf`
- `/media/user/DATA_DISK_8T/DesktopData/Chrome/TrustRegionBehaviorBlendingOPD/arxiv.tex`
- `/media/user/DATA_DISK_8T/DesktopData/Chrome/TrustRegionOPD/main.tex`
- `/media/user/DATA_DISK_8T/DesktopData/Chrome/TrustRegionPD/neurips_2026.tex`
