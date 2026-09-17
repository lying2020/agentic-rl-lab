# 组会汇报：三篇 OPD 变体对照精读（TGOPD / GC-OPD / RLCSD）

**材料范围**：

1. **TGOPD** — *Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation*（AllSpark Team, arXiv:2609.02998）
2. **GC-OPD** — *Beyond Teacher Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning*（Zhang & Wang et al., arXiv:2608.19181）
3. **RLCSD** — *Reinforcement Learning with Contrastive On-Policy Self-Distillation*（Pan et al., arXiv:2606.11709）

**单篇精读**（六问完整版 + 可编译 LaTeX）：

- [`paper_assist/tgopd/BRIEFING.md`](../paper_assist/tgopd/BRIEFING.md)
- [`paper_assist/gc-opd/BRIEFING.md`](../paper_assist/gc-opd/BRIEFING.md)
- [`paper_assist/rlcsd/BRIEFING.md`](../paper_assist/rlcsd/BRIEFING.md)

**核验方式**：以本地 PDF（`docs/7_TGOPD.pdf`、`docs/3_GC-OPD.pdf`、`docs/9.4_RLCSD.pdf`）正文与附录为准。TGOPD 无公开代码；GC-OPD / RLCSD 对照了 GitHub README。文中“源码核验”表示可在 PDF 或仓库定位；“分析判断”是研究评价，不要与作者主张混淆。

和上一份 Trust-Region 三篇精读的关系：TRB / TrOPD / TOP-D 管的是 **行为分布、token 分区、奖励有界**。这三篇管的是 **教师值不值得信、教师与验证器是否同序、特权自蒸馏的文风漂移**。不要把表中分数横向排名。

---

# 1. 结论先行：都还在修 OPD，但教师设定与干预层不同

## 1.1 一句话结论

- **TGOPD**：外部冻结教师。先问“这道题教师可靠吗？”——**prompt 级硬门**：可靠走稠密 OPD，不可靠撤教师改走 GRPO。
- **GC-OPD**：外部冻结教师 + 长上下文验证器。教师始终给稠密 $A_t$，再用组内 **验证器 vs 轨迹 OPD 分的残差** 做校准。
- **RLCSD**：没有更强外部教师。同一模型看特权 hint；用 **正确 hint 对错误 hint 对比** 消去 style drift，再调制 GRPO 的 $A_{\mathrm{ORM}}$。

正交控制面：**准入（admit/withhold）、校准（calibrate residual）、对比消去（contrastive cancel）**。

## 1.2 方法地图

```text
教师监督从哪来？
├─ 外部更强冻结教师
│   ├─ 不检查对错、每题都蒸 → Vanilla OPD
│   ├─ 先验证再决定蒸不蒸 → TGOPD（整题路由）
│   └─ 一直蒸，但用验证器纠正组内排序 → GC-OPD（残差校准）
└─ 自身 + 特权上下文（OPSD）
    ├─ 只看正确解答 → 文风漂移、长度崩/炸
    └─ 正确 vs 错误对称对比，再调制 GRPO → RLCSD
```

## 1.3 不应直接横向比分数

| 项目 | TGOPD | GC-OPD | RLCSD |
|---|---|---|---|
| 教师 | 同基座 GRPO 域专家 | Qwen3-30B-A3B-Thinking | 自身 snapshot / GT 条件 |
| 域 | 数学 / 代码 / IF | 长上下文九族 | 数学 + KK 逻辑 |
| 学生 | 4B 与 35B-A3B | 4B 与 8B | 1.7B/4B/8B + Olmo-7B |
| 步数 | 单域未统一写死；MOPD=199 | 固定 100 | 曲线约 200–500 |
| 主指标 | avg@64 / pass@1 / IF | 五长上下文未加权 Avg | mean@12 / pass@1 |
| 硬件 | 40 或 56 GPU 异步集群 | 8×80GB | 8×H20 |

---

# 2. 共同背景

OPD 用学生自己的轨迹、教师 token 级 $\log(\pi_T/\pi_S)$ 把稀疏结果奖励变稠密。Trust-Region 三篇已经说明：这个 log-ratio 会无界、会在 outlier token 上炸、会在错误前缀上监督。

这三篇补了另外三条裂缝：

1. **教师本身可能错**（尤其代码，自信 AUROC 0.51）→ 不能无条件 reverse-KL。
2. **教师对了局部、验证器看全局**（长上下文漏证据）→ 稠密偏好 ≠ 任务完成。
3. **教师就是自己 + 参考解答** → 特权会改文风，梯度打在 Wait/Therefore 而不是答案 token。

共同问题可以概括成：

> **在已经决定用 on-policy 稠密监督之后，如何保证这份监督指向“会做对题”，而不是指向“像教师/像看过答案的文风”？**

---

# 3. 三篇方法的严格对照

| 维度 | TGOPD | GC-OPD | RLCSD |
|---|---|---|---|
| 最小干预单元 | 整个 prompt | 一条回答的残差 + token 信用 | 被 mask 选中的 token |
| 主要风险 | 自信的错误教师 | 教师–验证器排序冲突 | privilege-induced style drift |
| 核心操作 | $g=1[q_T\ge\tau]$ 硬路由 | $\rho=\tilde R-\tilde s$，RACA | $e_{\mathrm{ctr}}=e_c-e_w^{\mathrm{mix}}$ |
| 教师如何介入 | 探针 decode + 无条件 scoring | 只 scoring，零额外前向 | 正 hint + $K$ 条负 hint 前向 |
| 是否改 OPD loss | 选 OPD 或 GRPO，不混合 | $A'=A+\beta c\rho$ | 不替代 $A_{\mathrm{ORM}}$，只调制 |
| 验证器角色 | 给教师打准入分；关门时给学生 GRPO | 组内相对位置 | 划分 G± 且锚定更新方向 |
| 主要超参 | $K_T=3,\tau=2/3$ | $\beta=0.10$ | $K=4,\tau=0.02,\lambda=0.5,\delta=0.02,\eta=1$ |
| 工程代价 | 教师 decode 探针；+5.9% step（35B code） | 几乎等于 OPD | 教师 logprob 14.2s vs ~9s；总 step 仍快于 dense OPSD |
| 最强证据 | 代码负迁移被门挡住；~90% 增益来自 mask | 残差优于直接加 $\tilde R$；再加一份 OPD 无效 | 插件 contrast 能救 OPSD/RLSD；去掉锚定跌最多 |

## 3.1 表面都像“加验证器”，数学上不同

| 方法 | 验证器进入的位置 |
|---|---|
| TGOPD | **开关**：通过则验证器不进梯度 |
| GC-OPD | **残差**：验证器相对 OPD 分的差，加在 $A_t$ 上 |
| RLCSD | **方向锚**：$A_{\mathrm{ORM}}$ 决定正负，对比信号不能反号 |

因此不能说“三篇都是 verifier-aware OPD 所以能叠”。TGOPD 门关时根本没有教师 $s$，GC-OPD 的 $\rho$ 未定义；RLCSD 没有外部 $\pi_T$。

---

# 4. 各篇最值得记住的数字

**TGOPD**（Table 1 / 3 / 5）

- 六个单域设定全部超过 Vanilla OPD；代码最大（4B +3.0，35B +2.9）
- 35B LCB：其它蒸馏方法全低于 base，仅 TGOPD +3.0 并超过教师
- 教师 GPU：4B SOPD $9.8\%\to78.9\%$
- 关门 mask vs GRPO：约 90% 增益来自挡住坏教师

**GC-OPD**（Table 2–4）

- 五基准 Avg：4B $39.31\to40.47$；8B $43.56\to44.65$（相对 OPD）
- Additional OPD +0.04；Direct reward +0.63；残差 +1.10
- Uniform 摊残差 +0.72；RACA +1.10

**RLCSD**（PDF Table 1 / 3 / 9）

- Qwen3-8B Math Avg 79.3、KK Avg 74.0（Base 76.6 / 59.6）
- 去掉 $A_{\mathrm{ORM}}$ 锚定：Math 78.0→75.7，Logic 66.9→62.4
- 4B 数学 step：891.55s，第三快（仅慢于 GRPO / RLSD）

---

# 5. 可能的组合与冲突

自然幻想：

```text
TGOPD 门开的 prompt：GC-OPD 残差校准
TGOPD 门关的 prompt：纯 GRPO（本来就没有可靠教师）
OPSD 设定：RLCSD 对比，而不是再请外部教师
```

必须先做的实验冲突：

1. **GC-OPD 假设教师 scoring 始终存在**。TGOPD 门关仍做 scoring（pipeline 规整），残差仍可算，但作者认为该题教师不可靠——再用 $s$ 分配 RACA 可能把校正摊到错误 token 上。
2. **双重验证器**：TGOPD 用教师探针 pass rate；GC-OPD 用学生回答的 $R$。两者不是同一随机变量。
3. **RLCSD 与外部 OPD 的 style 诊断可迁移，对比公式不能直接搬**（没有 $y_c^\ast,y_w^\ast$）。
4. 计算：TGOPD 探针 decode + RLCSD 的 $K$ 条负 hint 前向，不要同时加在异步教师节点上而不测墙钟。

统一实验仍应固定学生/教师/数据/step/评测次数与多种子，再做 门 × 残差 × 对比 的子集，而不是堆 SOTA。

---

# 6. 本组建议的阅读与复现顺序

1. **先读 RLCSD**，若本组已做 $\beta$-OPSD / AntiSD：同一 OPSD 故事，对比消去最容易讲清。仓库完整，8×H20 级。
2. **再读 GC-OPD**：最小改动（优势代数），仓库与论文对齐最好（`beta=0` 即 OPD）。需要 30B 教师和长上下文卡。
3. **最后读 TGOPD**：系统故事（填空闲 GPU）漂亮，但无官方代码、要自训域教师、集群拓扑重。算法冒烟可只做数学 4B + mask-only 关门。

Sanity 对照：

| 方法 | 必查 |
|---|---|
| TGOPD | $\tau=0$ ≈ Vanilla OPD；常数组奖励且门关 → 零梯度；探针与学生同一验证器 |
| GC-OPD | $\beta=0$ = OPD；组内 $R$ 全同 → $\rho=0$ |
| RLCSD | 先平均负支**概率**再 log；$\tilde A$ 不得反转 $A_{\mathrm{ORM}}$ 符号 |

---

# 7. 组会可直接讨论的问题

1. TGOPD 的增益主要是“挡住坏教师”（mask 已有 90%）。那 GRPO fallback 还值不值得做？代码域 OJBench 提示值得。
2. GC-OPD 的 RACA 用 OPD 相对优势分配验证器残差。若 OPD 已系统性偏向漏证据的流畅句，校正会不会打在错误 token 上？
3. RLCSD 逻辑主表给了更高 lr，公平比较要不要重跑？
4. 三篇都默认有自动验证器。开放式 IF 上 TGOPD 仍用规则裁判，GC-OPD 有 ROUGE 等 graded 奖励——可靠性定义已经在漂。
5. 和 Trust-Region 三篇如何分工：TOP-D 管无界 log-ratio，TrOPD 管 outlier token，TRB 管前缀；本三篇管“教师/特权是否在说任务”。更干净的下一篇可能是 **有界奖励 + 对比消去 + 仅在可靠 prompt 上启用**，但必须统一协议。

---

# 8. 最终判断

- **TGOPD** 最适合回答“什么时候根本不该蒸馏”。系统切口干净，代码域故事强；缺官方仓，主增益部分来自拒绝而非新目标。
- **GC-OPD** 最适合回答“蒸馏还做，但验证器说教师排错了怎么办”。工程最轻、代码对齐最好；平均分增量约 1 点，靠消融证明残差不是假动作。
- **RLCSD** 最适合回答“自蒸馏为什么学文风、怎样对消”。与本组 OPSD 线最贴；注意 PDF 与 GitHub 超参/表格不一致，汇报时锁 PDF。

**建议本组优先**：OPSD 线复现 RLCSD 的 contrast 插件（改 hint 即可接到现有 RLSD/OPSD）；外部教师 OPD 线先把 GC-OPD 的 $\rho$ 加成稳定基线，再决定是否值得上 TGOPD 探针集群。

---

# Sources

- `/home/liying/Desktop/agentic-rl-lab/docs/7_TGOPD.pdf`（arXiv:2609.02998）
- `/home/liying/Desktop/agentic-rl-lab/docs/3_GC-OPD.pdf`（arXiv:2608.19181）
- `/home/liying/Desktop/agentic-rl-lab/docs/9.4_RLCSD.pdf`（arXiv:2606.11709）
- https://github.com/SolereZhang/GC-OPD
- https://github.com/THU-BPM/RLCSD
- TGOPD 无公开实现；框架声明为 https://github.com/THUDM/slime
