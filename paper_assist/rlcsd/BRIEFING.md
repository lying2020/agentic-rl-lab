# 组会汇报：RLCSD（Contrastive On-Policy Self-Distillation）

> LaTeX 版：`BRIEFING.tex` → 运行 `bash compile_briefing.sh` 得到 `BRIEFING.pdf`（公式以 PDF 为准）

论文：Pan*, Tao, Zhai et al. *RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation*. arXiv:2606.11709v1, 2026-06-10.  
单位：清华大学 / 通义实验室 / 北京大学  
PDF：`docs/9.4_RLCSD.pdf`  
代码：https://github.com/THU-BPM/RLCSD（vendored verl）  
数据：https://huggingface.co/datasets/Leyiii/RLCSD  
和组里已读材料的位置：这是 **特权上下文自蒸馏（OPSD）**，与 $\beta$-OPSD / AntiSD 同族，**不是** 外部大教师 OPD。它承认 vanilla OPSD 的 token 信号被文风带跑，解法是 **正确 hint 对错误 hint 做对比消去**，再作为 GRPO 优势的调制而不是替代。

**【源码核验 / 论文–仓库不一致】** 以 PDF 附录 Table 8 为组会主口径，README 差异单列第 7 节。

---

## 0. 一句话贡献

OPSD 的 $e_{c,t}=\log\pi_T(\cdot\mid x,y_c^\ast)-\log\pi_S$ 会把梯度堆到 “Wait / Therefore / \\n\\n” 这类 **style token** 上。RLCSD 在 **同一 prompt 模板** 下用正确 hint 与 $K$ 条错误 hint 对比，得到 $e_{\mathrm{ctr},t}$，squash 后只调制 $A_{\mathrm{ORM}}$ 的幅度，并用 two-path 损失防止被 70–80% 的未选中 token 稀释。

---

## 1. 任务定义与动机剖析

### 1.1 本 paper 的任务是什么

推理模型的 on-policy **self**-distillation：师生同一模型，教师多看特权上下文（参考解答）。训练/验证都开 thinking。

- 模型：Qwen3-1.7B/4B/8B，Olmo-3-7B-Think
- 数学：DeepMath-103K 按难度分档；测 AMC23 / AIME24 / AIME25，**mean@12**
- 逻辑：Knights & Knaves 4–8 角色训练；测 ID 4–8（500 题）与 OOD 9/10/11（各 100），**pass@1**

一句话任务：**去掉特权条件带来的文风漂移，让 token 级自蒸馏真正打在任务 token 上，同时保住 GRPO 的稳定方向。**

### 1.2 要解决的问题

外部 OPD 要白盒大教师、词表一致、双模型常驻。OPSD 用特权 $y_c^\ast$（数据集 GT 或组内正确 rollout）绕开这些约束。标准信号：

$$
e_{c,t}=\log\pi_T(y_t\mid x,y_c^\ast,y_{<t})-\log\pi_S(y_t\mid x,y_{<t}).
$$

特权确实会锐化任务 token，但同时系统性改变生成风格：更短、更武断、少探索、概率质量挪到格式/话语词。作者称为 **privilege-induced style drift**。

### 1.3 Motivation

数学词汇分成 task（数字、算符、数学符号）与 style（Wait、Therefore、换行等）。Figure 2：每条 rollout 上 $|e_c|$ 的 style 均值 **0.263** vs task **0.083**，约 3×，几乎不重叠。

后果（Figure 1b / Figure 5）：

- OPSD / SDPO / SRPO：熵爆炸 + 长度爆炸 + 奖励崩
- RLSD：数学上明显 **长度收缩**，后期推理被截短

### 1.4 现有方法局限

| 方法 | 做法 | 仍用单侧正特权 → 继承 style drift |
|---|---|---|
| OPSD (Zhao et al.) | dense forward-KL + 逐 token KL clip | clip 治标 |
| SDPO | JSD 平衡 mode-seeking/covering | 同样单侧 hint |
| SRPO | 对的走 GRPO，错的走 SDPO | 失败样本仍单侧蒸馏 |
| RLSD | $e_c$ 只调制 $A_{\mathrm{ORM}}$ | 方向对了，style 仍污染幅度 |
| GRPO | 只有轨迹奖励 | 稳定但 token 信用稀疏 |

### 1.5 具象化例子

同一题，学生采样到 token `Wait`。  
教师看正确解答：讨厌磨蹭，$e_{c,\mathrm{Wait}}$ 很负（大力消灭 Wait）。  
教师看错误解答：同样被“参考解答”模板推成干脆风格，对 Wait 也负。  
相减后 style 分量对消，剩下的 $e_{\mathrm{ctr}}$ 更可能落在 `3`、`sqrt`、`factor` 这种题面 token（Figure 1a top-10）。

若只用错误类型完全不匹配的单条负 hint，负支可能不endorse当前错误轨迹的 token，$e_{\mathrm{ctr}}$ 符号随机——这就是要 $K$-marginalize 的原因。

---

## 2. 方法论树状拆解

**创新落点**：一级是 *对称正负特权对比*；二级是 $K$ 条负 hint 概率均值（先平均再 log）；三级是 verifier-anchored clamp + two-path 归一。

```text
RLCSD
├─ Stage 1：采样与划分
│   ├─ G=8 条学生 rollout，规则验证器 0/1
│   ├─ G⁺ / G⁻；任一侧为空则丢弃（与 GRPO 零优势组一致）
│   └─ 组相对 A_ORM（有 std 归一）
│
├─ Stage 2：对比 token 信号
│   ├─ 正 hint：主实验用数据集 GT CoT+答案；消融可用组内正确 sibling
│   ├─ 负 hint：K=4 条错误 sibling，排除当前 y
│   ├─ 模板字节级相同（都叫 “Correct final answer”）→ style 更共享
│   └─ ectr = log πT(·|yc*) − log( (1/K) Σ πT(·|yw,k*) )
│
└─ Stage 3：调制与 two-path 损失
    ├─ rt = λ tanh(ectr/τ) ∈ (−λ,λ)
    ├─ mt = 1[|rt|>δ]  （约 20–30% token）
    ├─ Ã = same_sign_clamp(A_ORM + rt)  禁止反转验证器方向
    └─ 在 U={mt=0} 与 M={mt=1} 上分别平均 PPO-clip，η 加权 M
```

交互：Stage 1 的验证器既划分 hint 池，又给出 $A_{\mathrm{ORM}}$ 方向；Stage 2 不替代该方向；Stage 3 保证稀疏 token 信号不被全局平均稀释。

---

## 3. 算法流程与公式细节推演

### 3.1 对比信号（Eq.7，注意平均顺序）

Vanilla：$e_{\mathrm{ctr}}=e_c-e_w=\log\pi_T(\cdot|y_c^\ast)-\log\pi_T(\cdot|y_w^\ast)$（$\pi_S$ 消掉）。

最终：

$$
e_{\mathrm{ctr},t}=\log\pi_T(y_t\mid x,y_c^\ast,y_{<t})
-\log\Big(\frac1K\sum_{k=1}^K\pi_T(y_t\mid x,y_{w,k}^\ast,y_{<t})\Big).
$$

**【注解】** 必须 **先对负支概率算术平均再 log**，不是对 $K$ 条 logprob 平均。后者不是对误差分布的 mixture。

### 3.2 调制（Eq.8–11）

$$
A_{\mathrm{ORM}}=\frac{R-\mu_R}{\sqrt{\mathrm{Var}(R)+\epsilon}},\qquad
r_t=\lambda\tanh(e_{\mathrm{ctr},t}/\tau),\qquad
m_t=\mathbf{1}(|r_t|>\delta).
$$

$$
\tilde A_t=\begin{cases}
\max(0,A_{\mathrm{ORM}}+r_t) & A_{\mathrm{ORM}}\ge0\\
\min(0,A_{\mathrm{ORM}}+r_t) & A_{\mathrm{ORM}}<0.
\end{cases}
$$

论文口径：$\tau=0.02,\lambda=0.5,\delta=0.02,\eta=1.0,K=4$。

### 3.3 Two-path（Eq.15）

对 rollout $g$，未调制集 $U$ 与调制集 $M$ **各自**对 $|U|$、$|M|$ 平均，再 $\eta$ 加权 $M$。若某侧为空则省略。查询级再对 $G$ 条平均。

若改成全体 token 一个大平均，$M$ 只占 20–30%，目标滑向纯 GRPO。

### 3.4 手算

设 $A_{\mathrm{ORM}}=+1.0$（组内相对正确），某 token $e_{\mathrm{ctr}}=+0.04$。

$$
r=0.5\tanh(0.04/0.02)=0.5\tanh(2)\approx0.482,\quad |r|>0.02\Rightarrow m=1,
$$

$$
\tilde A=\max(0,1.0+0.482)=1.482.
$$

若 $e_{\mathrm{ctr}}=-3$（很想反向），$r\approx-0.5$，$\tilde A=\max(0,0.5)=0.5$，**仍为正**，验证器方向保住。  
若 $A_{\mathrm{ORM}}=-1$ 且 $r=+0.8$，则 $\tilde A=\min(0,-0.2)=-0.2$，不能变成鼓励错误轨迹。

---

## 4. 实验设定与资源开销

### 4.1 Baselines

GRPO；OPSD；SDPO；SRPO；RLSD。对照共享 veRL+FSDP2+vLLM、全参、8×H20。OPSD 族正特权主实验用 **数据集 GT CoT**；RLCSD 负支用错误学生 rollout。

### 4.2 数据分档

| 模型 | DeepMath 难度 |
|---|---|
| 1.7B | 5–7 |
| 4B | 6–8 |
| 8B / Olmo-3-7B | 7–10 |

逻辑统一 KK 4–8 训练。

### 4.3 共享超参（附录 Table 7）

AdamW，wd 0.01，warmup 50 step；per-device batch 8，$G=8$，PPO mini-batch 16；rollout IS token-level threshold 2.0；train 温度 1.0，top-p 0.95，top-k 20；max prompt 2048，max completion **16384**；eval 温度 0.6，max **38912**；数学 12 sample，KK 1 sample。

数学 lr 全方法 $1\mathrm{e}{-6}$。逻辑：**RLCSD 用 $5\mathrm{e}{-6}$**，其它方法该 lr 更容易崩，故用 $1\mathrm{e}{-6}$。【注解】逻辑主表不是完全相同 lr。

教师（Table 8）：RLCSD / RLSD 为 **每 10 step 硬拷贝 snapshot**；OPSD 为 fixed pretrained；SDPO/SRPO 为 EMA 0.95。

### 4.4 训练成本（附录 D Table 9，Qwen3-4B 数学，秒/step）

| 方法 | Gen | Teacher logprob | Total |
|---|---:|---:|---:|
| GRPO | 503.78 | — | 759.01 |
| RLSD | 412.27 | 9.485 | 812.84 |
| **RLCSD** | 500.92 | **14.210** | **891.55** |
| SDPO | 495.64 | 9.701 | 961.14 |
| SRPO | 411.01 | 9.566 | 1011.51 |
| OPSD | 413.92 | 9.552 | 1035.54 |

多出来的教师前向相对 decode 很小；总时间仍快于 dense 全词表蒸馏。正文有一处 473.89/13.46 与表不完全一致，**以 Table 9 为准**。总 step 数数学曲线画到 ~500，逻辑 ~200；正文未写死 max step。

---

## 5. 实验结论的因果支撑

### 5.1 指标

- **mean@12**：每题 12 次采样平均正确率（AMC/AIME）↑
- **pass@1**：KK 单次 ↑
- Math/Logic **Avg.**：三数学 / 四逻辑（ID+三个 OOD）未加权均值 ↑
- 训练曲线：actor entropy、response length、training reward、验证集——稳定性证据

### 5.2 主表（PDF Table 1，Qwen3-8B 摘录）

| 方法 | Math Avg | KK Avg |
|---|---:|---:|
| Base | 76.6 | 59.6 |
| GRPO | 78.6 | 66.0 |
| OPSD | 78.7 | 64.8 |
| RLSD | 77.1 | 67.4 |
| **RLCSD** | **79.3** | **74.0** |

全 4 模型、两任务族上 RLCSD 几乎全列最优。OOD 更明显：8B 的 11-role **+21.0 vs Base**。**【因果】** 清洗后的 token 信号有助于泛化（中强支撑）。

相对 Base：1.7B +4.3 math / +10.9 logic；4B +2.5 / +6.8；8B +2.7 / +14.4；Olmo +1.8 / +9.9。

### 5.3 失败模式 vs 稳定（Figure 5，强支撑）

OPSD/SDPO/SRPO：熵↑、长度爆炸、奖励/验证下跌。RLSD：数学长度收缩。RLCSD：熵与长度接近 GRPO，验证更好。

### 5.4 对比作为插件（Table 2，Qwen3-4B）

只改 hint 构造：dataset CoT vs own correct vs own correct/incorrect contrast。单侧换 GT→self 几乎不变；加上 contrast：OPSD logic +2.3，RLSD math +2.2，RLCSD 自己去掉 contrast 则 math −3.0、KK −5.4。Figure 6–7：contrast 抑制 OPSD 熵爆、缓解 RLSD/单侧 RLCSD 的长度收缩。**【强支撑】** 对比原则可迁移。

OPSD 插件走分布级 CFG：$q=\mathrm{softmax}((1+\alpha)\ell_c-\alpha\ell_w)$ + KL 软门（Eq.18–19），与 RLCSD 的标量调制不是同一实现。

### 5.5 组件消融（Table 3，Qwen3-4B）

| 去掉 | Math Avg | Logic Avg |
|---|---:|---:|
| full RLCSD | 78.0 | 66.9 |
| − K-marginal | 76.8 | 64.4 |
| + self in hint pool | 76.8 | 63.7 |
| − $A_{\mathrm{ORM}}$ anchoring | **75.7** | **62.4**（最大跌） |
| − two-path | 76.4 | 63.4 |

验证器锚定最关键：干净的 $e_{\mathrm{ctr}}$ 仍有残差反号率。附录 A：least-similar 负 hint 时错误 rollout 的 wrong-sign ~51%，$K$-marginal 后 $<8\%$。

### 5.6 推广到外部 OPD（Table 4–5，分析判断）

Qwen3-1.7B-Base 学生。Instruct 教师 standalone 更强（math 34.9 vs 32.5），但蒸馏学生更差（15.7 vs 20.7）。Top KL token：Instruct 是 The / \\n\\n / Therefore；GRPO 教师是数字和内容词。**【中等支撑】** style vs task 分解可能也适用于 cross-model OPD；RLCSD 的 hint 对比 **不能直接**搬过去（无特权 hint）。

### 5.7 限制

- 逻辑主表 lr 不对齐
- 无多种子
- 主实验正 hint 用 GT，与“纯自蒸馏”叙事略有张力（消融说换 self 影响小）
- GitHub README 主表数字与 PDF Table 1 不完全相同，引用时锁定版本

---

## 6. 复现前置准备清单

- [ ] **数据**：`python scripts/download_data.py --all` → `Leyiii/RLCSD` parquet
- [ ] **权重**：Qwen3-1.7B/4B/8B，Olmo-3-7B-Think（HF）
- [ ] **硬件**：论文 **8×H20** 全参；最低【估计】1.7B 可试更少卡但需改 batch
- [ ] **软件**：Python 3.10–3.12；`torch>=2.5,<2.10`（CUDA 12）；flash-attn；vendored `third_party/verl`
- [ ] **代码**：`bash scripts/math_deepmath/run_qwen3_4b_rlcsd.sh`；逻辑 `scripts/logic_kk/...`
- [ ] **算法必做**：对称模板、Eq.7 的概率均值、$K$、排除 $y$、tanh 调制、sign clamp、two-path
- [ ] **配置对齐 PDF**：`K=4, τ=0.02, λ=0.5, δ=0.02, η=1.0`，teacher snapshot 每 10 step
- [ ] **Sanity**：无负样本组应 skip；`m=0` 路径等于 GRPO clip；$A_{\mathrm{ORM}}>0$ 时 $\tilde A$ 不得为负

---

## 7. Paper ↔ Code gaps / 开放问题

| 项 | PDF | GitHub README（2026 公开仓） |
|---|---|---|
| 正 hint 主实验 | GT CoT（§4.1）；Figure 3 画的是 self-rollout 选项 | 默认 `gt_cot`，与 PDF 一致 |
| 教师 | Table 8：snapshot $\tau_{\mathrm{sync}}=10$ | 写 “kept fixed throughout” |
| $\tau,\eta$ | 0.02 / 1.0 | 1.3 / 0.5 |
| Table 1 数字 | 见 PDF | README 另一套更高/不同的数 |

复现应对 YAML 逐项核对，不要混用两套超参。损失在 `third_party/verl/verl/trainer/ppo/core_algos.py`（`@register_policy_loss("rlcsd")`）；数据路径 `src/self_distill_main.py`。

**组会可追问**

1. 逻辑任务给 RLCSD 更高 lr，主表优势有多少来自调参？
2. two-path 的 $\eta=1$ 等于调制路径与 GRPO 路径等权，和“只调制 25% token”的叙事如何同时成立？（答案：独立归一）
3. 能否把 $e_{\mathrm{ctr}}$ 的 task/style 诊断直接做成训练期 token mask，而不是事后分析？
