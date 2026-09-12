# 组会汇报：Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information

论文：Shen et al., 2026. arXiv:2605.11609  
代码：https://github.com/FloyedShen/AntiSD（veRL fork）  
汇报定位：博士后组会精讲（问题 → 树状方法 → 带公式的走查 → 实验 → 复现清单）

---

# 1. 任务、问题、动机、现有方法局限，以及一个例子

## 1.1 这篇 paper 在做什么

这不是一篇新网络结构的论文，也不是多模态。它做的是 **数学推理的 RL 后训练（RLVR）里的 credit assignment**：

- 输入：一道可验证对错的数学题 \(x\)。
- 策略 \(\pi_\theta\) 采样一条推理轨迹 \(y\)（思维链 + `\boxed{}`）。
- 环境只给 **轨迹级 0/1 奖励** \(R(x,y)\)。
- 目标：让模型更快、更高地涨 AIME / HMMT 这类竞赛题准确率。

作者的具体贡献是：在 **on-policy self-distillation（自蒸馏）** 这条线上，指出它在数学上系统性失败的原因，并给出一个 **只改优势符号与形状、不改模型、不加外部教师** 的替换算法 **AntiSD**。

一句话任务：**不用更强外部教师、也不训单独 PRM，只靠「同一模型看到标准答案后」的 token 级信号，来加速并抬高数学 RL。**

## 1.2 要解决的问题（真正的 scientific question）

RLVR 的奖励太稀疏：一整条几千 token 的推理只得到一个 bit。谁该被强化？两条主流出路都贵：

1. **Process Reward Model（PRM）**：另训一个逐步打分器（人工标注或 MC 估计）。
2. **On-policy distillation（OPD）**：请一个更强教师，对学生自己采的 \(y\) 做 token 级模仿。

自蒸馏的诱惑是：把教师设成 **同一个 \(\pi_\theta\)，但多看一段训练期才有的特权上下文 \(c\)**（组内正确解 + 对错反馈）。这样似乎「模型自己给自己 dense credit」，既不用外部大模型，也不用标逐步分。

**问题：这件事在指令遵循 / 科学 QA / 工具使用上有时有效，在难数学上经常无效甚至有害。** 作者在 4B–30B、Qwen3 与 OLMo-3 上都复现了：默认 SD 打不过强 GRPO。

所以论文的真正问题不是「再发明一个 RL 算法」，而是：

> 自蒸馏给的那个 per-token 信号，在数学推理里到底奖励了什么、惩罚了什么？极性对不对？

## 1.3 Motivation（他们看到了什么）

把学生、教师的 token logprob 记为

\[
s_t=\log\pi_S(y_t\mid x,y_{<t}),\qquad
t_t=\log\pi_T(y_t\mid x,c,y_{<t}),\qquad
u_t=t_t-s_t.
\]

因为师生共享参数，

\[
u_t=\mathrm{PMI}(y_t;c\mid x,y_{<t}).
\]

\(u_t>0\)：看到标准答案后，这个 token 更像「答案已经暗示的东西」。  
\(u_t<0\)：看到标准答案后，这个 token 反而变不自然——往往是 **Wait / Let / Maybe / Alternatively** 这类审议、回溯词。

默认自蒸馏对 reverse-KL 做 **下降**，策略梯度里 \(\delta_t=+u_t\)。于是：

- 捷径词（Given, holds, succeeds）被强化 → 模型往「已知答案后的模板」塌。
- 审议词被惩罚 → 搜索变窄，回复变短（文献里常被说成「压缩」，作者说这其实是 **structural shortcut**）。

两个观察驱动后面的设计：

- **(O1) 极性错了**：数学要的是搜索，不是「像已经知道答案那样说话」。
- **(O2) 分布不对称**：\(y\) 来自学生，\(\pi_S>\pi_T\) 的审议 token 被过采样，\(u_t\) 可到 \(-20\)，直接用 \(+u_t\) 或无界反号都会炸。

## 1.4 现有方法的局限

| 路线 | 局限 |
|---|---|
| 纯 GRPO / RLVR | 只有 \(A^{\mathrm{seq}}\)，中间步 credit 靠碰运气，点火慢。 |
| 外部 PRM | 要标注或大量 MC；另训一个模型。 |
| 外部教师 OPD | 要更强模型；师生能力差会 mismatch。 |
| 默认 Self-Distillation | 不需要外部模型，但特权上下文把教师变成 oracle，**奖励捷径、惩罚审议**。数学上常弱于 GRPO。 |
| 无教师 self-reward | 作者消融：去掉 \(c\)、只拿学生自己的 logp 当信号，约 70 step 内 self-reinforcement 崩掉。说明增益必须来自 \(c\) 的信息，不是随便塑形 \(s_t\)。 |

## 1.5 用一道题把动机讲清楚

题目（Appendix C 风格）：求 \(1\le a,b\le 100\) 且 \(\gcd(a,b)+\mathrm{lcm}(a,b)=a+b+50\) 的整数对个数。正确答案 25。

学生自己写出一条轨迹 \(y\)，中间有：

- 「Wait, maybe I should try another substitution…」
- 「Let \(d=\gcd(a,b)\)…」
- 「Therefore the count holds / Given that \(d(1+mn)=\cdots\)」
- 最后 `\boxed{30}`（这题答错了）

训练时教师多看了两样东西 \(c\)：

1. 组里另一条答对的完整解答（或数据里的参考解），boxed 25。
2. 反馈：`Your answer is incorrect.`

教师已经「知道答案是 25」，于是：

- 对 “holds / Given / Therefore”：\(t_t \gg s_t\)，\(u_t\gg 0\)（捷径，红色）。
- 对 “Wait / Maybe / Alternatively”：教师觉得没必要再搜，\(t_t \ll s_t\)，\(u_t\ll 0\)（审议，蓝色）。

**默认 SD**：强化红色、打压蓝色 → 学生学会「装成已经知道答案」，搜索死掉，长度塌缩。  
**AntiSD**：符号反过来，强化蓝色、抑制红色 → 保留分叉搜索，同时 GRPO 的 0/1 仍拉最终对错。

这就是 Figure 1(a) 那张「oracle 把树压成单根；反号把其余分枝找回来」。

---

# 2. 他们怎么想到用这些工具：方法树

先给结论：作者没有新架构。整套东西是 **「GRPO 外壳 + 自教师前向 + 把 KL 下降改成有界 JSD 上升 + 一个熵门」**。想法链条是：

稀疏奖励 → 想要 token credit → 不想请外部教师 → 用特权上下文做自教师 → 发现该信号就是 PMI → PMI 极性对数学是反的 → **反号** → 反号后 reverse-KL 无界、审议侧过采样会炸 → 换 **JSD 的 \(f\)-散度导数 \(\varphi\)** 做单侧封顶 → 上升散度不再自终止 → 加 **熵门（Schmitt）**。

## 2.0 总览：三个模块，像一棵树

```
AntiSD（训练一步）
│
├── M1  轨迹与稀疏奖励（环境环）
│     ├─ vLLM rollout：y ~ π_S(·|x)
│     ├─ 规则奖励 R ∈ {0,1}（boxed + 数学等价）
│     └─ GRPO：A^seq = (R-μ)/σ     ← 只知道整条对不对
│
├── M2  特权自教师（同一套 θ，多看 c）
│     ├─ 选 c：组内正确解（否则数据集解）+ 对错反馈
│     ├─ 学生前向：s_t = log π_θ(y_t | x, y_<t)
│     └─ 教师前向（stopgrad）：t_t = log π_θ(y_t | x, c, y_<t)
│           └─ 输出 u_t = t_t - s_t = PMI(y_t ; c | ·)
│
└── M3  AntiSD 优势整形（本文核心）
      ├─ 极性：上升而不是下降  →  δ = -φ(u) 而不是 +u
      ├─ 形状：JSD 的 φ(u)=½(softplus(u)-log 2)  → 审议侧有界
      ├─ 门控：教师熵塌了就 λ=0（Schmitt）
      └─ 合成：A_t = A^seq + λ · A^AntiSD
            └─ 标准 PPO-clip 更新 θ
```

**模块关系（交互）**

- M1 **不看** \(c\)。推理期也没有 \(c\)。它保证「最终答对」仍是主任务。
- M2 **只在训练** 构造 \(c\)，对 **同一条学生轨迹 \(y\)** 重打 logprob。教师不重新生成，只评分。
- M3 把 M2 的 \(u_t\) 变成 token 优势，**加**到 M1 的 \(A^{\mathrm{seq}}\) 上。  
  - \(A^{\mathrm{seq}}\)：这条要不要整体加强。  
  - \(A^{\mathrm{AntiSD}}\)：这条里面哪些 token 是审议、哪些是捷径。  
- 消融表明：M3 去掉教师（M2 空心）会崩；M3 去掉门在 Qwen 上会晚崩；M3 不反号（默认 SD）全面弱于只留 M1 的 GRPO。

所以功能分工是：**M1 管对错，M2 管「答案改变了哪些 token 的信念」，M3 管「这个信念差该不该、以什么形状、在什么时候注入梯度」。**

## 2.1 M1 下面的小模块

- **采样**：每题 \(G=8\) 条，T=1.0，top-p=1.0。
- **奖励**：只看最后一个 `\boxed{}`，`math-verify` 等价。反馈字符串只有 correct/incorrect，避免泄漏「截断了」之类长度黑客。
- **组归一化**：标准 GRPO。`norm_adv_by_std` 论文写 \(\sigma_R\)，实现里有开关。
- **无 Critic、无参考 KL**：KL 系数 0，跟 DAPO 系设定一致。

## 2.2 M2 下面的小模块

- **解的来源**：优先同组成功 rollout（on-policy，分布匹配）；没有则用数据集解（论文 Setup；代码默认 `group_only`，组内全错时可能只有反馈）。
- **模板（App.C）**  
  - `Your previous attempt:` = **别人的正确解**（不是当前 \(y\)）。  
  - `Previous assessment:` = **当前 \(y\)** 的对错。  
  - 再要求 Now solve step by step。  
  - 这种「正确参考 vs 当前可能错误的尝试」给教师一个对比信号。
- **教师正则**：论文是同一 \(\pi_\theta\)；代码 `teacher_update_rate=1.0`，每步拷贝学生，不是慢 EMA。
- **stop-gradient**：\(t_t\) 当常数，避免师生一起漂。

## 2.3 M3 下面的小模块（三层）

**第一层：极性（对应 O1）**  
下降 \(D_f(\pi_S\|\pi_T)\) 的优势对 \(u\) 单调增 → 永远宠捷径。  
改成上升 → 符号翻转。这一步是主杠杆：Table 1 里 SD vs AntiSD 差几十个点。

**第二层：形状（对应 O2）**  
若只反号 reverse-KL，优势是 \(-u_t\)，审议侧 \(u=-20\) 会给 \(+20\) 的巨大奖励，过采样放大。  
JSD 的 \(f'(e^{-u})=-\varphi(u)\)，

\[
\varphi(u)=\tfrac12(\mathrm{softplus}(u)-\log 2),\quad
\varphi(u)\ge -\tfrac12\log 2.
\]

于是 \(A^{\mathrm{AntiSD}}=-\varphi(u)\) 在审议侧封顶 \(\tfrac12\log 2\approx 0.347\)，捷径侧仍近似线性惩罚。小 \(u\) 时 \(-\varphi(u)\approx -u/4\)，局部仍像「反号的 KL」。

**第三层：门（上升不再自终止）**  
下降 KL 会把 \(\pi_S\) 拉向 \(\pi_T\)，有自然停。上升会把两者推开，教师一旦塌成「答案模板」，\(u_t\) 变成数值地板噪声。  
用教师 token 熵中位数 \(H\) 做 Schmitt 触发：低于 \(0.93 H_{\mathrm{warm}}\) 关，回到 \(H_{\mathrm{warm}}\) 再开。  
代码里常用 **teacher perplexity** 当 \(H\) 的代理，并且 \(\lambda\) 可以连续，不完全是论文的二值 \(g\cdot\lambda_{\max}\)。

**合成方式**：加法 \(A_t=A^{\mathrm{seq}}+\lambda\delta_t\)。乘法会在 \(A^{\mathrm{seq}}\approx 0\)（组内全对或全错）时把 AntiSD 也掐死，恰恰是稀疏奖励最需要 dense 信号的时候。Table 3：乘法掉 6.3 点。

---

# 3. 算法流程：用例子把公式走一遍

## 3.1 Algorithm 1（一步，Appendix B）

对 batch 里每条 rollout \(i\)、每个 token \(t\)：

1. \(s_{i,t}\leftarrow \log\pi_\theta(y_{i,t}\mid x_i,y_{i,<t})\)
2. \(t_{i,t}\leftarrow \mathrm{sg}[\log\pi_\theta(y_{i,t}\mid x_i,c_i,y_{i,<t})]\)
3. \(u_{i,t}\leftarrow t_{i,t}-s_{i,t}\)
4. \(\varphi_{i,t}\leftarrow \tfrac12(\mathrm{softplus}(u_{i,t})-\log 2)\)
5. 算 batch 教师熵 \(H\)，更新门 \(g\)，\(\lambda\leftarrow g\cdot\lambda_{\max}\)
6. \(A_{i,t}\leftarrow A_i^{\mathrm{seq}}-\lambda\cdot\mathrm{sg}[\varphi_{i,t}]\)  
   （因为 \(A^{\mathrm{AntiSD}}=-\varphi\)，所以 \(A^{\mathrm{seq}}+\lambda A^{\mathrm{AntiSD}}=A^{\mathrm{seq}}-\lambda\varphi\)）
7. 用 \(\{A_{i,t}\}\) 做标准 policy gradient / PPO-clip

## 3.2 数值玩具（配合上面那道题）

设某一步学生正在写 “Wait”。

| | 学生（没看答案） | 教师（看过 boxed 25） |
|---|---|---|
| \(\pi(\texttt{Wait})\) | 0.20 | 0.01 |
| \(\pi(\texttt{Therefore})\) | 0.05 | 0.40 |

对 token `Wait`：

\[
s=\log 0.20\approx -1.61,\quad
t=\log 0.01\approx -4.61,\quad
u=t-s\approx -3.00.
\]

这是审议：答案让 “Wait” 更不可能。

\(\varphi(-3)=\tfrac12(\mathrm{softplus}(-3)-\log 2)\approx \tfrac12(0.049-0.693)=-0.322\)。

- 默认 SD：\(\delta=+u\approx -3.0\) → **惩罚** Wait。  
- AntiSD：\(A^{\mathrm{AntiSD}}=-\varphi\approx +0.322\) → **奖励** Wait，且不会涨到 +3。

对 token `Therefore`：

\[
u=\log 0.40-\log 0.05=\log 8\approx +2.08,
\]

\(\varphi(2.08)\approx 0.42\)，AntiSD 给 \(\approx -0.42\)（压捷径）。  
若用生的 \(-u\)，Wait 会拿到 +3，Therefore 拿到 -2.08，审议侧过大，正是 O2 要压的。

再叠加 GRPO。假设这组 8 条里 2 条对、6 条错，\(R=1\) 的 \(A^{\mathrm{seq}}>0\)，当前这条错了 \(A^{\mathrm{seq}}<0\)：

\[
A_t = A^{\mathrm{seq}} + \lambda (-\varphi(u_t)).
\]

整条仍被序列奖励往下压（答错了），但 **条内** Wait 相对 Therefore 更不那么负——credit 从「装会了」挪到「还在搜」。\(\lambda_{\max}=0.5\)。前 5 步 \(\lambda=0\)，先量 \(H_{\mathrm{warm}}\)。

轨迹上 \(\sum_t u_t=\mathrm{PMI}(y;c\mid x)\)，是势函数差分（Lemma 3），属 Ng 的 potential-based shaping：不改最优策略集合，只改学习过程。这是他们把 \(u_t\) 叫「免费 PRM」的理论借口；**贡献是发现这个 shaping 的 PMI 偏置并反号**，不是发明 shaping 本身。

---

# 4. Baseline、数据、训练成本

## 4.1 Baseline（每个模型四档）

| 条件 | 含义 |
|---|---|
| Base | 未做该 RL |
| +GRPO | Eq.(2) 且 \(\lambda=0\)，只有序列奖励 |
| +SD | 默认自蒸馏，\(\delta_t=+u_t\)，下降 KL |
| +AntiSD | Alg.1 |

没有另训 PRM、没有外部更强教师。代码里还有 reverse-KL 上升、乘法合成、关上门等消融，不算主表 baseline。

## 4.2 数据

**训练**：DAPO-Math-17k（BytedTsinghua-SIA），约 17k 竞赛题，可验证答案。200 on-policy step，batch 32 题 × 8 rollout。

**评测**：

- AIME 2024 / 2025 / 2026、HMMT 2025：avg@32  
- MinervaMath：avg@4  
- 额外：HumanEval+ / MBPP+（Table 2，Dolci-RLZero 上训，不是主 recipe）

## 4.3 模型与硬件（Table 5）

- Qwen3-8B、Qwen3-4B-Instruct-2507、OLMo-3-7B-Instruct、OLMo-3-7B-Think、Qwen3-30B-A3B  
- AdamW，\(10^{-6}\)，200 step，训练最长 32K；Think 评测 64K  
- **每节点 8× NVIDIA H20**；30B-A3B 多机  
- 框架 veRL；rollout vLLM  
- 论文 **没写墙钟时间和峰值显存**。按 32 题×8 条×16k–32k 估，8×H20 上 8B、200 step 大约是 **1–3 天** 量级（需实测）。

仓库 **不提供** AntiSD 训练后权重。

---

# 5. 实验做了什么、每列指标是什么、怎样支撑论点

## 5.1 评价指标（读 Table 1 必须先统一）

| 符号 | 定义 | 用来支撑什么 |
|---|---|---|
| AIME*/HMMT 上的数字 | **avg@32**：每题 32 次独立采样的平均准确率（对/错） | 单次解题能力，方差被平均掉一部分 |
| Minerva | **avg@4** | 同上，题更多、采样更少 |
| Average | 五科（或所列科）的平均；下标 @k = 该平均最高的训练步 | 总能力 + 何时最好 |
| Speedup | GRPO 达到自己最好 Avg 的步数 / 该方法 **第一次达到 GRPO 最好 Avg** 的步数；\(\times\) 表示从未达到 | 「点火」快不快 |
| pass@k（Fig.3） | k 次尝试里至少对一次的比例 | 是真会了更多题，还是只是把概率堆到少数已会的题上 |
| 训练曲线 | 截断校正后的 train reward、长度、师生熵 | 稳不稳、会不会熵崩/长度爆炸 |

准确率都是 **final-answer match**（数学等价），不是过程分。

## 5.2 主实验 Table 1 — 论点：反号后既快又高，默认 SD 有害

跨五模型三点稳定：

1. **更快**：追上 GRPO 要 2–10× 更少 step（4B 10×，Olmo-IT 9.5×，8B 5×，30B 2.9×，Think 2×）。弱基线加速更大——符合「dense \(-\varphi(u)\) 从第一步就有信息，不必等稀疏 \(R\) 回传」。  
2. **更高**：最终 Avg 超 GRPO **+2.1 到 +11.5**。8B：57.4→65.7；4B-IT：51.3→62.8。HMMT 上 8B 甚至 +15pp（Fig.1b）。  
3. **默认 SD 全面弱于 GRPO**（8B：30.6 vs 57.4）。这是「极性错了」的主证据，不是调参没调好的边角。

**Fig.3 pass@k**：8B 在 HMMT 上 k=1 大约 +13，k=32 仍 +7–10。曲线在高 k 不收敛到一起 → AntiSD 是 **覆盖了 GRPO 32 次也解不出的题**，并保住多样性，不是靠降方差刷 pass@1。

## 5.3 Table 2 代码 — 论点：机制不绑死数学

Qwen3-8B、Dolci-RLZero：HumanEval+ 40.4→41.6，MBPP+ 61.0→63.3。方向对、幅度小。作者自己说代码奖励更密，PMI 的边际更小。主叙事仍在数学。

## 5.4 Fig.4 训练动态 — 论点：SD 的双向失败，AntiSD 走中间带

- AntiSD：train reward 在 8B/4B 上约 30 step 从 0.5 拉到 0.95；GRPO 要约 150 step；SD 到不了。  
- SD 在 Qwen4B：师生熵塌到 ~0.1 nats（过度自信模板）；在 Olmo-IT：熵胀过 1 nat（漂离有用 token）。**同一错误极性，两种崩法。**  
- Qwen4B 的 SD 约 80 step：reward→0、长度顶 32K、熵尖峰，run 死。  
- 4B AntiSD 后期 train reward 停在 ~0.95：held-out 不掉，作者解释为 **训饱和（DAPO 几乎都会了），不是过拟合。**

## 5.5 消融 Table 3 / Fig.5 — 论点：三件套里谁在干活

在最容易崩的 Qwen3-4B-IT-2507 上：

| 改动 | 结果 | 说明 |
|---|---|---|
| 不反号（=SD） | Avg 65.7→30.6（8B） | 极性是主杠杆 |
| 上升 reverse-KL 而非 JSD | 4B Avg 49.5，追不上 GRPO | 无界审议侧会毁训练 |
| 去掉教师 | ~70 step 全模型崩 | 必须是 PMI(\(y_t;c\))，不是 self-reward |
| 去掉门 | Qwen 先冲到 0.97 再约 90 step 崩；Olmo 能撑满 200 且更高 | 门是跨模型保险，不是每模型都必要 |
| \(\tau_{\mathrm{down}}=0.90\) | 4B −8.3；8B 几乎不变 | 0.93 是迁移值不是 4B 甜区 |
| 乘法合成 | 4B 62.8→56.5，speedup 10×→5× | \(A^{\mathrm{seq}}\) 小时不该把 dense 项也乘没 |
| 门信号改学生困惑度 | 还能用，略低 | 教师熵/困惑度不是唯一选择 |

## 5.6 Table 4 续训 — 论点：AntiSD 可叠在已饱和的 GRPO 上

8B GRPO@200 再 AntiSD 50 step，+30 step 就到 Avg 65.0，接近从零训 180 step 的 65.7。说明审议信号在 GRPO 饱和后仍有信息。4B 续训只能到 61.9 vs 从零 62.8，有模型相关天花板。

## 5.7 这些实验怎样支撑「组会上可以守得住」的观点

- **诊断成立**：Fig.2 的 \(u_t\) 热力图 + SD 全面弱于 GRPO + 无教师崩，三角互证「问题在 PMI 极性，不在没加外部教师」。  
- **处方成立**：只翻 sign 就拉开 SD；JSD 形状解决炸梯度；门解决 Qwen 晚崩。  
- **不是刷单点**：五模型、五科、pass@k、续训，防止「一个种子一个集」。  
- **边界也写了**：PMI 是逐步局部刻画不是全局最优保证；主场是数学；agent / 更长代码是 future work。

---

# 6. 若要复现：数据、模型、硬件、代码、算法清单

## 6.1 数据

- 必须：`BytedTsinghua-SIA/DAPO-Math-17k` → `bash data/prepare_antisd.sh`  
- 评测：AIME24（HuggingFaceH4/aime_2024）、AIME25（math-ai/aime25）、AIME26、HMMT（MathArena/hmmt_feb_2025）、Minerva（math-ai/minervamath）  
- 注意：要用 **原版 DAPO**，不要 open-r1 清洗版（prompt 措辞不同）。  
- 要对齐 Table 1：val 需 **n=32、T=0.7、top-p=0.95**。仓库默认 val 是 n=4、T=0.6，只挂 AIME25。

## 6.2 模型

从 HF 拉基座，仓库不给 AntiSD ckpt：

- Qwen/Qwen3-8B  
- Qwen/Qwen3-4B-Instruct-2507  
- allenai/Olmo-3-7B-Instruct  
- allenai/Olmo-3-7B-Think  
- Qwen/Qwen3-30B-A3B（论文有，recipe **无脚本**）

不要用普通 `Qwen3-4B` 去对 Table 1 的 4B-IT-2507。

## 6.3 硬件

- 论文：8×H20 / 节点；30B 多机。  
- 认真复现 8B：8×80GB（H20/H100/A100-80）。  
- 单卡 48GB A6000：只能做推理/缩配置冒烟，不能原样 32×8×16k。  
- 依赖：CUDA 12.x，PyTorch 2.5.1（INSTALL）或集群已有 2.8+cu128；vLLM；flash-attn；W&B（脚本强制 key）。

## 6.4 代码

```text
recipe/antisd/run/<model>/antisd.sh   # PRM_RENYI_SIGN=-1
recipe/antisd/run/<model>/sd.sh       # +1，对照
recipe/antisd/run/<model>/grpo.sh     # λ=0
```

核心：`verl/workers/actor/dp_actor.py` 的 `grpo_ca` + `jsd_unbiased`。  
Teacher prompt：`ray_trainer.py` `_maybe_build_self_distillation_batch`。  
奖励：`verl/utils/reward_score/math_feedback/__init__.py`。

对齐 200 step 需加 `trainer.total_training_steps=200`（脚本现在是 `total_epochs=1` ≈ 531 step）。short launcher 训练长度 16k，论文写 32K。

## 6.5 算法超参（不要随手改的）

- `prm_forward_mode=jsd_unbiased`  
- AntiSD：`prm_renyi_sign=-1`；SD：`+1`  
- `ca_mode=additive`，`ca_lambda_max=0.5`  
- warmup 5（Think 用 10）  
- `TP_TARGET_RATIO=0.93`（论文 \(\tau_{\mathrm{down}}=0.93 H_{\mathrm{warm}}\)）  
- 代码门用 perplexity，且 \(\lambda\) 连续——和 Alg.1 二值门不完全一样，复现「论文数字」以 **官方脚本** 为准，不要只抄公式自己写一遍。

## 6.6 最小复现路径（建议组里这么排）

1. 环境 + 本地 4B 推理冒烟（已在 9007 上做过）。  
2. 4B-IT-2507 或 8B，8 卡，`antisd.sh` + `total_training_steps=200`。  
3. 同设置跑 `grpo.sh` / `sd.sh` 各一条，看 SD 是否弱于 GRPO、AntiSD 是否先点火。  
4. 正式表：把 val 改成 avg@32，补 AIME26 / HMMT / Minerva。  
5. 不要指望单卡 48GB 对齐 Table 1。
