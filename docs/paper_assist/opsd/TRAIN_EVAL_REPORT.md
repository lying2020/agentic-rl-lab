# OPSD 复现：训练与 AIME24 评测报告

日期：2026-09-11。机器：`100.88.54.3:1049`，目录 `/home1/cjl/ICLR2026`，GPU 1–4（A6000）。W&B offline。官方代码对应本地子模块 `algorithm/OPSD`（`ae7d251`）。

## 结论

两边正式 100-step OPSD 都训完，AIME24 官方长输出评测（`val_n=12`，`max_new_tokens=38912`）也都跑完。

- **Qwen3-1.7B** 最佳是 **checkpoint-100**，Avg@12 **54.17%**，相对 Base **+4.73pp**。
- **Qwen3-4B** 最佳是 **checkpoint-75**，Avg@12 **76.39%**，相对 Base **+4.17pp**；checkpoint-100 回落到 **75.83%**（仍 +3.61pp）。
- 主指标 Avg@12 随训练上升；Maj@12 几乎不动。1.7B 的 Pass@12 略降，4B 的 Pass@12 从 83.3% 升到 90%。

机制对照见 [CODE_PAPER_MAP.md](CODE_PAPER_MAP.md)。训练曲线见 [OPSD-training-curves.canvas.tsx](/home/liying/.cursor/projects/home-liying-Desktop-agentic-rl-lab/canvases/OPSD-training-curves.canvas.tsx)。

## 训练配置

与官方 `scripts/run_opsd_1b.sh` 同精神，适配 4×A6000：

| 项 | 取值 |
|---|---|
| 数据 | `datasets/Openthoughts_math_30k_opsd` |
| 步数 | 100（ckpt 25/50/75/100） |
| 优化 | lr `5e-6`，线性衰减；batch 1 × grad accum 8 × 4 卡 → 全局 32 |
| LoRA | `--use_peft` r=64，只更新学生 |
| 教师 | `--fixed_teacher`（`no_grad` + `disable_adapter()`，初始 base） |
| Loss | `--beta 0`（forward KL），`--jsd_token_clip 0.05` |
| 未开 | `--reason_first`、`--use_tinker_loss` |
| Thinking | 默认 `student_thinking=False`，`teacher_thinking=True` |
| 生成 | completion 1024；1.7B vLLM util 0.3，4B util 0.25 |

墙钟（训练日志头尾）：

| 模型 | 输出目录 | 墙钟 | 约 s/step | 生成速度 |
|---|---|---|---|---|
| Qwen3-1.7B | `outputs/qwen31b_a6000_100steps_retry2` | 00:46–03:15 ≈ 2h29min | 89 | ~128 tok/s |
| Qwen3-4B | `outputs/qwen34b_a6000_100steps` | 15:15–19:47 ≈ 4h32min | 163 | ~65 tok/s |

Loss 约 `0.005 → -0.0077`（约 step 8 过零）。这是 clipped forward-KL surrogate，不是记录错误：`jsd.clamp(max=0.05)` 封每个词表项的正项、保留负项，求和可为负。`grad_norm` 全程远低于 `0.1`。completion 均值约 840–850 tok。

## 评测协议

官方 `evaluate_math.py`，脚本 `OPSD/eval/run_eval_{1b,4b}_a6000.sh`。

- 集：AIME24，30 题
- 每题 12 条：`val_n=12`，共 360 次生成
- `temperature=1.0`，`enable_thinking=True`，`max_new_tokens=38912`
- vLLM TP=4，GPU 1–4
- 1.7B 日志 `eval_1b_aime24.log`，结果 `eval_results/qwen31b_aime24/`
- 4B 日志 `eval_4b_aime24.log`，结果 `eval_results/qwen34b_aime24/`

指标：

- **Avg@12**：360 条里答对比例（主指标）
- **Maj@12**：每题 12 票多数，30 题对几题
- **Pass@12**：每题 12 条里至少一条对

1pp Avg@12 ≈ 3.6/360 条。30 题上 1–2pp 波动有采样噪声。

## AIME24 结果

### Qwen3-1.7B

| Checkpoint | Avg@12 | Δ Base | Maj@12 | Pass@12 | Format |
|---|---:|---:|---:|---:|---:|
| Base | 49.44 | — | 66.67 | 80.00 | 98.61 |
| 25 | 48.89 | −0.55 | **70.00** | 80.00 | 99.17 |
| 50 | 51.11 | +1.67 | 66.67 | 76.67 | 99.72 |
| 75 | 52.22 | +2.78 | 66.67 | 73.33 | 99.72 |
| **100** | **54.17** | **+4.73** | 66.67 | 76.67 | 98.89 |

### Qwen3-4B

| Checkpoint | Avg@12 | Δ Base | Maj@12 | Pass@12 | Format |
|---|---:|---:|---:|---:|---:|
| Base | 72.22 | — | 80.00 | 83.33 | 99.17 |
| 25 | 73.61 | +1.39 | 80.00 | 83.33 | 99.44 |
| 50 | 75.56 | +3.34 | 80.00 | 86.67 | 99.17 |
| **75** | **76.39** | **+4.17** | 80.00 | **90.00** | 99.72 |
| 100 | 75.83 | +3.61 | 80.00 | 90.00 | 99.44 |

## 读数

1. **Avg@12 是真正被拉开的量。** 1.7B 在 ckpt-25 有一次 −0.55pp 回撤，之后单调升到 +4.73pp。4B 从 Base 一直升到 ckpt-75（+4.17pp），ckpt-100 回落 0.56pp。两边提升幅度接近，都在 4pp 量级。
2. **Maj@12 基本封顶。** 1.7B 除 ckpt-25 的 70% 外一直是 66.67%（20/30）；4B 五组全是 80%（24/30）。多数票已经卡在同一批题上，OPSD 主要在提高「已经会做的题」上 12 条里对几条，而不是翻更多新题的共识。
3. **Pass@12 两边方向不同。** 4B 从 83.3% → 90%（25/30 → 27/30），覆盖变宽。1.7B 从 80% 降到 73–77%，同时 Avg@12 上升：更像是在已覆盖题上更稳，而不是多解开新题。30 题噪声也够解释 1–2 题的出入。
4. **4B 最佳不在最后一步。** ckpt-75 高于 ckpt-100。100-step LoRA、AIME24 n=30，更像方差/轻度过拟合，不足以断言必须早停。若继续训，应看更多 step 或第二套评测（AIME25 / MATH）。
5. **格式不是瓶颈。** boxed 率全程 98.6–99.7%。分数变化来自对错，不是输出格式。
6. **和训练 loss 对齐。** 两边 loss 同期过零、后期贴在 −0.007 附近，`grad_norm` 稳定。评测增益出现在 loss 已为负、梯度仍远小于 0.1 的后半段，说明 clipped forward-KL 在更新学生 LoRA，而不是数值发散。

主实验是 **mode-covering 的 forward KL**（`beta=0`），不是 reverse KL。`use_tinker_loss` 才是 sampled-token / PG 变体，本次未开。

## 适配与路径

本地适配脚本在 `models/OPSD/`，已 scp 到远程 `OPSD/`：

- 训练：`scripts/run_opsd_{1b,4b}_a6000_{smoke,medium,100steps}.sh`
- 评测：`eval/run_eval_{1b,4b}_a6000.sh`
- 离线补丁：`OPSD_DATASET_PATH`、`OPSD_EVAL_DATASET_ROOT`（改过 `opsd_train.py` / `evaluate_math.py`）

远程关键路径：

| 用途 | 路径 |
|---|---|
| 1.7B 模型 | `/home1/cjl/ICLR2026/models/Qwen3-1.7B-modelscope` |
| 4B 模型 | `/home1/cjl/ICLR2026/models/Qwen3-4B-modelscope` |
| 训练数据 | `/home1/cjl/ICLR2026/datasets/Openthoughts_math_30k_opsd` |
| 评测集 | `/home1/cjl/ICLR2026/eval_datasets` |
| Conda | `source ~/anaconda3/etc/profile.d/conda.sh && conda activate opsd` |

复现评测：同一 conda、同一脚本、GPU 1–4；不要和训练抢显存。每个 4B checkpoint 约 2 小时。

## 限制

- 只跑了 100 step、LoRA，不是论文全量 SFT/全参或更长 schedule。
- 只报 AIME24；未跑 AIME25、MATH、LiveCodeBench。
- 未开 `reason_first` 显式 rationalization。
- 教师固定为初始 base，不是在线更新教师。
- GPU 0 有他人任务，全程未占用。
