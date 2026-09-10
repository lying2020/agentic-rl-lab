# OPSD 官方仓复现前评估

评估日期：2026-09-09  
目标服务器：`100.88.54.3:1049`  
目标目录：`/home1/cjl/ICLR2026`  
范围：仅做只读探测与方案梳理，尚未安装环境、下载完整权重或启动训练。

## 1. 结论

这台服务器可以复现 OPSD，建议第一阶段严格采用官方最小配置：

- 模型：`Qwen/Qwen3-1.7B`
- 训练集：`siyanzhao/Openthoughts_math_30k_opsd`
- GPU：`CUDA_VISIBLE_DEVICES=1,2,3,4`，共 4 张空闲 RTX A6000 48GB
- 训练：官方 `OPSDTrainer` + TRL GOLD + Accelerate + DeepSpeed ZeRO-2
- Rollout：vLLM colocate 模式
- 参数更新：LoRA，rank 64
- 首次验证：先把 completion 长度和训练步数缩小，完成 1～2 step smoke test；通过后再恢复官方参数

当前不能直接启动正式训练，主要原因有三点：

1. 服务器无法直连 `huggingface.co`，必须配置 `HF_ENDPOINT=https://hf-mirror.com` 或改用 ModelScope/离线传输。
2. 系统 Python 环境不可复用：只有旧版 `torch 2.1.2+cu118`，且存在 NumPy 2.x ABI 警告；OPSD 所需的大部分包未安装。
3. GPU 0 正被一个 vLLM 进程占用约 44GB，必须显式使用 GPU 1～4，不能让 Accelerate 默认选择 GPU 0～3。

## 2. 远程资源实测

### 2.1 硬件

- GPU：5 × NVIDIA RTX A6000，每张 48GB
- 当前空闲：GPU 1、2、3、4
- 当前占用：GPU 0 上有 `VLLM::EngineCore`，占约 44GB，利用率 100%
- CPU：2 × Intel Xeon Gold 6226R，合计 64 逻辑 CPU
- 内存：754GiB，总可用约 720GiB
- 存储：`/home1` 总计 14TB，剩余约 1.4TB，但整体已使用 90%
- 目标目录：存在、为空、当前用户可写

A6000 属于 Ampere 架构，支持 BF16 和 FlashAttention 2。4 张空闲卡足以匹配官方 `Qwen3-1.7B` 的 4 卡脚本。GPU 跨两个 NUMA 节点且没有 NVLink，实际速度会明显慢于 README 中的 4×H100。

### 2.2 软件现状

- OS：Ubuntu 22.04.5
- NVIDIA Driver：595.71.05
- 可用 CUDA Toolkit：11.7、11.8、12.1、13.x；建议先以 CUDA 12.1 为编译工具链
- GCC：11.4.0
- Conda：24.9.2，位于 `~/anaconda3`，但非交互 SSH 默认没有加入 `PATH`
- Docker：命令存在，但当前用户无 Docker daemon 权限
- 系统 Python：3.10.12，不建议使用

官方锁定依赖：

```text
Python 3.10
torch 2.8.0
accelerate 1.11.0
transformers 4.57.1
trl 0.26.0
datasets 3.6.0
deepspeed 0.18.2
peft 0.17.1
bitsandbytes 0.48.2
wandb 0.22.3
vllm 0.11.0
xformers 0.0.32.post1
triton 3.4.0
flash-attn 2.8.3（单独安装）
```

应新建独立 `opsd` Conda 环境，不能在系统 Python 上直接补包。FlashAttention 编译时必须核对 `torch.version.cuda` 与 `CUDA_HOME`；若二者小版本不匹配，应改用匹配的预编译 wheel，而不是强行源码编译。

## 3. 必须下载的内容

### 3.1 跑通官方最小 OPSD

只需要一个模型，不需要单独下载 Student 和 Teacher：

1. `Qwen/Qwen3-1.7B`
   - Hugging Face 仓总大小约 4.08GB。
   - OPSD 是 self-distillation：Student 只看题目，Teacher 是同一个模型，但额外看到参考解答。
   - `--fixed_teacher --use_peft` 时，Teacher 通过临时禁用 LoRA adapter 使用初始 base policy；Student 更新 LoRA。
2. `siyanzhao/Openthoughts_math_30k_opsd`
   - 29,434 条训练样本。
   - 下载大小约 280MB，解压/Arrow 缓存后约 654MB。
   - OPSD 使用的核心字段是 `problem` 和 `solution`。
3. OPSD 官方代码仓
   - 当前本地已有 `models/OPSD`，正式开始时复制或重新 clone 到服务器目标目录即可。

不需要下载：

- `DeepMath-103K`
- `DeepMath-Omn-1.5B`
- 额外的 7B/9B Teacher
- PyTRIO

它们属于前面讨论的通用 OPD 方案，不是这个 OPSD 官方实现的输入。

### 3.2 完整复现论文/README 评测

README 的主要结果覆盖以下三个小型数学评测集：

- `HuggingFaceH4/aime_2024`
- `yentinglin/aime_2025`
- `MathArena/hmmt_feb_2025`

仓库自带的 `eval/run_eval.sh` 默认只跑 AIME24。若要复现 README 的完整结果，需要把评测命令扩展到 AIME25 和 HMMT25，并分别评测：

- Base model
- checkpoint-25
- checkpoint-50
- checkpoint-75
- checkpoint-100

每题采样 12 次，指标为 Avg@12。评测脚本还支持以下可选数据集，但它们不是最小复现必需项：

- `HuggingFaceH4/MATH-500`
- `meituan-longcat/AMO-Bench`
- `math-ai/minervamath`
- `math-ai/amc23`

### 3.3 其他模型实验

- `Qwen/Qwen3-4B`：约 8.06GB，用于 4B thinking/non-thinking 实验
- `Qwen/Qwen3-8B`：约 16GB，用于 8B thinking/non-thinking 实验

建议先不下载 4B 和 8B。1.7B 跑通后再决定是否扩大实验，避免同时增加环境、显存和评测成本。

## 4. 网络实测与时间估算

服务器网络结果：

- `huggingface.co`：连接超时，当前不可用
- `github.com`：HTTP 200，连接与请求约 0.27 秒
- `pypi.org`：HTTP 200，连接与请求约 0.28 秒
- `hf-mirror.com`：HTTP 200
- `modelscope.cn`：HTTP 200

从 `hf-mirror.com` 分别对 OPSD 数据集和 Qwen3-1.7B 权重做了 10MiB Range 下载：

- 数据集：约 4.02MB/s
- 模型：约 3.99MB/s

按实测 4MB/s 粗略估算：

- OPSD 训练集 280MB：约 1～2 分钟
- Qwen3-1.7B 4.08GB：约 17～22 分钟
- Qwen3-4B 8.06GB：约 34～45 分钟
- Qwen3-8B 约 16GB：约 68～90 分钟
- 代码仓：通常 1 分钟内

环境依赖包含 PyTorch、vLLM、DeepSpeed、xFormers 和 FlashAttention，下载与安装量远大于训练集。若 wheel 匹配且无需编译，预计 15～45 分钟；若 FlashAttention 需要源码编译或版本冲突，可能额外需要 30～90 分钟。

因此，1.7B 路线从零到完成 smoke test 的合理预算是：

- 顺利情况：约 45～90 分钟
- 出现 CUDA / FlashAttention / vLLM 兼容问题：约 2～4 小时

正式 100-step 训练在 A6000 上尚未实测。README 的约 15 分钟是 4×H100，不能直接套用。结合 A6000 算力、无 NVLink、跨 NUMA 和 rollout 长度，初步估计约 45 分钟～3 小时，应先用 5～10 step 实测后再外推。

## 5. verl、vLLM、FSDP/FASP 分别做什么

### 5.1 verl

verl 是面向 LLM 强化学习后训练的编排框架。它负责把 Actor/Policy、Reference、Reward/Critic、rollout 服务和分布式训练后端组织起来，常与 FSDP/Megatron、vLLM/SGLang 配合，适合 PPO、GRPO 等复杂 RL 数据流。

本仓复现 OPSD **不需要 verl**。官方代码已经基于 TRL 的实验性 GOLD Trainer 实现了训练循环，再套了一层自定义 `OPSDTrainer`。引入 verl 意味着重写算法和数据流，不属于严格复现，并会显著增加调试量。

### 5.2 vLLM

vLLM 是高吞吐 LLM 推理/生成引擎，擅长通过 PagedAttention、连续批处理等机制加速 rollout 和评测。

本仓中 vLLM 有两个用途：

1. 训练时生成 Student 的 on-policy completion。
2. 评测时批量生成每道数学题的 12 个答案。

官方脚本启用：

```text
--use_vllm
--vllm_mode colocate
--vllm_gpu_memory_utilization 0.6
--vllm_tensor_parallel_size 1
```

`colocate` 表示 vLLM 和训练模型共享同一组 GPU。这样不用额外准备推理卡，但必须处理训练权重同步与显存竞争。理论上可以关闭 vLLM，退回 Transformers `generate`，但会明显变慢，也偏离官方配置。因此本次复现将 vLLM 视为必要依赖。

### 5.3 FSDP

如果用户所说的 “fasp” 实际是 **FSDP**：FSDP 是 PyTorch 的 Fully Sharded Data Parallel，把模型参数、梯度和优化器状态切分到多张 GPU，降低单卡显存占用。

此仓库的 `opsd_trainer.py` 虽然包含 FSDP 兼容代码，但官方 `accelerate.yaml` 实际配置的是：

```text
distributed_type: DEEPSPEED
zero_stage: 2
offload_optimizer_device: cpu
```

也就是说官方复现使用 **DeepSpeed ZeRO-2**，不是 FSDP。ZeRO-2 会分片优化器状态和梯度，并把优化器状态 offload 到 CPU。为了尽量忠实复现，第一轮不切换到 FSDP。

### 5.4 FASP

代码仓及依赖中没有名为 FASP 的训练工具。公开文献中的 FASP 通常指 “Fast and Accurate Structured Pruning”，它是模型结构化剪枝方法，不是 OPSD 训练框架，也不应加入本次复现。

如果 “fasp” 指的是 **FlashAttention**，它是注意力算子的显存与速度优化实现。官方明确要求 `flash-attn==2.8.3` 并传入 `--attn_implementation flash_attention_2`，因此 FlashAttention 2 是官方配置中的重要加速依赖。

## 6. OPSD 官方实现的数据流

```text
训练样本 (problem, solution)
  ├─ Student context：仅 problem
  │    └─ vLLM 用当前 Student 生成 on-policy completion
  └─ Teacher context：problem + ground-truth solution + transition prompt
       └─ 同一个模型对 Student completion 计算 token 分布

Student distribution 与 Teacher distribution
  └─ full-vocabulary JSD / KL（beta=0 为 forward KL）
       └─ point-wise token clipping
            └─ LoRA + DeepSpeed ZeRO-2 更新 Student
```

这与之前 Blog 中“大 Teacher 对小 Student 的 token logprob 蒸馏”不同。OPSD 的关键是**同一个模型、不同上下文权限**：Teacher 因为看到了参考解答而变强，不依赖另一个更大的模型。

## 7. 环境调试行动项

### 阶段 A：目录与网络

1. 将官方仓放到 `/home1/cjl/ICLR2026/OPSD`。
2. 将缓存统一放到大盘目标目录，避免污染 home：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home1/cjl/ICLR2026/.cache/huggingface
export TRANSFORMERS_CACHE=/home1/cjl/ICLR2026/.cache/huggingface/hub
```

3. 下载 Qwen3-1.7B 和 OPSD 数据集，并做文件完整性检查。
4. 预留至少 50GB 工作空间；若未来跑 4B/8B 和多个缓存，建议预留 150GB。

### 阶段 B：隔离环境

1. 加载 Conda：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
```

2. 从 `environment.yml` 创建全新 `opsd` 环境。
3. 验证 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

4. 验证 vLLM、DeepSpeed、TRL、PEFT 和 FlashAttention 能导入。
5. 运行 DeepSpeed distributed sanity test，确认 GPU 1～4 通信正常。

### 阶段 C：代码适配

官方脚本中的以下路径是作者机器路径，必须修改：

- `/data0/shared/Qwen3-1.7B`
- `/data0/siyanz/opsd/`
- eval 脚本中的 `BASE_MODEL` 和 `EXP_DIR`

启动前必须加：

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
```

W&B 在代码中被直接初始化。两种选择：

- 在线记录：需要执行 `wandb login`，这一步涉及账号登录，按约定应暂停等待用户完成。
- 本地验证：设置 `WANDB_MODE=offline` 或 `WANDB_MODE=disabled`，无需登录。

建议 smoke test 使用 offline/disabled，正式训练前再决定是否登录。

### 阶段 D：分级验证

1. 静态检查：参数解析、tokenizer chat template、数据集字段。
2. 单卡推理：vLLM 加载 Qwen3-1.7B 并生成一条短答案。
3. 4 卡分布式 smoke test：
   - `max_completion_length=128`
   - batch size 降到 1
   - 只跑 1～2 step
   - 暂不保存大量 checkpoint
4. 中型测试：
   - `max_completion_length=512`
   - 5～10 step
   - 记录每 step 时长、峰值显存和是否出现 NaN
5. 正式复现：
   - 恢复 `max_completion_length=1024`
   - 运行到 100 step
   - 保存 25/50/75/100 checkpoint
6. 评测 Base 与四个 checkpoint，在 AIME24/AIME25/HMMT25 上跑 Avg@12。

## 8. 正式推进前的建议

推荐路线是 `Qwen3-1.7B + 4×A6000 + 官方 OPSD/TRL/DeepSpeed/vLLM`。不要先改成 DeepMath 模型、verl 或独立 Teacher，否则会同时改变算法定义和工程栈，无法判断问题来自 OPSD 还是迁移适配。

正式推进前只需确认两件事：

1. 是否允许使用 GPU 1～4，并保持 GPU 0 上现有任务不受影响。
2. W&B 使用 offline/disabled，还是由用户先完成在线登录。

确认后再开始下载、安装和 smoke test；在确认前不启动付费服务，也不占用 GPU 做正式训练。
