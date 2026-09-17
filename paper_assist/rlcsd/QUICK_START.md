# Quick Start — RLCSD

推理优先：RLCSD 是训练损失。部署 = 原 Qwen3/Olmo thinking generate。

## 1. 环境

```bash
git clone https://github.com/THU-BPM/RLCSD.git
cd RLCSD
pip install -r requirements.txt
# torch 建议 <2.10，匹配 CUDA 12：
# pip install "torch>=2.5.0,<2.10" --index-url https://download.pytorch.org/whl/cu126
# pip install flash-attn --no-build-isolation
```

## 2. 权重

Hugging Face：`Qwen/Qwen3-1.7B`、`Qwen3-4B`、`Qwen3-8B`；`allenai/Olmo-3-7B-Think`（以仓库 YAML 为准）。

## 3. 数据

```bash
python scripts/download_data.py --all
# Leyiii/RLCSD → data/verl/.../{train,val}.parquet
```

## 4. 推理冒烟

加载学生 checkpoint，thinking 开，数学 `temperature=0.6`，max 38912，mean@12。不需要教师。

## 5. 训练（可选）

```bash
bash scripts/math_deepmath/run_qwen3_4b_rlcsd.sh
# 覆盖超参示例：
# bash scripts/math_deepmath/run_qwen3_4b_rlcsd.sh learning_rate=2e-6 group_size=16
```

**请按 PDF 核对 YAML**：$\tau=0.02$，$\eta=1.0$，teacher snapshot 每 10 step。README 另有一套 $\tau=1.3,\eta=0.5$。

正 hint 默认 `correct_privileged_hint_source=gt_cot`。
