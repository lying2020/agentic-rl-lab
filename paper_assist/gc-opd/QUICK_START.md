# Quick Start — GC-OPD

推理优先：方法只改训练优势，评测是 Qwen3 no-thinking generate。

## 仓库 Tree

见上游 README。核心：`verl/verl/trainer/ppo/gc_opd.py`。

## 1. 环境

```bash
git clone https://github.com/SolereZhang/GC-OPD.git
cd GC-OPD
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -e ./verl --no-deps
```

已验证：Linux x86_64，CUDA 12.8，vLLM 0.17.0，Ray 2.54.0，Transformers 4.57.1，FA 2.8.3。

## 2. 权重

```bash
hf download Qwen/Qwen3-4B --local-dir ../models/Qwen3-4B
hf download Qwen/Qwen3-8B --local-dir ../models/Qwen3-8B
hf download Qwen/Qwen3-30B-A3B-Thinking-2507 --local-dir ../models/Qwen3-30B-A3B-Thinking-2507
```

## 3. 数据（完整表才需要）

```bash
hf download Kwai-Klear/GoLongRL --repo-type dataset --local-dir ../data/GoLongRL
python scripts/prepare_golongrl_32k.py --overwrite
# 期望 9527 train / 231 val
```

## 4. 推理冒烟

```bash
# 任意 HF generate；enable_thinking=False
# 官方五基准：
# export MODEL_PATH=... JUDGE_MODEL_PATH=... DATA_ROOT=... OUTPUT_DIR=...
# bash evaluation/run_main_table_evaluation.sh
```

## 5. 训练（可选）

```bash
bash scripts/run_opd_4b_training.sh      # beta=0
bash scripts/run_gc_opd_4b_training.sh   # beta=0.10 RACA
DRY_RUN=1 bash scripts/run_gc_opd_8b_training.sh
```

默认 8 卡、100 step、seed 42。
