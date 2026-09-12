# 远端 GPU 最小推理记录（9007 / A6000）

时间：2026-09-12  
机器：`cjl@100.88.54.3:1049`（`tools/ssh-cjl_zju_5`）  
仓库：`/home1/cjl/ICLR2026/AntiSD`

## 原则：能复用就不下

| 资源 | 处理 |
|---|---|
| Python / torch / transformers | **复用已有 conda 环境 `opsd`**（torch 2.8.0+cu128，transformers 4.57.1）。没有新建 `antisd` 环境，也没有装 veRL / Ray / vLLM / flash-attn。 |
| 权重 | **复用本地** `Qwen3-1.7B-modelscope`（3.8G）和 `Qwen3-4B-modelscope`（7.6G）。`HF_HUB_OFFLINE=1`，零下载。 |
| 数据 | 推理冒烟用内置题目。AIME 等 parquet 已在 `/home1/cjl/ICLR2026/eval_datasets/`，尚未转换。 |
| GPU | 只用 **GPU 0**（约 48GB 空闲）。1–4 号卡被其他用户 AlphaFold 占满（各约 41GB）。 |

`Qwen3-1.7B` 的 git 目录只有 tokenizer、没有权重分片，不要用。用 `*-modelscope` 目录。

## 已跑通的结果

**1.7B**（约 26s，含加载）

```text
题: Compute 17 times 19.
pred=391   （算术错；17×19=323。说明管线通，模型能力弱）
```

**4B**（约 34s，含加载）

```text
题: x+y=10, x-y=4, find x
pred=7     （正确）
```

脚本已放到远端：

- `/home1/cjl/ICLR2026/AntiSD/inference_math.py`
- `/home1/cjl/ICLR2026/AntiSD/run_infer_gpu0.sh`

复现：

```bash
# 默认 1.7B
bash /home1/cjl/ICLR2026/AntiSD/run_infer_gpu0.sh "1+1=?"

# 换 4B
MODEL=/home1/cjl/ICLR2026/models/Qwen3-4B-modelscope \
  bash /home1/cjl/ICLR2026/AntiSD/run_infer_gpu0.sh "Solve x+y=10, x-y=4, find x."
```

## 还没做（训练）

训练才需要：`pip install -e .`、Ray、vLLM、`math-verify`、DAPO-Math-17k。  
当前 **没有** 装这些，也没有动 GPU 1–4。单卡 48GB A6000 不够按论文 8×H20 / 16k / n=8 原样训，后面要单独缩 batch。
