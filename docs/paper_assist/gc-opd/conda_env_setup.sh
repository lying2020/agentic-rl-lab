#!/usr/bin/env bash
set -euo pipefail
# GC-OPD 官方验证：Python 3.12, CUDA 12.8, torch 2.10.0
# 先保证推理（transformers）；训练依赖见仓库 requirements.txt
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install transformers accelerate
# 训练（可选）：
# python -m pip install --no-build-isolation -r requirements.txt
# python -m pip install -e ./verl --no-deps
# python -m pip install flash-attn --no-build-isolation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
