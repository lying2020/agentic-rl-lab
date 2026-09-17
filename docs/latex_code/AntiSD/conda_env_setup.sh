#!/usr/bin/env bash
# =============================================================================
# AntiSD conda 环境一键安装脚本
#
# 对应仓库: algorithm/AntiSD (veRL fork)
# 官方文档: INSTALL.md, requirements.txt, Dockerfile
#
# 推荐栈 (与 INSTALL.md Method A / Dockerfile 一致):
#   Python 3.12 + PyTorch 2.5.1 + CUDA 12.4 + transformers 4.57.1 + Ray 2.53
#
# 用法:
#   bash conda_env_setup.sh
#   bash conda_env_setup.sh --env-name antisd --skip-flash-attn
#   bash conda_env_setup.sh --cuda 128 --vllm-spec "vllm>=0.8.4,<0.13"
#
# 环境变量覆盖:
#   ANTISD_ROOT     仓库根目录 (默认: 自动探测)
#   CONDA_ENV_NAME  conda 环境名 (默认: antisd)
#   PYTHON_VERSION  Python 版本 (默认: 3.12)
#   TORCH_CUDA      cu124 | cu121 | cu128 | cpu
# =============================================================================
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-antisd}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
SKIP_FLASH_ATTN=0
SKIP_VLLM=0
SKIP_CONDA_CREATE=0
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.8.4,<0.13}"
ANTISD_ROOT="${ANTISD_ROOT:-}"

usage() {
    sed -n '2,28p' "$0"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --cuda) TORCH_CUDA="$2"; shift 2 ;;
        --antisd-root) ANTISD_ROOT="$2"; shift 2 ;;
        --skip-flash-attn) SKIP_FLASH_ATTN=1; shift ;;
        --skip-vllm) SKIP_VLLM=1; shift ;;
        --skip-conda-create) SKIP_CONDA_CREATE=1; shift ;;
        --vllm-spec) VLLM_SPEC="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

step() { echo -e "\n\033[1;32m===> $1\033[0m"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
info() { echo "     $1"; }

# ── 0. 定位仓库根 ────────────────────────────────────────────
if [[ -z "$ANTISD_ROOT" ]]; then
    if [[ -f "$(pwd)/verl/__init__.py" && -f "$(pwd)/requirements.txt" ]]; then
        ANTISD_ROOT="$(pwd)"
    elif [[ -f "/home/user/Documents/agentic-rl-lab/algorithm/AntiSD/verl/__init__.py" ]]; then
        ANTISD_ROOT="/home/user/Documents/agentic-rl-lab/algorithm/AntiSD"
    else
        echo "无法定位 AntiSD 仓库。请用 --antisd-root /path/to/AntiSD"
        exit 1
    fi
fi

if [[ ! -f "${ANTISD_ROOT}/verl/__init__.py" ]]; then
    echo "ANTISD_ROOT=${ANTISD_ROOT} 不是 AntiSD / veRL 仓库"
    exit 1
fi

step "仓库: ${ANTISD_ROOT}"
info "conda env = ${ENV_NAME}  python=${PYTHON_VERSION}  torch_cuda=${TORCH_CUDA}"

if ! command -v conda >/dev/null 2>&1; then
    echo "未找到 conda。请先安装 Miniconda / Anaconda。"
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ── 1. 创建环境 ──────────────────────────────────────────────
if [[ "$SKIP_CONDA_CREATE" -eq 0 ]]; then
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        warn "环境 ${ENV_NAME} 已存在，将直接复用。删除请先: conda env remove -n ${ENV_NAME}"
    else
        step "创建 conda 环境 ${ENV_NAME} (python=${PYTHON_VERSION})"
        conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
    fi
fi

conda activate "$ENV_NAME"
python -m pip install -U pip setuptools wheel ninja packaging

# ── 2. PyTorch ──────────────────────────────────────────────
step "安装 PyTorch (与 INSTALL.md / Dockerfile 对齐)"
case "$TORCH_CUDA" in
    cu124)
        pip install torch==2.5.1 torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124
        ;;
    cu121)
        pip install torch==2.5.1 torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu121
        ;;
    cu128)
        warn "CUDA 12.8 / Blackwell: 官方脚本未钉死。改用较新的 PyTorch nightly/stable。"
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu128
        ;;
    cpu)
        warn "CPU 只能跑通 import / 数据预处理，不能训练。"
        pip install torch==2.5.1 torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "未知 --cuda ${TORCH_CUDA}，可选: cu124 cu121 cu128 cpu"
        exit 1
        ;;
esac

python - <<'PY'
import torch
print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"cuda_runtime={getattr(torch.version, 'cuda', None)}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
PY

# ── 3. 仓库依赖 ──────────────────────────────────────────────
# 注意:
#   INSTALL.md 写了 requirements-stable.txt，仓库里实际只有 requirements.txt。
#   setup.py 的 install_requires 写了 numpy<2.0.0，但 requirements.txt 钉了 numpy==2.1.0。
#   这里以 requirements.txt 为准（论文复现栈），再 editable install --no-deps，避免被 setup.py 降级 numpy。
step "安装 requirements.txt (钉死版本)"
cd "$ANTISD_ROOT"
pip install -r requirements.txt

step "editable 安装 AntiSD/veRL (跳过 setup.py 依赖，避免 numpy<2 冲突)"
pip install -e . --no-deps

# setup.py extras 里的 math-verify 已在 requirements.txt 中:
#   math-verify[antlr4_9_3]==0.8.0
# 奖励函数 verl/utils/reward_score/math_feedback/__init__.py 依赖它。

# ── 4. FlashAttention ───────────────────────────────────────
if [[ "$SKIP_FLASH_ATTN" -eq 1 ]]; then
    warn "已跳过 flash-attn。长上下文训练会更慢，且部分 kernel 路径可能回退。"
else
    step "安装 Flash Attention 2 (编译较久，失败可 --skip-flash-attn)"
    if pip install flash-attn --no-build-isolation; then
        info "flash-attn 安装成功"
    else
        warn "flash-attn 编译失败。可稍后手动装 wheel，或加 --skip-flash-attn。"
        warn "常见原因: CUDA toolkit / nvcc 与 PyTorch 不匹配，或缺 ninja。"
    fi
fi

# ── 5. vLLM (rollout 必需) ──────────────────────────────────
# AntiSD 默认 rollout.name=vllm (verl/trainer/config/user.yaml)。
# 训练脚本不装 vLLM 会在 generate_sequences 阶段失败。
if [[ "$SKIP_VLLM" -eq 1 ]]; then
    warn "已跳过 vLLM。仅适合数据预处理 / 单卡 HF 推理，不能跑 recipe/antisd 训练。"
else
    step "安装 vLLM: ${VLLM_SPEC}"
    info "硬件对照 (requirements.txt 注释):"
    info "  GH200 / 官方集群: vllm==0.8.4"
    info "  Ampere/Hopper (A100/H100/H20): ${VLLM_SPEC}"
    info "  Blackwell (RTX 50 / B200): 往往需要 vllm>=0.12.0，且 PyTorch 也要更新"
    pip install "${VLLM_SPEC}"
fi

# ── 6. 可选: HuggingFace 下载加速 ────────────────────────────
pip install -q "huggingface_hub[cli]" hf_transfer || true

# ── 7. 冒烟检查 ──────────────────────────────────────────────
step "冒烟检查 import"
python - <<'PY'
import importlib
mods = [
    "torch", "transformers", "datasets", "ray", "hydra",
    "verl", "math_verify", "peft", "wandb",
]
optional = ["flash_attn", "vllm"]
for m in mods:
    importlib.import_module(m)
    print(f"  OK  {m}")
for m in optional:
    try:
        importlib.import_module(m)
        print(f"  OK  {m} (optional)")
    except Exception as e:
        print(f"  --  {m} 未安装: {e}")
import verl
print("verl package:", getattr(verl, "__file__", "ok"))
PY

step "完成"
cat <<EOF
下一步:
  conda activate ${ENV_NAME}
  export WANDB_API_KEY=...          # 训练脚本强制要求
  export HF_HOME=\$HOME/.cache/huggingface
  cd ${ANTISD_ROOT}
  bash data/prepare_antisd.sh       # 下载 DAPO-Math + AIME/HMMT
  bash recipe/antisd/run/qwen3-8b/antisd.sh

已知冲突 / 需要你确认:
  1. INSTALL.md 提到 requirements-stable.txt，仓库中不存在，已改用 requirements.txt。
  2. setup.py 要求 numpy<2，requirements.txt 钉 numpy==2.1.0；本脚本按后者安装。
  3. scripts/install_vllm_sglang_mcore.sh 是上游 veRL 通用脚本 (vllm==0.11.0,
     sglang==0.5.2, torch 2.8 wheel)，与 INSTALL.md 的 2.5.1/cu124 栈不一致。
     复现 AntiSD 请不要混用那条路径。
  4. 训练必须有 NVIDIA GPU + 匹配的驱动。论文硬件是 8×H20 / node。
EOF
