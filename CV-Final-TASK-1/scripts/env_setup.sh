#!/bin/bash
# ============================================================
# Project-1: Robust Environment Setup Script
# ============================================================
# Target: Ubuntu 20.04/22.04, CUDA 11.8, NVIDIA Driver >= 525
# Recommended: RTX 4090 (24GB) or A100 (40/80GB)
#
# Features:
#   - 3 isolated conda environments (2dgs / threestudio / magic123)
#   - Auto-detect China mainland → use mirrors
#   - Retry on network failures
#   - Verify each step
#
# Usage:
#   bash scripts/env_setup.sh              # Auto-detect mirrors
#   bash scripts/env_setup.sh --china      # Force China mirrors
#   bash scripts/env_setup.sh --prefix /my/envs  # Custom env prefix
# ============================================================

set -e

# ---- Configurable paths ----
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/conda_envs}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$HOME/project_cache}"
USE_CHINA_MIRROR=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --china) USE_CHINA_MIRROR=true; shift ;;
        --prefix) CONDA_ENV_PREFIX="$2"; shift 2 ;;
        --cache) DATA_CACHE_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Auto-detect: if pypi.org is unreachable, use mirrors
if ! curl -s --connect-timeout 3 https://pypi.org > /dev/null 2>&1; then
    echo "[*] Detected limited connectivity. Enabling China mirrors..."
    USE_CHINA_MIRROR=true
fi

if $USE_CHINA_MIRROR; then
    PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    CONDA_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
    APT_MIRROR="http://mirrors.tuna.tsinghua.edu.cn/ubuntu/"
    echo "[*] Using Tsinghua mirrors for pip/conda/apt"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

retry() {
    local n=1 max=3 delay=5
    while true; do
        if "$@"; then return 0; fi
        if [[ $n -lt $max ]]; then
            log_warn "Retry $n/$max after ${delay}s: $*"
            sleep $delay; n=$((n + 1))
        else
            log_error "Failed after $max attempts: $*"
            return 1
        fi
    done
}

# ---- Pre-flight checks ----
echo ""
echo "=============================================="
echo " Project-1: Environment Setup"
echo " $(date)"
echo "=============================================="

# ---- Configure cache directories ----
log_info "Configuring cache directories at $DATA_CACHE_DIR..."
mkdir -p "$DATA_CACHE_DIR/huggingface_cache" "$CONDA_ENV_PREFIX" \
         "$DATA_CACHE_DIR/pip_cache" "$DATA_CACHE_DIR/torch_home"

export HF_HOME="$DATA_CACHE_DIR/huggingface_cache"
export PIP_CACHE_DIR="$DATA_CACHE_DIR/pip_cache"
export TORCH_HOME="$DATA_CACHE_DIR/torch_home"

# Persist cache settings
if ! grep -q "HF_HOME=$DATA_CACHE_DIR" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << EOF
# Project-1: store caches on data disk
export HF_HOME=$DATA_CACHE_DIR/huggingface_cache
export PIP_CACHE_DIR=$DATA_CACHE_DIR/pip_cache
export TORCH_HOME=$DATA_CACHE_DIR/torch_home
EOF
fi

log_info "Checking prerequisites..."

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found! Install NVIDIA drivers first."
    log_error "See: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -3
log_ok "GPU detected"

# ---- Conda ----
log_info "Setting up Miniconda..."
if command -v conda &> /dev/null; then
    log_ok "Conda already installed: $(conda --version)"
else
    CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    if $USE_CHINA_MIRROR; then
        CONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
    retry wget "$CONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    log_ok "Miniconda installed"
fi

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"

# Configure conda channels
if $USE_CHINA_MIRROR; then
    conda config --set show_channel_urls yes
    conda config --add channels "$CONDA_MIRROR/pkgs/main/"
    conda config --add channels "$CONDA_MIRROR/pkgs/free/"
    conda config --add channels "$CONDA_MIRROR/cloud/conda-forge/"
    conda config --add channels "$CONDA_MIRROR/cloud/pytorch/"
fi

mkdir -p "$CONDA_ENV_PREFIX"

create_env() {
    local name="$1"; shift
    conda create --prefix "$CONDA_ENV_PREFIX/$name" "$@" -y 2>/dev/null || \
        log_warn "$name may already exist"
}

pip_install() {
    local env_name="$1"; shift
    conda activate "$CONDA_ENV_PREFIX/$env_name"
    log_info "[$env_name] Installing: $*"
    retry pip install $PIP_MIRROR "$@"
}

# ============================================================
# Environment 1: 2DGS (Python 3.8, PyTorch 2.0)
# ============================================================
log_info "[1/3] Creating '2dgs' environment..."
create_env 2dgs python=3.8.18

pip_install 2dgs torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118
pip_install 2dgs open3d==0.18.0 mediapy==1.1.2 lpips==0.1.4
pip_install 2dgs scikit-image==0.21.0 tqdm==4.66.2 trimesh==4.3.2 plyfile opencv-python
pip_install 2dgs tensorboard wandb swanlab

# Compile 2DGS CUDA extensions
log_info "[2dgs] Compiling CUDA submodules..."
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
_2DGS_DIR="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}/2d-gaussian-splatting/2d-gaussian-splatting"
if [ -d "$_2DGS_DIR" ]; then
    cd "$_2DGS_DIR"
    pip_install 2dgs submodules/diff-surfel-rasterization
    pip_install 2dgs submodules/simple-knn
    cd - > /dev/null
else
    log_warn "2DGS repo not found at $_2DGS_DIR. Install CUDA extensions manually."
fi

# Verify
conda activate "$CONDA_ENV_PREFIX/2dgs"
python -c "import torch; assert torch.cuda.is_available(); print(f'  CUDA: {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}')"
python -c "import diff_surfel_rasterization; print('  diff-surfel OK')" || log_error "diff-surfel-rasterization: compile failed!"
python -c "import simple_knn; print('  simple-knn OK')" || log_error "simple-knn: compile failed!"
log_ok "[2dgs] Environment ready"
conda deactivate

# ============================================================
# Environment 2: Threestudio (Python 3.10, PyTorch 2.0)
# ============================================================
log_info "[2/3] Creating 'threestudio' environment..."
create_env threestudio python=3.10

pip_install threestudio torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip_install threestudio lightning==2.0.0 omegaconf==2.3.0 jaxtyping typeguard
pip_install threestudio "diffusers>=0.18.0,<0.20.0" "transformers>=4.25,<5.0" accelerate
pip_install threestudio opencv-python tensorboard matplotlib "imageio>=2.28.0" "imageio[ffmpeg]"
pip_install threestudio "trimesh[easy]" networkx pysdf PyMCubes wandb swanlab
pip_install threestudio xformers "bitsandbytes>=0.35,<0.40" sentencepiece safetensors huggingface_hub
pip_install threestudio einops kornia taming-transformers-rom1504
pip_install threestudio git+https://github.com/openai/CLIP.git
pip_install threestudio controlnet_aux
pip_install threestudio git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
pip_install threestudio git+https://github.com/NVlabs/nvdiffrast.git
pip_install threestudio libigl xatlas
pip_install threestudio git+https://github.com/ashawkey/envlight.git

conda activate "$CONDA_ENV_PREFIX/threestudio"
python -c "import torch; assert torch.cuda.is_available(); print(f'  CUDA OK')"
python -c "import diffusers; print(f'  diffusers {diffusers.__version__}')"
python -c "import xformers; print('  xformers OK')"
log_ok "[threestudio] Environment ready"
conda deactivate

# ============================================================
# Environment 3: Magic123 (Python 3.10, PyTorch 2.0)
# ============================================================
log_info "[3/3] Creating 'magic123' environment..."
create_env magic123 python=3.10

pip_install magic123 torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip_install magic123 tqdm rich ninja numpy scipy scikit-learn matplotlib
pip_install magic123 opencv-python imageio imageio-ffmpeg pandas
pip_install magic123 torch-ema einops tensorboard tensorboardX
pip_install magic123 huggingface_hub "diffusers>=0.15,<0.18" accelerate "transformers>=4.25,<5.0"
pip_install magic123 xatlas PyMCubes "trimesh>=3.20,<4.0" "pymeshlab>=2022.0"
pip_install magic123 rembg carvekit-colab omegaconf pytorch-lightning
pip_install magic123 taming-transformers-rom1504 kornia
pip_install magic123 git+https://github.com/NVlabs/nvdiffrast.git
pip_install magic123 git+https://github.com/openai/CLIP.git
pip_install magic123 gdown "timm>=0.6,<0.7" easydict wandb swanlab termcolor
pip_install magic123 sentencepiece psutil lpips

# Compile Magic123 CUDA extensions
log_info "[magic123] Compiling CUDA submodules..."
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
_M123_DIR="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}/Magic123/Magic123"
if [ -d "$_M123_DIR" ]; then
    cd "$_M123_DIR"
    pip_install magic123 ./raymarching
    pip_install magic123 ./shencoder
    pip_install magic123 ./freqencoder
    pip_install magic123 ./gridencoder
    cd - > /dev/null
else
    log_warn "Magic123 repo not found at $_M123_DIR. Install CUDA extensions manually."
fi

conda activate "$CONDA_ENV_PREFIX/magic123"
python -c "import torch; assert torch.cuda.is_available(); print(f'  CUDA OK')"
python -c "import rembg; print('  rembg OK')"
python -c "import freqencoder; print('  freqencoder OK')" 2>/dev/null || log_warn "freqencoder not built yet"
python -c "import gridencoder; print('  gridencoder OK')" 2>/dev/null || log_warn "gridencoder not built yet"
python -c "import raymarching; print('  raymarching OK')" 2>/dev/null || log_warn "raymarching not built yet"
log_ok "[magic123] Environment ready"
conda deactivate

# ---- COLMAP ----
log_info "Installing COLMAP..."
if command -v colmap &> /dev/null; then
    log_ok "COLMAP already installed: $(colmap --version 2>&1 | head -1)"
else
    if $USE_CHINA_MIRROR; then
        sudo sed -i "s|http://archive.ubuntu.com|$APT_MIRROR|g" /etc/apt/sources.list 2>/dev/null || true
    fi
    sudo apt-get update -qq
    sudo apt-get install -y -qq colmap 2>/dev/null || {
        log_warn "apt install colmap failed. Trying conda..."
        conda install -n base -c conda-forge colmap -y 2>/dev/null || \
            log_warn "Conda colmap also failed. Install manually: https://colmap.github.io/install.html"
    }
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo " Environment Setup Complete!"
echo "=============================================="
echo ""
echo "Activate commands:"
echo "  conda activate $CONDA_ENV_PREFIX/2dgs         # 2DGS (Object A + Background)"
echo "  conda activate $CONDA_ENV_PREFIX/threestudio   # Text-to-3D (Object B)"
echo "  conda activate $CONDA_ENV_PREFIX/magic123      # Image-to-3D (Object C)"
echo ""
echo "Verify:  bash scripts/verify_env.sh"
echo "Run all: bash scripts/run_all.sh"
echo ""
echo "If CUDA extensions fail to compile:"
echo "  1. Check nvcc --version matches torch cuda version"
echo "  2. export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9'"
echo "  3. Try Docker alternative"
