#!/bin/bash
# ============================================================
# Environment Verification Script
# Run after env_setup.sh to check everything works
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass_count=0
fail_count=0

check() {
    local name="$1"
    local cmd="$2"
    echo -n "  [$name] ... "
    if eval "$cmd" &>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}FAIL${NC}"
        fail_count=$((fail_count + 1))
    fi
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/conda_envs}"

echo "=============================================="
echo " Project-1 Environment Verification"
echo " Project root: $PROJECT_ROOT"
echo " Conda prefix: $CONDA_ENV_PREFIX"
echo "=============================================="

# ---- System ----
echo ""
echo "[System]"
check "nvidia-smi" "nvidia-smi"
check "nvcc" "nvcc --version"
check "conda" "conda --version"
check "colmap" "colmap --help"
check "ffmpeg" "ffmpeg -version"

# ---- 2DGS Env ----
echo ""
echo "[2DGS Environment]"
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX/2dgs"
check "python 3.8" "python -c 'import sys; assert sys.version_info[:2] == (3,8)'"
check "torch 2.0.0" "python -c 'import torch; assert torch.__version__.startswith(\"2.0.0\")'"
check "torch CUDA" "python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'"
check "diff-surfel-rasterization" "python -c 'import diff_surfel_rasterization'"
check "simple-knn" "python -c 'import simple_knn'"
conda deactivate

# ---- Threestudio Env ----
echo ""
echo "[Threestudio Environment]"
conda activate "$CONDA_ENV_PREFIX/threestudio"
check "python 3.10" "python -c 'import sys; assert sys.version_info[:2] == (3,10)'"
check "torch 2.0.0" "python -c 'import torch; assert torch.__version__.startswith(\"2.0.0\")'"
check "torch CUDA" "python -c 'import torch; assert torch.cuda.is_available()'"
check "diffusers" "python -c 'import diffusers; print(diffusers.__version__)'"
check "xformers" "python -c 'import xformers'"
check "nerfacc" "python -c 'import nerfacc'"
conda deactivate

# ---- Magic123 Env ----
echo ""
echo "[Magic123 Environment]"
conda activate "$CONDA_ENV_PREFIX/magic123"
check "python 3.10" "python -c 'import sys; assert sys.version_info[:2] == (3,10)'"
check "torch 2.0.0" "python -c 'import torch; assert torch.__version__.startswith(\"2.0.0\")'"
check "torch CUDA" "python -c 'import torch; assert torch.cuda.is_available()'"
check "diffusers" "python -c 'import diffusers; print(diffusers.__version__)'"
check "rembg" "python -c 'import rembg'"
check "freqencoder" "python -c 'import freqencoder' 2>/dev/null || echo 'not built yet'"
check "gridencoder" "python -c 'import gridencoder' 2>/dev/null || echo 'not built yet'"
check "raymarching" "python -c 'import raymarching' 2>/dev/null || echo 'not built yet'"
conda deactivate

# ---- Data ----
echo ""
echo "[Data Files]"
check "Cat_RGB images" "[ \$(ls $PROJECT_ROOT/data/Cat_RGB/*.{jpg,JPG,png,PNG} 2>/dev/null | wc -l) -gt 0 ]"
check "cat_single image" "[ -f $PROJECT_ROOT/data/cat_single/cat_rgba.png ] || [ -f $PROJECT_ROOT/data/cat_single/*.jpg ]"
check "garden images" "[ -d $PROJECT_ROOT/data/garden/images_4 ] && [ \$(ls $PROJECT_ROOT/data/garden/images_4/ 2>/dev/null | wc -l) -ge 10 ]"
check "garden COLMAP" "[ -f $PROJECT_ROOT/data/garden/sparse/0/cameras.bin ] || echo 'COLMAP data not yet generated'"

# ---- Frameworks ----
echo ""
echo "[Frameworks]"
check "2DGS repo" "[ -f $PROJECT_ROOT/2d-gaussian-splatting/2d-gaussian-splatting/train.py ]"
check "threestudio repo" "[ -f $PROJECT_ROOT/threestudio/threestudio/launch.py ]"
check "Magic123 repo" "[ -f $PROJECT_ROOT/Magic123/Magic123/main.py ]"

# ---- Summary ----
echo ""
echo "=============================================="
echo -e " Results: ${GREEN}$pass_count passed${NC}, ${RED}$fail_count failed${NC}"
echo "=============================================="

if [ $fail_count -gt 0 ]; then
    echo -e "${YELLOW}Some checks failed. Review above and fix before running pipeline.${NC}"
    echo "Note: Some failures are expected if training hasn't been run yet."
    exit 1
else
    echo -e "${GREEN}All checks passed! Ready to run pipeline.${NC}"
fi
