#!/bin/bash
# ============================================================
# Step 5: Train 2DGS on Background Scene
# Produces: output/background_garden/ (Gaussian splat)
# ============================================================
set -e

# ---- Configurable paths ----
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/conda_envs}"
_2DGS_DIR="${_2DGS_DIR:-$PROJECT_ROOT/2d-gaussian-splatting/2d-gaussian-splatting}"
SOURCE_PATH="${SOURCE_PATH:-$PROJECT_ROOT/data/garden}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/background_garden}"
ITERATIONS="${ITERATIONS:-30000}"
IMAGES_SUBDIR="${IMAGES_SUBDIR:-images_4}"  # images_4 = 4x downsampled
RESOLUTION="${RESOLUTION:-4}"

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX/2dgs"

echo "=============================================="
echo " Step 5: 2DGS Training - Background Scene"
echo " Source: $SOURCE_PATH ($IMAGES_SUBDIR)"
echo " Output: $MODEL_PATH"
echo " GPU Memory: ~12-16 GB"
echo " Time: ~1-1.5 hrs (${ITERATIONS} iters, RTX 4090)"
echo "=============================================="

cd "$_2DGS_DIR"

python train.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --images "$IMAGES_SUBDIR" \
    --resolution "$RESOLUTION" \
    --sh_degree 3 \
    --iterations "$ITERATIONS" \
    --position_lr_init 0.00016 \
    --position_lr_final 1.6e-07 \
    --position_lr_delay_mult 0.01 \
    --position_lr_max_steps "$ITERATIONS" \
    --feature_lr 0.0025 \
    --opacity_lr 0.05 \
    --scaling_lr 0.005 \
    --rotation_lr 0.001 \
    --percent_dense 0.01 \
    --lambda_dssim 0.2 \
    --densification_interval 100 \
    --opacity_reset_interval 3000 \
    --densify_from_iter 500 \
    --densify_until_iter 15000 \
    --lambda_normal 0.05 \
    --lambda_dist 1000.0 \
    --save_iterations 7000 15000 "$ITERATIONS" \
    --checkpoint_iterations 15000 "$ITERATIONS" \
    --eval

echo ""
echo "Background training complete! Output: $MODEL_PATH"
ls -lh "$MODEL_PATH/"
