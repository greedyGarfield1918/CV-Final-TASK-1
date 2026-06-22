#!/bin/bash
# ============================================================
# Step 2: Train 2DGS on multi-view images (Object A)
# Produces: output/object_a_cat/ (Gaussian splat + checkpoints)
# ============================================================
set -e

# ---- Configurable paths ----
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/conda_envs}"
_2DGS_DIR="${_2DGS_DIR:-$PROJECT_ROOT/2d-gaussian-splatting/2d-gaussian-splatting}"
SOURCE_PATH="${SOURCE_PATH:-$PROJECT_ROOT/data/Cat_RGB}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/output/object_a_cat}"
ITERATIONS="${ITERATIONS:-30000}"

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX/2dgs"

echo "=============================================="
echo " Step 2: 2DGS Training - Object A"
echo " Source: $SOURCE_PATH"
echo " Output: $MODEL_PATH"
echo " GPU Memory: ~8-12 GB"
echo " Time: ~40-60 min (${ITERATIONS} iters, RTX 4090)"
echo "=============================================="

cd "$_2DGS_DIR"

python train.py \
    --source_path "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --images images \
    --resolution 1 \
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
echo "Object A training complete! Output: $MODEL_PATH"
ls -lh "$MODEL_PATH/"
