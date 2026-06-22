#!/bin/bash
# ============================================================
# Step 3: Threestudio Text-to-3D (Object B)
# Produces: output/object_b_text23d/ (Mesh + renderings)
# ============================================================
set -e

# ---- Configurable paths ----
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/conda_envs}"
_THREESTUDIO_DIR="${_THREESTUDIO_DIR:-$PROJECT_ROOT/threestudio/threestudio}"

# ---- Configurable prompt ----
PROMPT="${PROMPT:-A cute cartoon cat with fluffy orange fur, sitting pose, 3D asset, highly detailed}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-ugly, bad anatomy, blurry, low resolution, distorted}"
MAX_STEPS="${MAX_STEPS:-10000}"
SD_MODEL="${SD_MODEL:-stabilityai/stable-diffusion-2-1-base}"
SEED="${SEED:-42}"

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX/threestudio"

echo "=============================================="
echo " Step 3: Text-to-3D Generation (Object B)"
echo " Prompt: '$PROMPT'"
echo " GPU Memory: ~16-24 GB"
echo " Time: ~1.5-2 hrs (${MAX_STEPS} steps, RTX 4090)"
echo "=============================================="

cd "$_THREESTUDIO_DIR"

python launch.py \
    --config configs/dreamfusion-sd.yaml \
    --train \
    --gpu 0 \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.negative_prompt="$NEGATIVE_PROMPT" \
    system.geometry_convert_from="dmtet" \
    seed="$SEED" \
    data.batch_size=1 \
    data.width=64 \
    data.height=64 \
    system.guidance.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.guidance_scale=100 \
    trainer.max_steps="$MAX_STEPS" \
    trainer.log_every_n_steps=50

echo ""
echo "Object B generation complete!"
ls -lh "$PROJECT_ROOT/output/object_b_text23d/" 2>/dev/null || \
    echo "Check threestudio output directory for results."
