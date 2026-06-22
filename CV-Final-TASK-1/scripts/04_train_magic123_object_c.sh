#!/bin/bash
# ============================================================
# Step 4: Magic123 Single-Image-to-3D (Object C)
# Produces: output/object_c_single23d/ (Mesh + renderings)
# ============================================================
set -e

# ---- Configurable paths ----
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/conda_envs}"
_MAGIC123_DIR="${_MAGIC123_DIR:-$PROJECT_ROOT/Magic123/Magic123}"

# ---- Configurable input ----
INPUT_IMAGE="${INPUT_IMAGE:-$PROJECT_ROOT/data/cat_single/cat_rgba.png}"
TEXT_PROMPT="${TEXT_PROMPT:-A high-resolution DSLR image of a cat}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/object_c_single23d}"

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX/magic123"

echo "=============================================="
echo " Step 4: Single-Image-to-3D (Object C)"
echo " Input: $INPUT_IMAGE"
echo " GPU Memory: ~12-16 GB"
echo " Time: ~1-1.5 hrs (2 stages, RTX 4090)"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# Step 0: Preprocess - remove background + resize (if input is not already RGBA)
echo "[0/3] Preprocessing input image..."
INPUT_BASENAME=$(basename "$INPUT_IMAGE")
cp "$INPUT_IMAGE" "$OUTPUT_DIR/"

if [[ "$INPUT_IMAGE" != *rgba* ]] && [[ "$INPUT_IMAGE" != *mask* ]]; then
    cd "$OUTPUT_DIR"
    python "$_MAGIC123_DIR/preprocess_image.py" "$INPUT_BASENAME" 2>/dev/null || {
        echo "  WARNING: preprocess_image.py failed. Using rembg directly..."
        python -c "
from rembg import remove
from PIL import Image
import numpy as np
img = Image.open('$INPUT_BASENAME')
img = img.convert('RGB')
out = remove(img)
out.save('rgba.png')
print('  RGBA saved: rgba.png')
"
    }
    RGBA_IMAGE="$OUTPUT_DIR/rgba.png"
    cd - > /dev/null
else
    RGBA_IMAGE="$INPUT_IMAGE"
fi

if [ ! -f "$RGBA_IMAGE" ]; then
    echo "ERROR: Background removal failed. Please provide an RGBA image."
    echo "You can preprocess manually: python -m rembg i input.jpg output_rgba.png"
    exit 1
fi
echo "  RGBA image: $RGBA_IMAGE"

cd "$_MAGIC123_DIR"

# Step 1: Coarse stage (2D prior + 3D prior)
echo "[1/3] Stage 1: Coarse reconstruction (SD + Zero123 priors)..."
python main.py -O \
    --text "$TEXT_PROMPT" \
    --sd_version 1.5 \
    --image "$RGBA_IMAGE" \
    --workspace "$OUTPUT_DIR/stage1_coarse" \
    --optim adam \
    --iters 5000 \
    --guidance SD zero123 \
    --lambda_guidance 1.0 40 \
    --guidance_scale 100 5 \
    --latent_iter_ratio 0 \
    --normal_iter_ratio 0.2 \
    --fp16 \
    --cuda_ray

# Step 2: Refine stage (DMTet mesh refinement)
echo "[2/3] Stage 2: DMTet mesh refinement..."
python main.py -O \
    --text "$TEXT_PROMPT" \
    --sd_version 1.5 \
    --image "$RGBA_IMAGE" \
    --workspace "$OUTPUT_DIR/stage2_refine" \
    --optim adam \
    --iters 3000 \
    --guidance SD zero123 \
    --lambda_guidance 1.0 40 \
    --guidance_scale 100 5 \
    --dmtet \
    --init_with "$OUTPUT_DIR/stage1_coarse/checkpoints/df.pth" \
    --fp16 \
    --cuda_ray

# Step 3: Export mesh
echo "[3/3] Exporting final mesh..."
python meshutils.py \
    --mesh_path "$OUTPUT_DIR/stage2_refine/mesh" \
    --output "$OUTPUT_DIR/final_mesh.obj" 2>/dev/null || {
    echo "  WARNING: meshutils failed. Looking for mesh in checkpoint..."
    # Try to find dmtet mesh in checkpoints
    ls "$OUTPUT_DIR/stage2_refine/mesh/" 2>/dev/null || \
        echo "  No mesh found — you may need to export manually."
}

echo ""
echo "Object C generation complete!"
ls -lh "$OUTPUT_DIR/"
