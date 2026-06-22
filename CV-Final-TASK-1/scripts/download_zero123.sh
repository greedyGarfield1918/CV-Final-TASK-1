#!/bin/bash
# ============================================================
# Download Zero123 pretrained weights
# Required by Magic123 for the 3D prior
# ============================================================
set -e

PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
ZERO123_DIR="${ZERO123_DIR:-$PROJECT_ROOT/Magic123/Magic123/pretrained/zero123}"
mkdir -p "$ZERO123_DIR"

TARGET_FILE="$ZERO123_DIR/105000.ckpt"

echo "=============================================="
echo " Downloading Zero123 Weights (105000.ckpt)"
echo " Target: $TARGET_FILE"
echo "=============================================="

# Check if already exists
if [ -f "$TARGET_FILE" ]; then
    SIZE=$(stat -c%s "$TARGET_FILE" 2>/dev/null || stat -f%z "$TARGET_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000000000 ]; then
        echo "Zero123 weights already exist ($SIZE bytes). Skipping."
        exit 0
    else
        echo "Existing file too small ($SIZE bytes). Re-downloading..."
        rm -f "$TARGET_FILE"
    fi
fi

# Try multiple sources
echo ""
echo "Trying download sources..."

# Method 1: HuggingFace mirror (CN-friendly)
echo "[1] HuggingFace mirror..."
wget -c --timeout=60 --tries=3 \
    "https://hf-mirror.com/cvlab/zero123-weights/resolve/main/105000.ckpt" \
    -O "$TARGET_FILE" 2>/dev/null && {
    SIZE=$(stat -c%s "$TARGET_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000000000 ]; then
        echo "Success! ($SIZE bytes)"
        exit 0
    fi
}

# Method 2: HuggingFace original
echo "[2] HuggingFace original..."
wget -c --timeout=60 --tries=3 \
    "https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt" \
    -O "$TARGET_FILE" 2>/dev/null && {
    SIZE=$(stat -c%s "$TARGET_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000000000 ]; then
        echo "Success! ($SIZE bytes)"
        exit 0
    fi
}

# Method 3: ModelScope (requires Python)
echo "[3] ModelScope (via Python)..."
python -c "
from modelscope import snapshot_download
import glob, shutil, os
model_dir = snapshot_download('iic/zero123_weights', cache_dir='/tmp/zero123_ms', revision='master')
ckpt_files = glob.glob(f'{model_dir}/**/*.ckpt', recursive=True)
if ckpt_files:
    shutil.copy2(ckpt_files[0], '$TARGET_FILE')
    print(f'Success: {os.path.getsize(ckpt_files[0])} bytes')
else:
    print('No .ckpt found in ModelScope download')
" 2>/dev/null && {
    if [ -f "$TARGET_FILE" ]; then
        echo "Success via ModelScope!"
        exit 0
    fi
}

echo ""
echo "ERROR: All download methods failed."
echo "Please manually download Zero123 weights (105000.ckpt) from:"
echo "  https://huggingface.co/cvlab/zero123-weights"
echo "And place it at: $TARGET_FILE"
exit 1
