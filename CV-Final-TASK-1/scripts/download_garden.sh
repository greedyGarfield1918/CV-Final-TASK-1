#!/bin/bash
# ============================================================
# Download Mip-NeRF 360 garden scene
# ============================================================
set -e

PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
GARDEN_DIR="${GARDEN_DIR:-$PROJECT_ROOT/data/garden}"
mkdir -p "$GARDEN_DIR"

echo "=============================================="
echo " Downloading Mip-NeRF 360 Garden Scene"
echo " Target: $GARDEN_DIR"
echo "=============================================="

echo ""
echo "Choose download method:"
echo "  1) nerfbaselines (recommended, auto-downloads + extracts)"
echo "  2) Google Drive (manual download link)"
echo "  3) Direct wget from mirror"

METHOD="${1:-1}"

if [ "$METHOD" == "1" ]; then
    echo ""
    echo "[Method 1] Using nerfbaselines..."
    pip install nerfbaselines
    nerfbaselines download-dataset -o "$GARDEN_DIR" external://mipnerf360/garden

elif [ "$METHOD" == "2" ]; then
    echo ""
    echo "[Method 2] Google Drive URL — please download manually if wget fails"
    echo "Mirror: https://drive.google.com/drive/folders/1Yy2NhPqmsmg3n8KqEYO4gLmgA-YZBhmw"
    # Try gdown
    pip install gdown 2>/dev/null || true
    echo "Manual download required. See README for details."

elif [ "$METHOD" == "3" ]; then
    echo ""
    echo "[Method 3] Direct download..."
    # This URL may need updating
    wget -c "https://storage.googleapis.com/gresearch/refraw360/garden.zip" \
        -O "$GARDEN_DIR/garden.zip" 2>/dev/null || {
        echo "Direct download failed. Try method 1 or 2."
        exit 1
    }
    unzip -o "$GARDEN_DIR/garden.zip" -d "$GARDEN_DIR"
fi

# Verify
echo ""
echo "Downloaded files:"
ls -lh "$GARDEN_DIR" 2>/dev/null || echo "  (empty or download failed)"
echo ""
echo "Done! Garden scene at: $GARDEN_DIR"
echo ""
echo "If the download failed, you can:"
echo "  1. Use nerfbaselines: pip install nerfbaselines && nerfbaselines download-dataset ..."
echo "  2. Download manually from: https://drive.google.com/drive/folders/1Yy2NhPqmsmg3n8KqEYO4gLmgA-YZBhmw"
echo "  3. Use any other Mip-NeRF 360 compatible scene"
