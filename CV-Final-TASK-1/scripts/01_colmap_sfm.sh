#!/bin/bash
# ============================================================
# Step 1: COLMAP SfM on input images
# Produces: data/<dataset>/sparse/ (camera poses + sparse point cloud)
# ============================================================
set -e

# ---- Configurable paths ----
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/Cat_RGB}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/colmap_cat}"

echo "=============================================="
echo " Step 1: COLMAP SfM Reconstruction"
echo " Input : $DATA_DIR"
echo " Output: $OUTPUT_DIR"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

IMAGE_COUNT=$(ls "$DATA_DIR"/*.{jpg,jpeg,JPG,JPEG,png,PNG} 2>/dev/null | wc -l)
echo "Found $IMAGE_COUNT images"

# 1. Feature extraction
echo "[1/4] Extracting features..."
colmap feature_extractor \
    --database_path "$OUTPUT_DIR/database.db" \
    --image_path "$DATA_DIR" \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_image_size 2000 \
    --SiftExtraction.max_num_features 8192

# 2. Feature matching (use vocab_tree for >200 images, exhaustive for fewer)
echo "[2/4] Matching features..."
if [ "$IMAGE_COUNT" -gt 200 ]; then
    colmap vocab_tree_matcher \
        --database_path "$OUTPUT_DIR/database.db" \
        --SiftMatching.use_gpu 1
else
    colmap exhaustive_matcher \
        --database_path "$OUTPUT_DIR/database.db" \
        --SiftMatching.use_gpu 1
fi

# 3. Sparse reconstruction (mapper)
echo "[3/4] Sparse reconstruction..."
mkdir -p "$OUTPUT_DIR/sparse"
colmap mapper \
    --database_path "$OUTPUT_DIR/database.db" \
    --image_path "$DATA_DIR" \
    --output_path "$OUTPUT_DIR/sparse"

# 4. Convert to text format for inspection
echo "[4/4] Converting to text format..."
if [ -d "$OUTPUT_DIR/sparse/0" ]; then
    colmap model_converter \
        --input_path "$OUTPUT_DIR/sparse/0" \
        --output_path "$OUTPUT_DIR/sparse/0/text" \
        --output_type TXT
    echo "  Sparse model: $OUTPUT_DIR/sparse/0/"
else
    echo "  WARNING: No sparse reconstruction found (check COLMAP output)"
    echo "  Available dirs: $(ls "$OUTPUT_DIR/sparse/" 2>/dev/null || echo 'none')"
fi

echo ""
echo "COLMAP done!"
echo ""

# 2DGS expects sparse/0/ inside the source_path
echo "Copying COLMAP data to source directory for 2DGS..."
mkdir -p "$DATA_DIR/sparse"
if [ -d "$OUTPUT_DIR/sparse/0" ]; then
    cp -r "$OUTPUT_DIR/sparse/0" "$DATA_DIR/sparse/0"
    echo "  Copied to: $DATA_DIR/sparse/0/"
else
    echo "  WARNING: No sparse/0 to copy"
fi

echo ""
echo "Ready for 2DGS training."
