#!/bin/bash
# ============================================================
# Master Pipeline: Run All Steps Sequentially
# ============================================================
# Usage:
#   bash scripts/run_all.sh                    # Run everything
#   bash scripts/run_all.sh --step 1           # Run specific step only
#   bash scripts/run_all.sh --skip_train        # Skip training, only fusion+render
#   bash scripts/run_all.sh --step 6 --skip_train  # Only fusion
#
# Environment variables:
#   PROJECT_ROOT        - Project root directory (default: auto-detect)
#   CONDA_ENV_PREFIX    - Conda env prefix (default: $HOME/conda_envs)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$PROJECT_ROOT"

export PROJECT_ROOT
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/conda_envs}"

# Parse arguments
STEP=""
SKIP_TRAIN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --step) STEP="$2"; shift 2 ;;
        --skip_train) SKIP_TRAIN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo " Project-1: Full Pipeline"
echo " Working dir: $PROJECT_ROOT"
echo " Conda envs:  $CONDA_ENV_PREFIX"
echo "=============================================="

# ---- Step 1: COLMAP SfM ----
if [[ -z "$STEP" || "$STEP" == "1" ]]; then
    echo -e "\n>>> Step 1: COLMAP SfM on input images"
    bash scripts/01_colmap_sfm.sh
fi

# ---- Step 2: 2DGS Object A ----
if [[ -z "$STEP" || "$STEP" == "2" ]]; then
    if $SKIP_TRAIN; then
        echo -e "\n>>> Step 2: SKIPPED (--skip_train)"
    else
        echo -e "\n>>> Step 2: 2DGS Training - Object A"
        bash scripts/02_train_2dgs_object_a.sh
    fi
fi

# ---- Step 3: Threestudio Object B ----
if [[ -z "$STEP" || "$STEP" == "3" ]]; then
    if $SKIP_TRAIN; then
        echo -e "\n>>> Step 3: SKIPPED (--skip_train)"
    else
        echo -e "\n>>> Step 3: Text-to-3D - Object B (Threestudio)"
        bash scripts/03_train_threestudio_object_b.sh
    fi
fi

# ---- Step 4: Magic123 Object C ----
if [[ -z "$STEP" || "$STEP" == "4" ]]; then
    if $SKIP_TRAIN; then
        echo -e "\n>>> Step 4: SKIPPED (--skip_train)"
    else
        echo -e "\n>>> Step 4: Image-to-3D - Object C (Magic123)"
        bash scripts/04_train_magic123_object_c.sh
    fi
fi

# ---- Step 5: 2DGS Background ----
if [[ -z "$STEP" || "$STEP" == "5" ]]; then
    if $SKIP_TRAIN; then
        echo -e "\n>>> Step 5: SKIPPED (--skip_train)"
    else
        echo -e "\n>>> Step 5: 2DGS Training - Background Scene"
        bash scripts/05_train_2dgs_background.sh
    fi
fi

# ---- Step 6: Scene Fusion ----
if [[ -z "$STEP" || "$STEP" == "6" ]]; then
    echo -e "\n>>> Step 6: Scene Fusion & Camera Trajectory"
    source ~/.bashrc 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_PREFIX/2dgs"
    python scripts/06_scene_fusion.py
fi

# ---- Step 7: Render Video ----
if [[ -z "$STEP" || "$STEP" == "7" ]]; then
    echo -e "\n>>> Step 7: Render Multi-View Video"
    source ~/.bashrc 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_PREFIX/2dgs"
    python scripts/07_render_video.py
fi

echo ""
echo "=============================================="
echo " Pipeline Complete!"
echo "=============================================="
echo ""
echo "Deliverables:"
echo "  Object A (2DGS):       output/object_a_cat/"
echo "  Object B (Text-to-3D): output/object_b_text23d/"
echo "  Object C (Image-to-3D): output/object_c_single23d/"
echo "  Background (Garden):   output/background_garden/"
echo "  Fused Scene:           output/scene_fused/"
echo "  Render Video:          output/scene_fused/"
