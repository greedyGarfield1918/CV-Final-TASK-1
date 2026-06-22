# CV-Final TASK-1: 2DGS + AIGC Multi-Source Asset Generation & Real Scene Fusion

**Full pipeline for multi-source 3D asset generation, background reconstruction, and unified scene fusion rendering.**

This project implements a complete "3D visual chain" that combines:
- **Multi-view 3D reconstruction** (COLMAP + 2D Gaussian Splatting)
- **Text-to-3D generation** (Threestudio with Stable Diffusion)
- **Image-to-3D generation** (Magic123 with Zero123 prior)
- **Background scene reconstruction** (2DGS on Mip-NeRF 360 dataset)
- **Unified scene fusion & rendering** (Gaussian merging + hybrid nvdiffrast)

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Framework Repositories](#framework-repositories)
5. [Data Preparation](#data-preparation)
6. [Pipeline Execution](#pipeline-execution)
7. [Fusion Approaches](#fusion-approaches)
8. [Technical Report Guidance](#technical-report-guidance)
9. [Troubleshooting](#troubleshooting)
10. [Citation & References](#citation--references)

---

## Directory Structure

```
CV-Final-TASK-1/
├── README.md                          # This file
├── requirements/
│   ├── requirements_2dgs.txt          # 2DGS conda environment
│   ├── requirements_threestudio.txt   # Threestudio conda environment
│   └── requirements_magic123.txt      # Magic123 conda environment
├── configs/
│   ├── experiment_config.yaml         # All hyperparameters & settings
│   └── 2dgs_config.json              # 2DGS training config template
├── scripts/
│   ├── env_setup.sh                   # Automated 3-env conda setup
│   ├── verify_env.sh                  # Environment verification
│   ├── run_all.sh                     # Master pipeline runner
│   ├── run_pipeline.py                # Python pipeline orchestrator
│   ├── 01_colmap_sfm.sh               # Step 1: COLMAP SfM
│   ├── 02_train_2dgs_object_a.sh      # Step 2: 2DGS Object A
│   ├── 03_train_threestudio_object_b.sh # Step 3: Threestudio Object B
│   ├── 04_train_magic123_object_c.sh  # Step 4: Magic123 Object C
│   ├── 05_train_2dgs_background.sh    # Step 5: 2DGS Background
│   ├── 06_scene_fusion.py             # Step 6: Scene Fusion (PLY merge)
│   ├── 07_render_video.py             # Step 7: Multi-view rendering
│   ├── download_garden.sh             # Download Mip-NeRF 360 garden scene
│   ├── download_zero123.sh            # Download Zero123 weights
│   ├── download_zero123_modelscope.py # ModelScope Zero123 downloader
│   └── utils/
│       ├── export_mesh.py             # Marching cubes mesh export
│       ├── downsample_colmap.py       # COLMAP point cloud downsampler
│       ├── check_scales.py            # Gaussian model scale inspector
│       └── fix_object_colors.py       # PLY color tinting tool
├── fusion/
│   ├── fusion_final.py                # Standalone hybrid fusion renderer
│   ├── fusion_render.py               # Full hybrid renderer (2DGS + nvdiffrast)
│   └── generate_charts.py             # Report chart generator
├── data/          (to be populated by user)
│   ├── Cat_RGB/                       # Multi-view images for Object A
│   ├── cat_single/                    # Single image for Object C
│   └── garden/                        # Mip-NeRF 360 garden scene
└── output/        (generated at runtime)
    ├── colmap_cat/                    # COLMAP SfM outputs
    ├── object_a_cat/                  # 2DGS reconstruction (Object A)
    ├── object_b_text23d/              # Threestudio output (Object B)
    ├── object_c_single23d/            # Magic123 output (Object C)
    ├── background_garden/             # 2DGS background reconstruction
    └── scene_fused/                   # Fused scene + rendered video
```

---

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with ≥12 GB VRAM (24 GB recommended, e.g., RTX 4090)
- **RAM**: ≥32 GB system memory
- **Storage**: ≥100 GB free disk space (models + datasets + outputs)
- **OS**: Ubuntu 20.04 / 22.04 (Windows via WSL2 may work with adjustments)

### Software
- **CUDA**: 11.8 (required for PyTorch 2.0.0+cu118)
- **NVIDIA Driver**: ≥525
- **Conda**: Miniconda3 or Anaconda
- **COLMAP**: ≥3.8 (for SfM)
- **FFmpeg**: for video encoding

### Base System Packages
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 colmap
```

---

## Environment Setup

### Quick Setup (Recommended)

```bash
# Clone this repository
git clone <repo-url> CV-Final-TASK-1
cd CV-Final-TASK-1

# Run automated setup (creates 3 conda environments)
bash scripts/env_setup.sh

# For China mainland users (uses Tsinghua mirrors):
bash scripts/env_setup.sh --china

# With custom paths:
bash scripts/env_setup.sh --prefix /mnt/data/conda_envs --cache /mnt/data/cache
```

This creates three isolated conda environments:
| Environment | Python | Purpose |
|---|---|---|
| `2dgs` | 3.8.18 | COLMAP SfM, 2DGS training, rendering |
| `threestudio` | 3.10 | Text-to-3D generation (Object B) |
| `magic123` | 3.10 | Image-to-3D generation (Object C) |

### Manual Setup

If the automated script fails, install each environment manually:

```bash
# 1. 2DGS Environment
conda create -n 2dgs python=3.8.18 -y
conda activate 2dgs
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/requirements_2dgs.txt
# Build CUDA extensions (see Framework Repositories section)

# 2. Threestudio Environment
conda create -n threestudio python=3.10 -y
conda activate threestudio
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/requirements_threestudio.txt

# 3. Magic123 Environment
conda create -n magic123 python=3.10 -y
conda activate magic123
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/requirements_magic123.txt
# Build CUDA extensions (raymarching, shencoder, freqencoder, gridencoder)
```

### Verify Installation

```bash
bash scripts/verify_env.sh
```

---

## Framework Repositories

Clone the following repositories into the project root:

```bash
cd CV-Final-TASK-1

# 1. 2D Gaussian Splatting (SIGGRAPH 2024)
git clone https://github.com/hbb1/2d-gaussian-splatting.git
# Build CUDA extensions:
cd 2d-gaussian-splatting/2d-gaussian-splatting
conda activate 2dgs
pip install -e submodules/diff-surfel-rasterization
pip install -e submodules/simple-knn
cd ../..

# 2. Threestudio
git clone https://github.com/threestudio-project/threestudio.git
# Note: threestudio may require specific patches. See Troubleshooting.

# 3. Magic123
git clone https://github.com/guochengqian/Magic123.git
# Build CUDA extensions:
cd Magic123/Magic123
conda activate magic123
pip install -e ./raymarching
pip install -e ./shencoder
pip install -e ./freqencoder
pip install -e ./gridencoder
cd ../..

# 4. Download Zero123 pretrained weights
bash scripts/download_zero123.sh
# Or via Python:
python scripts/download_zero123_modelscope.py
```

### Alternative: pre-built Docker

If building CUDA extensions proves difficult, consider using the official Docker images:
```bash
# 2DGS
docker pull hbb1/2d-gaussian-splatting

# Threestudio
docker pull threestudio/threestudio:latest
```

---

## Data Preparation

### Object A: Multi-view Images (Cat)

```
data/Cat_RGB/
├── images/
│   ├── im_0001.JPG
│   ├── im_0002.JPG
│   └── ...          # 50-100+ images covering 360° of the object
```

**Guidelines for capture:**
- 50-100 images covering full 360° around the object
- Overlap ≥60% between adjacent frames
- Consistent lighting, avoid motion blur
- Object should occupy ~60-80% of frame
- Shoot at highest resolution your phone supports (will be downsampled by COLMAP)

### Object C: Single Image

```
data/cat_single/
└── cat_rgba.png      # Background-removed RGBA image, max dimension 512px
```

Preprocess with:
```bash
conda activate magic123
python -m rembg i data/cat_single/input.jpg data/cat_single/cat_rgba.png
```

### Background: Mip-NeRF 360 Garden Scene

```bash
# Method 1: nerfbaselines (recommended)
pip install nerfbaselines
bash scripts/download_garden.sh 1

# Method 2: Manual download from Google Drive
bash scripts/download_garden.sh 2
# Then download manually from: https://drive.google.com/drive/folders/1Yy2NhPqmsmg3n8KqEYO4gLmgA-YZBhmw
```

Expected structure:
```
data/garden/
├── images_4/           # 4× downsampled images for training
│   ├── DSCF5944.png
│   └── ...
├── images/             # Full resolution images (optional)
└── sparse/             # Pre-computed COLMAP (if available)
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

---

## Pipeline Execution

### Full Pipeline (All Steps)

```bash
# Run everything sequentially
bash scripts/run_all.sh

# Or use the Python orchestrator
python scripts/run_pipeline.py
```

### Step-by-Step Execution

```bash
# Step 1: COLMAP SfM on Object A images
bash scripts/01_colmap_sfm.sh

# Step 2: 2DGS multi-view reconstruction (Object A)
bash scripts/02_train_2dgs_object_a.sh

# Step 3: Threestudio text-to-3D (Object B)
# Customize prompt via env var:
PROMPT="A cute cartoon cat, 3D asset" bash scripts/03_train_threestudio_object_b.sh

# Step 4: Magic123 image-to-3D (Object C)
bash scripts/04_train_magic123_object_c.sh

# Step 5: 2DGS background scene reconstruction
bash scripts/05_train_2dgs_background.sh

# Step 6: Scene fusion (PLY merge)
conda activate 2dgs
python scripts/06_scene_fusion.py

# Step 7: Render multi-view flythrough video
conda activate 2dgs
python scripts/07_render_video.py
```

### Running Individual Steps

```bash
# Run only step 3 (Threestudio)
bash scripts/run_all.sh --step 3

# Skip training, only run fusion + rendering
bash scripts/run_all.sh --skip_train
```

### Configuration

All paths and hyperparameters are configurable via environment variables:

```bash
# Path overrides
export PROJECT_ROOT="/path/to/CV-Final-TASK-1"
export CONDA_ENV_PREFIX="/path/to/conda_envs"
export SOURCE_PATH="/path/to/input/images"
export MODEL_PATH="/path/to/output"

# Training overrides
export ITERATIONS=15000        # Reduce for quick testing
export MAX_STEPS=2500          # Threestudio quick test

# Framework locations
export _2DGS_DIR="$PROJECT_ROOT/2d-gaussian-splatting/2d-gaussian-splatting"
export _THREESTUDIO_DIR="$PROJECT_ROOT/threestudio/threestudio"
export _MAGIC123_DIR="$PROJECT_ROOT/Magic123/Magic123"
```

### Approximate Timings (RTX 4090, 24 GB)

| Step | Operation | Time | GPU Memory |
|---|---|---|---|
| 1 | COLMAP SfM | 5-15 min | ~4 GB |
| 2 | 2DGS Object A (30K iter) | 40-60 min | 8-12 GB |
| 3 | Threestudio Object B (10K steps) | 1.5-2 hrs | 16-24 GB |
| 4 | Magic123 Object C (5K+3K steps) | 1-1.5 hrs | 12-16 GB |
| 5 | 2DGS Background (30K iter) | 1-1.5 hrs | 12-16 GB |
| 6 | Scene Fusion | 1-2 min | 2-4 GB |
| 7 | Render Video (120 views) | 5-10 min | 4-6 GB |
| **Total** | | **~5-7 hrs** | |

---

## Fusion Approaches

Two fusion approaches are provided, corresponding to the report discussion:

### Approach 1: PLY Merge (Gaussian-to-Gaussian)

**File**: `scripts/06_scene_fusion.py`

All assets are converted to a unified Gaussian representation:
- Object A (2DGS): Used directly as Gaussians
- Object B (Threestudio mesh): Sampled → dense point cloud → Gaussian initialization
- Object C (Magic123 mesh): Sampled → dense point cloud → Gaussian initialization
- Background (2DGS): Used directly

Meshes are sampled via `trimesh.sample.sample_surface` and converted to Gaussian
primitives with identity rotations and uniform small scales. The merged `.ply` file
can be rendered directly by the 2DGS rasterizer.

### Approach 2: Hybrid Rendering (Gaussians + Mesh)

**File**: `fusion/fusion_render.py` and `fusion/fusion_final.py`

2DGS Gaussians and AIGC meshes are rendered separately and alpha-composited:
- Background + Object A: Rendered via 2DGS CUDA rasterizer
- Objects B + C: Rendered as triangle meshes via nvdiffrast with diffuse shading
- All layers are depth-sorted and alpha-blended

This avoids information loss from mesh-to-Gaussian conversion and preserves
the mesh structure of AIGC outputs.

### When to Use Each

| Criterion | PLY Merge | Hybrid |
|---|---|---|
| Visual quality | Good (slight loss from sampling) | Best (native representations) |
| Rendering speed | Fast (single rasterizer) | Moderate (two renderers) |
| Implementation complexity | Low | Medium |
| Report relevance | Demonstrates unified representation | Demonstrates multi-representation fusion |

---

## Technical Report Guidance

The `configs/experiment_config.yaml` file contains all hyperparameters ready for
inclusion in tables. Below are key discussion points for the report:

### 1. Comparison of Three Generation Methods

| Method | Geometry Accuracy | Texture Detail | Compute Time | File Size |
|---|---|---|---|---|
| 2DGS (multi-view) | High (photometric loss) | High (from images) | ~1 hr | ~200 MB (.ply) |
| Threestudio (text) | Medium (SDS prior) | Medium (from SD) | ~2 hrs | ~5 MB (.obj) |
| Magic123 (single-image) | Medium-High (2D+3D priors) | Good (from image+SD) | ~1.5 hrs | ~8 MB (.obj) |

**Key insights for the report:**
- Multi-view reconstruction achieves the highest fidelity but requires the most input data
- Text-to-3D has the lowest input barrier but the lowest geometric accuracy
- Single-image-to-3D offers a good balance; quality heavily depends on the input image
- AIGC methods produce meshes that need conversion for Gaussian-based pipelines

### 2. Unified Representation for Merged Rendering

The report should detail the mesh→Gaussian conversion pipeline:
1. **Mesh Sampling**: `trimesh.sample.sample_surface` to generate dense point cloud
2. **Gaussian Initialization**: Points → 2D Gaussians with:
   - Position = sampled point
   - Opacity = 0.6 (tunable)
   - Scale = 0.015 (small isotropic)
   - Rotation = identity quaternion
   - SH coefficients = sampled color (DC only, higher bands = 0)
3. **Scene Merging**: All Gaussians concatenated into single `.ply` with 2DGS format

Alternative hybrid approach (no conversion needed):
- Render Gaussians via 2DGS rasterizer
- Render meshes via nvdiffrast
- Alpha-composite in depth order

### 3. Generated Charts

The `fusion/generate_charts.py` script produces:
1. Training loss curves (2DGS)
2. Loss component breakdown (L1, SSIM, distortion, normal)
3. Gaussian point count growth
4. Convergence speed comparison
5. Method comparison (PSNR vs training time)

```bash
conda activate 2dgs
python fusion/generate_charts.py --data training_data.json --out output/charts
```

---

## Troubleshooting

### CUDA Extension Build Failures

```bash
# Ensure CUDA toolkit matches PyTorch's CUDA version
nvcc --version          # Should show 11.8
python -c "import torch; print(torch.version.cuda)"  # Should print 11.8

# Set CUDA architectures for your GPU
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"  # Covers most modern GPUs

# For RTX 4090 specifically:
export TORCH_CUDA_ARCH_LIST="8.9"
```

### Out of Memory (OOM)

```bash
# Reduce iterations for testing
export ITERATIONS=7000   # 2DGS quick test
export MAX_STEPS=2500    # Threestudio quick test

# For Threestudio on 12 GB GPU: use smaller resolution
# Add to launch.py: data.width=32 data.height=32
```

### COLMAP Issues

```bash
# If COLMAP fails on large images:
# Reduce max_image_size
colmap feature_extractor --SiftExtraction.max_image_size 1600 ...

# If matching fails:
# Use sequential_matcher for video sequences
colmap sequential_matcher ...

# If reconstruction fails (0 images registered):
# Check image overlap, reduce max_image_size, increase max_num_features
```

### Magic123 Zero123 Download

If the auto-download fails (common in China mainland):
```bash
# Option 1: ModelScope
python scripts/download_zero123_modelscope.py

# Option 2: HF mirror
bash scripts/download_zero123.sh

# Option 3: Manual
# Download from: https://huggingface.co/cvlab/zero123-weights
# Place at: Magic123/Magic123/pretrained/zero123/105000.ckpt
```

### Threestudio Dependency Issues

```bash
# Common fix for xformers version
pip install xformers==0.0.20 --index-url https://download.pytorch.org/whl/cu118

# If nerfacc install fails:
pip install ninja
pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2

# If envlight install fails:
pip install git+https://github.com/ashawkey/envlight.git
```

### China Mainland Mirror Setup

All scripts auto-detect connectivity to pypi.org. To force mirrors:
```bash
bash scripts/env_setup.sh --china

# Or manually:
export HF_ENDPOINT=https://hf-mirror.com
export PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
```

---

## Citation & References

### Core Frameworks
- **2DGS**: Huang et al., "2D Gaussian Splatting for Geometrically Accurate Radiance Fields", SIGGRAPH 2024
  - Repo: https://github.com/hbb1/2d-gaussian-splatting
- **Threestudio**: Guo et al., "Threestudio: A unified framework for 3D content generation", 2023
  - Repo: https://github.com/threestudio-project/threestudio
- **Magic123**: Qian et al., "Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors", ICLR 2024
  - Repo: https://github.com/guochengqian/Magic123

### Key Techniques
- **COLMAP**: Schönberger et al., "Structure-from-Motion Revisited", CVPR 2016
- **DreamFusion**: Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion", ICLR 2023
- **Zero123**: Liu et al., "Zero-1-to-3: Zero-shot One Image to 3D Object", ICCV 2023
- **Mip-NeRF 360**: Barron et al., "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields", CVPR 2022

### Dataset
- Mip-NeRF 360 Dataset: https://jonbarron.info/mipnerf360/

---

## Model Weights

Trained model weights are hosted separately from the code repository:

| Asset | Description | Size |
|---|---|---|
| Object A (2DGS Cat) | Multi-view reconstruction, 30K iters | ~106 MB |
| Object B (Threestudio) | Text-to-3D mesh | ~8 MB |
| Object C (Magic123) | Image-to-3D mesh | ~5 MB |
| Background (2DGS Garden) | Mip-NeRF 360 scene, 30K iters | ~536 MB |
| Fused Flythrough Video | 180-frame spiral render | ~108 MB |

**Cloud Download**: [Google Drive](https://drive.google.com/drive/folders/1HkL489TVlCqVoQG_jXXswmkQ7_GN_Ttq)

The weights folder (`../CV-Final-TASK-1-Weights/`) contains these files with detailed usage instructions.

---

## License

This project is for academic/research purposes. Each framework repository has its own license.
Please comply with the licenses of:
- 2D Gaussian Splatting
- Threestudio
- Magic123
- COLMAP (BSD license)
- Mip-NeRF 360 dataset terms
