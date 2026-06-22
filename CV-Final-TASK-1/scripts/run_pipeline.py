#!/usr/bin/env python3
"""
Python pipeline orchestrator for Project-1.

Runs the full pipeline: COLMAP → 2DGS Object A → Threestudio Object B →
Magic123 Object C → 2DGS Background → Fusion → Render

Usage:
    python scripts/run_pipeline.py                    # Full pipeline
    python scripts/run_pipeline.py --step 1           # Specific step only
    python scripts/run_pipeline.py --skip_train        # Only fusion+render
"""
import subprocess as sp
import os
import sys
import argparse
from pathlib import Path


# ---- Configuration (all paths configurable) ----
PROJECT_ROOT = Path(__file__).parent.parent
CONDA_ENV_PREFIX = os.environ.get('CONDA_ENV_PREFIX', str(Path.home() / 'conda_envs'))

# Python interpreters per environment
PY_2DGS = os.environ.get('PY_2DGS', f'{CONDA_ENV_PREFIX}/2dgs/bin/python')
PY_THREESTUDIO = os.environ.get('PY_THREESTUDIO', f'{CONDA_ENV_PREFIX}/threestudio/bin/python')
PY_MAGIC123 = os.environ.get('PY_MAGIC123', f'{CONDA_ENV_PREFIX}/magic123/bin/python')

# Framework directories
_2DGS_DIR = os.environ.get('_2DGS_DIR', str(PROJECT_ROOT / '2d-gaussian-splatting/2d-gaussian-splatting'))
_THREESTUDIO_DIR = os.environ.get('_THREESTUDIO_DIR', str(PROJECT_ROOT / 'threestudio/threestudio'))
_MAGIC123_DIR = os.environ.get('_MAGIC123_DIR', str(PROJECT_ROOT / 'Magic123/Magic123'))

# Mirror for China mainland
USE_CHINA_MIRROR = os.environ.get('USE_CHINA_MIRROR', 'false').lower() == 'true'
PIP_MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/simple' if USE_CHINA_MIRROR else ''


def run(cmd, timeout=None, env=None):
    """Run a shell command and print results."""
    print(f'  RUN: {cmd[:120]}')
    r = sp.run(cmd, shell=True, capture_output=True, text=True,
               timeout=timeout, env=env)
    if r.returncode != 0:
        print(f'  FAIL:\n{r.stderr[-300:]}')
    else:
        out = r.stdout.strip()
        if out:
            print(f'  OK: {out[-200:]}')
        else:
            print('  OK')
    return r


def step_colmap(args):
    """Step 1: COLMAP SfM."""
    print("\n=== [1] COLMAP SfM ===")
    src = os.environ.get('COLMAP_SRC', str(PROJECT_ROOT / 'data/Cat_RGB'))
    out = os.environ.get('COLMAP_OUT', str(PROJECT_ROOT / 'output/colmap_cat'))

    try:
        import pycolmap
        import sqlite3
    except ImportError:
        print("pycolmap not found. Using shell script instead...")
        run(f'bash {PROJECT_ROOT}/scripts/01_colmap_sfm.sh')
        return

    os.makedirs(out, exist_ok=True)
    db = str(Path(out) / 'database.db')

    opts = pycolmap.SiftExtractionOptions()
    opts.max_image_size = 2000
    opts.use_gpu = True

    n_images = len([f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f'  Feature extraction ({n_images} images)...')
    pycolmap.extract_features(db, src, camera_mode=pycolmap.CameraMode.SINGLE, sift_options=opts)

    c = sqlite3.connect(db)
    n_feat = c.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0]
    print(f'  Features extracted: {n_feat}')

    print('  Matching...')
    pycolmap.match_exhaustive(db)

    print('  Sparse reconstruction...')
    maps = pycolmap.incremental_mapping(db, src, os.path.dirname(out))
    if maps:
        out_dir = os.path.join(out, 'sparse', '0')
        os.makedirs(out_dir, exist_ok=True)
        maps[0].write_binary(out_dir)
        print(f'  Done: {maps[0].num_images()} images, {maps[0].num_points3D()} points')

        # Copy to source for 2DGS
        sparse_dest = os.path.join(src, 'sparse', '0')
        os.makedirs(sparse_dest, exist_ok=True)
        import shutil
        for f in os.listdir(out_dir):
            src_f = os.path.join(out_dir, f)
            dst_f = os.path.join(sparse_dest, f)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, dst_f)
    else:
        print('  FAILED: no reconstructions')
        sys.exit(1)


def step_2dgs_object_a(args):
    """Step 2: 2DGS training on Object A."""
    print("\n=== [2] 2DGS Object A training ===")
    run(f'bash {PROJECT_ROOT}/scripts/02_train_2dgs_object_a.sh', timeout=7200)


def step_threestudio_object_b(args):
    """Step 3: Threestudio text-to-3D."""
    print("\n=== [3] Threestudio text-to-3D ===")
    run(f'bash {PROJECT_ROOT}/scripts/03_train_threestudio_object_b.sh', timeout=14400)


def step_magic123_object_c(args):
    """Step 4: Magic123 image-to-3D."""
    print("\n=== [4] Magic123 image-to-3D ===")
    run(f'bash {PROJECT_ROOT}/scripts/04_train_magic123_object_c.sh', timeout=14400)


def step_2dgs_background(args):
    """Step 5: 2DGS training on background."""
    print("\n=== [5] 2DGS Background training ===")
    run(f'bash {PROJECT_ROOT}/scripts/05_train_2dgs_background.sh', timeout=7200)


def step_fusion(args):
    """Step 6: Scene fusion."""
    print("\n=== [6] Scene Fusion ===")
    run(f'{PY_2DGS} {PROJECT_ROOT}/scripts/06_scene_fusion.py')


def step_render(args):
    """Step 7: Render video."""
    print("\n=== [7] Render Video ===")
    run(f'{PY_2DGS} {PROJECT_ROOT}/scripts/07_render_video.py')


STEPS = {
    '1': ('COLMAP SfM', step_colmap),
    '2': ('2DGS Object A', step_2dgs_object_a),
    '3': ('Threestudio Object B', step_threestudio_object_b),
    '4': ('Magic123 Object C', step_magic123_object_c),
    '5': ('2DGS Background', step_2dgs_background),
    '6': ('Scene Fusion', step_fusion),
    '7': ('Render Video', step_render),
}


def main():
    parser = argparse.ArgumentParser(description="Project-1 Pipeline Orchestrator")
    parser.add_argument('--step', type=str, default=None,
                        help='Run specific step (1-7)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training steps (2-5)')
    parser.add_argument('--list', action='store_true',
                        help='List all steps')
    args = parser.parse_args()

    if args.list:
        print("Available steps:")
        for k, (name, _) in STEPS.items():
            print(f"  {k}: {name}")
        return

    skip_train = args.skip_train or os.environ.get('SKIP_TRAIN', 'false').lower() == 'true'

    print("=" * 60)
    print(" Project-1: Full Pipeline")
    print(f" Project root: {PROJECT_ROOT}")
    print(f" Conda prefix: {CONDA_ENV_PREFIX}")
    print("=" * 60)

    if args.step:
        if args.step in STEPS:
            name, func = STEPS[args.step]
            print(f"\nRunning step {args.step}: {name}")
            func(args)
        else:
            print(f"Unknown step: {args.step}. Valid: {list(STEPS.keys())}")
            sys.exit(1)
    else:
        for step_key in sorted(STEPS.keys(), key=int):
            name, func = STEPS[step_key]
            if skip_train and step_key in ('2', '3', '4', '5'):
                print(f"\n>>> Step {step_key}: SKIPPED (--skip_train)")
                continue
            func(args)

    print("\n" + "=" * 60)
    print(" Pipeline Complete!")
    print("=" * 60)
    print(f"\nCheck {PROJECT_ROOT}/output/ for results.")


if __name__ == '__main__':
    main()
