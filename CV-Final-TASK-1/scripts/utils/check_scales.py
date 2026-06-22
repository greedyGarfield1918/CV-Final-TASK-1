#!/usr/bin/env python3
"""
Check and report scales of 2DGS Gaussian models.

Useful for determining appropriate scale values when placing
objects in the fused scene.

Usage:
    python scripts/utils/check_scales.py --ply <path_to_ply>
"""
import argparse
import numpy as np
from pathlib import Path


def check_ply_scales(ply_path):
    """Analyze a 2DGS PLY file and report scale statistics."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply['vertex']

    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)
    center = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    radius = np.linalg.norm(xyz - center, axis=-1).max()

    n = len(v.data)
    print(f"File: {ply_path}")
    print(f"  Gaussians: {n:,}")
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Extent: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
    print(f"  Radius: {radius:.3f}")

    # Opacity stats
    opacity = np.array(v['opacity'])
    print(f"  Opacity: mean={opacity.mean():.3f}, std={opacity.std():.3f}")

    # Scale stats
    if 'scale_0' in v.data.dtype.names:
        scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1)
        scale_norms = np.linalg.norm(scales, axis=-1)
        print(f"  Scale norm: mean={scale_norms.mean():.4f}, "
              f"min={scale_norms.min():.4f}, max={scale_norms.max():.4f}")

    # Color stats (DC component)
    if 'f_dc_0' in v.data.dtype.names:
        colors = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
        print(f"  Color DC: mean=[{colors[:,0].mean():.3f}, "
              f"{colors[:,1].mean():.3f}, {colors[:,2].mean():.3f}]")

    return {
        'n_gaussians': n,
        'center': center,
        'extent': extent,
        'radius': radius,
    }


def main():
    parser = argparse.ArgumentParser(description="Check 2DGS model scales")
    parser.add_argument('--ply', required=True, help='Path to .ply file')
    args = parser.parse_args()

    check_ply_scales(args.ply)


if __name__ == '__main__':
    main()
