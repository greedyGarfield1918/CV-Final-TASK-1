#!/usr/bin/env python3
"""
Export mesh from Magic123/Threestudio checkpoint using marching cubes.

Usage:
    python scripts/utils/export_mesh.py --ckpt <path> --out <output.obj>
    python scripts/utils/export_mesh.py --ckpt last.ckpt --config config.yaml --out mesh.obj
"""
import sys
import os
import argparse
import numpy as np
import torch


def export_from_density(ckpt_path, config_path, out_path, resolution=128, threshold_ratio=0.3):
    """Extract mesh from implicit volume checkpoint via marching cubes."""
    # Add threestudio to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    th_path = os.path.join(project_root, 'threestudio', 'threestudio')
    if os.path.isdir(th_path):
        sys.path.insert(0, th_path)

    from threestudio.models.geometry.implicit_volume import ImplicitVolume
    from omegaconf import OmegaConf

    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Find geometry keys
    geo_keys = [k for k in ckpt['state_dict'].keys() if k.startswith('geometry.')]
    print(f'Found {len(geo_keys)} geometry keys')

    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        # Build minimal config
        cfg = OmegaConf.create({
            'geometry': {
                'radius': 1.0,
                'isosurface_method': 'mt',
                'isosurface_resolution': 128,
                'density_bias': 'blob_magic3d',
                'density_activation': 'softplus',
                'pos_encoding_config': {
                    'otype': 'ProgressiveBandFrequency',
                    'n_frequencies': 8,
                },
            }
        })

    geo = ImplicitVolume(cfg.system.geometry if hasattr(cfg, 'system') else cfg.geometry)
    state_dict = {k.replace('geometry.', ''): v for k, v in ckpt['state_dict'].items()
                  if k.startswith('geometry.')}
    geo.load_state_dict(state_dict, strict=False)
    geo.eval()

    print(f'Performing marching cubes at resolution {resolution}...')
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, resolution),
        torch.linspace(-1, 1, resolution),
        torch.linspace(-1, 1, resolution),
    ), dim=-1).reshape(-1, 3)

    density = []
    chunk = 65536
    for i in range(0, grid.shape[0], chunk):
        with torch.no_grad():
            out = geo(grid[i:i + chunk])
            d = out['density'].squeeze(-1).cpu()
        density.append(d)
    density = torch.cat(density).reshape(resolution, resolution, resolution).numpy()

    print(f'Density range: {density.min():.3f} to {density.max():.3f}')

    from skimage import measure
    threshold = density.max() * threshold_ratio
    verts, faces, normals, values = measure.marching_cubes(density, level=threshold)

    # Scale to [-1, 1]
    verts = verts / resolution * 2 - 1

    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    mesh.export(out_path)
    print(f'Mesh saved: {len(verts)} vertices, {len(faces)} faces → {out_path}')


def main():
    parser = argparse.ArgumentParser(description="Export mesh from checkpoint")
    parser.add_argument('--ckpt', required=True, help='Path to .ckpt file')
    parser.add_argument('--config', default=None, help='Path to config YAML')
    parser.add_argument('--out', required=True, help='Output .obj path')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Marching cubes resolution')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Density threshold ratio')
    args = parser.parse_args()

    export_from_density(args.ckpt, args.config, args.out,
                        args.resolution, args.threshold)


if __name__ == '__main__':
    main()
