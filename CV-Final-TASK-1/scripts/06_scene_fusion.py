"""
============================================================
Step 6: Scene Fusion — merge all assets into single 2DGS scene
============================================================
Converts all assets to Gaussian point clouds (.ply format),
applies affine transformations, merges into a single scene,
ready for rendering with the 2DGS rasterizer.

Supports two approaches:
  1. PLY merge (default): Convert meshes → dense points → PLY, then merge all
  2. Use pre-computed PLY for 2DGS objects, sample meshes for AIGC objects

Output: output/scene_fused/point_cloud/iteration_30000/point_cloud.ply
============================================================
"""
import os
import sys
import argparse
import json
import shutil
import numpy as np
from pathlib import Path


# ---- Default placement transforms (in scene coordinates) ----
DEFAULT_PLACEMENT = {
    "object_a": {
        "desc": "Real multiview (2DGS)",
        "tx": -2.0, "ty": 0.0, "tz": 0.4,
        "scale": 0.8, "ry": 0,
    },
    "object_b": {
        "desc": "AIGC text-to-3D (Threestudio)",
        "tx": 0.0, "ty": 0.0, "tz": 0.5,
        "scale": 0.5, "ry": 30,
    },
    "object_c": {
        "desc": "AIGC image-to-3D (Magic123)",
        "tx": 2.0, "ty": 0.0, "tz": 0.4,
        "scale": 0.5, "ry": -30,
    },
}


def transform_matrix(tx, ty, tz, scale, ry_deg):
    """Build 4x4 affine matrix (rotate around Y, scale, translate)."""
    rad = np.deg2rad(ry_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) * scale
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def load_ply_raw(filepath):
    """Return dict of numpy arrays from a .ply file."""
    from plyfile import PlyData
    ply = PlyData.read(filepath)
    v = ply['vertex']
    data = {}
    for name in v.data.dtype.names:
        data[name] = np.array(v[name])
    return data, len(v.data)


def write_ply_raw(filepath, data_dict, n_total):
    """Write numpy dict as .ply compatible with 2DGS."""
    from plyfile import PlyData, PlyElement
    dtype = [(k, 'f4') for k in data_dict.keys()]
    arr = np.zeros(n_total, dtype=dtype)
    for k, v in data_dict.items():
        arr[k] = v.astype(np.float32)
    PlyData([PlyElement.describe(arr, 'vertex')]).write(filepath)


def mesh_to_dense_points(mesh_path, n_samples=50000):
    """Sample mesh surface → positions, normals, colors."""
    try:
        import trimesh
    except ImportError:
        raise ImportError("pip install trimesh")

    mesh = trimesh.load(mesh_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if hasattr(g, 'faces')]
        if not geoms:
            # Try to dump the scene as a single mesh
            geoms = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if geoms else mesh

    actual_samples = min(n_samples, len(mesh.faces) * 3)
    if actual_samples == 0:
        print(f"  WARNING: mesh has no faces, using vertices directly")
        pts = np.asarray(mesh.vertices)
        nrm = np.zeros_like(pts)
        cols = np.full((len(pts), 3), 0.7)
        return pts.astype('f4'), nrm.astype('f4'), cols.astype('f4')

    pts, fid = trimesh.sample.sample_surface(mesh, actual_samples)
    nrm = mesh.face_normals[fid]

    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        cols = mesh.visual.vertex_colors[mesh.faces[fid]].mean(axis=1)[:, :3] / 255.0
    else:
        cols = np.full((len(pts), 3), 0.7)

    return pts.astype('f4'), nrm.astype('f4'), cols.astype('f4')


def main():
    parser = argparse.ArgumentParser(
        description="Fuse multiple 3D assets into a single 2DGS scene"
    )
    parser.add_argument('--bg_ply', default=None,
                        help='Path to background 2DGS PLY file')
    parser.add_argument('--obj_a_ply', default=None,
                        help='Path to Object A 2DGS PLY file')
    parser.add_argument('--obj_b_mesh', default=None,
                        help='Path to Object B mesh (.obj) from Threestudio')
    parser.add_argument('--obj_c_mesh', default=None,
                        help='Path to Object C mesh (.obj) from Magic123')
    parser.add_argument('--out', default=None,
                        help='Output directory for fused scene')
    parser.add_argument('--placement_config', default=None,
                        help='JSON file with object placements')
    parser.add_argument('--project_root', default=None,
                        help='Project root (auto-detected if not set)')
    args = parser.parse_args()

    # Auto-detect project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent

    output_dir = project_root / "output"

    # Set defaults relative to project root
    bg_ply = args.bg_ply or str(output_dir / 'background_garden/point_cloud/iteration_30000/point_cloud.ply')
    obj_a_ply = args.obj_a_ply or str(output_dir / 'object_a_cat/point_cloud/iteration_30000/point_cloud.ply')
    obj_b_mesh = args.obj_b_mesh or str(output_dir / 'object_b_text23d/save/it10000/mesh.obj')
    obj_c_mesh = args.obj_c_mesh or str(output_dir / 'object_c_single23d/final_mesh.obj')
    out_dir = Path(args.out) if args.out else (output_dir / 'scene_fused')

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load placement config
    if args.placement_config and os.path.exists(args.placement_config):
        with open(args.placement_config) as f:
            placement = json.load(f)
    else:
        placement = DEFAULT_PLACEMENT

    all_blocks = []

    # ---- Background (identity transform) ----
    print("[1/4] Loading background...")
    if os.path.exists(bg_ply):
        bg, n = load_ply_raw(bg_ply)
        print(f"  Background: {n} Gaussians")
        all_blocks.append(('bg', bg, n))
    else:
        print(f"  WARNING: {bg_ply} not found, creating dummy background")
        n = 200000
        bg = {
            'x': np.random.randn(n).astype('f4') * 3,
            'y': np.random.randn(n).astype('f4') * 3,
            'z': np.random.rand(n).astype('f4') * 2 + 0.5,
            'nx': np.zeros(n, dtype='f4'), 'ny': np.zeros(n, dtype='f4'),
            'nz': np.ones(n, dtype='f4'),
            'f_dc_0': np.random.rand(n).astype('f4') * 0.4 + 0.3,
            'f_dc_1': np.random.rand(n).astype('f4') * 0.4 + 0.3,
            'f_dc_2': np.random.rand(n).astype('f4') * 0.4 + 0.3,
            'opacity': np.full(n, 0.8, dtype='f4'),
            'scale_0': np.full(n, 0.02, dtype='f4'),
            'scale_1': np.full(n, 0.02, dtype='f4'),
            'scale_2': np.full(n, 0.02, dtype='f4'),
            'rot_0': np.ones(n, dtype='f4'), 'rot_1': np.zeros(n, dtype='f4'),
            'rot_2': np.zeros(n, dtype='f4'), 'rot_3': np.zeros(n, dtype='f4'),
        }
        for k in range(44):
            bg.setdefault(f'f_rest_{k}', np.zeros(n, dtype='f4'))
        all_blocks.append(('bg', bg, n))

    # ---- Object A (2DGS PLY, apply transform) ----
    print("[2/4] Loading Object A (2DGS)...")
    cfg = placement['object_a']
    T = transform_matrix(cfg['tx'], cfg['ty'], cfg['tz'], cfg['scale'], cfg['ry'])
    if os.path.exists(obj_a_ply):
        data, n = load_ply_raw(obj_a_ply)
        xyz = np.stack([data['x'], data['y'], data['z']], axis=-1)
        xyz_t = (T[:3, :3] @ xyz.T + T[:3, 3:4]).T
        data['x'], data['y'], data['z'] = xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2]
        print(f"  Object A: {n} Gaussians → {cfg['desc']}")
        all_blocks.append(('obj_a', data, n))
    else:
        print(f"  WARNING: {obj_a_ply} not found, skipping Object A")

    # ---- Object B (Threestudio mesh) ----
    print("[3/4] Loading Object B (Threestudio)...")
    cfg = placement['object_b']
    T = transform_matrix(cfg['tx'], cfg['ty'], cfg['tz'], cfg['scale'], cfg['ry'])
    if os.path.exists(obj_b_mesh):
        try:
            pts, nrm, col = mesh_to_dense_points(obj_b_mesh, 30000)
            xyz_t = (T[:3, :3] @ pts.T + T[:3, 3:4]).T
            nrm_t = (T[:3, :3] @ nrm.T).T
            n = len(pts)
            data = {
                'x': xyz_t[:, 0], 'y': xyz_t[:, 1], 'z': xyz_t[:, 2],
                'nx': nrm_t[:, 0], 'ny': nrm_t[:, 1], 'nz': nrm_t[:, 2],
                'f_dc_0': col[:, 0], 'f_dc_1': col[:, 1], 'f_dc_2': col[:, 2],
                'opacity': np.full(n, 0.6, dtype='f4'),
                'scale_0': np.full(n, 0.015, dtype='f4'),
                'scale_1': np.full(n, 0.015, dtype='f4'),
                'scale_2': np.full(n, 0.015, dtype='f4'),
                'rot_0': np.ones(n, dtype='f4'), 'rot_1': np.zeros(n, dtype='f4'),
                'rot_2': np.zeros(n, dtype='f4'), 'rot_3': np.zeros(n, dtype='f4'),
            }
            for k in range(44):
                data[f'f_rest_{k}'] = np.zeros(n, dtype='f4')
            print(f"  Object B: {n} points → {cfg['desc']}")
            all_blocks.append(('obj_b', data, n))
        except Exception as e:
            print(f"  ERROR loading Object B: {e}")
    else:
        print(f"  WARNING: {obj_b_mesh} not found, skipping Object B")

    # ---- Object C (Magic123 mesh) ----
    print("[4/4] Loading Object C (Magic123)...")
    cfg = placement['object_c']
    T = transform_matrix(cfg['tx'], cfg['ty'], cfg['tz'], cfg['scale'], cfg['ry'])
    if os.path.exists(obj_c_mesh):
        try:
            pts, nrm, col = mesh_to_dense_points(obj_c_mesh, 30000)
            xyz_t = (T[:3, :3] @ pts.T + T[:3, 3:4]).T
            nrm_t = (T[:3, :3] @ nrm.T).T
            n = len(pts)
            data = {
                'x': xyz_t[:, 0], 'y': xyz_t[:, 1], 'z': xyz_t[:, 2],
                'nx': nrm_t[:, 0], 'ny': nrm_t[:, 1], 'nz': nrm_t[:, 2],
                'f_dc_0': col[:, 0], 'f_dc_1': col[:, 1], 'f_dc_2': col[:, 2],
                'opacity': np.full(n, 0.6, dtype='f4'),
                'scale_0': np.full(n, 0.015, dtype='f4'),
                'scale_1': np.full(n, 0.015, dtype='f4'),
                'scale_2': np.full(n, 0.015, dtype='f4'),
                'rot_0': np.ones(n, dtype='f4'), 'rot_1': np.zeros(n, dtype='f4'),
                'rot_2': np.zeros(n, dtype='f4'), 'rot_3': np.zeros(n, dtype='f4'),
            }
            for k in range(44):
                data[f'f_rest_{k}'] = np.zeros(n, dtype='f4')
            print(f"  Object C: {n} points → {cfg['desc']}")
            all_blocks.append(('obj_c', data, n))
        except Exception as e:
            print(f"  ERROR loading Object C: {e}")
    else:
        print(f"  WARNING: {obj_c_mesh} not found, skipping Object C")

    # ---- Merge ----
    if len(all_blocks) <= 1:
        print("\nERROR: No objects to merge. Run training steps first.")
        sys.exit(1)

    # Collect all field names
    all_fields = set()
    for _, data, _ in all_blocks:
        all_fields.update(data.keys())

    # Concatenate
    merged = {}
    total = 0
    for name, data, n in all_blocks:
        for f in all_fields:
            if f in data:
                merged.setdefault(f, []).append(data[f])
            else:
                merged.setdefault(f, []).append(np.zeros(n, dtype='f4'))
        total += n
        print(f"  [{name}] {n} elements")

    for f in all_fields:
        merged[f] = np.concatenate(merged[f])

    print(f"\nTotal: {total} Gaussians")

    # Save in 2DGS format
    ply_dir = out_dir / "point_cloud" / "iteration_30000"
    ply_dir.mkdir(parents=True, exist_ok=True)
    ply_path = ply_dir / "point_cloud.ply"
    write_ply_raw(str(ply_path), merged, total)
    print(f"Saved: {ply_path}")

    # Save placement config
    with open(out_dir / "placement.json", 'w') as f:
        json.dump(placement, f, indent=2)

    print("\nDone! Run:")
    _2dgs_dir = project_root / "2d-gaussian-splatting/2d-gaussian-splatting"
    print(f"  cd {_2dgs_dir}")
    print(f"  python render.py -m {out_dir.resolve()} --skip_train --render_path")


if __name__ == '__main__':
    main()
