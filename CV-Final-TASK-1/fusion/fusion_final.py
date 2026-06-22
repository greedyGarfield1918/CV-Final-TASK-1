#!/usr/bin/env python3
"""
Fusion: Gaussian Splatting + Mesh objects → Unified Render

Simplified version that merges and renders using the 2DGS rasterizer
for Gaussian objects and nvdiffrast for mesh objects, alpha-composited.

This is a self-contained version with fewer dependencies on the
2DGS codebase internals.

Usage:
    python fusion/fusion_final.py --bg_ply <path> --obj_a_ply <path> \
        --obj_b_mesh <path> --obj_c_mesh <path> --out <dir>
"""
import numpy as np
import torch
import sys
import os
import argparse
from pathlib import Path

# Try to locate 2DGS
SCRIPT_DIR = Path(__file__).parent.parent
_2DGS_DIR = SCRIPT_DIR / "2d-gaussian-splatting/2d-gaussian-splatting"
if _2DGS_DIR.is_dir():
    sys.path.insert(0, str(_2DGS_DIR))


def main():
    parser = argparse.ArgumentParser(description="Fused scene renderer")
    parser.add_argument('--bg_ply', required=True, help='Background PLY')
    parser.add_argument('--obj_a_ply', default=None, help='Object A PLY')
    parser.add_argument('--obj_b_mesh', default=None, help='Object B OBJ')
    parser.add_argument('--obj_c_mesh', default=None, help='Object C OBJ')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--num_frames', type=int, default=180)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fov', type=float, default=60.0)
    parser.add_argument('--trajectory', default='spiral',
                        choices=['circular', 'spiral'])
    args = parser.parse_args()

    out_dir = Path(args.out)
    frames_dir = out_dir / 'flythrough_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    W, H = args.width, args.height
    FOV = args.fov
    CENTER = np.array([0.0, 0.0, 0.0])
    RADIUS = 3.5
    HEIGHT = 0.6
    UP = np.array([0.0, 1.0, 0.0])

    PLACEMENTS = {
        'A': np.array([-0.8, -0.2, -0.5]),
        'B': np.array([0.6, -0.2, -0.7]),
        'C': np.array([0.0, -0.15, -1.0]),
    }

    # --- Camera helpers ---
    def look_at(eye, center, up):
        z = eye - center
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.eye(4)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        m[:3, 3] = eye
        return np.linalg.inv(m)

    def perspective(fov, aspect, n=0.1, f=100.0):
        ff = 1 / np.tan(np.radians(fov) / 2)
        m = np.zeros((4, 4))
        m[0, 0] = ff / aspect
        m[1, 1] = ff
        m[2, 2] = (f + n) / (n - f)
        m[2, 3] = 2 * f * n / (n - f)
        m[3, 2] = -1
        return m

    def composite(layers):
        layers = sorted(layers, key=lambda x: -x[1])
        result = np.ones((H, W, 4), dtype=np.float32)
        for rgba, _ in layers:
            if rgba.shape[-1] == 3:
                rgba = np.concatenate([rgba, np.ones((H, W, 1))], -1)
            a = rgba[:, :, 3:4]
            result = rgba * a + result * (1 - a)
        return np.clip(result, 0, 1)

    # --- Load 2DGS models ---
    from scene import GaussianModel
    from gaussian_renderer import render as gs_render
    from utils.general_utils import safe_state
    import nvdiffrast.torch as dr

    safe_state(True)
    glctx = dr.RasterizeCudaContext()

    bg = GaussianModel(sh_degree=3)
    bg.load_ply(args.bg_ply)
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
    print(f"Background: {bg._xyz.shape[0]} Gaussians")

    obj_a = None
    if args.obj_a_ply and os.path.exists(args.obj_a_ply):
        obj_a = GaussianModel(sh_degree=3)
        obj_a.load_ply(args.obj_a_ply)
        print(f"Object A: {obj_a._xyz.shape[0]} Gaussians")
    obj_a_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')

    # --- Load meshes ---
    import trimesh
    meshes = {}
    for label, path, scale, color in [
        ('B', args.obj_b_mesh, 0.15, [0.9, 0.7, 0.3]),
        ('C', args.obj_c_mesh, 0.12, [0.5, 0.5, 0.55]),
    ]:
        if path and os.path.exists(path):
            m = trimesh.load(path, force='mesh')
            v = torch.tensor(np.asarray(m.vertices) * scale, dtype=torch.float32, device='cuda')
            f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device='cuda')
            v = v - v.mean(dim=0)
            meshes[label] = (v, f, color)
            print(f"Mesh {label}: {v.shape[0]}v")

    # --- Render loop ---
    proj = perspective(FOV, W / H)
    print(f"\nRendering {args.num_frames} frames...")

    for fi in range(args.num_frames):
        # Camera pose
        if args.trajectory == 'spiral':
            angle = 2 * np.pi * fi / args.num_frames * 1.5
            r = RADIUS * (0.6 + 0.4 * fi / args.num_frames)
            eye = np.array([
                CENTER[0] + r * np.cos(angle),
                CENTER[1] + HEIGHT * (1.5 - 0.5 * fi / args.num_frames),
                CENTER[2] + r * np.sin(angle)
            ])
        else:
            angle = 2 * np.pi * fi / args.num_frames
            eye = np.array([
                CENTER[0] + RADIUS * np.cos(angle),
                CENTER[1] + HEIGHT,
                CENTER[2] + RADIUS * np.sin(angle)
            ])

        vm = look_at(eye, CENTER, UP)
        layers = []

        # Render background
        with torch.no_grad():
            from scene.cameras import Camera
            from utils.graphics_utils import getWorld2View2

            z_axis = eye - CENTER
            z_axis = z_axis / np.linalg.norm(z_axis)
            x_axis = np.cross(UP, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            R = np.stack([x_axis, y_axis, z_axis], axis=1)
            t = -R.T @ eye

            fov_rad = np.radians(FOV)
            fy = H / (2 * np.tan(fov_rad / 2))
            fov_y = 2 * np.arctan(H / (2 * fy))
            fov_x = 2 * np.arctan(W / (2 * fy))

            cam = Camera(0, R, t, fov_x, fov_y,
                         torch.zeros(3, H, W), None, '', 0,
                         data_device='cuda', trans=np.zeros(3), scale=1.0)

            result = gs_render(cam, bg, None, bg_color)
            bg_np = result['render'].permute(1, 2, 0).cpu().numpy()
            layers.append((
                np.concatenate([bg_np, np.ones((H, W, 1))], -1),
                np.linalg.norm(eye - CENTER)
            ))

        # Render Object A (2DGS)
        if obj_a is not None:
            a_eye = eye - PLACEMENTS['A']
            a_center = np.zeros(3)
            az = a_eye - a_center
            az = az / np.linalg.norm(az)
            ax = np.cross(UP, az)
            ax = ax / np.linalg.norm(ax)
            ay = np.cross(az, ax)
            aR = np.stack([ax, ay, az], axis=1)
            at = -aR.T @ a_eye

            a_cam = Camera(0, aR, at, fov_x, fov_y,
                           torch.zeros(3, H, W), None, '', 0,
                           data_device='cuda', trans=np.zeros(3), scale=1.0)

            ar = gs_render(a_cam, obj_a, None, obj_a_bg)
            ar_np = ar['render'].permute(1, 2, 0).cpu().numpy()
            aa = (ar_np.sum(-1) < 2.9).astype(np.float32)
            layers.append((
                np.concatenate([ar_np, aa[..., None]], -1),
                np.linalg.norm(eye - PLACEMENTS['A'])
            ))

        # Render meshes
        for label in ['B', 'C']:
            if label in meshes:
                v, f, color = meshes[label]
                vw = v + torch.tensor(PLACEMENTS[label], dtype=torch.float32, device='cuda')

                # nvdiffrast render
                vh = torch.cat([vw, torch.ones(vw.shape[0], 1, device='cuda')], 1)
                vt = torch.tensor(vm, dtype=torch.float32, device='cuda')
                pt = torch.tensor(proj, dtype=torch.float32, device='cuda')
                vc = vh @ vt.T @ pt.T
                ro, _ = dr.rasterize(glctx, vc[None], f, (H, W))

                # Diffuse shading
                v0, v1, v2 = vc[f[:, 0]], vc[f[:, 1]], vc[f[:, 2]]
                n = torch.cross(v1[:, :3] - v0[:, :3], v2[:, :3] - v0[:, :3])
                n = n / (n.norm(dim=1, keepdim=True) + 1e-8)
                light = torch.tensor([0.4, 0.6, -0.7], device='cuda')
                sh = torch.clamp((n @ light).abs(), 0.25, 1.0)
                co = torch.tensor(color, dtype=torch.float32, device='cuda').expand_as(n) * sh[:, None]
                col, alp = dr.interpolate(co[None], ro, f)
                rgba = np.clip(np.flipud(torch.cat([col[0], alp[0]], -1).cpu().numpy()), 0, 1)
                layers.append((rgba, np.linalg.norm(eye - PLACEMENTS[label])))

        # Composite and save
        frame = composite(layers)
        from PIL import Image
        Image.fromarray((frame[:, :, :3] * 255).astype(np.uint8)).save(
            frames_dir / f'frame_{fi:04d}.png'
        )

        if (fi + 1) % 45 == 0:
            print(f"  {fi + 1}/{args.num_frames}")

    print(f"\nDone → {frames_dir}")
    print(f"Video: ffmpeg -y -framerate 30 -i {frames_dir}/frame_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p {out_dir}/flythrough.mp4")


if __name__ == '__main__':
    main()
