#!/usr/bin/env python3
"""
Fusion Rendering (Hybrid Approach): Place 2DGS objects and mesh objects
into a 2DGS background scene using nvdiffrast for mesh rendering.

Approach:
  - Background: rendered via 2DGS CUDA rasterizer
  - Object A (2DGS): rendered via 2DGS CUDA rasterizer
  - Objects B + C (meshes): rendered via nvdiffrast
  - Alpha-composite all layers in depth order

This hybrid approach keeps 2DGS objects as native Gaussians while
rendering AIGC mesh outputs directly, avoiding mesh→Gaussian conversion.

Output: output/fusion/flythrough_frames/
"""
import numpy as np
import torch
import sys
import os
import argparse
import json
from pathlib import Path


def find_module(module_path):
    """Find and add a Python module to sys.path."""
    if os.path.isdir(module_path):
        sys.path.insert(0, module_path)
        return True
    return False


def load_gaussian_model(checkpoint_path, sh_degree=3, device='cuda'):
    """Load a trained 2DGS Gaussian model from checkpoint."""
    from scene import GaussianModel
    gaussians = GaussianModel(sh_degree=sh_degree)

    if not os.path.exists(checkpoint_path):
        print(f"  WARNING: checkpoint not found at {checkpoint_path}")
        return None

    gaussians.load_ply(checkpoint_path)
    return gaussians


def render_gaussian(gaussians, camera, pipe, bg_color):
    """Render a 2DGS Gaussian model from a given camera."""
    from gaussian_renderer import render
    with torch.no_grad():
        result = render(camera, gaussians, pipe, bg_color)
    return result['render']


def make_camera_2dgs(eye, center, up, fov_deg, width, height):
    """Create a 2DGS Camera object from look-at parameters."""
    from scene.cameras import Camera as Cam
    from utils.graphics_utils import getWorld2View2

    z_axis = eye - center
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    t = -R.T @ eye

    fov_rad = np.radians(fov_deg)
    focal_y = height / (2.0 * np.tan(fov_rad / 2.0))
    fov_y = 2.0 * np.arctan(height / (2.0 * focal_y))
    fov_x = 2.0 * np.arctan(width / (2.0 * focal_y))

    world_view_transform = torch.tensor(
        getWorld2View2(R, t).transpose(), dtype=torch.float32, device='cuda'
    )
    full_proj_transform = torch.eye(4, device='cuda')
    camera_center = torch.tensor(eye, dtype=torch.float32, device='cuda')

    return Cam(colmap_id=0, R=R, T=t,
               FoVx=fov_x, FoVy=fov_y,
               image=torch.zeros(3, height, width),
               gt_alpha_mask=None,
               image_name='', uid=0,
               data_device='cuda',
               trans=np.array([0.0, 0.0, 0.0]), scale=1.0)


def perspective_matrix(fov_deg, aspect, near=0.1, far=100.0):
    """OpenGL-style perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at_matrix(eye, center, up):
    """World-to-camera view matrix."""
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


def render_mesh_nvdiffrast(glctx, verts, faces, view_mat, proj_mat, resolution, color=None):
    """Render a triangle mesh with nvdiffrast. Returns RGBA numpy array."""
    import nvdiffrast.torch as dr
    h, w = resolution[1], resolution[0]

    verts_h = torch.cat([verts, torch.ones(verts.shape[0], 1, device=verts.device)], dim=1)
    view_t = torch.tensor(view_mat, dtype=torch.float32, device=verts.device)
    proj_t = torch.tensor(proj_mat, dtype=torch.float32, device=verts.device)

    verts_clip = verts_h @ view_t.T @ proj_t.T
    rast_out, _ = dr.rasterize(glctx, verts_clip[None], faces, resolution=(h, w))

    # Compute face normals for simple diffuse shading
    v0 = verts_clip[faces[:, 0]]
    v1 = verts_clip[faces[:, 1]]
    v2 = verts_clip[faces[:, 2]]
    normals = torch.cross(v1[:, :3] - v0[:, :3], v2[:, :3] - v0[:, :3])
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)

    light = torch.tensor([0.4, 0.6, -0.7], device=verts.device)
    shading = torch.clamp((normals @ light).abs(), 0.25, 1.0)

    if color is None:
        colors = torch.ones_like(normals) * 0.75 * shading[:, None]
    else:
        colors = torch.tensor(color, dtype=torch.float32, device=verts.device).expand_as(normals) * shading[:, None]

    color_out, alpha = dr.interpolate(colors[None], rast_out, faces)
    rgba = torch.cat([color_out[0], alpha[0]], dim=-1).cpu().numpy()
    return np.clip(np.flipud(rgba), 0, 1)


def composite_layers(layers, resolution):
    """Alpha-blend sorted layers (far to near)."""
    layers = sorted(layers, key=lambda x: -x[1])
    result = np.ones((resolution[1], resolution[0], 4), dtype=np.float32)
    for rgba, _ in layers:
        if rgba.shape[-1] == 3:
            rgba = np.concatenate([rgba, np.ones((*rgba.shape[:2], 1))], axis=-1)
        alpha = rgba[:, :, 3:4]
        result = rgba * alpha + result * (1 - alpha)
    return np.clip(result, 0, 1)


def generate_camera_path(num_frames, center, radius, height, style='spiral'):
    """Generate camera positions and look-at targets."""
    cameras = []
    for i in range(num_frames):
        if style == 'circular':
            angle = 2 * np.pi * i / num_frames
            eye = np.array([
                center[0] + radius * np.cos(angle),
                center[1] + height + 0.2 * np.sin(angle * 3),
                center[2] + radius * np.sin(angle)
            ])
        elif style == 'spiral':
            angle = 2 * np.pi * i / num_frames * 1.5
            r = radius * (0.6 + 0.4 * i / num_frames)
            eye = np.array([
                center[0] + r * np.cos(angle),
                center[1] + height * (1.5 - 0.5 * i / num_frames),
                center[2] + r * np.sin(angle)
            ])
        up = np.array([0.0, 1.0, 0.0])
        cameras.append({'eye': eye, 'center': center, 'up': up})
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Hybrid fusion renderer")
    parser.add_argument('--project_root', default=None,
                        help='Project root directory')
    parser.add_argument('--bg_ply', default=None,
                        help='Background Gaussian PLY')
    parser.add_argument('--obj_a_ply', default=None,
                        help='Object A Gaussian PLY')
    parser.add_argument('--obj_b_mesh', default=None,
                        help='Object B mesh (Threestudio)')
    parser.add_argument('--obj_c_mesh', default=None,
                        help='Object C mesh (Magic123)')
    parser.add_argument('--out', default=None,
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=180,
                        help='Number of frames')
    parser.add_argument('--resolution', default='1280x720',
                        help='Output resolution WxH')
    parser.add_argument('--trajectory', default='spiral',
                        choices=['circular', 'spiral'])
    parser.add_argument('--skip_render', action='store_true',
                        help='Skip rendering, only generate camera path')
    args = parser.parse_args()

    # Path resolution
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"

    # Find 2DGS module
    _2dgs_path = project_root / "2d-gaussian-splatting/2d-gaussian-splatting"
    if not find_module(str(_2dgs_path)):
        print(f"ERROR: 2DGS not found at {_2dgs_path}")
        sys.exit(1)

    from utils.general_utils import safe_state
    safe_state(True)

    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()

    # Configuration
    W, H = [int(x) for x in args.resolution.split('x')]
    FOV = 60.0
    CENTER = np.array([0.0, 0.0, 0.0])
    RADIUS = 3.5
    HEIGHT = 0.6

    PLACEMENTS = {
        'A': np.array([-0.8, -0.2, -0.5]),
        'B': np.array([0.6, -0.2, -0.7]),
        'C': np.array([0.0, -0.15, -1.0]),
    }

    # Set defaults
    bg_ply = args.bg_ply or str(output_dir / 'background_garden/point_cloud/iteration_30000/point_cloud.ply')
    obj_a_ply = args.obj_a_ply or str(output_dir / 'object_a_cat/point_cloud/iteration_30000/point_cloud.ply')
    obj_b_mesh = args.obj_b_mesh or str(output_dir / 'object_b_text23d/save/it10000/mesh.obj')
    obj_c_mesh = args.obj_c_mesh or str(output_dir / 'object_c_single23d/final_mesh.obj')
    out_dir = Path(args.out) if args.out else (output_dir / 'fusion')
    frames_dir = out_dir / 'flythrough_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    bg_gauss = load_gaussian_model(bg_ply)
    if bg_gauss:
        print(f"  Background: {bg_gauss._xyz.shape[0]} Gaussians")
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

    obj_a_gauss = load_gaussian_model(obj_a_ply)
    if obj_a_gauss:
        print(f"  Object A: {obj_a_gauss._xyz.shape[0]} Gaussians")
    obj_a_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device='cuda')

    # Load meshes
    import trimesh
    meshes = {}
    for label, path, scale, color in [
        ('B', obj_b_mesh, 0.15, [0.9, 0.7, 0.3]),
        ('C', obj_c_mesh, 0.12, [0.5, 0.5, 0.55]),
    ]:
        if os.path.exists(path):
            try:
                m = trimesh.load(path, force='mesh')
                v = torch.tensor(np.asarray(m.vertices) * scale, dtype=torch.float32, device='cuda')
                f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device='cuda')
                v = v - v.mean(dim=0)
                meshes[label] = (v, f, color)
                print(f"  Mesh {label}: {v.shape[0]}v {f.shape[0]}f")
            except Exception as e:
                print(f"  Mesh {label}: FAILED ({e})")
        else:
            print(f"  Mesh {label}: not found at {path}")

    # Generate camera path
    print(f"\nGenerating {args.num_frames} camera poses ({args.trajectory})...")
    cameras = generate_camera_path(args.num_frames, CENTER, RADIUS, HEIGHT, args.trajectory)
    proj = perspective_matrix(FOV, W / H)

    if args.skip_render:
        print("--skip_render set. Exiting.")
        return

    # Render
    print(f"Rendering {args.num_frames} frames...")
    for fi, cam in enumerate(cameras):
        layers = []

        # Background (rendered at center)
        if bg_gauss:
            cam_2dgs = make_camera_2dgs(cam['eye'], cam['center'], cam['up'], FOV, W, H)
            bg_render = render_gaussian(bg_gauss, cam_2dgs, None, bg_color)
            bg_np = bg_render.permute(1, 2, 0).cpu().numpy()
            bg_rgba = np.concatenate([bg_np, np.ones((H, W, 1))], axis=-1)
            layers.append((bg_rgba, np.linalg.norm(cam['eye'] - CENTER)))

        # Object A (2DGS at offset position)
        if obj_a_gauss:
            obj_center = PLACEMENTS['A']
            obj_eye = cam['eye'] - obj_center
            obj_cam = make_camera_2dgs(obj_eye, np.zeros(3), cam['up'], FOV, W, H)
            ar = render_gaussian(obj_a_gauss, obj_cam, None, obj_a_bg)
            ar_np = ar.permute(1, 2, 0).cpu().numpy()
            # Alpha from non-white pixels
            aa = (ar_np.sum(-1) < 2.9).astype(np.float32)
            layers.append((
                np.concatenate([ar_np, aa[..., None]], -1),
                np.linalg.norm(cam['eye'] - obj_center)
            ))

        # Object meshes (B, C)
        view_mat = look_at_matrix(cam['eye'], cam['center'], cam['up'])
        for label in ['B', 'C']:
            if label in meshes:
                v, f, color = meshes[label]
                vw = v + torch.tensor(PLACEMENTS[label], dtype=torch.float32, device='cuda')
                rgba = render_mesh_nvdiffrast(glctx, vw, f, view_mat, proj, (W, H), color)
                depth = np.linalg.norm(cam['eye'] - PLACEMENTS[label])
                layers.append((rgba, depth))

        # Composite
        frame = composite_layers(layers, (W, H))

        # Save
        from PIL import Image
        Image.fromarray((frame[:, :, :3] * 255).astype(np.uint8)).save(
            frames_dir / f'frame_{fi:04d}.png'
        )

        if (fi + 1) % 45 == 0:
            print(f"  Frame {fi + 1}/{args.num_frames}")

    print(f"\nDone! {args.num_frames} frames saved to {frames_dir}")
    print(f"To create video:")
    print(f"  ffmpeg -y -framerate 30 -i {frames_dir}/frame_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p {out_dir}/flythrough.mp4")


if __name__ == '__main__':
    main()
