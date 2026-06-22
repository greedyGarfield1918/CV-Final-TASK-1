"""
============================================================
Step 7: Render multi-view video from fused 2DGS scene
============================================================
Uses the 2DGS render.py with --render_path flag for automatic
spiral trajectory rendering. Also supports custom camera paths.

Dependencies: 2DGS environment (conda activate 2dgs)
============================================================
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Render multi-view flythrough video from fused 2DGS scene"
    )
    parser.add_argument('--fused_dir', default=None,
                        help='Path to fused scene directory (default: auto-detect)')
    parser.add_argument('--project_root', default=None,
                        help='Project root directory (default: auto-detect)')
    parser.add_argument('--output_video', default=None,
                        help='Output video path')
    parser.add_argument('--trajectory', default='spiral',
                        choices=['spiral', 'circular', 'custom'],
                        help='Camera trajectory type')
    parser.add_argument('--num_views', type=int, default=120,
                        help='Number of views/frames')
    parser.add_argument('--resolution', default='1920x1080',
                        help='Output resolution (WxH)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    args = parser.parse_args()

    # Auto-detect paths
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent

    _2dgs_dir = project_root / "2d-gaussian-splatting/2d-gaussian-splatting"

    if args.fused_dir:
        fused_dir = Path(args.fused_dir)
    else:
        fused_dir = project_root / "output" / "scene_fused"

    ply_path = fused_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"

    if not ply_path.exists():
        print(f"ERROR: Fused PLY not found at {ply_path}")
        print("Run 06_scene_fusion.py first.")
        print(f"Or specify --fused_dir with path to a valid 2DGS model directory.")
        sys.exit(1)

    print(f"Rendering fused scene from: {fused_dir}")
    print(f"  Trajectory: {args.trajectory}")
    print(f"  Views: {args.num_views}")
    print(f"  Resolution: {args.resolution}")

    # Use 2DGS render.py with --render_path for automatic spiral trajectory
    cmd = [
        sys.executable,
        str(_2dgs_dir / "render.py"),
        "-m", str(fused_dir.resolve()),
        "--skip_train",
        "--skip_test",
        "--render_path",
        "--quiet",
    ]

    print(f"Running: {' '.join(cmd)}")
    os.chdir(str(_2dgs_dir))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nWARNING: 2DGS render.py failed.")
        print("You may need to render manually with:")
        print(f"  cd {_2dgs_dir}")
        print(f"  python render.py -m {fused_dir.resolve()} --skip_train --render_path")
        sys.exit(1)

    # Check output
    for subdir_name in ['traj', 'render_path', 'video']:
        video_dir = fused_dir / subdir_name
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            if videos:
                print(f"\nVideo rendered: {videos[0]}")
                break
            frames = list(video_dir.glob("*.png"))
            if frames:
                print(f"\nFrames saved to: {video_dir} ({len(frames)} frames)")
                print("To create video: ffmpeg -y -framerate 30 -i frame_%04d.png -c:v libx264 output.mp4")
                break
    else:
        print(f"\nCheck {fused_dir} for rendered output.")

    print("\nDone!")


if __name__ == "__main__":
    main()
