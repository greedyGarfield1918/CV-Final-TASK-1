#!/usr/bin/env python3
"""
Downsample COLMAP point cloud for faster 2DGS initialization.

Reads COLMAP points3D.bin and writes a smaller .ply for 2DGS.
Useful when the original COLMAP reconstruction has too many points.

Usage:
    python scripts/utils/downsample_colmap.py --input points3D.bin --output points3D.ply --max_points 30000
"""
import struct
import os
import sys
import argparse


def read_points3D_bin(filepath, max_points=None):
    """Read COLMAP points3D.bin file."""
    pts = []
    with open(filepath, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        print(f'Total COLMAP points: {n}')
        limit = min(n, max_points) if max_points else n
        for i in range(n):
            if i >= limit:
                break
            pid = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<ddd', f.read(24))
            rgb = struct.unpack('<BBB', f.read(3))
            error = struct.unpack('<d', f.read(8))[0]
            track = struct.unpack('<Q', f.read(8))[0]
            pts.append((xyz, rgb))
    return pts


def write_ply(filepath, pts):
    """Write points as ASCII PLY."""
    with open(filepath, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(pts)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in pts:
            f.write(f'{x} {y} {z} {r} {g} {b}\n')
    print(f'Written {len(pts)} points to {filepath}')


def main():
    parser = argparse.ArgumentParser(description="Downsample COLMAP point cloud")
    parser.add_argument('--input', required=True, help='Path to points3D.bin')
    parser.add_argument('--output', required=True, help='Output .ply path')
    parser.add_argument('--max_points', type=int, default=30000,
                        help='Maximum number of points to keep')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    pts = read_points3D_bin(args.input, args.max_points)
    write_ply(args.output, pts)

    # Backup original
    bak = args.input + '.bak'
    if not os.path.exists(bak):
        os.rename(args.input, bak)
        print(f'Backed up original to {bak}')


if __name__ == '__main__':
    main()
