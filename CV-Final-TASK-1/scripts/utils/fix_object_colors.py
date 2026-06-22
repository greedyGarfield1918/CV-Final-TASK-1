#!/usr/bin/env python3
"""
Fix/adjust colors in a 2DGS PLY file.
Useful for making objects visually distinct in the fused scene.

Usage:
    python scripts/utils/fix_object_colors.py --ply input.ply --output output.ply --tint "1.0,0.5,0.3"
"""
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def tint_ply(input_path, output_path, tint_color, brightness=1.0):
    """Apply tint color to a 2DGS PLY file."""
    ply = PlyData.read(input_path)
    v = ply['vertex']

    n = len(v.data)
    data = {}
    for name in v.data.dtype.names:
        data[name] = np.array(v[name])

    # Tint the DC colors
    if 'f_dc_0' in data:
        data['f_dc_0'] = np.clip(data['f_dc_0'] * tint_color[0] * brightness, 0, 1)
        data['f_dc_1'] = np.clip(data['f_dc_1'] * tint_color[1] * brightness, 0, 1)
        data['f_dc_2'] = np.clip(data['f_dc_2'] * tint_color[2] * brightness, 0, 1)
        print(f"Applied tint {tint_color} with brightness {brightness}")

    # Write
    dtype = [(k, 'f4') for k in data.keys()]
    arr = np.zeros(n, dtype=dtype)
    for k, v in data.items():
        arr[k] = v.astype(np.float32)
    PlyData([PlyElement.describe(arr, 'vertex')]).write(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fix object colors in PLY")
    parser.add_argument('--ply', required=True, help='Input .ply file')
    parser.add_argument('--output', required=True, help='Output .ply file')
    parser.add_argument('--tint', default='1.0,1.0,1.0',
                        help='RGB tint comma-separated, e.g. "1.0,0.5,0.3"')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='Brightness multiplier')
    args = parser.parse_args()

    tint = [float(x) for x in args.tint.split(',')]
    tint_ply(args.ply, args.output, tint, args.brightness)


if __name__ == '__main__':
    main()
