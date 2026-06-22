#!/usr/bin/env python3
"""
Download Zero123 model from ModelScope or alternative sources.
Falls back through multiple download strategies.

Usage:
    python scripts/download_zero123_modelscope.py
    python scripts/download_zero123_modelscope.py --target /path/to/105000.ckpt
"""
import os
import sys
import argparse
import urllib.request
import shutil


def download_from_modelscope(target_path):
    """Try downloading from ModelScope."""
    print("Trying ModelScope...")
    try:
        from modelscope import snapshot_download
        import glob
        model_dir = snapshot_download(
            'iic/zero123_weights',
            cache_dir='/tmp/zero123_ms',
            revision='master'
        )
        ckpt_files = glob.glob(f'{model_dir}/**/*.ckpt', recursive=True)
        if ckpt_files:
            src = ckpt_files[0]
            size_gb = os.path.getsize(src) / 1e9
            print(f"Found: {src} ({size_gb:.1f} GB)")
            shutil.copy2(src, target_path)
            print(f"Copied to {target_path}")
            return True
    except Exception as e:
        print(f"ModelScope failed: {e}")
    return False


def download_from_hf_mirror(target_path):
    """Try HuggingFace mirror (CN-friendly)."""
    print("\nTrying HuggingFace mirror...")
    url = "https://hf-mirror.com/cvlab/zero123-weights/resolve/main/105000.ckpt"
    try:
        urllib.request.urlretrieve(url, target_path)
        size = os.path.getsize(target_path)
        print(f"Downloaded: {size / 1e9:.1f} GB")
        if size > 1e9:
            print("Success!")
            return True
        else:
            os.remove(target_path)
    except Exception as e:
        print(f"HF mirror failed: {e}")
    return False


def download_from_hf_original(target_path):
    """Try HuggingFace original."""
    print("\nTrying HuggingFace original...")
    url = "https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt"
    try:
        urllib.request.urlretrieve(url, target_path)
        size = os.path.getsize(target_path)
        print(f"Downloaded: {size / 1e9:.1f} GB")
        if size > 1e9:
            print("Success!")
            return True
        else:
            os.remove(target_path)
    except Exception as e:
        print(f"HF original failed: {e}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Zero123 weights")
    parser.add_argument('--target', default=None,
                        help='Target path for 105000.ckpt')
    args = parser.parse_args()

    if args.target:
        target_dir = os.path.dirname(args.target)
        target_file = args.target
    else:
        # Default: Magic123/pretrained/zero123/105000.ckpt
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_dir = os.path.join(project_root, 'Magic123', 'Magic123', 'pretrained', 'zero123')
        target_file = os.path.join(target_dir, '105000.ckpt')

    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_file) and os.path.getsize(target_file) > 1e9:
        print(f"Zero123 weights already exist at {target_file}")
        print(f"Size: {os.path.getsize(target_file) / 1e9:.1f} GB")
        return

    # Try all methods
    for method in [download_from_modelscope, download_from_hf_mirror, download_from_hf_original]:
        if method(target_file):
            return

    print("\nAll sources failed. Please manually download Zero123 weights:")
    print("  https://huggingface.co/cvlab/zero123-weights")
    print(f"  Place at: {target_file}")
    sys.exit(1)


if __name__ == '__main__':
    main()
