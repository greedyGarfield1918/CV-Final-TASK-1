#!/usr/bin/env python3
"""
Generate comparison charts for the technical report.

Charts produced:
  1. 2DGS Training Loss Curves
  2. Loss Components Breakdown
  3. Gaussian Point Growth
  4. Training Convergence Speed
  5. Method Comparison (PSNR vs Time)

Usage:
    python fusion/generate_charts.py --data_dir <training_data_json_dir> --out <charts_output_dir>
"""
import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Font configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
for font in ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei']:
    try:
        plt.rcParams['font.sans-serif'] = [font]
        break
    except Exception:
        continue


def chart_loss_curves(datasets, out_dir):
    """Chart 1: 2DGS Total Loss Comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Object A - Cat': '#2196F3', 'Background - Garden': '#4CAF50'}
    for name, data in datasets.items():
        tag = 'train_loss_patches/total_loss'
        if tag in data:
            steps = data[tag]['steps']
            values = data[tag]['values']
            idx = np.linspace(0, len(steps) - 1, min(500, len(steps)), dtype=int)
            ax.plot(np.array(steps)[idx], np.array(values)[idx], label=name,
                    color=colors.get(name), linewidth=1.2, alpha=0.85)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('2DGS Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_loss_curves_2dgs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_loss_curves_2dgs.png")


def chart_loss_components(datasets, out_dir):
    """Chart 2: Loss components breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    components = [
        ('train_loss_patches/total_loss', 'Total', '#333'),
        ('train_loss_patches/reg_loss', 'L1 Reg', '#E91E63'),
        ('train_loss_patches/dist_loss', 'Distortion', '#FF9800'),
        ('train_loss_patches/normal_loss', 'Normal', '#9C27B0'),
    ]
    for idx, name in enumerate(datasets.keys()):
        ax = axes[idx] if len(datasets) > 1 else axes
        data = datasets[name]
        for tag, label, c in components:
            if tag in data:
                steps = np.array(data[tag]['steps'])
                values = np.array(data[tag]['values'])
                idxx = np.linspace(0, len(steps) - 1, min(300, len(steps)), dtype=int)
                ax.plot(steps[idxx], values[idxx], label=label, color=c, linewidth=1.2, alpha=0.8)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('2DGS Loss Components', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_loss_components.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_loss_components.png")


def chart_gaussian_growth(datasets, out_dir):
    """Chart 3: Gaussian points growth."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Object A - Cat': '#2196F3', 'Background - Garden': '#4CAF50'}
    for name, data in datasets.items():
        tag = 'total_points'
        if tag in data:
            steps = np.array(data[tag]['steps'])
            values = np.array(data[tag]['values'])
            idx = np.linspace(0, len(steps) - 1, min(500, len(steps)), dtype=int)
            ax.plot(steps[idx], values[idx] / 1000, label=name,
                    color=colors.get(name), linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gaussian Count (thousands)', fontsize=12)
    ax.set_title('2DGS Gaussian Count During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_gaussian_growth.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_gaussian_growth.png")


def chart_convergence(datasets, out_dir):
    """Chart 4: Training convergence speed comparison (normalized)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Object A - Cat': '#2196F3', 'Background - Garden': '#4CAF50'}
    for name, data in datasets.items():
        tag = 'train_loss_patches/total_loss'
        if tag in data:
            steps = np.array(data[tag]['steps'])
            values = np.array(data[tag]['values'])
            vmin, vmax = values.min(), values.max()
            if vmax - vmin > 1e-8:
                values = (values - vmin) / (vmax - vmin)
            idx = np.linspace(0, len(steps) - 1, min(500, len(steps)), dtype=int)
            ax.plot(steps[idx], values[idx], label=name,
                    color=colors.get(name), linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Normalized Loss', fontsize=12)
    ax.set_title('Training Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_convergence.png")


def chart_method_comparison(out_dir):
    """Chart 5: Method comparison (PSNR vs Training Time)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = ['Object A\n(COLMAP+2DGS)', 'Object B\n(Threestudio)', 'Object C\n(Magic123)']
    psnr_vals = [38.5, 22.0, 28.0]  # Estimated/example values
    time_vals = [0.75, 3.7, 2.5]    # Hours
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width / 2, psnr_vals, width,
                   label='PSNR (dB, estimated)', color=['#2196F3', '#FF9800', '#FF5722'])
    bars2 = ax.bar(x + width / 2, time_vals, width,
                   label='Training Time (hours)', color=['#64B5F6', '#FFB74D', '#FF8A65'])

    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Method Comparison: Quality vs Speed', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                f'{bar.get_height():.1f}h', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_method_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_method_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison charts")
    parser.add_argument('--data', default=None,
                        help='Path to training_data.json')
    parser.add_argument('--out', default=None,
                        help='Output directory for charts')
    args = parser.parse_args()

    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'output', 'charts')
    os.makedirs(out_dir, exist_ok=True)

    # Load training data if available
    datasets = {}
    if args.data and os.path.exists(args.data):
        with open(args.data) as f:
            datasets = json.load(f)
        print(f"Loaded training data with {len(datasets)} datasets")

    if datasets:
        chart_loss_curves(datasets, out_dir)
        chart_loss_components(datasets, out_dir)
        chart_gaussian_growth(datasets, out_dir)
        chart_convergence(datasets, out_dir)
    else:
        print("No training data JSON provided. Skipping data-driven charts.")

    chart_method_comparison(out_dir)

    print(f"\nAll charts saved to: {out_dir}/")


if __name__ == '__main__':
    main()
