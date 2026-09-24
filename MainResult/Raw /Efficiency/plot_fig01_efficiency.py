"""从本目录的 data/ 重画效率对比图。

不依赖仓库里的其它脚本。在本目录执行：

    python plot_fig01_efficiency.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'

COLORS = ['#0072B2', '#D55E00', '#009E73']
METHODS = [
    ('ours', 'Ours'),
    ('posthoc', 'Post-hoc'),
    ('random', 'Random'),
]
BACKBONES = [('ResNet', 'resnet'), ('VGG11', 'vgg11')]


def apply_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.28,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'lines.linewidth': 1.6,
    })


def load_frame(backbone_slug: str, method_slug: str) -> pd.DataFrame:
    path = DATA / 'cifar100' / backbone_slug / f'{method_slug}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).sort_values('Round')
    if len(df) != 100:
        raise ValueError(f'{path} has {len(df)} rounds')
    return df


def pad_ylim(ax, curves):
    ycat = np.concatenate(curves)
    pad = max(float(ycat.max() - ycat.min()) * 0.06, 0.4)
    ax.set_ylim(float(ycat.min()) - pad, float(ycat.max()) + pad)


def save(fig, stem: str):
    for ext in ('png', 'pdf'):
        path = HERE / f'{stem}.{ext}'
        fig.savefig(path)
        print(path)
    plt.close(fig)


def add_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc='outside upper center',
        ncol=len(labels),
        frameon=True,
        fancybox=False,
        fontsize=9,
        borderpad=0.45,
        handlelength=2.2,
        columnspacing=1.2,
    )
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(0.8)
    frame.set_facecolor('white')


def plot_accuracy():
    apply_paper_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(3.15 * 2, 3.15),
        sharex=True, sharey=False, layout='constrained',
    )
    for c, (backbone_name, backbone_slug) in enumerate(BACKBONES):
        ax = axes[c]
        curves = []
        for i, (method_slug, label) in enumerate(METHODS):
            df = load_frame(backbone_slug, method_slug)
            y = df['TestAcc'].to_numpy(dtype=float)
            curves.append(y)
            ax.plot(
                df['Round'].to_numpy(), y,
                color=COLORS[i],
                linewidth=1.6,
                label=label if c == 0 else None,
            )
        pad_ylim(ax, curves)
        ax.set_title(f'CIFAR-100 / {backbone_name}', fontsize=10)
        ax.set_xlabel('Communication round')
        if c == 0:
            ax.set_ylabel('Accuracy (%)')
    add_legend(fig, axes[0])
    save(fig, 'fig01_test_accuracy')


def plot_efficiency():
    apply_paper_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(3.15 * 2, 3.15),
        sharex=False, sharey=False, layout='constrained',
    )
    for c, (backbone_name, backbone_slug) in enumerate(BACKBONES):
        ax = axes[c]
        curves = []
        for i, (method_slug, label) in enumerate(METHODS):
            df = load_frame(backbone_slug, method_slug)
            y = df['TestAcc'].to_numpy(dtype=float)
            # 累计本地训练时间，不含聚合和测试。
            x = df['LocalTrainTimeSec'].cumsum().to_numpy(dtype=float) / 60.0
            curves.append(y)
            ax.plot(
                x, y,
                color=COLORS[i],
                linewidth=1.6,
                label=label if c == 0 else None,
            )
        pad_ylim(ax, curves)
        ax.set_title(f'CIFAR-100 / {backbone_name}', fontsize=10)
        ax.set_xlabel('Cumulative local training time (min)')
        if c == 0:
            ax.set_ylabel('Accuracy (%)')
    add_legend(fig, axes[0])
    save(fig, 'fig02_accuracy_vs_local_time')


if __name__ == '__main__':
    plot_accuracy()
    plot_efficiency()
