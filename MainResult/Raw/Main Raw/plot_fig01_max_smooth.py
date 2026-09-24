"""从本目录的 data/ 重画 fig01_max_smooth。

不依赖仓库里的其它脚本。在本目录执行：

    python plot_fig01_max_smooth.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
SMOOTH_PICK = HERE.parent / 'smooth_pick'

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9']
METHODS = [
    ('ours', 'Ours'),
    ('fedrolex', 'FedRolex'),
    ('random', 'Random'),
    ('fedavg', 'FedAvg'),
    ('fedadam', 'FedAdam'),
]
# 列顺序，以及该数据集使用的中心滑动平均窗口。
DATASETS = [
    ('CIFAR-10', 'cifar10', 31),
    ('CIFAR-100', 'cifar100', 15),
    ('STL-10', 'stl10', 41),
    ('TinyImageNet', 'tinyimagenet', 15),
    ('CINIC-10', 'cinic10', 21),
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
        'lines.linewidth': 1.8,
        'lines.markersize': 4.5,
    })


def rolling_mean(y, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()


def load_curve(dataset_slug: str, backbone_slug: str, method_slug: str):
    path = DATA / dataset_slug / backbone_slug / f'{method_slug}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).sort_values('Round')
    return df['Round'].to_numpy(), df['TestAcc'].to_numpy(dtype=float)


def plot():
    apply_paper_style()
    n_col = len(DATASETS)
    fig, axes = plt.subplots(
        2, n_col, figsize=(3.15 * n_col, 5.8),
        sharex=True, layout='constrained',
    )
    axes = np.asarray(axes).reshape(2, n_col)

    for c, (title, dataset_slug, window) in enumerate(DATASETS):
        for r, (backbone_name, backbone_slug) in enumerate(BACKBONES):
            ax = axes[r, c]
            smoothed = []
            for i, (method_slug, label) in enumerate(METHODS):
                x, y = load_curve(dataset_slug, backbone_slug, method_slug)
                y = rolling_mean(y, window)
                smoothed.append(y)
                ax.plot(
                    x, y,
                    color=COLORS[i],
                    linewidth=1.6,
                    label=label if r == 0 and c == 0 else None,
                )
            ycat = np.concatenate(smoothed)
            pad = max(float(ycat.max() - ycat.min()) * 0.06, 0.4)
            ax.set_ylim(float(ycat.min()) - pad, float(ycat.max()) + pad)
            ax.set_title(f'{title} / {backbone_name}', fontsize=10)
            if r == 1:
                ax.set_xlabel('Communication round')
            if c == 0:
                ax.set_ylabel('Accuracy (%)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
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

    out_dirs = [HERE, SMOOTH_PICK]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'pdf'):
            path = out_dir / f'fig01_max_smooth.{ext}'
            fig.savefig(path)
            print(path)
    plt.close(fig)


if __name__ == '__main__':
    plot()
