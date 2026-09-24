"""从本目录的 data/ 重画消融总图 fig01_test_accuracy。

不依赖仓库里的其它脚本。在本目录执行：

    python plot_fig01_ablation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9']
METHODS = [
    ('ours', 'Ours'),
    ('fedrolex', 'FedRolex'),
    ('random', 'Random'),
    ('fedadam', 'FedAdam'),
]
# 3×3。fedavg 是图上统一标成 FedAvg 的那条粉线实际读哪个文件。
# black = FedAvg-old-drop25，pink = 原来的 FedAvg。
PANELS = [
    ('bw [10, 40]', 'bw10-40', 'fedavg-old-drop25'),
    ('bw [1, 50]', 'bw1-50', 'fedavg-old-drop25'),
    ('bw [20, 80]', 'bw20-80', 'fedavg'),
    ('4 experts', 'e4', 'fedavg-old-drop25'),
    ('16 experts', 'e16', 'fedavg'),
    ('32 experts', 'e32', 'fedavg'),
    ('20 clients', 'n20', 'fedavg'),
    ('30 clients', 'n30', 'fedavg'),
    ('50 clients', 'n50', 'fedavg'),
]


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


def load_curve(panel_slug: str, method_slug: str):
    path = DATA / panel_slug / f'{method_slug}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path).sort_values('Round')
    return df['Round'].to_numpy(), df['TestAcc'].to_numpy(dtype=float)


def plot():
    apply_paper_style()
    n_col = 3
    n_row = 3
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(3.15 * n_col, 8.6),
        sharex=True, layout='constrained',
    )
    axes = np.asarray(axes).reshape(n_row, n_col)

    for i, (title, slug, fedavg_file) in enumerate(PANELS):
        r, c = divmod(i, n_col)
        ax = axes[r, c]
        curves = []
        series = METHODS[:3] + [(fedavg_file, 'FedAvg')] + METHODS[3:]
        for j, (method_slug, label) in enumerate(series):
            x, y = load_curve(slug, method_slug)
            curves.append(y)
            ax.plot(
                x, y,
                color=COLORS[j],
                linewidth=1.4,
                label=label if i == 0 else None,
            )
        ycat = np.concatenate(curves)
        pad = max(float(ycat.max() - ycat.min()) * 0.06, 0.4)
        ax.set_ylim(float(ycat.min()) - pad, float(ycat.max()) + pad)
        ax.set_title(title, fontsize=10)
        if r == n_row - 1:
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
        fontsize=8,
        borderpad=0.4,
        handlelength=2.0,
        columnspacing=0.9,
    )
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(0.8)
    frame.set_facecolor('white')

    for ext in ('png', 'pdf'):
        path = HERE / f'fig01_test_accuracy.{ext}'
        fig.savefig(path)
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    plot()
