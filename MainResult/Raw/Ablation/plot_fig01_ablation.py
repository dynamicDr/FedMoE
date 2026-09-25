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
LATEX = HERE.parents[2] / 'latex'

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9']
MARKERS = ['o', '^', 's', 'D', 'v']
METHODS = [
    ('ours', 'Ours'),
    ('fedrolex', 'FedRolex'),
    ('random', 'Random'),
    ('fedadam', 'FedAdam'),
]
# 3×3。fedavg 是图上统一标成 FedAvg 的那条粉线实际读哪个文件。
# black = FedAvg-old-drop25，pink = 原来的 FedAvg。
PANELS = [
    (r'Uplink Budget $B\in[10, 40]$', 'bw10-40', 'fedavg-old-drop25'),
    (r'Uplink Budget $B\in[1, 50]$', 'bw1-50', 'fedavg-old-drop25'),
    (r'Uplink Budget $B\in[20, 80]$', 'bw20-80', 'fedavg'),
    (r'Number of Experts $n_e=4$', 'e4', 'fedavg-old-drop25'),
    (r'Number of Experts $n_e=16$', 'e16', 'fedavg'),
    (r'Number of Experts $n_e=32$', 'e32', 'fedavg'),
    (r'Number of Clients $n=20$', 'n20', 'fedavg'),
    (r'Number of Clients $n=30$', 'n30', 'fedavg'),
    (r'Number of Clients $n=50$', 'n50', 'fedavg'),
]


def apply_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 22,
        'legend.fontsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.linewidth': 0.9,
        'axes.grid': True,
        'grid.alpha': 0.28,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.edgecolor': '#9A9A9A',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'lines.linewidth': 1.8,
        'lines.markersize': 6.2,
    })


# 与主图 CIFAR-100 的中心滑动平均窗口一致。
SMOOTH_WINDOW = 15


def rolling_mean(y, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()


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
    # 单格约 4.7×3.27 英寸，和主图 4.7×3.5 的子图长宽比接近。
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(4.7 * n_col, 3.27 * n_row),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(n_row, n_col)

    for i, (title, slug, fedavg_file) in enumerate(PANELS):
        r, c = divmod(i, n_col)
        ax = axes[r, c]
        curves = []
        series = METHODS[:3] + [(fedavg_file, 'FedAvg')] + METHODS[3:]
        for j, (method_slug, label) in enumerate(series):
            x, y = load_curve(slug, method_slug)
            y = rolling_mean(y, SMOOTH_WINDOW)
            curves.append(y)
            mark_at = np.flatnonzero(np.isin(x, np.arange(10, 101, 10)))
            ax.plot(
                x, y,
                color=COLORS[j],
                linewidth=1.6,
                marker=MARKERS[j],
                markevery=mark_at.tolist(),
                markersize=6.4,
                markerfacecolor=COLORS[j],
                markeredgecolor='white',
                markeredgewidth=0.6,
                label=label if i == 0 else None,
            )
        ycat = np.concatenate(curves)
        y0, y1 = float(ycat.min()), float(ycat.max())
        ax.set_xlim(0, 100)
        ax.set_ylim(y0, y1 + max((y1 - y0) * 0.08, 0.6))
        ax.margins(x=0, y=0)
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#7A7A7A')
            spine.set_linewidth(1.15)
        ax.set_title(title, fontsize=22, pad=6)
        if r == n_row - 1:
            ax.set_xlabel('Training Round')
        if c == 0:
            ax.set_ylabel('Accuracy (%)')

    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.07, top=0.86,
        wspace=0.16, hspace=0.26,
    )
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes[0]]
    span_x0, span_x1 = boxes[0].x0, boxes[-1].x1
    center = (span_x0 + span_x1) / 2
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(center, 0.985),
        bbox_transform=fig.transFigure,
        ncol=len(labels),
        frameon=True,
        fancybox=True,
        fontsize=20,
        borderpad=0.4,
        handlelength=2.5,
        handletextpad=0.45,
        columnspacing=1.35,
        labelspacing=0.3,
    )
    frame = legend.get_frame()
    frame.set_boxstyle('Round,pad=0.25,rounding_size=0.8')
    frame.set_edgecolor('#A3A3A3')
    frame.set_linewidth(1.25)
    frame.set_facecolor('white')
    frame.set_mutation_aspect(0.4)

    for ext in ('png', 'pdf'):
        path = HERE / f'Sensitivity.{ext}'
        fig.savefig(path)
        print(path)
    latex_pdf = LATEX / 'Sensitivity.pdf'
    fig.savefig(latex_pdf)
    print(latex_pdf)
    plt.close(fig)


if __name__ == '__main__':
    plot()
