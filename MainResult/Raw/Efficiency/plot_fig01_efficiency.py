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
LATEX = HERE.parents[2] / 'latex'

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


def save(fig, stem: str, latex_copy: bool = False):
    for ext in ('png', 'pdf'):
        path = HERE / f'{stem}.{ext}'
        fig.savefig(path)
        print(path)
    if latex_copy:
        latex_pdf = LATEX / f'{stem}.pdf'
        fig.savefig(latex_pdf)
        print(latex_pdf)
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


def plot_avg_local_time():
    # 单栏左右：左 ResNet、右 VGG11。字号和灰框跟 Main Raw 的子图一致。
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 22,
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
    })
    fig, axes = plt.subplots(1, 2, figsize=(4.7 * 2, 3.6), sharey=False)
    order = [
        ('random', 'Random', '#009E73'),
        ('ours', 'Ours', '#0072B2'),
        ('posthoc', 'Post-hoc', '#D55E00'),
    ]
    x = np.arange(len(order))
    for r, (backbone_name, backbone_slug) in enumerate(BACKBONES):
        ax = axes[r]
        means = []
        for method_slug, _label, _color in order:
            df = load_frame(backbone_slug, method_slug)
            means.append(float(df['LocalTrainTimeSec'].mean()))
        bars = ax.bar(
            x, means,
            width=0.62,
            color=[color for _slug, _label, color in order],
            edgecolor='black',
            linewidth=0.8,
            zorder=3,
        )
        ymax = max(means)
        ax.set_ylim(0, ymax * 1.22)
        ax.set_xticks(x, [label for _slug, label, _color in order])
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.margins(x=0.08)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#7A7A7A')
            spine.set_linewidth(1.15)
        ax.set_title(f'CIFAR-100 / {backbone_name}', fontsize=22, pad=6)
        ax.set_xlabel('Method')
        if r == 0:
            ax.set_ylabel('Time (s)')
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.025,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=18,
            )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.88, wspace=0.18)
    save(fig, 'Efficiency', latex_copy=True)


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


def plot_expert_upload():
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.15), layout='constrained')
    upload = pd.read_csv(DATA / 'cifar100' / 'resnet' / 'ours-e32_upload.csv')
    bw = pd.read_csv(DATA / 'cifar100' / 'resnet' / 'ours-e32_bw.csv')
    ax.scatter(
        upload['Round'], upload['ExpertId'],
        s=1.5, color=COLORS[0], linewidths=0, zorder=3,
    )
    ax.set_ylim(-0.6, 31.4)
    ax.set_xlim(0.5, 100.5)
    twin = ax.twinx()
    twin.plot(
        bw['Round'], bw['MeanUplinkBandwidth'],
        color='black', linewidth=1.1, zorder=2,
    )
    twin.set_ylim(0, float(bw['MeanUplinkBandwidth'].max()) * 1.15)
    twin.grid(False)
    twin.set_ylabel('Uplink bandwidth')
    ax.set_title('ResNet / Ours', fontsize=10)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Expert id')
    handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555', markersize=4, label='Uploaded expert'),
        plt.Line2D([0], [0], color='black', linewidth=1.1, label='Uplink bandwidth'),
    ]
    legend = fig.legend(
        handles, [h.get_label() for h in handles],
        loc='outside upper center', ncol=2,
        frameon=True, fancybox=False, fontsize=9,
        borderpad=0.45, handlelength=2.2, columnspacing=1.2,
    )
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_linewidth(0.8)
    frame.set_facecolor('white')
    save(fig, 'fig04_expert_upload')


if __name__ == '__main__':
    plot_accuracy()
    plot_efficiency()
    plot_avg_local_time()
    plot_expert_upload()
