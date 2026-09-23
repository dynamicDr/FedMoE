"""主实验 fig01 的平滑系数对照。

每个数据集一张图：行是骨干（ResNet / VGG11），列是不同的中心滑动平均窗口。
浅色线是原始 TestAcc，实线是平滑后的曲线。窗口为 1 时只画原始曲线。

看图选定窗口后，改下面 DATASETS 里对应数据集的 windows，再运行：

    python plot/plot_main_smooth.py

图写到 MainResult/smooth_pick/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plot.plot_multi import PALETTE, add_run_legend, load_runs
from plot.plot_results import apply_paper_style

ROOT = _ROOT
OUT_DIR = ROOT / 'MainResult' / 'smooth_pick'

EXP01_METHODS = [
    ('ours', 'Ours'),
    ('fedrolex', 'FedRolex'),
    ('random', 'Random'),
    ('zerofill', 'FedAvg'),
    ('fedadam-tau1e2', 'FedAdam'),
]
EXP02_METHODS = [
    ('ours', 'Ours'),
    ('fedrolex', 'FedRolex'),
    ('random', 'Random'),
    ('fedavg', 'FedAvg'),
    ('fedadam', 'FedAdam'),
]

# 每个数据集单独一组候选窗口（中心滑动平均，1 = 不平滑）。
# 抖动大的数据集窗口偏大，已经比较稳的数据集窗口偏小。
DATASETS = [
    {
        'name': 'CIFAR-10',
        'slug': 'cifar10',
        'windows': [1, 5, 11, 21, 31],
        'prefix': 'exp01',
        'methods': EXP01_METHODS,
        'backbones': [('ResNet', 'c10', 'resnet'), ('VGG11', 'c10', 'vgg11')],
    },
    {
        'name': 'CIFAR-100',
        'slug': 'cifar100',
        'windows': [1, 3, 5, 9, 15],
        'prefix': 'exp01',
        'methods': EXP01_METHODS,
        'backbones': [('ResNet', 'c100', 'resnet'), ('VGG11', 'c100', 'vgg11')],
    },
    {
        'name': 'STL-10',
        'slug': 'stl10',
        'windows': [1, 11, 21, 31, 41],
        'prefix': 'exp01',
        'methods': EXP01_METHODS,
        'backbones': [('ResNet', 'stl10', 'resnet'), ('VGG11', 'stl10', 'vgg11')],
    },
    {
        'name': 'TinyImageNet',
        'slug': 'tinyimagenet',
        'windows': [1, 3, 5, 9, 15],
        'prefix': 'exp02',
        'methods': EXP02_METHODS,
        'backbones': [('ResNet', 'tiny', 'resnet'), ('VGG11', 'tiny', 'vgg11')],
    },
    {
        'name': 'CINIC-10',
        'slug': 'cinic10',
        'windows': [1, 5, 9, 15, 21],
        'prefix': 'exp02',
        'methods': EXP02_METHODS,
        'backbones': [('ResNet', 'cinic', 'resnet'), ('VGG11', 'cinic', 'vgg11')],
    },
]


def latest_run(exp_dir: Path) -> Path:
    runs = [p for p in exp_dir.iterdir() if p.is_dir() and (p / 'experiment.json').exists()]
    if not runs:
        raise FileNotFoundError(exp_dir)

    def key(p):
        data = json.loads((p / 'experiment.json').read_text())
        return data.get('summary', {}).get('end_time') or p.name

    return max(runs, key=key)


def rolling_mean(y, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()


def load_backbone(prefix, dtag, backbone, methods):
    dirs, labels = [], []
    for mtag, label in methods:
        exp_dir = ROOT / 'results' / f'{prefix}-{dtag}-{backbone}-e8-{mtag}'
        run = latest_run(exp_dir)
        dirs.append(str(run))
        labels.append(label)
    return load_runs(dirs, labels=labels)


def row_ylim(runs):
    lo, hi = None, None
    for run in runs:
        y = run.df['TestAcc'].to_numpy(dtype=float)
        lo = float(np.min(y)) if lo is None else min(lo, float(np.min(y)))
        hi = float(np.max(y)) if hi is None else max(hi, float(np.max(y)))
    pad = max((hi - lo) * 0.04, 0.4)
    return lo - pad, hi + pad


def plot_dataset(spec):
    windows = list(spec['windows'])
    n_col = len(windows)
    loaded = []
    for backbone_name, dtag, backbone in spec['backbones']:
        runs = load_backbone(spec['prefix'], dtag, backbone, spec['methods'])
        loaded.append((backbone_name, runs, row_ylim(runs)))

    fig_w = 2.55 * n_col + 0.6
    fig_h = 2.35 * len(loaded) + 0.85
    fig, axes = plt.subplots(
        len(loaded), n_col, figsize=(fig_w, fig_h),
        sharex=True, sharey='row', layout='constrained',
    )
    axes = np.asarray(axes).reshape(len(loaded), n_col)

    for r, (backbone_name, runs, ylim) in enumerate(loaded):
        for c, window in enumerate(windows):
            ax = axes[r, c]
            for i, run in enumerate(runs):
                x = run.df['Round'].to_numpy()
                y = run.df['TestAcc'].to_numpy(dtype=float)
                color = PALETTE[i % len(PALETTE)]
                if window > 1:
                    ax.plot(x, y, color=color, alpha=0.22, linewidth=0.9)
                ax.plot(
                    x, rolling_mean(y, window),
                    color=color, linewidth=1.5,
                    label=run.label if r == 0 and c == 0 else None,
                )
            title = 'raw' if window <= 1 else f'window = {window}'
            ax.set_title(title, fontsize=10)
            ax.set_ylim(*ylim)
            if c == 0:
                ax.set_ylabel(f'{backbone_name}\nAccuracy (%)')
            if r == len(loaded) - 1:
                ax.set_xlabel('Communication round')

    fig.suptitle(spec['name'], fontsize=13)
    add_run_legend(fig, len(spec['methods']), axes[0, 0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ('png', 'pdf'):
        path = OUT_DIR / f"fig01_{spec['slug']}_smooth.{ext}"
        fig.savefig(path)
        saved.append(path)
    plt.close(fig)
    print(f"{spec['name']}: windows={windows}")
    for path in saved:
        print(f'  {path}')


def plot_max_grid():
    """2×5：每个数据集用 windows 里的最大窗口，上排 ResNet，下排 VGG11。"""
    n_col = len(DATASETS)
    fig, axes = plt.subplots(
        2, n_col, figsize=(3.15 * n_col, 5.8),
        sharex=True, layout='constrained',
    )
    axes = np.asarray(axes).reshape(2, n_col)

    for c, spec in enumerate(DATASETS):
        window = max(spec['windows'])
        for r, (backbone_name, dtag, backbone) in enumerate(spec['backbones']):
            runs = load_backbone(spec['prefix'], dtag, backbone, spec['methods'])
            ax = axes[r, c]
            smoothed = []
            for i, run in enumerate(runs):
                x = run.df['Round'].to_numpy()
                y = rolling_mean(run.df['TestAcc'].to_numpy(dtype=float), window)
                smoothed.append(y)
                ax.plot(
                    x, y,
                    color=PALETTE[i % len(PALETTE)],
                    linewidth=1.6,
                    label=run.label if r == 0 and c == 0 else None,
                )
            ycat = np.concatenate(smoothed)
            pad = max(float(ycat.max() - ycat.min()) * 0.06, 0.4)
            ax.set_ylim(float(ycat.min()) - pad, float(ycat.max()) + pad)
            ax.set_title(f'{spec["name"]} / {backbone_name}', fontsize=10)
            if r == 1:
                ax.set_xlabel('Communication round')
            if c == 0:
                ax.set_ylabel('Accuracy (%)')
            print(f'  {spec["name"]} / {backbone_name}: window={window}')

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = OUT_DIR / f'fig01_max_smooth.{ext}'
        fig.savefig(path)
        print(f'  {path}')
    plt.close(fig)


def write_note():
    lines = [
        '主实验 fig01 平滑系数对照',
        '========================',
        '',
        '每张 fig01_<dataset>_smooth 图是一个数据集。行是 ResNet / VGG11，列是不同的平滑窗口。',
        '平滑方式：中心滑动平均。',
        '浅色线是原始测试精度，实线是平滑后的曲线。window = 1 的列是未平滑的原始曲线。',
        '',
        'fig01_max_smooth 是 2×5 总图。每个数据集用上面候选窗口里的最大值。',
        '上排 ResNet，下排 VGG11。只画平滑后的实线。图例在上方，带边框，横排。',
        '图例：Ours, FedRolex, Random, FedAvg, FedAdam。',
        '',
        '当前候选窗口（总图取每行最后一个）',
        '--------------------------------',
    ]
    for spec in DATASETS:
        wins = ', '.join(str(w) for w in spec['windows'])
        lines.append(f"  {spec['name']:<14}  {wins}  ->  {max(spec['windows'])}")
    lines.append('')
    path = OUT_DIR / '说明.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {path}')


def main():
    apply_paper_style()
    for spec in DATASETS:
        plot_dataset(spec)
    print('===== max smooth 2x5 =====')
    plot_max_grid()
    write_note()


if __name__ == '__main__':
    main()
