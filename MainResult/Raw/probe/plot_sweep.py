"""把 exp11 / exp12 的探测一致率图集中画到本目录。

在仓库根目录执行：

    python MainResult/Raw/probe/plot_sweep.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

OURS = '#0072B2'
CHANNEL = '#222222'

# 名字, 结果目录名, 图上的短说明, 计划轮数
RUNS = [
    ('exp11', 'exp11-c100-resnet-probe-agreement', 'radius x1, K=1-3, batch 64', 100),
    ('s05', 'exp12-c100-resnet-probe-s05', 'radius x0.5, K=1-3, batch 64', 100),
    ('s025', 'exp12-c100-resnet-probe-s025', 'radius x0.25, K=1-3, batch 64', 100),
    ('s01', 'exp12-c100-resnet-probe-s01', 'radius x0.1, K=1-3, batch 64', 100),
    ('s005', 'exp12-c100-resnet-probe-s005', 'radius x0.05, K=1-3, batch 64', 100),
    ('d10', 'exp12-c100-resnet-probe-d10', 'delta=0.1, K=1-3, batch 64', 100),
    ('d01', 'exp12-c100-resnet-probe-d01', 'delta=0.01, K=1-3, batch 64', 100),
    ('d001', 'exp12-c100-resnet-probe-d001', 'delta=0.001, K=1-3, batch 64', 100),
    ('k1-s025', 'exp12-c100-resnet-probe-k1-s025', 'radius x0.25, K=1 only', 100),
    ('k12-s025', 'exp12-c100-resnet-probe-k12-s025', 'radius x0.25, K=1-2', 100),
    ('k12-s01', 'exp12-c100-resnet-probe-k12-s01', 'radius x0.1, K=1-2', 100),
    ('late30-s025', 'exp12-c100-resnet-probe-late30-s025', 'radius x0.25, from round 30', 100),
    ('late40-k12-s01', 'exp12-c100-resnet-probe-late40-k12-s01', 'radius x0.1, K=1-2, from round 40', 100),
    ('b32-s025', 'exp12-c100-resnet-probe-b32-s025', 'radius x0.25, batch 32', 80),
    ('b16-s025', 'exp12-c100-resnet-probe-b16-s025', 'radius x0.25, batch 16', 60),
    ('b16-k12-s01', 'exp12-c100-resnet-probe-b16-k12-s01', 'radius x0.1, K=1-2, batch 16, bw<=30', 60),
    ('b32-k12-s01', 'exp12-c100-resnet-probe-b32-k12-s01', 'radius x0.1, K=1-2, batch 32', 80),
    ('e16-s025', 'exp12-c100-resnet-probe-e16-s025', 'radius x0.25, 16 experts', 80),
    ('e4-k12-s025', 'exp12-c100-resnet-probe-e4-k12-s025', 'radius x0.25, 4 experts, K=1-2', 80),
    ('beta05-s025', 'exp12-c100-resnet-probe-beta05-s025', 'radius x0.25, Dirichlet 0.05', 80),
    ('b32-k12-s015', 'exp12-c100-resnet-probe-b32-k12-s015', 'radius x0.15, K=1-2, batch 32, bw<=40', 80),
]


def latest_dir(name: str) -> Path:
    root = RESULTS / name
    stamps = sorted(p for p in root.iterdir() if p.is_dir())
    if not stamps:
        raise FileNotFoundError(root)
    return stamps[-1]


def apply_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'Nimbus Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 18,
        'legend.fontsize': 16,
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
        'savefig.dpi': 160,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'lines.linewidth': 2.2,
    })


def style_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#7A7A7A')
        spine.set_linewidth(1.15)
    ax.set_axisbelow(True)


def draw(fixed: pd.DataFrame, adaptive: pd.DataFrame, title: str):
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35))
    ax = axes[0]
    ax.plot(
        fixed['ProbeBatches'], fixed['Agreement'] * 100.0,
        color=CHANNEL, marker='o', markersize=8, label='Stop at fixed batches',
    )
    mean_m = float(adaptive['StopBatches'].mean())
    mean_acc = float(adaptive['Agree'].mean()) * 100.0
    ax.scatter([mean_m], [mean_acc], s=160, color=OURS, zorder=5, label='Single-swap stop')
    ax.axvline(mean_m, color=OURS, linestyle='--', linewidth=1.4)
    ax.set_xticks(list(fixed['ProbeBatches']))
    ax.set_xlim(0.5, 17)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Probe mini-batches')
    ax.set_ylabel('Agreement (%)')
    ax.set_title(title, pad=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right', fontsize=14)
    style_spines(ax)

    ax = axes[1]
    bins = np.arange(0.5, 17.5, 1.0)
    ax.hist(adaptive['StopBatches'], bins=bins, color=OURS, edgecolor='black', linewidth=0.6)
    ax.axvline(mean_m, color=CHANNEL, linestyle='--', linewidth=1.6)
    ax.set_xlim(0.5, 16.5)
    ax.text(
        min(mean_m + 0.35, 12.2), ax.get_ylim()[1] * 0.92,
        f'mean {mean_m:.1f}',
        fontsize=16, color=CHANNEL, va='top',
    )
    ax.set_xlabel('Batches until stop')
    ax.set_ylabel('Rounds')
    ax.set_title('Single-swap stopping', pad=8)
    style_spines(ax)
    fig.tight_layout()
    return fig, mean_m, mean_acc


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    data_root = HERE / 'data'
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir()
    notes = []
    images = []
    for tag, dirname, spec, planned in RUNS:
        src = latest_dir(dirname)
        fixed = pd.read_csv(src / 'data' / 'probe_fixed.csv')
        adaptive = pd.read_csv(src / 'data' / 'probe_adaptive.csv')
        detail = pd.read_csv(src / 'probe_detail.csv')
        last_round = int(detail['Round'].max()) if len(detail) else 0
        done = last_round >= planned
        title = spec if done else f'{spec} (round {last_round}/{planned})'
        fig, mean_m, mean_acc = draw(fixed, adaptive, title)
        for ext in ('png', 'pdf'):
            path = HERE / f'fig04_{tag}.{ext}'
            fig.savefig(path)
        plt.close(fig)
        images.append(HERE / f'fig04_{tag}.png')
        dest = data_root / tag
        dest.mkdir()
        fixed.to_csv(dest / 'probe_fixed.csv', index=False)
        adaptive.to_csv(dest / 'probe_adaptive.csv', index=False)
        agree = {
            int(row.ProbeBatches): float(row.Agreement)
            for row in fixed.itertuples()
        }
        cert = float((detail['Certified'] == 1).mean()) if 'Certified' in detail.columns else float('nan')
        notes.append(
            f'{tag}\t{spec}\tround {last_round}/{planned}\tn={len(adaptive)}\t'
            f'stop {mean_m:.2f}\tagree {mean_acc:.1f}%\t'
            f'fixed '
            + '/'.join(f'{agree.get(m, float("nan")) * 100:.1f}' for m in (1, 2, 4, 8, 16))
            + f'\tcertified {cert * 100:.1f}%\t{src}'
        )
        print(f'{tag}: stop {mean_m:.2f}, agree {mean_acc:.1f}%, round {last_round}/{planned}')

    # 一页总览，方便并排看。
    apply_paper_style()
    ncol = 3
    nrow = int(np.ceil(len(images) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.6, nrow * 2.45))
    for ax, image, (tag, _dirname, spec, planned) in zip(axes.ravel(), images, RUNS):
        ax.imshow(plt.imread(image))
        ax.set_title(tag, fontsize=11, pad=2)
        ax.axis('off')
    for ax in axes.ravel()[len(images):]:
        ax.axis('off')
    fig.tight_layout()
    overview = HERE / 'overview.png'
    fig.savefig(overview, dpi=120)
    plt.close(fig)
    print(overview)

    text = HERE / '说明.txt'
    text.write_text(
        'CIFAR-100 / ResNet 探测一致率。每张图左轴是固定批次数与全量解的一致率，'
        '蓝点是单交换早停的平均批次数和一致率。\n'
        'fixed 五列依次是 1/2/4/8/16 个 mini-batch 的一致率（%）。\n\n'
        + '\n'.join(notes)
        + '\n',
        encoding='utf-8',
    )
    print(text)


if __name__ == '__main__':
    main()
