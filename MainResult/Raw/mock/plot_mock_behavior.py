"""用模拟数据预览五张行为图。数字不是实验结果。

在本目录执行：

    python plot_mock_behavior.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'

# 与主图、效率图一致的配色。
OURS = '#0072B2'
FEDROLEX = '#D55E00'
RANDOM = '#009E73'
POSTHOC = '#D55E00'
CHANNEL = '#222222'

N_CLIENTS = 10
N_EXPERTS = 8
N_ROUNDS = 100
RNG = np.random.default_rng(42)


def apply_paper_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 22,
        'legend.fontsize': 18,
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
        'lines.linewidth': 2.2,
    })


def style_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#7A7A7A')
        spine.set_linewidth(1.15)
    ax.set_axisbelow(True)


def save(fig, stem: str):
    for ext in ('png', 'pdf'):
        path = HERE / f'{stem}.{ext}'
        fig.savefig(path)
        print(path)
    plt.close(fig)


def sample_budget(rng, size) -> np.ndarray:
    """单位功率信道落在预算中点 45，再截到 [10, 80]。每个专家代价 10。"""
    power = rng.exponential(45.0, size=size)
    budget = np.clip(power, 10.0, 80.0)
    k = np.floor(budget / 10.0).astype(int)
    return np.clip(k, 1, N_EXPERTS)


def client_gates(rng) -> np.ndarray:
    """每个客户端两座峰，峰的位置互不相同。和为 1。"""
    gates = np.zeros((N_CLIENTS, N_EXPERTS))
    for i in range(N_CLIENTS):
        peaks = rng.choice(N_EXPERTS, size=2, replace=False)
        logits = rng.normal(0.0, 0.12, size=N_EXPERTS)
        logits[peaks[0]] += 1.85
        logits[peaks[1]] += 1.25
        weights = np.exp(logits)
        gates[i] = weights / weights.sum()
    return gates


def select_ours(gate: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(gate)[-k:]


def select_random(rng, k: int) -> np.ndarray:
    return rng.choice(N_EXPERTS, size=k, replace=False)


def select_fedrolex(start: int, k: int) -> np.ndarray:
    return np.arange(start, start + k) % N_EXPERTS


def coverage(gate: np.ndarray, chosen: np.ndarray) -> float:
    return float(gate[chosen].sum())


def simulate_rounds():
    rng = np.random.default_rng(42)
    gates = client_gates(rng)
    # 每个客户端、每一轮独立抽一次信道。
    k = sample_budget(rng, (N_ROUNDS, N_CLIENTS))
    cov = {name: np.zeros(N_ROUNDS) for name in ('ours', 'fedrolex', 'random')}
    uploads = {name: np.zeros((N_CLIENTS, N_EXPERTS), dtype=int) for name in ('ours', 'random', 'fedrolex')}
    rows = []
    for r in range(N_ROUNDS):
        acc = {name: [] for name in cov}
        for i in range(N_CLIENTS):
            kk = int(k[r, i])
            chosen = {
                'ours': select_ours(gates[i], kk),
                'random': select_random(rng, kk),
                'fedrolex': select_fedrolex((r + i) % N_EXPERTS, kk),
            }
            for name, idx in chosen.items():
                acc[name].append(coverage(gates[i], idx))
                uploads[name][i, idx] += 1
            rows.append({
                'Round': r + 1,
                'Client': i + 1,
                'BudgetK': kk,
                'CoverageOurs': acc['ours'][-1],
                'CoverageFedRolex': acc['fedrolex'][-1],
                'CoverageRandom': acc['random'][-1],
            })
        for name in cov:
            cov[name][r] = float(np.mean(acc[name]))
    mean_k = k.mean(axis=1)
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame({
        'Round': np.arange(1, N_ROUNDS + 1),
        'MeanBudgetK': mean_k,
        'CoverageOurs': cov['ours'],
        'CoverageFedRolex': cov['fedrolex'],
        'CoverageRandom': cov['random'],
    })
    return gates, summary, uploads, frame


def simulate_probe(_gates: np.ndarray):
    """按预期形状造数：固定批次数越多越接近全量解，单交换多在 2–4 批停下。"""
    fixed = pd.DataFrame({
        'ProbeBatches': [1, 2, 4, 8, 16],
        'Agreement': [0.46, 0.71, 0.86, 0.93, 0.96],
    })
    # 400 轮紧预算上的停止批次数，均值约 3.2。
    counts = {1: 18, 2: 96, 3: 124, 4: 86, 5: 40, 6: 20, 8: 10, 12: 6}
    stop_ms = []
    for m, n in counts.items():
        stop_ms.extend([m] * n)
    adaptive = pd.DataFrame({
        'StopBatches': stop_ms,
        'Agree': np.ones(len(stop_ms), dtype=float),
    })
    # 停得越早越容易和全量解不一致，整体一致率约 0.88。
    rng = np.random.default_rng(7)
    early = adaptive['StopBatches'] <= 2
    adaptive.loc[early, 'Agree'] = (rng.random(int(early.sum())) < 0.62).astype(float)
    mid = (adaptive['StopBatches'] >= 3) & (adaptive['StopBatches'] <= 4)
    adaptive.loc[mid, 'Agree'] = (rng.random(int(mid.sum())) < 0.90).astype(float)
    late = adaptive['StopBatches'] >= 5
    adaptive.loc[late, 'Agree'] = (rng.random(int(late.sum())) < 0.96).astype(float)
    print(
        'probe fixed', fixed['Agreement'].tolist(),
        'adaptive mean m', float(adaptive['StopBatches'].mean()),
        'adaptive agreement', float(adaptive['Agree'].mean()),
    )
    return fixed, adaptive


def simulate_time():
    """先选再训的耗时随可上传个数上升；事后上传几乎不随预算变。"""
    k = np.arange(1, N_EXPERTS + 1)
    ours = 30.0 + 5.3 * (k - 1)
    posthoc = np.full_like(ours, 67.5, dtype=float)
    frame = pd.DataFrame({
        'BudgetK': k,
        'OursSec': ours,
        'PosthocSec': posthoc,
    })
    return frame


def plot_coverage(summary: pd.DataFrame):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.15))
    rounds = summary['Round'].to_numpy()
    series = [
        ('CoverageOurs', 'Ours', OURS),
        ('CoverageFedRolex', 'FedRolex', FEDROLEX),
        ('CoverageRandom', 'Random', RANDOM),
    ]
    for col, label, color in series:
        ax.plot(rounds, summary[col].to_numpy() * 100.0, color=color, label=label)
    ax.set_xlim(1, N_ROUNDS)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Gate mass retained (%)')
    ax.set_title('CIFAR-100 / ResNet-18', pad=8)
    style_spines(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    fig.tight_layout()
    save(fig, 'fig01_gate_coverage')


def plot_heatmap(uploads: dict):
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    panels = [
        ('ours', 'Ours'),
        ('random', 'Random'),
    ]
    vmax = max(int(uploads[name].max()) for name, _label in panels)
    image = None
    for ax, (name, label) in zip(axes, panels):
        image = ax.imshow(
            uploads[name],
            cmap='Blues',
            vmin=0,
            vmax=vmax,
            aspect='auto',
            interpolation='nearest',
        )
        ax.set_title(label, pad=8)
        ax.set_xlabel('Expert')
        ax.set_xticks(np.arange(N_EXPERTS), [str(j + 1) for j in range(N_EXPERTS)])
        ax.set_yticks(np.arange(N_CLIENTS), [str(i + 1) for i in range(N_CLIENTS)])
        ax.set_ylabel('Client' if name == 'ours' else '')
        ax.grid(False)
        ax.tick_params(length=0)
        style_spines(ax)
        for i in range(N_CLIENTS):
            for j in range(N_EXPERTS):
                value = int(uploads[name][i, j])
                color = 'white' if value > 0.62 * vmax else '#1A1A1A'
                ax.text(j, i, str(value), ha='center', va='center', fontsize=12, color=color)
    cbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label('Upload count', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    fig.suptitle('CIFAR-100 / ResNet-18', fontsize=22, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.14, top=0.84, wspace=0.18)
    save(fig, 'fig02_client_expert_heatmap')


def plot_channel(summary: pd.DataFrame):
    apply_paper_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(7.6, 6.3), sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.15], 'hspace': 0.16},
    )
    rounds = summary['Round'].to_numpy()
    k = summary['MeanBudgetK'].to_numpy()

    ax = axes[0]
    ax.fill_between(rounds, 0, k, color='#D0D0D0', step='mid', zorder=1)
    ax.step(rounds, k, where='mid', color=CHANNEL, linewidth=1.8, label='Allowed by channel', zorder=3)
    ax.plot(rounds, k, color=OURS, linewidth=0, marker='o', markersize=3.2, label='Uploaded (Ours)', zorder=4)
    ax.set_ylim(0, 9)
    ax.set_ylabel('Expert count')
    ax.set_title('CIFAR-100 / ResNet-18', pad=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right', fontsize=16)
    style_spines(ax)
    ax.grid(axis='y')

    ax = axes[1]
    series = [
        ('CoverageOurs', 'Ours', OURS),
        ('CoverageFedRolex', 'FedRolex', FEDROLEX),
        ('CoverageRandom', 'Random', RANDOM),
    ]
    for col, label, color in series:
        ax.plot(rounds, summary[col].to_numpy() * 100.0, color=color, label=label)
    ax.set_xlim(1, N_ROUNDS)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Gate mass retained (%)')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right', fontsize=16, ncol=3)
    style_spines(ax)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.94, hspace=0.22)
    save(fig, 'fig03_channel_and_coverage')


def plot_probe(fixed: pd.DataFrame, adaptive: pd.DataFrame):
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15))

    ax = axes[0]
    ax.plot(
        fixed['ProbeBatches'], fixed['Agreement'] * 100.0,
        color=CHANNEL, marker='o', markersize=8, label='Stop at fixed batches',
    )
    mean_m = float(adaptive['StopBatches'].mean())
    mean_acc = float(adaptive['Agree'].mean()) * 100.0
    ax.scatter(
        [mean_m], [mean_acc],
        s=160, color=OURS, zorder=5, label='Single-swap stop',
    )
    ax.axvline(mean_m, color=OURS, linestyle='--', linewidth=1.4)
    ax.set_xticks(list(fixed['ProbeBatches']))
    ax.set_xlim(0.5, 17)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Probe mini-batches')
    ax.set_ylabel('Agreement (%)')
    ax.set_title('Tight budget (1–3 experts)', pad=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right', fontsize=15)
    style_spines(ax)

    ax = axes[1]
    bins = np.arange(0.5, 17.5, 1.0)
    ax.hist(
        adaptive['StopBatches'], bins=bins,
        color=OURS, edgecolor='black', linewidth=0.6,
    )
    ax.axvline(mean_m, color=CHANNEL, linestyle='--', linewidth=1.6)
    ax.text(
        mean_m + 0.35, ax.get_ylim()[1] * 0.92,
        f'mean {mean_m:.1f}',
        fontsize=16, color=CHANNEL, va='top',
    )
    ax.set_xlim(0.5, 16.5)
    ax.set_xlabel('Batches until stop')
    ax.set_ylabel('Rounds')
    ax.set_title('Single-swap stopping', pad=8)
    style_spines(ax)
    fig.tight_layout()
    save(fig, 'fig04_probe_agreement')


def plot_time(frame: pd.DataFrame):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.15))
    ax.plot(
        frame['BudgetK'], frame['PosthocSec'],
        color=POSTHOC, marker='s', markersize=8, label='Post-hoc',
    )
    ax.plot(
        frame['BudgetK'], frame['OursSec'],
        color=OURS, marker='o', markersize=8, label='Ours',
    )
    ax.set_xticks(list(frame['BudgetK']))
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(0, 90)
    ax.set_xlabel('Experts allowed by the channel')
    ax.set_ylabel('Local training time (s)')
    ax.set_title('CIFAR-100 / ResNet-18', pad=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')
    style_spines(ax)
    fig.tight_layout()
    save(fig, 'fig05_time_vs_budget')


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    _gates, summary, uploads, detail = simulate_rounds()
    fixed, adaptive = simulate_probe(_gates)
    timing = simulate_time()

    summary.to_csv(DATA / 'coverage_by_round.csv', index=False)
    detail.to_csv(DATA / 'coverage_by_client_round.csv', index=False)
    for name, matrix in uploads.items():
        pd.DataFrame(
            matrix,
            index=[f'c{i+1}' for i in range(N_CLIENTS)],
            columns=[f'e{j+1}' for j in range(N_EXPERTS)],
        ).to_csv(DATA / f'uploads_{name}.csv')
    fixed.to_csv(DATA / 'probe_fixed.csv', index=False)
    adaptive.to_csv(DATA / 'probe_adaptive.csv', index=False)
    timing.to_csv(DATA / 'time_vs_budget.csv', index=False)

    plot_coverage(summary)
    plot_heatmap(uploads)
    plot_channel(summary)
    plot_probe(fixed, adaptive)
    plot_time(timing)
    print('mean coverage', summary[['CoverageOurs', 'CoverageFedRolex', 'CoverageRandom']].mean().to_dict())
    print('mean budget k', float(summary['MeanBudgetK'].mean()))


if __name__ == '__main__':
    main()
