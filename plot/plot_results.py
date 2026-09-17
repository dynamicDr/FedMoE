"""Read one experiment result folder and export paper-ready figures.

Figures are written to <result>/figure/.

Usage:
    python plot/plot_results.py --result results/20260913-013621
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
PURPLE = '#CC79A7'
SKY = '#56B4E9'
GRAY = '#4D4D4D'


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


def parse_args():
    p = argparse.ArgumentParser(description='Plot paper figures from one result folder.')
    p.add_argument('--result', required=True, help='Path to a single result folder.')
    p.add_argument('--out', default=None,
                   help='Figure directory. Default: <result>/figure')
    return p.parse_args()


def load_experiment(result_dir: Path):
    csv_path = result_dir / 'rounds.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'No rounds.csv in {result_dir}')
    df = pd.read_csv(csv_path)
    config, summary = {}, {}
    json_path = result_dir / 'experiment.json'
    if json_path.exists():
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        config = data.get('config') or {}
        summary = data.get('summary') or {}
    return df, config, summary


def _ints(text):
    return [int(x) for x in re.findall(r'-?\d+', str(text))]


def parse_selected_experts(text):
    """Parse '8:0,2;1:2,3' or '0:[1,2,3,4];1:[0,1,2,4]'."""
    pairs = []
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return pairs
    for part in str(text).split(';'):
        part = part.strip()
        if not part or ':' not in part:
            continue
        cid_s, rest = part.split(':', 1)
        cid_s = cid_s.strip().lstrip('cC')
        if not cid_s.isdigit():
            continue
        experts = _ints(rest)
        pairs.append((int(cid_s), experts))
    return pairs


def parse_bandwidth_map(text):
    """Parse '0:40;1:30' into {0: 40, 1: 30}."""
    mapping = {}
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return mapping
    for cid, values in parse_selected_experts(text):
        if values:
            mapping[cid] = values[0]
    return mapping


def expand_uploads(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_bw = 'Bandwidth' in df.columns
    for _, rec in df.iterrows():
        rnd = int(rec['Round'])
        pairs = parse_selected_experts(rec.get('SelectedExperts', ''))
        bw_map = parse_bandwidth_map(rec['Bandwidth']) if has_bw else {}
        for cid, experts in pairs:
            k = len(experts)
            bw = bw_map.get(cid, np.nan)
            if not experts:
                rows.append({
                    'Round': rnd, 'client_id': cid, 'expert_id': np.nan,
                    'k': k, 'bandwidth': bw,
                })
                continue
            for eid in experts:
                rows.append({
                    'Round': rnd, 'client_id': cid, 'expert_id': int(eid),
                    'k': k, 'bandwidth': bw,
                })
    return pd.DataFrame(rows)


def infer_counts(df, uploads, config):
    n_experts = config.get('num_experts')
    n_clients = config.get('num_clients')
    if n_experts is None and not uploads.empty and uploads['expert_id'].notna().any():
        n_experts = int(uploads['expert_id'].max()) + 1
    if n_clients is None:
        if not uploads.empty:
            n_clients = int(uploads['client_id'].max()) + 1
        elif 'SelectedClients' in df.columns and len(df):
            ids = []
            for text in df['SelectedClients']:
                ids.extend(_ints(text))
            n_clients = (max(ids) + 1) if ids else 1
        else:
            n_clients = 1
    return int(n_clients or 1), int(n_experts or 0)


def save_fig(fig, out_dir: Path, name: str, saved: list):
    for ext in ('pdf', 'png'):
        path = out_dir / f'{name}.{ext}'
        fig.savefig(path)
        saved.append(path)
    plt.close(fig)


def panel_label(ax, text):
    ax.text(
        -0.14, 1.08, text, transform=ax.transAxes,
        fontweight='bold', fontsize=11, va='bottom',
    )


def set_count_ylim(ax, values, vmax_hint=None):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    hi = float(np.nanmax(vals))
    if vmax_hint is not None:
        hi = max(hi, float(vmax_hint))
    top = max(hi + 1.0, 1.0)
    ax.set_ylim(0, top)
    ticks = np.arange(0, int(np.floor(top)) + 1)
    if 1 < len(ticks) <= 12:
        ax.set_yticks(ticks)


def plot_test_accuracy(df, out_dir, saved):
    fig, ax = plt.subplots(figsize=(3.5, 2.45), layout='constrained')
    ax.plot(df['Round'], df['TestAcc'], color=BLUE, marker='o', label='Test acc.')
    if 'BestAcc' in df.columns:
        ax.plot(df['Round'], df['BestAcc'], color=ORANGE, linestyle='--',
                marker='s', label='Best so far')
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(df['Round'].min() - 0.2, df['Round'].max() + 0.2)
    ax.legend(frameon=False)
    save_fig(fig, out_dir, 'fig01_test_accuracy', saved)


def plot_train_loss(df, out_dir, saved):
    fig, ax = plt.subplots(figsize=(3.5, 2.45), layout='constrained')
    ax.plot(df['Round'], df['AvgLoss'], color=ORANGE, marker='o')
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Training loss')
    ax.set_xlim(df['Round'].min() - 0.2, df['Round'].max() + 0.2)
    save_fig(fig, out_dir, 'fig02_train_loss', saved)


def plot_acc_and_loss(df, out_dir, saved):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), layout='constrained')
    axes[0].plot(df['Round'], df['TestAcc'], color=BLUE, marker='o', label='Test acc.')
    if 'BestAcc' in df.columns:
        axes[0].plot(df['Round'], df['BestAcc'], color=ORANGE, linestyle='--',
                     marker='s', label='Best so far')
    axes[0].set_xlabel('Communication round')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(frameon=False)
    panel_label(axes[0], '(a)')

    axes[1].plot(df['Round'], df['AvgLoss'], color=GREEN, marker='o')
    axes[1].set_xlabel('Communication round')
    axes[1].set_ylabel('Training loss')
    panel_label(axes[1], '(b)')
    for ax in axes:
        ax.set_xlim(df['Round'].min() - 0.2, df['Round'].max() + 0.2)
    save_fig(fig, out_dir, 'fig03_acc_and_loss', saved)


def round_comm_stats(df, uploads):
    stats = df[['Round']].copy()
    if uploads.empty:
        stats['mean_k'] = np.nan
        stats['mean_bw'] = np.nan
        stats['n_uploads'] = 0
        return stats
    per_client = (
        uploads.groupby(['Round', 'client_id'], as_index=False)
        .agg(k=('k', 'first'), bandwidth=('bandwidth', 'first'))
    )
    grouped = per_client.groupby('Round', as_index=False).agg(
        mean_k=('k', 'mean'),
        mean_bw=('bandwidth', 'mean'),
        min_bw=('bandwidth', 'min'),
        max_bw=('bandwidth', 'max'),
        n_clients=('client_id', 'nunique'),
    )
    n_up = (
        uploads.dropna(subset=['expert_id'])
        .groupby('Round').size()
        .rename('n_uploads')
        .reset_index()
    )
    stats = stats.merge(grouped, on='Round', how='left').merge(n_up, on='Round', how='left')
    stats['n_uploads'] = stats['n_uploads'].fillna(0)
    return stats


def plot_bandwidth_and_k(stats, out_dir, saved):
    has_bw = stats['mean_bw'].notna().any() if 'mean_bw' in stats.columns else False
    nrows = 2 if has_bw else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(3.5, 2.2 * nrows + 0.2),
                             sharex=True, layout='constrained')
    if nrows == 1:
        axes = [axes]
    idx = 0
    if has_bw:
        ax = axes[idx]
        ax.plot(stats['Round'], stats['mean_bw'], color=PURPLE, marker='o', label='Mean bandwidth')
        if stats['min_bw'].notna().any() and not np.allclose(stats['min_bw'].fillna(0), stats['max_bw'].fillna(0)):
            ax.fill_between(
                stats['Round'], stats['min_bw'], stats['max_bw'],
                color=PURPLE, alpha=0.18, linewidth=0, label='Min–max',
            )
        ax.set_ylabel('Bandwidth')
        ax.legend(frameon=False)
        panel_label(ax, '(a)')
        idx += 1
    ax = axes[idx]
    ax.plot(stats['Round'], stats['mean_k'], color=BLUE, marker='o')
    ax.set_ylabel('Uploaded experts $k$')
    ax.set_xlabel('Communication round')
    set_count_ylim(ax, stats['mean_k'])
    if has_bw:
        panel_label(ax, '(b)')
    save_fig(fig, out_dir, 'fig04_bandwidth_and_k', saved)


def plot_expert_round_heatmap(uploads, n_experts, rounds, out_dir, saved):
    if uploads.empty or n_experts <= 0:
        return
    valid = uploads.dropna(subset=['expert_id'])
    mat = np.zeros((len(rounds), n_experts), dtype=float)
    round_index = {int(r): i for i, r in enumerate(rounds)}
    for (rnd, eid), cnt in valid.groupby(['Round', 'expert_id']).size().items():
        if int(eid) < n_experts and int(rnd) in round_index:
            mat[round_index[int(rnd)], int(eid)] = cnt
    fig, ax = plt.subplots(figsize=(3.8, 2.7), layout='constrained')
    im = ax.imshow(mat, aspect='auto', cmap='YlGnBu', origin='lower')
    ax.set_xticks(range(n_experts), [str(i) for i in range(n_experts)])
    if len(rounds) <= 16:
        yticks = np.arange(len(rounds))
    else:
        step = max(1, int(np.ceil(len(rounds) / 8)))
        yticks = np.arange(0, len(rounds), step)
    ax.set_yticks(yticks, [str(rounds[i]) for i in yticks])
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Communication round')
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Upload count')
    save_fig(fig, out_dir, 'fig05_expert_round_heatmap', saved)


def plot_client_expert_heatmap(uploads, n_clients, n_experts, out_dir, saved):
    if uploads.empty or n_experts <= 0:
        return
    valid = uploads.dropna(subset=['expert_id'])
    mat = np.zeros((n_clients, n_experts), dtype=float)
    for (cid, eid), cnt in valid.groupby(['client_id', 'expert_id']).size().items():
        if int(cid) < n_clients and int(eid) < n_experts:
            mat[int(cid), int(eid)] = cnt
    fig_w = min(6.4, 2.6 + 0.28 * n_clients)
    fig_h = min(4.6, 2.2 + 0.18 * n_clients)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout='constrained')
    im = ax.imshow(mat, aspect='auto', cmap='YlGnBu', origin='upper')
    ax.set_xticks(range(n_experts), [str(i) for i in range(n_experts)])
    ax.set_yticks(range(n_clients), [str(i) for i in range(n_clients)])
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Client ID')
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Upload count')
    save_fig(fig, out_dir, 'fig06_client_expert_heatmap', saved)


def plot_expert_frequency(uploads, n_experts, out_dir, saved):
    if uploads.empty or n_experts <= 0:
        return
    valid = uploads.dropna(subset=['expert_id'])
    counts = valid.groupby('expert_id').size().reindex(range(n_experts), fill_value=0)
    fig, ax = plt.subplots(figsize=(3.5, 2.45), layout='constrained')
    ax.bar(counts.index.astype(int), counts.values, color=SKY, edgecolor='white', width=0.72)
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Times uploaded')
    ax.set_xticks(range(n_experts))
    save_fig(fig, out_dir, 'fig07_expert_frequency', saved)


def plot_time_breakdown(df, out_dir, saved):
    cols = [c for c in ('LocalTrainTimeSec', 'AggTimeSec', 'EvalTimeSec') if c in df.columns]
    if not cols:
        return
    labels = {
        'LocalTrainTimeSec': 'Local training',
        'AggTimeSec': 'Aggregation',
        'EvalTimeSec': 'Evaluation',
    }
    colors = [BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(3.6, 2.65), layout='constrained')
    x = df['Round'].to_numpy()
    bottoms = np.zeros(len(df), dtype=float)
    for col, color in zip(cols, colors):
        vals = df[col].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottoms, color=color, width=0.72,
               label=labels[col], edgecolor='white', linewidth=0.4)
        bottoms += vals
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Time (s)')
    ax.legend(loc='upper left', frameon=True, framealpha=0.92,
              edgecolor='#dddddd', fancybox=False, fontsize=8)
    save_fig(fig, out_dir, 'fig08_time_breakdown', saved)


def plot_acc_vs_communication(df, stats, out_dir, saved):
    if stats is None or 'n_uploads' not in stats.columns:
        return
    merged = df[['Round', 'TestAcc']].merge(stats[['Round', 'n_uploads']], on='Round', how='left')
    merged['n_uploads'] = merged['n_uploads'].fillna(0)
    merged['cum_uploads'] = merged['n_uploads'].cumsum()
    if merged['cum_uploads'].max() <= 0:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.45), layout='constrained')
    ax.plot(merged['cum_uploads'], merged['TestAcc'], color=BLUE, marker='o')
    ax.set_xlabel('Cumulative uploaded experts')
    ax.set_ylabel('Accuracy (%)')
    save_fig(fig, out_dir, 'fig09_accuracy_vs_communication', saved)


def plot_overview(df, stats, uploads, n_experts, out_dir, saved):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), layout='constrained')
    ax = axes[0, 0]
    ax.plot(df['Round'], df['TestAcc'], color=BLUE, marker='o', label='Test acc.')
    if 'BestAcc' in df.columns:
        ax.plot(df['Round'], df['BestAcc'], color=ORANGE, linestyle='--',
                marker='s', label='Best so far')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(frameon=False)
    panel_label(ax, '(a)')

    ax = axes[0, 1]
    ax.plot(df['Round'], df['AvgLoss'], color=GREEN, marker='o')
    ax.set_xlabel('Round')
    ax.set_ylabel('Training loss')
    panel_label(ax, '(b)')

    ax = axes[1, 0]
    if stats is not None and stats['mean_k'].notna().any():
        ax.plot(stats['Round'], stats['mean_k'], color=BLUE, marker='o')
        ax.set_ylabel('Uploaded experts $k$')
        set_count_ylim(ax, stats['mean_k'])
    else:
        ax.text(0.5, 0.5, 'No upload records', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Round')
    panel_label(ax, '(c)')

    ax = axes[1, 1]
    if not uploads.empty and n_experts > 0:
        valid = uploads.dropna(subset=['expert_id'])
        counts = valid.groupby('expert_id').size().reindex(range(n_experts), fill_value=0)
        ax.bar(counts.index.astype(int), counts.values, color=SKY, edgecolor='white', width=0.72)
        ax.set_xticks(range(n_experts))
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Times uploaded')
    panel_label(ax, '(d)')
    save_fig(fig, out_dir, 'fig10_overview', saved)


def write_caption_notes(out_dir: Path, config: dict, summary: dict, df: pd.DataFrame):
    method = config.get('experiment') or config.get('method') or (
        df['Method'].iloc[0] if 'Method' in df.columns and len(df) else 'unknown'
    )
    lines = [
        f"method: {method}",
        f"dataset: {config.get('dataset', '-')}",
        f"clients: {config.get('num_clients', '-')}",
        f"experts: {config.get('num_experts', '-')}",
        f"rounds: {len(df)}",
        f"best_acc: {summary.get('best_acc', df['BestAcc'].max() if 'BestAcc' in df.columns else df['TestAcc'].max())}",
        '',
        'fig01_test_accuracy: test accuracy and best-so-far vs round',
        'fig02_train_loss: local training loss vs round',
        'fig03_acc_and_loss: paper two-panel of (a) accuracy (b) loss',
        'fig04_bandwidth_and_k: available bandwidth and uploaded expert count',
        'fig05_expert_round_heatmap: how often each expert is uploaded each round',
        'fig06_client_expert_heatmap: client-expert upload frequency',
        'fig07_expert_frequency: overall expert upload histogram',
        'fig08_time_breakdown: local / aggregation / evaluation time',
        'fig09_accuracy_vs_communication: accuracy vs cumulative expert uploads',
        'fig10_overview: four-panel summary figure',
    ]
    (out_dir / 'captions.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_experiment(result_dir, out_dir=None, verbose=True):
    """Generate paper figures into <result_dir>/figure. Returns the figure directory."""
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = (Path.cwd() / result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)

    apply_paper_style()
    df, config, summary = load_experiment(result_dir)
    df = df.sort_values('Round').reset_index(drop=True)
    uploads = expand_uploads(df)
    n_clients, n_experts = infer_counts(df, uploads, config)
    stats = round_comm_stats(df, uploads)
    rounds = df['Round'].astype(int).tolist()

    if out_dir is None:
        out_dir = result_dir / 'figure'
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    plot_test_accuracy(df, out_dir, saved)
    plot_train_loss(df, out_dir, saved)
    plot_acc_and_loss(df, out_dir, saved)
    plot_bandwidth_and_k(stats, out_dir, saved)
    plot_expert_round_heatmap(uploads, n_experts, rounds, out_dir, saved)
    plot_client_expert_heatmap(uploads, n_clients, n_experts, out_dir, saved)
    plot_expert_frequency(uploads, n_experts, out_dir, saved)
    plot_time_breakdown(df, out_dir, saved)
    plot_acc_vs_communication(df, stats, out_dir, saved)
    plot_overview(df, stats, uploads, n_experts, out_dir, saved)
    write_caption_notes(out_dir, config, summary, df)

    if verbose:
        print(f'[Plot] result = {result_dir}')
        print(f'[Plot] figures -> {out_dir}')
        for path in saved:
            if path.suffix == '.pdf':
                print(f'  {path.name}')
    return out_dir


def main():
    args = parse_args()
    plot_experiment(args.result, out_dir=args.out)


if __name__ == '__main__':
    main()
