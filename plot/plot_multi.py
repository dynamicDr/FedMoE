"""Overlay paper figures from a group of experiment result folders.

Same figure set as plot_results.py; each panel draws one curve (or bar
group) per run.

Usage:
    python plot/plot_multi.py \\
        --result results/20260918-070936-3 results/20260918-083318 \\
        --out plot/multi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plot.plot_results import (
    apply_paper_style,
    expand_uploads,
    infer_counts,
    load_experiment,
    panel_label,
    round_comm_stats,
    save_fig,
    set_count_ylim,
)

PALETTE = [
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9',
    '#E69F00', '#000000', '#882255', '#117733', '#332288',
    '#AA4499', '#44AA99', '#999933', '#661100',
]
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '>', '<', 'p', '8', 'H']
LINESTYLES = ['-', '--', '-.', ':']
BAR = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9',
       '#E69F00', '#888888', '#882255', '#117733', '#332288',
       '#AA4499', '#44AA99', '#999933', '#661100']

DIFF_KEYS = (
    'dataset', 'backbone', 'width_mult', 'method', 'select', 'merge',
    'beta', 'num_clients', 'num_experts', 'seed',
)


class Run:
    def __init__(self, path: Path, label: str, df, config, summary,
                 uploads, stats, n_clients, n_experts):
        self.path = path
        self.label = label
        self.df = df
        self.config = config
        self.summary = summary
        self.uploads = uploads
        self.stats = stats
        self.n_clients = n_clients
        self.n_experts = n_experts


def parse_args():
    p = argparse.ArgumentParser(
        description='Overlay paper figures from several result folders.')
    p.add_argument('--result', nargs='+', required=True,
                   help='One or more experiment result folders.')
    p.add_argument('--label', nargs='+', default=None,
                   help='Legend labels, same order as --result. '
                        'Default: auto from differing config fields.')
    p.add_argument('--out', default='plot/multi',
                   help='Figure directory. Default: plot/multi')
    return p.parse_args()


def style_of(i: int, n_points: int) -> dict:
    every = max(1, n_points // 12) if n_points > 16 else 1
    return dict(
        color=PALETTE[i % len(PALETTE)],
        marker=MARKERS[i % len(MARKERS)],
        linestyle=LINESTYLES[i % len(LINESTYLES)],
        markevery=every,
        linewidth=1.6,
    )


def auto_labels(configs, paths):
    keys = []
    for key in DIFF_KEYS:
        vals = {str(cfg.get(key, '')) for cfg in configs}
        if len(vals) > 1:
            keys.append(key)
    labels = []
    for cfg, path in zip(configs, paths):
        if keys:
            labels.append(' / '.join(str(cfg.get(k, '-')) for k in keys))
        else:
            labels.append(path.name)
    seen = {}
    unique = []
    for label, path in zip(labels, paths):
        n = seen.get(label, 0) + 1
        seen[label] = n
        unique.append(label if n == 1 else f'{label} ({path.name})')
    return unique


def load_run(path: Path) -> tuple:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    df, config, summary = load_experiment(path)
    df = df.sort_values('Round').reset_index(drop=True)
    uploads = expand_uploads(df)
    n_clients, n_experts = infer_counts(df, uploads, config)
    stats = round_comm_stats(df, uploads)
    return path, df, config, summary, uploads, stats, n_clients, n_experts


def load_runs(result_dirs, labels=None) -> list[Run]:
    packed = [load_run(p) for p in result_dirs]
    paths = [item[0] for item in packed]
    configs = [item[2] for item in packed]
    if labels is None:
        labels = auto_labels(configs, paths)
    elif len(labels) != len(packed):
        raise ValueError(
            f'--label count ({len(labels)}) must match --result count ({len(packed)})')
    runs = []
    for label, item in zip(labels, packed):
        path, df, config, summary, uploads, stats, n_clients, n_experts = item
        runs.append(Run(path, label, df, config, summary, uploads,
                        stats, n_clients, n_experts))
    return runs


def add_run_legend(fig, n_runs, ax=None):
    target = ax or fig.axes[0]
    handles, labels = target.get_legend_handles_labels()
    if not handles:
        return
    ncol = 3 if n_runs > 10 else 2 if n_runs > 5 else 1
    fontsize = 7 if n_runs > 8 else 8
    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
        ncol=ncol, frameon=False, fontsize=fontsize, borderaxespad=0.0,
    )


def round_xlim(runs, key='Round'):
    lo, hi = None, None
    for run in runs:
        vals = run.df[key] if key in run.df.columns else run.stats[key]
        lo = vals.min() if lo is None else min(lo, vals.min())
        hi = vals.max() if hi is None else max(hi, vals.max())
    return lo - 0.2, hi + 0.2


def cumulative_time_sec(df):
    if 'CumulativeTimeSec' in df.columns and df['CumulativeTimeSec'].notna().any():
        return df['CumulativeTimeSec'].astype(float)
    if 'RoundTimeSec' in df.columns and df['RoundTimeSec'].notna().any():
        return df['RoundTimeSec'].fillna(0).astype(float).cumsum()
    return None


def time_axis_meta(runs):
    max_sec = 0.0
    any_time = False
    for run in runs:
        t = cumulative_time_sec(run.df)
        if t is None:
            continue
        any_time = True
        max_sec = max(max_sec, float(np.nanmax(t.to_numpy())))
    if not any_time:
        return None
    if max_sec >= 7200:
        return 3600.0, 'Wall-clock time (h)'
    if max_sec >= 180:
        return 60.0, 'Wall-clock time (min)'
    return 1.0, 'Wall-clock time (s)'


def time_xlim(runs, scale):
    lo, hi = None, None
    for run in runs:
        t = cumulative_time_sec(run.df)
        if t is None:
            continue
        vals = t / scale
        lo = float(vals.min()) if lo is None else min(lo, float(vals.min()))
        hi = float(vals.max()) if hi is None else max(hi, float(vals.max()))
    if lo is None:
        return 0.0, 1.0
    pad = max((hi - lo) * 0.02, 0.0)
    return max(0.0, lo - pad), hi + pad


def plot_test_accuracy(runs, out_dir, saved):
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    for i, run in enumerate(runs):
        ax.plot(run.df['Round'], run.df['TestAcc'],
                label=run.label, **style_of(i, len(run.df)))
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(*round_xlim(runs))
    add_run_legend(fig, len(runs), ax)
    save_fig(fig, out_dir, 'fig01_test_accuracy', saved)


def plot_test_accuracy_vs_time(runs, out_dir, saved):
    """Fig 2: test accuracy vs wall-clock time, not communication round."""
    time_meta = time_axis_meta(runs)
    if time_meta is None:
        return
    scale, xlabel = time_meta
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    drawn = 0
    for i, run in enumerate(runs):
        t = cumulative_time_sec(run.df)
        if t is None:
            continue
        ax.plot(t / scale, run.df['TestAcc'],
                label=run.label, **style_of(i, len(run.df)))
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(*time_xlim(runs, scale))
    add_run_legend(fig, len(runs), ax)
    save_fig(fig, out_dir, 'fig02_test_accuracy_vs_time', saved)


def plot_train_loss(runs, out_dir, saved):
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    for i, run in enumerate(runs):
        ax.plot(run.df['Round'], run.df['AvgLoss'],
                label=run.label, **style_of(i, len(run.df)))
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Training loss')
    ax.set_xlim(*round_xlim(runs))
    add_run_legend(fig, len(runs), ax)
    save_fig(fig, out_dir, 'fig02_train_loss', saved)


def plot_acc_and_loss(runs, out_dir, saved):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7), layout='constrained')
    for i, run in enumerate(runs):
        st = style_of(i, len(run.df))
        axes[0].plot(run.df['Round'], run.df['TestAcc'], label=run.label, **st)
        axes[1].plot(run.df['Round'], run.df['AvgLoss'], label=run.label, **st)
    axes[0].set_xlabel('Communication round')
    axes[0].set_ylabel('Accuracy (%)')
    panel_label(axes[0], '(a)')
    axes[1].set_xlabel('Communication round')
    axes[1].set_ylabel('Training loss')
    panel_label(axes[1], '(b)')
    xlim = round_xlim(runs)
    for ax in axes:
        ax.set_xlim(*xlim)
    add_run_legend(fig, len(runs), axes[0])
    save_fig(fig, out_dir, 'fig03_acc_and_loss', saved)


def plot_bandwidth_and_k(runs, out_dir, saved):
    has_bw = any(
        'mean_bw' in run.stats.columns and run.stats['mean_bw'].notna().any()
        for run in runs
    )
    nrows = 2 if has_bw else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(4.6, 2.3 * nrows + 0.35),
                             sharex=True, layout='constrained')
    if nrows == 1:
        axes = [axes]
    idx = 0
    if has_bw:
        ax = axes[idx]
        for i, run in enumerate(runs):
            if 'mean_bw' not in run.stats.columns:
                continue
            ax.plot(run.stats['Round'], run.stats['mean_bw'],
                    label=run.label, **style_of(i, len(run.stats)))
        ax.set_ylabel('Bandwidth')
        panel_label(ax, '(a)')
        idx += 1
    ax = axes[idx]
    k_vals = []
    for i, run in enumerate(runs):
        ax.plot(run.stats['Round'], run.stats['mean_k'],
                label=run.label, **style_of(i, len(run.stats)))
        k_vals.extend(run.stats['mean_k'].dropna().tolist())
    ax.set_ylabel('Uploaded experts $k$')
    ax.set_xlabel('Communication round')
    set_count_ylim(ax, k_vals)
    if has_bw:
        panel_label(ax, '(b)')
    axes[0].set_xlim(*round_xlim(runs))
    add_run_legend(fig, len(runs), axes[0])
    save_fig(fig, out_dir, 'fig04_bandwidth_and_k', saved)


def plot_uploads_per_round(runs, out_dir, saved):
    """Curve version of the expert-round heatmap: total uploads vs round."""
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    y_vals = []
    for i, run in enumerate(runs):
        if 'n_uploads' not in run.stats.columns:
            continue
        ax.plot(run.stats['Round'], run.stats['n_uploads'],
                label=run.label, **style_of(i, len(run.stats)))
        y_vals.extend(run.stats['n_uploads'].dropna().tolist())
    ax.set_xlabel('Communication round')
    ax.set_ylabel('Uploaded experts')
    ax.set_xlim(*round_xlim(runs))
    if y_vals:
        set_count_ylim(ax, y_vals)
    add_run_legend(fig, len(runs), ax)
    save_fig(fig, out_dir, 'fig05_expert_round_heatmap', saved)


def plot_client_uploads(runs, out_dir, saved):
    """Curve version of the client-expert heatmap: uploads per client."""
    n_clients = max(run.n_clients for run in runs)
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    x = np.arange(n_clients)
    y_vals = []
    for i, run in enumerate(runs):
        counts = np.zeros(n_clients, dtype=float)
        if not run.uploads.empty:
            valid = run.uploads.dropna(subset=['expert_id'])
            for cid, cnt in valid.groupby('client_id').size().items():
                if int(cid) < n_clients:
                    counts[int(cid)] = cnt
        ax.plot(x, counts, label=run.label, **style_of(i, n_clients))
        y_vals.extend(counts.tolist())
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Times uploaded')
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlim(-0.3, n_clients - 0.7)
    if y_vals:
        set_count_ylim(ax, y_vals)
    add_run_legend(fig, len(runs), ax)
    save_fig(fig, out_dir, 'fig06_client_expert_heatmap', saved)


def plot_expert_frequency(runs, out_dir, saved):
    n_experts = max(run.n_experts for run in runs)
    if n_experts <= 0:
        return
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    x = np.arange(n_experts)
    width = min(0.8 / max(len(runs), 1), 0.22)
    offset0 = -0.5 * (len(runs) - 1) * width
    y_vals = []
    for i, run in enumerate(runs):
        counts = np.zeros(n_experts, dtype=float)
        if not run.uploads.empty:
            valid = run.uploads.dropna(subset=['expert_id'])
            grouped = valid.groupby('expert_id').size()
            for eid, cnt in grouped.items():
                if int(eid) < n_experts:
                    counts[int(eid)] = cnt
        ax.bar(
            x + offset0 + i * width, counts, width=width * 0.95,
            color=BAR[i % len(BAR)], edgecolor='white', linewidth=0.3,
            label=run.label,
        )
        y_vals.extend(counts.tolist())
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Times uploaded')
    ax.set_xticks(x, [str(i) for i in x])
    if y_vals:
        set_count_ylim(ax, y_vals)
    add_run_legend(fig, len(runs), ax)
    save_fig(fig, out_dir, 'fig07_expert_frequency', saved)


def plot_time_breakdown(runs, out_dir, saved):
    cols = [
        ('LocalTrainTimeSec', 'Local training', '(a)'),
        ('AggTimeSec', 'Aggregation', '(b)'),
        ('EvalTimeSec', 'Evaluation', '(c)'),
    ]
    present = [item for item in cols if any(item[0] in run.df.columns for run in runs)]
    if not present:
        if any('RoundTimeSec' in run.df.columns for run in runs):
            present = [('RoundTimeSec', 'Round time', None)]
        else:
            return
    fig, axes = plt.subplots(
        1, len(present), figsize=(3.0 * len(present) + 1.2, 2.7),
        layout='constrained', squeeze=False)
    axes = axes[0]
    for ax, (col, ylabel, tag) in zip(axes, present):
        for i, run in enumerate(runs):
            if col not in run.df.columns:
                continue
            ax.plot(run.df['Round'], run.df[col],
                    label=run.label, **style_of(i, len(run.df)))
        ax.set_xlabel('Communication round')
        ax.set_ylabel(f'{ylabel} (s)')
        ax.set_xlim(*round_xlim(runs))
        if tag:
            panel_label(ax, tag)
    add_run_legend(fig, len(runs), axes[0])
    save_fig(fig, out_dir, 'fig08_time_breakdown', saved)


def plot_acc_vs_communication(runs, out_dir, saved):
    fig, ax = plt.subplots(figsize=(4.6, 2.7), layout='constrained')
    drawn = 0
    for i, run in enumerate(runs):
        if 'n_uploads' not in run.stats.columns:
            continue
        merged = run.df[['Round', 'TestAcc']].merge(
            run.stats[['Round', 'n_uploads']], on='Round', how='left')
        merged['n_uploads'] = merged['n_uploads'].fillna(0)
        merged['cum_uploads'] = merged['n_uploads'].cumsum()
        if merged['cum_uploads'].max() <= 0:
            continue
        ax.plot(merged['cum_uploads'], merged['TestAcc'],
                label=run.label, **style_of(i, len(merged)))
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    ax.set_xlabel('Cumulative uploaded experts')
    ax.set_ylabel('Accuracy (%)')
    add_run_legend(fig, len(runs), ax)
    save_fig(fig, out_dir, 'fig09_accuracy_vs_communication', saved)


def plot_overview(runs, out_dir, saved):
    n_experts = max(run.n_experts for run in runs)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4), layout='constrained')

    ax = axes[0, 0]
    for i, run in enumerate(runs):
        ax.plot(run.df['Round'], run.df['TestAcc'],
                label=run.label, **style_of(i, len(run.df)))
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    panel_label(ax, '(a)')

    ax = axes[0, 1]
    for i, run in enumerate(runs):
        ax.plot(run.df['Round'], run.df['AvgLoss'],
                label=run.label, **style_of(i, len(run.df)))
    ax.set_xlabel('Round')
    ax.set_ylabel('Training loss')
    panel_label(ax, '(b)')

    ax = axes[1, 0]
    k_vals = []
    for i, run in enumerate(runs):
        if run.stats is None or 'mean_k' not in run.stats.columns:
            continue
        ax.plot(run.stats['Round'], run.stats['mean_k'],
                label=run.label, **style_of(i, len(run.stats)))
        k_vals.extend(run.stats['mean_k'].dropna().tolist())
    ax.set_xlabel('Round')
    ax.set_ylabel('Uploaded experts $k$')
    if k_vals:
        set_count_ylim(ax, k_vals)
    panel_label(ax, '(c)')

    ax = axes[1, 1]
    if n_experts > 0:
        x = np.arange(n_experts)
        width = min(0.8 / max(len(runs), 1), 0.22)
        offset0 = -0.5 * (len(runs) - 1) * width
        y_vals = []
        for i, run in enumerate(runs):
            counts = np.zeros(n_experts, dtype=float)
            if not run.uploads.empty:
                valid = run.uploads.dropna(subset=['expert_id'])
                for eid, cnt in valid.groupby('expert_id').size().items():
                    if int(eid) < n_experts:
                        counts[int(eid)] = cnt
            ax.bar(
                x + offset0 + i * width, counts, width=width * 0.95,
                color=BAR[i % len(BAR)], edgecolor='white', linewidth=0.3,
                label=run.label,
            )
            y_vals.extend(counts.tolist())
        ax.set_xticks(x, [str(i) for i in x])
        if y_vals:
            set_count_ylim(ax, y_vals)
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Times uploaded')
    panel_label(ax, '(d)')

    xlim = round_xlim(runs)
    axes[0, 0].set_xlim(*xlim)
    axes[0, 1].set_xlim(*xlim)
    axes[1, 0].set_xlim(*xlim)
    add_run_legend(fig, len(runs), axes[0, 0])
    save_fig(fig, out_dir, 'fig10_overview', saved)


def mean_round_time(run: Run) -> float:
    if 'RoundTimeSec' in run.df.columns and len(run.df):
        return float(run.df['RoundTimeSec'].mean())
    val = run.summary.get('avg_round_time_sec')
    return float(val) if val is not None else 0.0


def plot_avg_round_time(runs, out_dir, saved):
    times = [mean_round_time(run) for run in runs]
    if not any(t > 0 for t in times):
        return
    labels = [run.label for run in runs]
    fig_w = max(4.6, 0.72 * len(runs) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 2.8), layout='constrained')
    x = np.arange(len(runs))
    ax.bar(
        x, times, width=0.72,
        color=[BAR[i % len(BAR)] for i in range(len(runs))],
        edgecolor='white', linewidth=0.4,
    )
    ax.set_xticks(x, labels, rotation=25, ha='right')
    ax.set_xlabel('Method')
    ax.set_ylabel('Avg. time per round (s)')
    ymax = max(times)
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
    save_fig(fig, out_dir, 'fig11_avg_round_time', saved)


def write_caption_notes(out_dir: Path, runs: list[Run]):
    lines = ['runs:']
    for run in runs:
        cfg = run.config
        best = run.summary.get('best_acc', run.df['TestAcc'].max() if len(run.df) else '-')
        lines.append(
            f"  {run.label}  dir={run.path.name}  method={cfg.get('method', '-')}  "
            f"backbone={cfg.get('backbone', '-')}  select={cfg.get('select', '-')}  "
            f"merge={cfg.get('merge', '-')}  best_acc={best}"
        )
    lines += [
        '',
        'fig01_test_accuracy: test accuracy vs round, one curve per run',
        'fig02_test_accuracy_vs_time: test accuracy vs wall-clock time',
        'fig02_train_loss: local training loss vs round',
        'fig03_acc_and_loss: (a) accuracy (b) loss',
        'fig04_bandwidth_and_k: mean bandwidth and uploaded expert count',
        'fig05_expert_round_heatmap: total uploaded experts vs round',
        'fig06_client_expert_heatmap: total uploads per client',
        'fig07_expert_frequency: overall expert upload histogram',
        'fig08_time_breakdown: local / aggregation / evaluation time',
        'fig09_accuracy_vs_communication: accuracy vs cumulative expert uploads',
        'fig10_overview: four-panel summary figure',
        'fig11_avg_round_time: average wall time per round for each method',
    ]
    (out_dir / 'captions.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_multi(result_dirs, out_dir='plot/multi', labels=None, verbose=True):
    """Generate overlay figures for a group of result folders."""
    apply_paper_style()
    runs = load_runs(result_dirs, labels=labels)
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    plot_test_accuracy(runs, out_dir, saved)
    plot_test_accuracy_vs_time(runs, out_dir, saved)
    plot_train_loss(runs, out_dir, saved)
    plot_acc_and_loss(runs, out_dir, saved)
    plot_bandwidth_and_k(runs, out_dir, saved)
    plot_uploads_per_round(runs, out_dir, saved)
    plot_client_uploads(runs, out_dir, saved)
    plot_expert_frequency(runs, out_dir, saved)
    plot_time_breakdown(runs, out_dir, saved)
    plot_acc_vs_communication(runs, out_dir, saved)
    plot_overview(runs, out_dir, saved)
    plot_avg_round_time(runs, out_dir, saved)
    write_caption_notes(out_dir, runs)

    if verbose:
        print(f'[Plot] {len(runs)} runs -> {out_dir}')
        for run in runs:
            print(f'  {run.label}: {run.path}')
        for path in saved:
            if path.suffix == '.pdf':
                print(f'  {path.name}')
    return out_dir


def main():
    args = parse_args()
    plot_multi(args.result, out_dir=args.out, labels=args.label)


if __name__ == '__main__':
    main()
