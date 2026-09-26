"""CIFAR-100 / ResNet：紧预算下探测批次与全量解的一致率。

训练流程与主实验 Ours 相同（8 专家、10 客户端、Dirichlet 0.1、带宽 [10, 80]）。
每一轮、每个客户端若信道只允许上传 K∈[k_min, k_max]（默认 1–3）个专家，
就在本地训练之前，用当前全局模型的门控 softmax 做一次测量：

- 全量解：该客户端全部本地样本的平均门控，再取 Top-K。
- 固定批次数：同一随机顺序的前 m 个 mini-batch，与全量解的专家集合是否相同。
- 单交换早停：每多看一个 mini-batch，用经验 Bernstein 半径构造置信区间。
  当前集合的下置信界严格大于所有「换入一个、换出一个」邻居的上置信界时停止。
  停下来的集合再与全量解比较。

论文把半径写成关于探测 batch 数 m 的式子。直接把 m 当成样本量时，
加项 3 log(m)/m 会大于门控概率本身，本地数据耗尽前几乎不会停。
这里的观测是每个样本的门控概率，半径里的 n 是已经看过的样本数，
横轴仍是用掉的 mini-batch 数。log 用自然对数，与 Audibert 的经验 Bernstein 界一致。
专家上传代价相同，背包退化为按平均门控取 Top-K。
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data_utils import DATASET_CFG, get_dataset, maybe_subset, partition_dirichlet
from methods import build_method
from methods.Select.ours import select_topk_experts
from upload_budget import sample_bandwidth, upload_expert_count
from utils import evaluate, format_hms, get_lr, make_result_dir, release_cuda


FIXED_BATCHES = (1, 2, 4, 8, 16)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--backbone', default='resnet', choices=['resnet', 'vgg11'])
    p.add_argument('--data_root', default='./data')
    p.add_argument('--output_dir', default='./results/exp11-c100-resnet-probe-agreement')
    p.add_argument('--num_clients', type=int, default=10)
    p.add_argument('--num_experts', type=int, default=8)
    p.add_argument('--topk', type=int, default=2)
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--frac', type=float, default=1.0)
    p.add_argument('--local_epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lr_min', type=float, default=1e-4)
    p.add_argument('--warmup_rounds', type=int, default=5)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--label_smooth', type=float, default=0.1)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--width_mult', type=float, default=1.0)
    p.add_argument('--expert_bw', type=int, default=10)
    p.add_argument('--bw_min', type=int, default=10)
    p.add_argument('--bw_max', type=int, default=80)
    p.add_argument('--snr_db', type=float, default=10.0)
    p.add_argument('--probe_batches', type=int, default=2,
                   help='训练时 Ours 选择器使用的固定探测批次数。')
    p.add_argument('--n_ref', type=int, default=0)
    p.add_argument('--k_min', type=int, default=1)
    p.add_argument('--k_max', type=int, default=3)
    p.add_argument('--max_probe_batches', type=int, default=16)
    p.add_argument('--fixed_batches', default='1,2,4,8,16')
    p.add_argument('--eval_from_round', type=int, default=1,
                   help='只把这一轮及之后的紧预算样本画进图里。前面的轮次仍训练、仍写入明细。')
    p.add_argument('--radius_mode', default='sample', choices=['sample', 'delta'],
                   help='sample: 半径里用 log(n)。delta: 用 log(1/delta)。n 都是已看过的样本数。')
    p.add_argument('--radius_scale', type=float, default=1.0,
                   help='乘在经验 Bernstein 半径上。小于 1 更容易提前停止。')
    p.add_argument('--delta', type=float, default=0.1,
                   help='radius_mode=delta 时的失误概率。')
    p.add_argument('--max_train_samples', type=int, default=0)
    p.add_argument('--max_test_samples', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--method', default='ours')
    p.add_argument('--select', default='ours')
    p.add_argument('--merge', default='ours')
    p.add_argument('--drop_expert_prob', type=float, default=0.0)
    return p.parse_args()


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def bernstein_radius(var, n, scale=1.0, log_term=None):
    """经验 Bernstein 半径。n 为样本数，var 为各专家的经验方差。"""
    n = float(n)
    var = np.maximum(np.asarray(var, dtype=np.float64), 0.0)
    if n <= 1.0 or scale < 0:
        return np.full(var.shape, np.inf, dtype=np.float64)
    if log_term is None:
        log_term = float(np.log(n))
    log_term = float(log_term)
    radius = np.sqrt(2.0 * var * log_term / n) + 3.0 * log_term / n
    return float(scale) * radius


def radius_from_args(var, n, args):
    log_term = None
    if args.radius_mode == 'delta':
        delta = float(args.delta)
        if not 0.0 < delta < 1.0:
            raise ValueError(f'delta must lie in (0, 1), got {delta}')
        log_term = float(np.log(1.0 / delta))
    return bernstein_radius(var, n, scale=float(args.radius_scale), log_term=log_term)


def single_swap_certified(mean, beta, k):
    """当前 Top-K 的 LCB 是否严格大于每一个单交换邻居的 UCB。"""
    chosen = select_topk_experts(mean, k)
    if not chosen:
        return False
    chosen_set = set(chosen)
    n_experts = len(mean)
    if len(chosen_set) >= n_experts:
        return True
    lcb = float(sum(mean[j] - beta[j] for j in chosen))
    best_ucb = None
    for a in chosen:
        base = float(sum(mean[j] + beta[j] for j in chosen if j != a))
        for b in range(n_experts):
            if b in chosen_set:
                continue
            ucb = base + float(mean[b] + beta[b])
            if best_ucb is None or ucb > best_ucb:
                best_ucb = ucb
    if best_ucb is None:
        return True
    return lcb > best_ucb


def prefix_stats(batch_sum, batch_sq, batch_n, m):
    total = np.sum(batch_sum[:m], axis=0)
    sq = np.sum(batch_sq[:m], axis=0)
    n = int(np.sum(batch_n[:m]))
    if n <= 0:
        raise ValueError('prefix has no samples')
    mean = total / n
    var = np.maximum(sq / n - mean * mean, 0.0)
    return mean, var, n


@torch.no_grad()
def collect_gate_batches(model, loader, device):
    """每个 mini-batch 上，各专家门控概率的和与平方和。"""
    was_training = model.training
    model.eval()
    sums, sqs, counts = [], [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        feat = model.backbone(x)
        logits = model.moe_head.gating.gate(feat)
        probs = torch.softmax(logits.float(), dim=-1)
        sums.append(probs.sum(dim=0).detach().cpu().numpy())
        sqs.append(probs.square().sum(dim=0).detach().cpu().numpy())
        counts.append(int(x.size(0)))
    if was_training:
        model.train()
    if not counts:
        return None
    return np.stack(sums), np.stack(sqs), np.asarray(counts, dtype=np.int64)


def measure_client(model, loader, device, k, fixed_batches, max_probe_batches, radius_args):
    collected = collect_gate_batches(model, loader, device)
    if collected is None:
        return None
    batch_sum, batch_sq, batch_n = collected
    n_batches = int(batch_sum.shape[0])
    if n_batches < 1:
        return None
    oracle_mean, _, n_samples = prefix_stats(batch_sum, batch_sq, batch_n, n_batches)
    oracle = select_topk_experts(oracle_mean, k)
    oracle_set = set(oracle)

    agree = {}
    for m in fixed_batches:
        if n_batches < m:
            agree[m] = None
            continue
        mean, _, _ = prefix_stats(batch_sum, batch_sq, batch_n, m)
        agree[m] = float(set(select_topk_experts(mean, k)) == oracle_set)

    limit = min(n_batches, int(max_probe_batches))
    stop_m = limit
    certified = 0
    for m in range(1, limit + 1):
        mean, var, n = prefix_stats(batch_sum, batch_sq, batch_n, m)
        if single_swap_certified(mean, radius_from_args(var, n, radius_args), k):
            stop_m = m
            certified = 1
            break
    stop_mean, _, _ = prefix_stats(batch_sum, batch_sq, batch_n, stop_m)
    adaptive_agree = float(set(select_topk_experts(stop_mean, k)) == oracle_set)
    return {
        'NumBatches': n_batches,
        'NumSamples': n_samples,
        'StopBatches': int(stop_m),
        'Certified': int(certified),
        'Agree': adaptive_agree,
        'fixed': agree,
    }


def probe_loader(dataset, indices, batch_size, seed, rnd, cid, num_workers):
    g = torch.Generator()
    g.manual_seed((int(seed) * 1_000_003 + int(rnd) * 10_007 + int(cid) * 97) % (2**31 - 1))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=False,
    )


def summarize(detail: pd.DataFrame, fixed_batches):
    fixed_rows = []
    for m in fixed_batches:
        col = f'Agree{m}'
        if col not in detail.columns or detail.empty:
            fixed_rows.append({'ProbeBatches': m, 'Agreement': np.nan, 'Count': 0})
            continue
        values = detail[col].dropna()
        fixed_rows.append({
            'ProbeBatches': m,
            'Agreement': float(values.mean()) if len(values) else np.nan,
            'Count': int(len(values)),
        })
    fixed = pd.DataFrame(fixed_rows)
    if detail.empty:
        adaptive = pd.DataFrame(columns=['StopBatches', 'Agree'])
    else:
        adaptive = detail[['StopBatches', 'Agree']].copy()
        adaptive['Agree'] = adaptive['Agree'].astype(float)
    return fixed, adaptive


def save_tables(result_dir, detail, fixed, adaptive):
    data_dir = os.path.join(result_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    detail.to_csv(os.path.join(result_dir, 'probe_detail.csv'), index=False)
    fixed[['ProbeBatches', 'Agreement']].to_csv(
        os.path.join(data_dir, 'probe_fixed.csv'), index=False,
    )
    adaptive.to_csv(os.path.join(data_dir, 'probe_adaptive.csv'), index=False)
    fixed.to_csv(os.path.join(result_dir, 'probe_fixed_counts.csv'), index=False)


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


def plot_probe(fixed, adaptive, result_dir, k_min, k_max):
    if fixed['Agreement'].dropna().empty or adaptive.empty:
        print('[Plot] 没有足够的紧预算样本，跳过作图。')
        return
    apply_paper_style()
    ours = '#0072B2'
    channel = '#222222'
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.15))

    ax = axes[0]
    ax.plot(
        fixed['ProbeBatches'], fixed['Agreement'] * 100.0,
        color=channel, marker='o', markersize=8, label='Stop at fixed batches',
    )
    mean_m = float(adaptive['StopBatches'].mean())
    mean_acc = float(adaptive['Agree'].mean()) * 100.0
    ax.scatter([mean_m], [mean_acc], s=160, color=ours, zorder=5, label='Single-swap stop')
    ax.axvline(mean_m, color=ours, linestyle='--', linewidth=1.4)
    ax.set_xticks(list(fixed['ProbeBatches']))
    ax.set_xlim(0.5, max(17.0, float(fixed['ProbeBatches'].max()) + 1.0))
    ax.set_ylim(0, 105)
    ax.set_xlabel('Probe mini-batches')
    ax.set_ylabel('Agreement (%)')
    ax.set_title(f'Tight budget ({k_min}–{k_max} experts)', pad=8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right', fontsize=15)
    style_spines(ax)

    ax = axes[1]
    hi = max(16.5, float(adaptive['StopBatches'].max()) + 0.5)
    bins = np.arange(0.5, hi + 1.0, 1.0)
    ax.hist(
        adaptive['StopBatches'], bins=bins,
        color=ours, edgecolor='black', linewidth=0.6,
    )
    ax.axvline(mean_m, color=channel, linestyle='--', linewidth=1.6)
    ax.set_xlim(0.5, hi)
    ymax = ax.get_ylim()[1]
    ax.text(
        mean_m + 0.35, ymax * 0.92,
        f'mean {mean_m:.1f}',
        fontsize=16, color=channel, va='top',
    )
    ax.set_xlabel('Batches until stop')
    ax.set_ylabel('Rounds')
    ax.set_title('Single-swap stopping', pad=8)
    style_spines(ax)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(result_dir, f'fig04_probe_agreement.{ext}')
        fig.savefig(path)
        print(path)
    plt.close(fig)


def client_budget_k(args, cid, rnd):
    bandwidth = sample_bandwidth(args, cid, rnd)
    return upload_expert_count(bandwidth, args.expert_bw, args.num_experts)


def main():
    args = parse_args()
    fixed_batches = tuple(int(x) for x in args.fixed_batches.split(',') if x.strip())
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = DATASET_CFG[args.dataset]
    pin_memory = device.type == 'cuda'

    train_ds, test_ds = get_dataset(args.dataset, args.data_root)
    train_ds = maybe_subset(train_ds, args.max_train_samples, args.seed)
    test_ds = maybe_subset(test_ds, args.max_test_samples, args.seed + 1)
    client_idx = partition_dirichlet(train_ds, args.num_clients, args.beta, args.seed)

    method = build_method(args, device)
    model = method.build_model(cfg)
    test_loader = DataLoader(
        test_ds, batch_size=min(256, args.batch_size * 4), shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    client_loaders = [
        DataLoader(
            Subset(train_ds, idx), batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin_memory,
        )
        for idx in client_idx
    ]

    result_dir = make_result_dir(args.output_dir)
    print(f'[Export] {result_dir}')
    print(f'[Probe] {args.dataset} / {args.backbone} | experts={args.num_experts} | '
          f'clients={args.num_clients} | tight K=[{args.k_min}, {args.k_max}] | '
          f'bw=[{args.bw_min}, {args.bw_max}] | batch={args.batch_size} | '
          f'beta={args.beta:g} | rounds={args.rounds} | from_round={args.eval_from_round}')
    print(f'[Probe] radius={args.radius_mode} scale={args.radius_scale:g} '
          f'delta={args.delta:g} seed={args.seed}')

    rows = []
    best_acc = 0.0
    t0 = datetime.now()
    import time
    experiment_t0 = time.perf_counter()
    m_clients = max(1, int(args.num_clients * args.frac))

    for rnd in range(1, args.rounds + 1):
        round_t0 = time.perf_counter()
        current_lr = get_lr(rnd, args.rounds, args.lr, args.lr_min, args.warmup_rounds)
        chosen = np.random.choice(args.num_clients, m_clients, replace=False).tolist()
        n_measured = 0
        for cid in chosen:
            k = client_budget_k(args, cid, rnd)
            if k < args.k_min or k > args.k_max:
                continue
            loader = probe_loader(
                train_ds, client_idx[cid], args.batch_size,
                args.seed, rnd, cid, args.num_workers,
            )
            measured = measure_client(
                model, loader, device, k, fixed_batches, args.max_probe_batches, args,
            )
            del loader
            if measured is None:
                continue
            row = {
                'Round': rnd,
                'Client': cid,
                'BudgetK': int(k),
                'NumBatches': measured['NumBatches'],
                'NumSamples': measured['NumSamples'],
                'StopBatches': measured['StopBatches'],
                'Certified': measured['Certified'],
                'Agree': measured['Agree'],
            }
            for m_fixed in fixed_batches:
                row[f'Agree{m_fixed}'] = measured['fixed'][m_fixed]
            rows.append(row)
            n_measured += 1
        release_cuda()

        payloads = []
        for cid in chosen:
            payloads.append(method.local_update(
                model, client_loaders[cid], cid, rnd, current_lr,
            ))
        new_state = method.aggregate(model, payloads)
        model.load_state_dict(new_state)
        avg_loss = float(np.mean([p['loss'] for p in payloads]))
        del payloads, new_state
        release_cuda()

        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        detail = pd.DataFrame(rows)
        plotted = detail[detail['Round'] >= args.eval_from_round] if len(detail) else detail
        fixed, adaptive = summarize(plotted, fixed_batches)
        save_tables(result_dir, detail, fixed, adaptive)
        plot_probe(fixed, adaptive, result_dir, args.k_min, args.k_max)
        elapsed = time.perf_counter() - round_t0
        adapt_txt = 'n/a'
        if not adaptive.empty:
            adapt_txt = (f"stop {adaptive['StopBatches'].mean():.2f} | "
                         f"agree {100.0 * adaptive['Agree'].mean():.1f}% "
                         f"(n={len(adaptive)})")
        print(
            f'{rnd:>5} | loss {avg_loss:.4f} | acc {acc:6.2f}% | best {best_acc:6.2f}% | '
            f'tight {n_measured} | total {len(rows)} | {adapt_txt} | {format_hms(elapsed)}',
            flush=True,
        )

    detail = pd.DataFrame(rows)
    plotted = detail[detail['Round'] >= args.eval_from_round] if len(detail) else detail
    fixed, adaptive = summarize(plotted, fixed_batches)
    save_tables(result_dir, detail, fixed, adaptive)
    plot_probe(fixed, adaptive, result_dir, args.k_min, args.k_max)
    total = time.perf_counter() - experiment_t0
    print(f'[Done] samples={len(rows)} best_acc={best_acc:.2f}% time={format_hms(total)}')
    print(f'[Done] started {t0:%Y-%m-%d %H:%M:%S} dir={result_dir}')
    if not fixed.empty:
        print('[Fixed]', fixed.to_string(index=False))
    if not adaptive.empty:
        print(
            f"[Adaptive] n={len(adaptive)} mean_stop={adaptive['StopBatches'].mean():.3f} "
            f"agreement={adaptive['Agree'].mean():.4f} "
            f"certified={(plotted['Certified'] == 1).mean():.4f}"
        )


if __name__ == '__main__':
    main()
