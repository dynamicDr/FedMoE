import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_utils import (
    DATASET_CFG, download_all_datasets, get_dataset,
    maybe_subset, partition_dirichlet,
)
from methods import METHOD_REGISTRY, build_method
from methods.Merge import MERGE_REGISTRY
from methods.Select import SELECT_REGISTRY
from upload_budget import resolve_bw_range
from utils import (
    export_experiment, format_hms, get_lr, make_result_dir,
    release_cuda, save_rounds,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='bla-fedmoe', choices=sorted(METHOD_REGISTRY),
                   help='Federated method class to run.')
    p.add_argument('--select', default=None, choices=sorted(SELECT_REGISTRY),
                   help='Expert selection: ours | posthoc | fedrolex | snip | '
                        'random | routingfreq | magnitude. Default: method preset.')
    p.add_argument('--merge', default=None, choices=sorted(MERGE_REGISTRY),
                   help='Aggregation: ours | fedavg | fedavg-old | fedavgm | fedadam. '
                        'Default: method preset.')
    p.add_argument('--dataset',       default='cifar10',
                   choices=['cifar10','cifar100','svhn','tinyimagenet','cinic-10','stl10'])
    p.add_argument('--beta',          type=float, default=0.1)
    p.add_argument('--data_root',     default='./data')
    p.add_argument('--num_clients',   type=int,   default=10)
    p.add_argument('--backbone',      default='resnet',
                   choices=['resnet', 'vgg11'],
                   help='Shared feature extractor: resnet | vgg11.')
    p.add_argument('--width_mult',    type=float, default=1.0,
                   help='Backbone width multiplier only. ResNet base channels '
                        'are 64-128-256-512; 0.5 -> 32-64-128-256. '
                        'Expert FFN hidden size stays 512.')
    p.add_argument('--num_experts',   type=int,   default=4)
    p.add_argument('--topk',          type=int,   default=2)
    p.add_argument('--expert_bw',     type=int,   default=10,
                   help='Bandwidth cost of uploading one expert.')
    p.add_argument('--bw_min',        type=int,   default=10,
                   help='Min uplink budget. Default 10 => K=1. '
                        '<0 means no fading (use bw_max).')
    p.add_argument('--bw_max',        type=int,   default=40,
                   help='Max uplink budget. Default 40 => K=M. '
                        '<0 means num_experts * expert_bw.')
    p.add_argument('--snr_db',        type=float, default=10.0,
                   help='Mean SNR (dB) for Rayleigh-Shannon fading when '
                        'bw_min < bw_max.')
    p.add_argument('--drop_expert_prob', type=float, default=0.0,
                   help='Per client per round, probability of uploading '
                        'one fewer expert than the bandwidth budget. '
                        '0 disables it.')
    p.add_argument('--rounds',        type=int,   default=100)
    p.add_argument('--frac',          type=float, default=1.0)
    p.add_argument('--local_epochs',  type=int,   default=2)
    p.add_argument('--batch_size',    type=int,   default=64)
    p.add_argument('--lr',            type=float, default=0.01)
    p.add_argument('--lr_min',        type=float, default=1e-4)
    p.add_argument('--warmup_rounds', type=int,   default=5)
    p.add_argument('--momentum',      type=float, default=0.9)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--lam',           type=float, default=1e-3)
    p.add_argument('--agg_iters',     type=int,   default=300)
    p.add_argument('--agg_lr',        type=float, default=1e-2)
    p.add_argument('--kron_samples',  type=int,   default=256)
    p.add_argument('--probe_batches', type=int,   default=2,
                   help='Probe batches before local training for ours/snip.')
    p.add_argument('--n_ref', type=int, default=0,
                   help='Expert refresh interval in rounds. '
                        '0: auto (=num_experts, so N_ref*K>=M at K=1). '
                        '<0: disable refresh.')
    p.add_argument('--server_momentum', type=float, default=0.9,
                   help='FedAvgM server momentum on trainable params.')
    p.add_argument('--server_lr', type=float, default=1.0,
                   help='FedAvgM server learning rate on trainable params.')
    p.add_argument('--fedadam_lr', type=float, default=0.01,
                   help='FedAdam server learning rate on trainable params.')
    p.add_argument('--fedadam_beta1', type=float, default=0.9,
                   help='FedAdam first-moment coefficient.')
    p.add_argument('--fedadam_beta2', type=float, default=0.99,
                   help='FedAdam second-moment coefficient.')
    p.add_argument('--fedadam_tau', type=float, default=1e-3,
                   help='FedAdam numerical stability term.')
    p.add_argument('--label_smooth',  type=float, default=0.1)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--device',        default='auto',
                   choices=['auto','cpu','cuda','mps'])
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--max_train_samples', type=int, default=0,
                   help='If >0, randomly subsample the training set.')
    p.add_argument('--max_test_samples',  type=int, default=0,
                   help='If >0, randomly subsample the test set.')
    p.add_argument('--demo', action='store_true',
                   help='Minimal few-step demo on a CIFAR-10 subset.')
    p.add_argument('--download-all', action='store_true',
                   help='Download and extract all supported datasets, then exit.')
    p.add_argument('--output_dir',    default='./results',
                   help='Directory for experiment result files.')
    return p.parse_args()


def apply_demo_defaults(args):
    args.dataset = 'cifar10'
    args.num_clients = 2
    args.num_experts = 2
    args.topk = 1
    args.rounds = 1
    args.local_epochs = 1
    args.batch_size = 16
    args.warmup_rounds = 0
    args.agg_iters = 2
    args.kron_samples = 16
    args.probe_batches = 1
    args.expert_bw = 10
    args.bw_min = 10
    args.bw_max = 20
    args.max_train_samples = 256
    args.max_test_samples = 64
    args.num_workers = 0
    return args


def resolve_device(name):
    if name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(name)


def main():
    args = get_args()
    if getattr(args, 'download_all', False):
        download_all_datasets(args.data_root)
        return
    if args.demo:
        args = apply_demo_defaults(args)
        print('[Demo] Using a CIFAR-10 subset with 1 round / 1 local epoch.')

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = DATASET_CFG[args.dataset]
    pin_memory = device.type == 'cuda'

    method = build_method(args, device)
    print(f'\n{"="*66}')
    print(f' {method.name}  ')
    print(f'  Dataset={args.dataset} | beta={args.beta} | '
          f'Backbone={args.backbone} | width_mult={args.width_mult:g} | '
          f'Clients={args.num_clients} | Experts={args.num_experts}')
    select_name = getattr(getattr(method, 'selector', None), 'name', None)
    merge_name = getattr(getattr(method, 'merger', None), 'name', None)
    if select_name or merge_name:
        print(f'  Select={select_name or "-"} | Merge={merge_name or "-"}')
    if select_name == 'ours':
        from methods.Select.ours import resolve_n_ref
        n_ref_eff = resolve_n_ref(args)
        n_ref_txt = 'off' if n_ref_eff < 0 else str(n_ref_eff)
        print(f'  Expert refresh N_ref={n_ref_txt}')
    bw_min, bw_max, expert_bw = resolve_bw_range(args)
    print(f'  Rounds={args.rounds} | LR={args.lr}')
    fading = bw_min < bw_max
    fade_txt = (f'Rayleigh-Shannon, SNR={args.snr_db:g} dB'
                if fading else 'static')
    drop_prob = float(getattr(args, 'drop_expert_prob', 0.0) or 0.0)
    drop_txt = f' | drop-one p={drop_prob:g}' if drop_prob > 0 else ''
    print(f'  Bandwidth: expert_bw={expert_bw} | bw=[{bw_min}, {bw_max}] '
          f'| {fade_txt} | k=floor(bw/{expert_bw}){drop_txt}')
    print(f'{"="*66}\n')

    train_ds, test_ds = get_dataset(args.dataset, args.data_root)
    train_ds = maybe_subset(train_ds, args.max_train_samples, args.seed)
    test_ds  = maybe_subset(test_ds,  args.max_test_samples,  args.seed + 1)
    client_idx = partition_dirichlet(train_ds, args.num_clients, args.beta, args.seed)
    selector = getattr(method, 'selector', None)
    if selector is not None:
        select_note = getattr(selector, 'description', None) or selector.name
    elif args.method == 'moe':
        select_note = 'random experts'
    else:
        select_note = 'random experts (还没设计)'
    print(f'[Upload] expert_bw={expert_bw} | available bw=[{bw_min}, {bw_max}] '
          f'| {fade_txt} | upload k=floor(bw/{expert_bw}), cap={args.num_experts} '
          f'| {select_note}')
    print()
    client_loaders = [
        DataLoader(Subset(train_ds, idx), batch_size=args.batch_size,
                   shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        for idx in client_idx
    ]
    test_loader = DataLoader(test_ds, batch_size=min(256, args.batch_size * 4), shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = method.build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Model] Total params: {n_params:,}\n')

    result_dir = make_result_dir(args.output_dir)
    print(f'[Export] 本次实验目录: {result_dir}\n')
    experiment_start = datetime.now()
    experiment_t0 = time.perf_counter()

    m = max(1, int(args.num_clients * args.frac))
    best_acc = 0.0
    best_round = 0
    print(f'{"Round":>5} | {"LR":>7} | {"AvgLoss":>8} | {"TestAcc":>8} | {"Best":>8} | {"RoundTime":>10}')
    print('-' * 80)

    history_records = []
    client_sizes = [len(idx) for idx in client_idx]

    for rnd in range(1, args.rounds + 1):
        round_t0 = time.perf_counter()
        current_lr = get_lr(rnd, args.rounds, args.lr, args.lr_min, args.warmup_rounds)
        chosen = np.random.choice(args.num_clients, m, replace=False).tolist()

        local_t0 = time.perf_counter()
        payloads = []
        client_times = []
        for cid in chosen:
            client_t0 = time.perf_counter()
            payloads.append(method.local_update(
                model, client_loaders[cid], cid, rnd, current_lr,
            ))
            client_times.append(time.perf_counter() - client_t0)
        local_time = time.perf_counter() - local_t0

        agg_t0 = time.perf_counter()
        new_state = method.aggregate(model, payloads)
        model.load_state_dict(new_state)
        agg_time = time.perf_counter() - agg_t0

        avg_loss = float(np.mean([p['loss'] for p in payloads]))
        bandwidth_parts = []
        expert_parts = []
        upload_details = []
        for p in payloads:
            meta = p['meta']
            cid = meta['client_id']
            bw = meta.get('bandwidth', '-')
            eids = list(meta['expert_ids'])
            eid_str = ','.join(str(e) for e in eids)
            bandwidth_parts.append(f'{cid}:{bw}')
            expert_parts.append(f'{cid}:[{eid_str}]')
            upload_details.append(f'c{cid}:bw={bw},experts=[{eid_str}]')
        bandwidth_log = ';'.join(bandwidth_parts)
        selected_experts = ';'.join(expert_parts)
        upload_detail = '; '.join(upload_details)
        del payloads, new_state
        release_cuda()

        eval_t0 = time.perf_counter()
        acc = method.evaluate(model, test_loader)
        eval_time = time.perf_counter() - eval_t0
        if acc > best_acc:
            best_acc = acc
            best_round = rnd
        round_time = time.perf_counter() - round_t0
        elapsed = time.perf_counter() - experiment_t0
        print(f'{rnd:>5} | {current_lr:>7.5f} | {avg_loss:>8.4f} | '
              f'{acc:>7.2f}% | {best_acc:>7.2f}% | {format_hms(round_time):>10}',
              flush=True)
        print(f'      bandwidth/experts | {upload_detail}', flush=True)

        history_records.append({
            'Round': rnd,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Method': method.name,
            'LR': current_lr,
            'AvgLoss': avg_loss,
            'TestAcc': acc,
            'BestAcc': best_acc,
            'BestRound': best_round,
            'RoundTimeSec': round(round_time, 4),
            'RoundTimeHMS': format_hms(round_time),
            'LocalTrainTimeSec': round(local_time, 4),
            'AggTimeSec': round(agg_time, 4),
            'EvalTimeSec': round(eval_time, 4),
            'CumulativeTimeSec': round(elapsed, 4),
            'CumulativeTimeHMS': format_hms(elapsed),
            'NumClientsSelected': len(chosen),
            'SelectedClients': ','.join(str(c) for c in chosen),
            'Bandwidth': bandwidth_log,
            'SelectedExperts': selected_experts,
            'UploadDetail': upload_detail,
            'ClientTrainTimesSec': ','.join(f'{t:.4f}' for t in client_times),
        })
        save_rounds(result_dir, history_records)

    total_time = time.perf_counter() - experiment_t0
    experiment_end = datetime.now()
    avg_round_time = total_time / max(len(history_records), 1)
    final_acc = history_records[-1]['TestAcc'] if history_records else 0.0

    print(f'\nDone. Best Acc: {best_acc:.2f}% @ round {best_round}')
    print(f'Total time: {format_hms(total_time)}')

    config = {
        'experiment': method.name,
        'method': args.method,
        'select': getattr(method.selector, 'name', None),
        'merge': getattr(method.merger, 'name', None),
        'dataset': args.dataset,
        'beta': args.beta,
        'data_root': os.path.abspath(args.data_root),
        'num_clients': args.num_clients,
        'backbone': args.backbone,
        'width_mult': args.width_mult,
        'num_experts': args.num_experts,
        'topk': args.topk,
        'expert_bw': args.expert_bw,
        'bw_min': bw_min,
        'bw_max': bw_max,
        'snr_db': args.snr_db,
        'drop_expert_prob': float(getattr(args, 'drop_expert_prob', 0.0) or 0.0),
        'rounds': args.rounds,
        'frac': args.frac,
        'local_epochs': args.local_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lr_min': args.lr_min,
        'warmup_rounds': args.warmup_rounds,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'lam': args.lam,
        'agg_iters': args.agg_iters,
        'agg_lr': args.agg_lr,
        'kron_samples': args.kron_samples,
        'probe_batches': args.probe_batches,
        'n_ref': int(getattr(args, 'n_ref', 0)),
        'server_momentum': args.server_momentum,
        'server_lr': args.server_lr,
        'fedadam_lr': args.fedadam_lr,
        'fedadam_beta1': args.fedadam_beta1,
        'fedadam_beta2': args.fedadam_beta2,
        'fedadam_tau': args.fedadam_tau,
        'label_smooth': args.label_smooth,
        'seed': args.seed,
        'device': str(device),
        'num_workers': args.num_workers,
        'max_train_samples': args.max_train_samples,
        'max_test_samples': args.max_test_samples,
        'demo': bool(args.demo),
        'num_classes': cfg['num_classes'],
        'in_channels': cfg['in_channels'],
        'img_size': cfg['img_size'],
        'train_samples': len(train_ds),
        'test_samples': len(test_ds),
        'client_sizes': ','.join(str(s) for s in client_sizes),
        'model_params': n_params,
        'clients_per_round': m,
        'best_acc': best_acc,
        'best_round': best_round,
        'final_acc': final_acc,
        'start_time': experiment_start.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': experiment_end.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_sec': round(total_time, 4),
        'total_time_hms': format_hms(total_time),
        'avg_round_time_sec': round(avg_round_time, 4),
    }
    paths = export_experiment(result_dir, config, history_records)
    print(f'[Export] 实验目录: {paths["result_dir"]}')
    print(f'[Export] 每轮明细: {paths["rounds_xlsx"]}')
    print(f'[Export] 总体摘要: {paths["summary_xlsx"]}')
    try:
        from plot.plot_results import plot_experiment
        figure_dir = plot_experiment(paths['result_dir'])
        print(f'[Export] 图片目录: {figure_dir}')
    except Exception as exc:
        print(f'[Plot] 画图失败，实验数据已保存: {exc}')


if __name__ == '__main__':
    main()
