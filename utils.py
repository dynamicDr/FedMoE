import copy
import json
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def get_lr(rnd, total_rounds, lr_max, lr_min, warmup_rounds):
    if warmup_rounds > 0 and rnd <= warmup_rounds:
        return lr_min + (lr_max - lr_min) * rnd / warmup_rounds
    return lr_max


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_local_sgd(global_model, loader, device, local_epochs, lr,
                  momentum, weight_decay, label_smooth,
                  trainable_expert_ids=None, expert_usage_out=None):
    model = copy.deepcopy(global_model).to(device)
    if trainable_expert_ids is not None:
        keep = set(int(i) for i in trainable_expert_ids)
        for i, expert in enumerate(model.moe_head.experts):
            requires = i in keep
            for p in expert.parameters():
                p.requires_grad = requires
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.SGD(params, lr=lr,
                    momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    usage = None
    hook = None
    if expert_usage_out is not None:
        num_experts = len(model.moe_head.experts)
        usage = torch.zeros(num_experts, device=device)

        def _count_routed(mod, inp, out):
            _, topk_idx = out
            for i in range(num_experts):
                usage[i] += (topk_idx == i).any(dim=-1).sum()

        hook = model.moe_head.gating.register_forward_hook(_count_routed)

    total_loss, n = 0.0, 0
    try:
        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
    finally:
        if hook is not None:
            hook.remove()
    if expert_usage_out is not None and usage is not None:
        expert_usage_out.clear()
        counts = usage.detach().cpu().tolist()
        expert_usage_out.update({i: float(c) for i, c in enumerate(counts)})
    return model, total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


def format_hms(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f'{h:d}:{m:02d}:{s:05.2f}'
    return f'{m:d}:{s:05.2f}'


def make_result_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    result_dir = os.path.join(output_dir, stamp)
    suffix = 2
    while os.path.exists(result_dir):
        result_dir = os.path.join(output_dir, f'{stamp}-{suffix}')
        suffix += 1
    os.makedirs(result_dir, exist_ok=True)
    return os.path.abspath(result_dir)


def _kv_df(data):
    return pd.DataFrame({'Key': list(data.keys()), 'Value': [str(v) for v in data.values()]})


def save_rounds(result_dir, records):
    rounds_df = pd.DataFrame(records)
    csv_path = os.path.join(result_dir, 'rounds.csv')
    xlsx_path = os.path.join(result_dir, 'rounds.xlsx')
    rounds_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    rounds_df.to_excel(xlsx_path, sheet_name='rounds', index=False)
    return xlsx_path, csv_path


def export_experiment(result_dir, config, records):
    os.makedirs(result_dir, exist_ok=True)
    summary = {
        'best_acc': config.get('best_acc'),
        'best_round': config.get('best_round'),
        'final_acc': config.get('final_acc'),
        'num_rounds_completed': len(records),
        'total_time_sec': config.get('total_time_sec'),
        'total_time_hms': config.get('total_time_hms'),
        'avg_round_time_sec': config.get('avg_round_time_sec'),
        'result_dir': os.path.abspath(result_dir),
        'start_time': config.get('start_time'),
        'end_time': config.get('end_time'),
    }
    rounds_xlsx, rounds_csv = save_rounds(result_dir, records)
    summary_path = os.path.join(result_dir, 'summary.xlsx')
    config_path = os.path.join(result_dir, 'config.xlsx')
    _kv_df(summary).to_excel(summary_path, sheet_name='summary', index=False)
    _kv_df(config).to_excel(config_path, sheet_name='config', index=False)
    json_path = os.path.join(result_dir, 'experiment.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'config': config, 'rounds': records},
                  f, ensure_ascii=False, indent=2)
    return {
        'result_dir': os.path.abspath(result_dir),
        'rounds_xlsx': os.path.abspath(rounds_xlsx),
        'rounds_csv': os.path.abspath(rounds_csv),
        'summary_xlsx': os.path.abspath(summary_path),
        'config_xlsx': os.path.abspath(config_path),
        'json': os.path.abspath(json_path),
    }


def release_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
