"""训练最普通的联邦 MoE，使用最简单的数据集 CIFAR-10。

普通 MoE：每轮按带宽上传专家，服务器对参数做简单平均（不用 FedLPA）。
带宽设为 -1，表示够传全部专家、也不波动，等价于全专家上传。

用法（在项目根目录）：
    python train_simple_moe.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 参数含义见 Note.txt。这里只列出本次实验实际用到的项。
cmd = [
    sys.executable, str(ROOT / 'main.py'),

    # 算法：普通联邦 MoE（参数平均，不走 FedLPA）
    '--method', 'moe',

    # 数据集：CIFAR-10（五个数据集里最简单、最小）
    '--dataset', 'cifar10',
    '--data_root', './data',

    # Dirichlet 划分：0.1 表示客户端之间类别较不均衡
    '--beta', '0.1',

    # 联邦规模
    '--num_clients', '10',      # 客户端总数
    '--rounds', '100',          # 全局通信轮数
    '--frac', '1.0',            # 每轮全体客户端都参与
    '--local_epochs', '2',      # 每个客户端本轮本地训练 epoch 数
    '--batch_size', '64',       # 本地训练 batch 大小

    # MoE 结构
    '--num_experts', '4',       # 专家个数
    '--topk', '2',              # 前向时每个样本激活 2 个专家

    # 带宽：-1 表示不限制、不波动，上传全部专家
    '--expert_bw', '10',        # 上传 1 个专家的带宽代价
    '--bw_min', '-1',           # 不波动
    '--bw_max', '-1',           # 上限 = num_experts * expert_bw，够传全部专家

    # 优化器
    '--lr', '0.01',             # warmup 结束后的学习率
    '--lr_min', '0.0001',       # warmup 起始学习率
    '--warmup_rounds', '5',     # 前 5 轮从 lr_min 升到 lr
    '--momentum', '0.9',
    '--weight_decay', '0.0001',
    '--label_smooth', '0.1',

    # 可复现与设备
    '--seed', '42',
    '--device', 'auto',         # 有 GPU 用 GPU，否则 CPU
    '--num_workers', '0',       # Windows 建议保持 0

    # 全量数据（0 表示不抽样）
    '--max_train_samples', '0',
    '--max_test_samples', '0',

    '--output_dir', './results',
]


def main():
    print('运行普通 MoE @ CIFAR-10')
    print(' '.join(cmd))
    print()
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == '__main__':
    main()
