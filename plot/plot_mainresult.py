from pathlib import Path
import json
from plot.plot_multi import plot_multi

root = Path('/userhome/cs3/duanty/FedMoE')
out_root = root / 'MainResult'
order = [
    ('ours', 'Ours'),
    ('fedrolex', 'FedRolex'),
    ('random', 'Random'),
    ('zerofill', 'FedAvg'),
    ('fedadam-tau1e2', 'FedAdam'),
]
groups = [
    ('c10', 'resnet', 'main_cifar10_resnet', 'CIFAR-10 / ResNet'),
    ('c10', 'vgg11', 'main_cifar10_vgg11', 'CIFAR-10 / VGG11'),
    ('c100', 'resnet', 'main_cifar100_resnet', 'CIFAR-100 / ResNet'),
    ('c100', 'vgg11', 'main_cifar100_vgg11', 'CIFAR-100 / VGG11'),
    ('svhn', 'resnet', 'main_svhn_resnet', 'SVHN / ResNet'),
    ('svhn', 'vgg11', 'main_svhn_vgg11', 'SVHN / VGG11'),
    ('stl10', 'resnet', 'main_stl10_resnet', 'STL-10 / ResNet'),
    ('stl10', 'vgg11', 'main_stl10_vgg11', 'STL-10 / VGG11'),
]


def latest_run(exp_dir: Path) -> Path:
    runs = [p for p in exp_dir.iterdir() if p.is_dir() and (p / 'experiment.json').exists()]
    if not runs:
        raise FileNotFoundError(exp_dir)

    def key(p):
        data = json.loads((p / 'experiment.json').read_text())
        return data.get('summary', {}).get('end_time') or p.name

    return max(runs, key=key)


def fmt_acc(val):
    if val is None:
        return '-'
    x = float(val)
    return f'{x:.2f}' if x < 90 else f'{x:.2f}'


def write_notes(table):
    lines = [
        '主实验（Main Result）',
        '====================',
        '',
        '本组图是论文主实验的 baseline 对比，来自 exp01（8 专家、带宽随专家数放大）。',
        '图例方法名：Ours, FedRolex, Random, FedAvg, FedAdam。',
        '结果目录：MainResult/main_<dataset>_<backbone>/，主图为 fig01_test_accuracy。',
        '',
        '方法名与实现',
        '------------',
        '图例      选择器 --select    聚合 --merge     说明',
        'Ours      ours              ours            探测门控后选 Top-K 专家训练；服务器按路由 token 加权，跳过未上传专家',
        'FedRolex  fedrolex          ours            滚动窗口选 K 个专家；聚合与 Ours 相同',
        'Random    random            ours            均匀随机选 K 个专家；聚合与 Ours 相同',
        'FedAvg    ours              fedavg          选择与 Ours 相同；未上传专家权重填 0，再按样本量对全部客户端平均（原 Zero-fill）',
        'FedAdam   ours              fedadam         选择与 Ours 相同；skip-missing FedAvg 后，服务器对可训练参数做 Adam',
        '',
        'FedAdam 超参（主实验采用 tau=0.01 这一组，不是默认 tau=0.001）',
        '  --fedadam_lr     0.01',
        '  --fedadam_beta1  0.9',
        '  --fedadam_beta2  0.99',
        '  --fedadam_tau    0.01',
        '对应结果目录后缀：exp01-*-fedadam-tau1e2',
        '',
        '共用训练设置',
        '------------',
        '  脚本              exp01_e8_baselines.sh',
        '  专家数            8',
        '  Top-K             2',
        '  客户端            10，每轮全参与 frac=1.0',
        '  数据划分          Dirichlet beta=0.1',
        '  通信轮数          100',
        '  本地 epoch        2',
        '  batch size        64',
        '  学习率            0.01，warmup 5 轮，lr_min=1e-4',
        '  SGD momentum      0.9',
        '  weight decay      1e-4',
        '  label smoothing   0.1',
        '  seed              42',
        '  骨干宽度          width_mult=1.0',
        '  专家刷新          n_ref=0（自动 = num_experts = 8）',
        '  探测 batch        probe_batches=2（ours 选择器用）',
        '',
        '带宽',
        '----',
        '  expert_bw=10，--bw_max -1  =>  bw=[10, 80]',
        '  即 K = floor(bw/10) ∈ [1, 8]，上界可上传全部专家。',
        '  时变：Rayleigh-Shannon，SNR=10 dB。',
        '',
        '设置网格',
        '--------',
        '  数据集   CIFAR-10, CIFAR-100, SVHN, STL-10',
        '  骨干     ResNet, VGG11',
        '  共 4×2=8 组对比，每组 5 条 baseline。',
        '',
        '最佳测试精度（%）',
        '----------------',
        f"{'设置':<22} {'Ours':>8} {'FedRolex':>8} {'Random':>8} {'FedAvg':>8} {'FedAdam':>8}",
    ]
    for title, folder, accs, _dirs in table:
        row = f'{title:<22}' + ''.join(f'{fmt_acc(a):>8}' for a in accs)
        lines.append(row)
    lines += [
        '',
        '原始结果目录',
        '------------',
        '每条曲线对应 results/exp01-<dtag>-<backbone>-e8-<method>/ 下的一次完整 100 轮实验。',
        'method 标签：ours, fedrolex, random, zerofill, fedadam-tau1e2。',
        '',
        '未纳入主实验图的 exp01 变体（仅作消融，不在本组图中）',
        '  FedAdam 默认 tau=0.001',
        '  FedAdam lr=0.001',
        '  FedAdam beta2=0.999',
        '',
    ]
    for title, folder, accs, dirs in table:
        lines.append(f'[{title}]  图目录 MainResult/{folder}/')
        for label, d in zip([x[1] for x in order], dirs):
            lines.append(f'  {label:8}  {d}')
        lines.append('')
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / '实验说明.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {path}')


def main():
    table = []
    for dtag, backbone, folder, title in groups:
        dirs, labels, accs = [], [], []
        print(f'===== {title} -> MainResult/{folder} =====')
        for mtag, label in order:
            exp_dir = root / 'results' / f'exp01-{dtag}-{backbone}-e8-{mtag}'
            run = latest_run(exp_dir)
            data = json.loads((run / 'experiment.json').read_text())
            acc = (data.get('summary') or {}).get('best_acc')
            dirs.append(str(run))
            labels.append(label)
            accs.append(acc)
            print(f'  {label:12} best={acc}  {exp_dir.name}/{run.name}')
        plot_multi(dirs, out_dir=out_root / folder, labels=labels, verbose=False)
        table.append((title, folder, accs, dirs))
        print()
    write_notes(table)
    print('done')


if __name__ == '__main__':
    main()

