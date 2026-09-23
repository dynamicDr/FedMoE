from pathlib import Path
import json
from plot.plot_multi import plot_multi

root = Path('/userhome/cs3/duanty/FedMoE')
order = [
    ('ours', 'Ours'),
    ('fedrolex', 'FedRolex'),
    ('random', 'Random'),
    ('zerofill', 'Zero-fill'),
    ('fedadam-tau1e2', 'FedAdam'),
]
groups = [
    ('c10', 'resnet', 'plot/exp01new_cifar10_resnet'),
    ('c10', 'vgg11', 'plot/exp01new_cifar10_vgg11'),
    ('c100', 'resnet', 'plot/exp01new_cifar100_resnet'),
    ('c100', 'vgg11', 'plot/exp01new_cifar100_vgg11'),
    ('svhn', 'resnet', 'plot/exp01new_svhn_resnet'),
    ('svhn', 'vgg11', 'plot/exp01new_svhn_vgg11'),
    ('stl10', 'resnet', 'plot/exp01new_stl10_resnet'),
    ('stl10', 'vgg11', 'plot/exp01new_stl10_vgg11'),
]


def latest_run(exp_dir: Path) -> Path:
    runs = [p for p in exp_dir.iterdir() if p.is_dir() and (p / 'experiment.json').exists()]
    if not runs:
        raise FileNotFoundError(exp_dir)

    def key(p):
        data = json.loads((p / 'experiment.json').read_text())
        return data.get('summary', {}).get('end_time') or p.name

    return max(runs, key=key)


def main():
    for dtag, backbone, out in groups:
        dirs, labels = [], []
        print(f'===== {out} =====')
        for mtag, label in order:
            exp_dir = root / 'results' / f'exp01-{dtag}-{backbone}-e8-{mtag}'
            run = latest_run(exp_dir)
            data = json.loads((run / 'experiment.json').read_text())
            acc = (data.get('summary') or {}).get('best_acc')
            dirs.append(str(run))
            labels.append(label)
            print(f'  {label:12} best={acc}  {exp_dir.name}/{run.name}')
        plot_multi(dirs, out_dir=root / out, labels=labels, verbose=False)
        print()
    print('done')


if __name__ == '__main__':
    main()
