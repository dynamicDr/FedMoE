import os, copy, argparse, zipfile, tarfile, urllib.request, shutil, subprocess, time, json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',       default='cifar10',
                   choices=['cifar10','cifar100','tinyimagenet','cinic-10','stl10'])
    p.add_argument('--beta',          type=float, default=0.1)
    p.add_argument('--data_root',     default='./data')
    p.add_argument('--num_clients',   type=int,   default=10)
    p.add_argument('--num_experts',   type=int,   default=4)
    p.add_argument('--topk',          type=int,   default=2)
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
    # Keep the original algorithm, but shrink data and steps so a smoke run finishes quickly.
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
    args.max_train_samples = 256
    args.max_test_samples = 64
    args.num_workers = 0
    return args


DATASET_CFG = {
    'cifar10':      {'num_classes': 10,  'in_channels': 3, 'img_size': 32},
    'cifar100':     {'num_classes': 100, 'in_channels': 3, 'img_size': 32},
    'tinyimagenet': {'num_classes': 200, 'in_channels': 3, 'img_size': 64},
    'cinic-10':     {'num_classes': 10,  'in_channels': 3, 'img_size': 32},
    'stl10':        {'num_classes': 10,  'in_channels': 3, 'img_size': 96},
}

def _download_file(urls, dest, min_bytes=1024):
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
        print(f'[Dataset] Using existing file {dest} ({os.path.getsize(dest)} bytes)')
        return dest
    last_err = None
    curl = 'curl.exe' if os.name == 'nt' else 'curl'
    for url in urls:
        try:
            print(f'[Dataset] Downloading {os.path.basename(dest)} from {url}')
            cmd = [curl, '-fL', '--retry', '5', '-C', '-', '--connect-timeout', '30',
                   '-A', 'Mozilla/5.0', '-o', dest, url]
            subprocess.check_call(cmd)
            if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
                return dest
            raise RuntimeError('downloaded file is too small')
        except Exception as e:
            last_err = e
            print(f'[Dataset] curl failed: {e}')
            try:
                urllib.request.urlretrieve(url, dest)
                if os.path.exists(dest) and os.path.getsize(dest) > min_bytes:
                    return dest
            except Exception as e2:
                last_err = e2
                print(f'[Dataset] urllib failed: {e2}')
    raise RuntimeError(f'Failed to download {dest}') from last_err


def _extract_tar(tar_path, dest):
    print(f'[Dataset] Extracting {os.path.basename(tar_path)} ...')
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(dest)


def _ensure_cifar10(root):
    marker = os.path.join(root, 'cifar-10-batches-py', 'batches.meta')
    if os.path.exists(marker):
        return
    tar_path = os.path.join(root, 'cifar-10-python.tar.gz')
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1024:
        _download_file([
            'https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz',
            'https://github.com/Digital-Media/cv_data/releases/download/cifar-10/cifar-10-python.tar.gz',
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        ], tar_path)
    _extract_tar(tar_path, root)


def _ensure_cifar100(root):
    marker = os.path.join(root, 'cifar-100-python', 'meta')
    if os.path.exists(marker):
        return
    tar_path = os.path.join(root, 'cifar-100-python.tar.gz')
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1024:
        _download_file([
            'https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz',
            'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
        ], tar_path)
    _extract_tar(tar_path, root)


def _ensure_stl10(root):
    marker = os.path.join(root, 'stl10_binary', 'train_X.bin')
    if os.path.exists(marker):
        return
    tar_path = os.path.join(root, 'stl10_binary.tar.gz')
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1024:
        _download_file([
            'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz',
            'https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz',
        ], tar_path, min_bytes=10 * 1024 * 1024)
    _extract_tar(tar_path, root)


def _prepare_tinyimagenet(root):
    data_path = os.path.join(root, 'tiny-imagenet-200')
    train_ok = os.path.isdir(os.path.join(data_path, 'train'))
    if not train_ok:
        zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1024:
            _download_file([
                'https://cs231n.stanford.edu/tiny-imagenet-200.zip',
                'http://cs231n.stanford.edu/tiny-imagenet-200.zip',
            ], zip_path, min_bytes=10 * 1024 * 1024)
        print('[Dataset] Extracting TinyImageNet...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)
    val_dir = os.path.join(data_path, 'val')
    anno    = os.path.join(val_dir, 'val_annotations.txt')
    img_dir = os.path.join(val_dir, 'images')
    if os.path.exists(anno):
        with open(anno, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                cls_dir = os.path.join(val_dir, parts[1])
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(img_dir, parts[0])
                if os.path.exists(src):
                    os.rename(src, os.path.join(cls_dir, parts[0]))
        os.remove(anno)
        if os.path.isdir(img_dir) and not os.listdir(img_dir):
            os.rmdir(img_dir)
    return data_path


def _prepare_cinic10(root):
    data_path = os.path.join(root, 'CINIC-10')
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return data_path
    os.makedirs(data_path, exist_ok=True)
    tar_path = os.path.join(root, 'CINIC-10.tar.gz')
    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1024:
        _download_file([
            'https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=1&isAllowed=y',
            'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=1&isAllowed=y',
            'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz',
        ], tar_path, min_bytes=10 * 1024 * 1024)
    _extract_tar(tar_path, data_path)
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        # Some archives nest an extra CINIC-10 directory.
        nested = os.path.join(data_path, 'CINIC-10')
        if os.path.isdir(os.path.join(nested, 'train')):
            for name in os.listdir(nested):
                shutil.move(os.path.join(nested, name), os.path.join(data_path, name))
    return data_path


def get_dataset(name, data_root):
    os.makedirs(data_root, exist_ok=True)
    if name == 'cifar10':
        _ensure_cifar10(data_root)
        mean, std = (0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)
        tr = transforms.Compose([
             transforms.RandomCrop(32, 4),
             transforms.RandomHorizontalFlip(),
             transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return (torchvision.datasets.CIFAR10(data_root, True,  download=False, transform=tr),
                torchvision.datasets.CIFAR10(data_root, False, download=False, transform=te))
    elif name == 'cifar100':
        mean, std = (0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
        tr = transforms.Compose([
             transforms.RandomCrop(32, 4),
             transforms.RandomHorizontalFlip(),
             transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        _ensure_cifar100(data_root)
        return (torchvision.datasets.CIFAR100(data_root, True,  download=False, transform=tr),
                torchvision.datasets.CIFAR100(data_root, False, download=False, transform=te))
    elif name == 'tinyimagenet':
        dp = _prepare_tinyimagenet(data_root)
        mean, std = (0.4802,0.4481,0.3975),(0.2302,0.2265,0.2262)
        tr = transforms.Compose([
             transforms.RandomCrop(64, 8),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return (ImageFolder(os.path.join(dp,'train'), transform=tr),
                ImageFolder(os.path.join(dp,'val'),   transform=te))
    elif name == 'stl10':
        mean, std = (0.4467,0.4398,0.4066),(0.2603,0.2566,0.2713)
        tr = transforms.Compose([
             transforms.RandomCrop(96, 12),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        _ensure_stl10(data_root)
        return (torchvision.datasets.STL10(data_root, split='train', download=False, transform=tr),
                torchvision.datasets.STL10(data_root, split='test',  download=False, transform=te))
    elif name == 'cinic-10':
        dp = _prepare_cinic10(data_root)
        mean, std = (0.4789,0.4723,0.4305),(0.2421,0.2383,0.2587)
        tr = transforms.Compose([
             transforms.RandomCrop(32, 4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return (ImageFolder(os.path.join(dp,'train'), transform=tr),
                ImageFolder(os.path.join(dp,'test'),  transform=te))
    raise ValueError(f'Unsupported dataset: {name}')


def maybe_subset(dataset, max_samples, seed):
    if not max_samples or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=int(max_samples), replace=False).tolist()
    subset = Subset(dataset, indices)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    subset.targets = labels[indices].tolist()
    return subset


def partition_dirichlet(dataset, num_clients, beta, seed=42):
    # Create Non-IID client partitions with a Dirichlet distribution.
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets if hasattr(dataset,'targets') else dataset.labels)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(np.full(num_clients, beta))
        props = (props * len(idx_c)).astype(int)
        props[-1] = len(idx_c) - props[:-1].sum()
        cur = 0
        for i, cnt in enumerate(props):
            client_indices[i].extend(idx_c[cur:cur+cnt].tolist())
            cur += cnt
    empty_clients = [i for i, idx in enumerate(client_indices) if not idx]
    for empty_id in empty_clients:
        donor_id = max(range(num_clients), key=lambda i: len(client_indices[i]))
        if len(client_indices[donor_id]) <= 1:
            raise ValueError('The dataset has too few samples for the requested number of clients.')
        client_indices[empty_id].append(client_indices[donor_id].pop())
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    print(f'\n[Partition] Dirichlet beta={beta}, {num_clients} clients')
    for i, idx in enumerate(client_indices):
        cnt = np.bincount(labels[idx], minlength=num_classes)
        print(f'  Client {i}: {len(idx):>6d} samples | {np.count_nonzero(cnt)}/{num_classes} classes')
    print()
    return client_indices


def get_lr(rnd, total_rounds, lr_max, lr_min, warmup_rounds):
    if warmup_rounds > 0 and rnd <= warmup_rounds:
        return lr_min + (lr_max - lr_min) * rnd / warmup_rounds
    return lr_max

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, img_size):
        super().__init__()
        stem_stride = 1 if img_size <= 32 else 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    @staticmethod
    def _make_layer(in_ch, out_ch, stride):
        return nn.Sequential(
            BasicBlock(in_ch,  out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


class ExpertFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TopKGating(nn.Module):
    def __init__(self, in_dim, num_experts, topk):
        super().__init__()
        self.topk = topk
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)
        
        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
        
        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_vals)
        weights = weights.to(x.dtype)
        
        return weights, topk_idx


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, topk):
        super().__init__()
        self.gating  = TopKGating(in_dim, num_experts, topk)
        self.experts = nn.ModuleList([
            ExpertFFN(in_dim, hidden_dim, out_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        weights, topk_idx = self.gating(x)
        out = torch.zeros(x.size(0), self.experts[0].fc2.out_features, device=x.device, dtype=x.dtype)
        
        for i, expert in enumerate(self.experts):
            batch_mask = (topk_idx == i).any(dim=-1)
            if batch_mask.any():
                expert_output = expert(x[batch_mask])
                gate_scores = weights[batch_mask, i].unsqueeze(-1)
                out[batch_mask] += expert_output * gate_scores
                
        return out


class MoEFedModel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, num_experts, topk):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, img_size)
        feat_dim = self.backbone.feat_dim
        self.moe_head = MoELayer(feat_dim, 512, num_classes, num_experts, topk)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.moe_head(feat)
        return logits

def compute_kronecker_factors(model, loader, device, lam, max_samples=256):
    # Estimate layer-wise Kronecker factors with forward and backward hooks.
    model.eval()
    _acts  = {}
    _grads = {}
    _hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        def _fwd(n, has_bias):
            def hook(mod, inp, out):
                a = inp[0].detach().clone()
                if has_bias:
                    ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
                    a = torch.cat([a, ones], dim=1)
                _acts.setdefault(n, []).append(a)
            return hook

        def _bwd(n):
            def hook(mod, grad_in, grad_out):
                _grads.setdefault(n, []).append(grad_out[0].detach().clone())
            return hook

        _hooks.append(module.register_forward_hook(_fwd(name, module.bias is not None)))
        _hooks.append(module.register_full_backward_hook(_bwd(name)))

    count = 0
    for x, y in loader:
        if count >= max_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        nn.CrossEntropyLoss()(logits, y).backward()
        del logits
        count += x.size(0)

    for h in _hooks:
        h.remove()
    _hooks.clear()
    model.zero_grad()

    kron = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in _acts or name not in _grads:
            continue
        a_cat = torch.cat(_acts[name],  dim=0).float()
        g_cat = torch.cat(_grads[name], dim=0).float()
        del _acts[name], _grads[name]
        
        A = (a_cat.T @ a_cat) / a_cat.size(0)
        A = A + lam * torch.eye(A.size(0), device=device)
        B = (g_cat.T @ g_cat) / g_cat.size(0)
        B = B + lam * torch.eye(B.size(0), device=device)
        del a_cat, g_cat
        
        W = module.weight.detach().float()
        if module.bias is not None:
            M = torch.cat([W, module.bias.detach().float().unsqueeze(1)], dim=1)
        else:
            M = W.clone()
        del W
        kron[name] = {'A': A.cpu(), 'B': B.cpu(), 'M': M.cpu()}
        del A, B

    _acts.clear()
    _grads.clear()
    return kron


def local_train(global_model, loader, device, local_epochs, lr,
                momentum, weight_decay, lam, kron_samples, label_smooth):
    model = copy.deepcopy(global_model).to(device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr,
                    momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    total_loss, n = 0.0, 0
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

    kron  = compute_kronecker_factors(model, loader, device, lam, kron_samples)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, kron, total_loss / max(n, 1)

def fedlpa_layer_agg(Ms, As, Bs, agg_iters, agg_lr, device):
    # Iteratively solve the curvature-weighted layer-wise expert parameters.
    K  = len(Ms)
    Ms = [m.float().to(device) for m in Ms]
    As = [a.float().to(device) for a in As]
    Bs = [b.float().to(device) for b in Bs]
    with torch.no_grad():
        z_bar = sum(Bs[k] @ Ms[k] @ As[k] for k in range(K))
    M_bar = torch.stack(Ms).mean(0).clone().detach().requires_grad_(True)
    optimizer = optim.Adam([M_bar], lr=agg_lr)
    for _ in range(agg_iters):
        optimizer.zero_grad()
        pred = sum(Bs[k] @ M_bar @ As[k] for k in range(K))
        loss = 0.5 * (pred - z_bar).pow(2).sum()
        loss.backward()
        optimizer.step()
    result = M_bar.detach().cpu()
    del Ms, As, Bs, z_bar, M_bar, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def moe_same_expert_layerwise_aggregate(global_model, client_states, client_krons,
                                        agg_iters, agg_lr, device):
    new_state = {}
    linear_modules = {
        name: mod for name, mod in global_model.named_modules()
        if isinstance(mod, nn.Linear)
    }
    for key, g_param in global_model.state_dict().items():
        stacked = torch.stack([cs[key].float() for cs in client_states])
        matched_linear = None
        for ln in linear_modules:
            if key == ln + '.weight' or key == ln + '.bias':
                matched_linear = ln
                break
        is_expert_linear = (
            matched_linear is not None
            and 'moe_head.experts.' in matched_linear
            and all(matched_linear in ck for ck in client_krons)
        )
        if not is_expert_linear:
            new_state[key] = stacked.mean(0).to(g_param.dtype)
            continue
        
        As = [ck[matched_linear]['A'] for ck in client_krons]
        Bs = [ck[matched_linear]['B'] for ck in client_krons]
        Ms = [ck[matched_linear]['M'] for ck in client_krons]
        M_opt    = fedlpa_layer_agg(Ms, As, Bs, agg_iters, agg_lr, device)
        has_bias = linear_modules[matched_linear].bias is not None
        
        if key.endswith('.weight'):
            new_state[key] = (M_opt[:, :-1] if has_bias else M_opt).to(g_param.dtype)
        elif key.endswith('.bias') and has_bias:
            new_state[key] = M_opt[:, -1].to(g_param.dtype)
        else:
            new_state[key] = stacked.mean(0).to(g_param.dtype)
    return new_state

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


def _format_hms(seconds):
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


def download_all_datasets(data_root):
    os.makedirs(data_root, exist_ok=True)
    print('[Dataset] Preparing CIFAR-10')
    _ensure_cifar10(data_root)
    print('[Dataset] Preparing CIFAR-100')
    _ensure_cifar100(data_root)
    print('[Dataset] Preparing STL-10')
    _ensure_stl10(data_root)
    print('[Dataset] Preparing TinyImageNet')
    _prepare_tinyimagenet(data_root)
    print('[Dataset] Preparing CINIC-10')
    _prepare_cinic10(data_root)
    print('[Dataset] All datasets are ready.')


def main():
    args = get_args()
    if getattr(args, 'download_all', False):
        download_all_datasets(args.data_root)
        return
    if args.demo:
        args = apply_demo_defaults(args)
        print('[Demo] Using a CIFAR-10 subset with 1 round / 1 local epoch.')

    if args.device == 'auto':
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = DATASET_CFG[args.dataset]
    pin_memory = device.type == 'cuda'

    print(f'\n{"="*66}')
    print(f' BLA-FedMoE  ')
    print(f'  Dataset={args.dataset} | beta={args.beta} | '
          f'Clients={args.num_clients} | Experts={args.num_experts}')
    print(f'  Rounds={args.rounds} | LR={args.lr}')
    print(f'{"="*66}\n')

    train_ds, test_ds = get_dataset(args.dataset, args.data_root)
    train_ds = maybe_subset(train_ds, args.max_train_samples, args.seed)
    test_ds  = maybe_subset(test_ds,  args.max_test_samples,  args.seed + 1)
    client_idx = partition_dirichlet(train_ds, args.num_clients, args.beta, args.seed)
    client_loaders = [
        DataLoader(Subset(train_ds, idx), batch_size=args.batch_size,
                   shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
        for idx in client_idx
    ]
    test_loader = DataLoader(test_ds, batch_size=min(256, args.batch_size * 4), shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    global_model = MoEFedModel(
        in_channels=cfg['in_channels'],
        num_classes=cfg['num_classes'],
        img_size=cfg['img_size'],
        num_experts=args.num_experts,
        topk=args.topk,
    ).to(device)

    n_params = sum(p.numel() for p in global_model.parameters())
    print(f'[Model] Total params: {n_params:,}\n')

    result_dir = make_result_dir(args.output_dir)
    print(f'[Export] 本次实验目录: {result_dir}\n')
    experiment_start = datetime.now()
    experiment_t0 = time.perf_counter()

    m        = max(1, int(args.num_clients * args.frac))
    best_acc = 0.0
    best_round = 0
    print(f'{"Round":>5} | {"LR":>7} | {"AvgLoss":>8} | {"TestAcc":>8} | {"Best":>8} | {"RoundTime":>10}')
    print('-' * 68)

    history_records = []
    client_sizes = [len(idx) for idx in client_idx]

    for rnd in range(1, args.rounds + 1):
        round_t0 = time.perf_counter()
        current_lr = get_lr(rnd, args.rounds, args.lr, args.lr_min, args.warmup_rounds)
        chosen = np.random.choice(args.num_clients, m, replace=False).tolist()
        all_states, all_krons, all_losses = [], [], []
        client_times = []

        local_t0 = time.perf_counter()
        for cid in chosen:
            client_t0 = time.perf_counter()
            state, kron, loss = local_train(
                global_model, client_loaders[cid], device,
                args.local_epochs, current_lr, args.momentum,
                args.weight_decay, args.lam, args.kron_samples,
                args.label_smooth,
            )
            client_times.append(time.perf_counter() - client_t0)
            all_states.append(state)
            all_krons.append(kron)
            all_losses.append(loss)
        local_time = time.perf_counter() - local_t0

        agg_t0 = time.perf_counter()
        new_state = moe_same_expert_layerwise_aggregate(
            global_model, all_states, all_krons,
            args.agg_iters, args.agg_lr, device,
        )
        global_model.load_state_dict(new_state)
        agg_time = time.perf_counter() - agg_t0

        avg_loss = float(np.mean(all_losses))
        del all_states, all_krons, all_losses, new_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_t0 = time.perf_counter()
        acc      = evaluate(global_model, test_loader, device)
        eval_time = time.perf_counter() - eval_t0
        if acc > best_acc:
            best_acc = acc
            best_round = rnd
        round_time = time.perf_counter() - round_t0
        elapsed = time.perf_counter() - experiment_t0
        print(f'{rnd:>5} | {current_lr:>7.5f} | {avg_loss:>8.4f} | '
              f'{acc:>7.2f}% | {best_acc:>7.2f}% | {_format_hms(round_time):>10}')

        history_records.append({
            'Round': rnd,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'LR': current_lr,
            'AvgLoss': avg_loss,
            'TestAcc': acc,
            'BestAcc': best_acc,
            'BestRound': best_round,
            'RoundTimeSec': round(round_time, 4),
            'RoundTimeHMS': _format_hms(round_time),
            'LocalTrainTimeSec': round(local_time, 4),
            'AggTimeSec': round(agg_time, 4),
            'EvalTimeSec': round(eval_time, 4),
            'CumulativeTimeSec': round(elapsed, 4),
            'CumulativeTimeHMS': _format_hms(elapsed),
            'NumClientsSelected': len(chosen),
            'SelectedClients': ','.join(str(c) for c in chosen),
            'ClientTrainTimesSec': ','.join(f'{t:.4f}' for t in client_times),
        })
        save_rounds(result_dir, history_records)

    total_time = time.perf_counter() - experiment_t0
    experiment_end = datetime.now()
    avg_round_time = total_time / max(len(history_records), 1)
    final_acc = history_records[-1]['TestAcc'] if history_records else 0.0

    print(f'\nDone. Best Acc: {best_acc:.2f}% @ round {best_round}')
    print(f'Total time: {_format_hms(total_time)}')

    config = {
        'experiment': 'BLA-FedMoE',
        'dataset': args.dataset,
        'beta': args.beta,
        'data_root': os.path.abspath(args.data_root),
        'num_clients': args.num_clients,
        'num_experts': args.num_experts,
        'topk': args.topk,
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
        'total_time_hms': _format_hms(total_time),
        'avg_round_time_sec': round(avg_round_time, 4),
    }
    paths = export_experiment(result_dir, config, history_records)
    print(f'[Export] 实验目录: {paths["result_dir"]}')
    print(f'[Export] 每轮明细: {paths["rounds_xlsx"]}')
    print(f'[Export] 总体摘要: {paths["summary_xlsx"]}')


if __name__ == '__main__':
    main()
