import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder


DATASET_CFG = {
    'cifar10':      {'num_classes': 10,  'in_channels': 3, 'img_size': 32},
    'cifar100':     {'num_classes': 100, 'in_channels': 3, 'img_size': 32},
    'svhn':         {'num_classes': 10,  'in_channels': 3, 'img_size': 32},
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


def _ensure_svhn(root):
    svhn_dir = os.path.join(root, 'SVHN')
    os.makedirs(svhn_dir, exist_ok=True)
    files = {
        'train_32x32.mat': [
            'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
            'https://ufldl.stanford.edu/housenumbers/train_32x32.mat',
        ],
        'test_32x32.mat': [
            'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
            'https://ufldl.stanford.edu/housenumbers/test_32x32.mat',
        ],
    }
    for name, urls in files.items():
        dest = os.path.join(svhn_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            continue
        _download_file(urls, dest, min_bytes=10 * 1024 * 1024)


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
    elif name == 'svhn':
        mean, std = (0.4377,0.4438,0.4728),(0.1980,0.2010,0.1970)
        tr = transforms.Compose([
             transforms.RandomCrop(32, 4),
             transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        _ensure_svhn(data_root)
        return (torchvision.datasets.SVHN(data_root, split='train', download=False, transform=tr),
                torchvision.datasets.SVHN(data_root, split='test',  download=False, transform=te))
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
    print('[Dataset] Preparing SVHN')
    _ensure_svhn(data_root)
    print('[Dataset] All datasets are ready.')
