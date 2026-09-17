from .base import MergeBase
from .fedadam import FedAdam
from .fedavg import FedAvg
from .fedavgm import FedAvgM
from .ours import Ours
from .zerofill import ZeroFill

MERGE_REGISTRY = {
    'ours': Ours,
    'fedavg': FedAvg,
    'fedavgm': FedAvgM,
    'fedadam': FedAdam,
    'zerofill': ZeroFill,
}

__all__ = [
    'MergeBase', 'Ours', 'FedAvg', 'FedAvgM', 'FedAdam', 'ZeroFill',
    'MERGE_REGISTRY', 'build_merger',
]


def build_merger(name, args, device):
    if not name:
        return None
    key = str(name).lower()
    try:
        cls = MERGE_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f'Unknown merge method: {name}. Available: {sorted(MERGE_REGISTRY)}'
        ) from exc
    return cls(args, device)
