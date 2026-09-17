from .base import MergeBase
from .fedavg import FedAvg
from .ours import Ours
from .zerofill import ZeroFill

MERGE_REGISTRY = {
    'ours': Ours,
    'fedavg': FedAvg,
    'zerofill': ZeroFill,
}

__all__ = [
    'MergeBase', 'Ours', 'FedAvg', 'ZeroFill',
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
