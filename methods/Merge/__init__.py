from .base import MergeBase
from .fedadam import FedAdam
from .fedavg import FedAvg
from .fedavg_old import FedAvgOld
from .fedavg_old_bias import FedAvgOldBias
from .fedavgm import FedAvgM
from .ours import Ours

MERGE_REGISTRY = {
    'ours': Ours,
    'fedavg': FedAvg,
    'fedavg-old': FedAvgOld,
    'fedavg-old-bias': FedAvgOldBias,
    'fedavgm': FedAvgM,
    'fedadam': FedAdam,
}

__all__ = [
    'MergeBase', 'Ours', 'FedAvg', 'FedAvgOld', 'FedAvgOldBias', 'FedAvgM', 'FedAdam',
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
