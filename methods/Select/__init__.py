from .base import SelectBase
from .magnitude import Magnitude
from .ours import Ours
from .posthoc import Posthoc
from .random import Random
from .routingfreq import RoutingFreq

SELECT_REGISTRY = {
    'ours': Ours,
    'random': Random,
    'posthoc': Posthoc,
    'routingfreq': RoutingFreq,
    'magnitude': Magnitude,
}

__all__ = [
    'SelectBase', 'Ours', 'Random', 'Posthoc', 'RoutingFreq', 'Magnitude',
    'SELECT_REGISTRY', 'build_selector',
]


def build_selector(name, args, device):
    if not name:
        return None
    key = str(name).lower()
    try:
        cls = SELECT_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f'Unknown select method: {name}. Available: {sorted(SELECT_REGISTRY)}'
        ) from exc
    return cls(args, device)
