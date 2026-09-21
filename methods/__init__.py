from methods.bla_fedmoe import BLAFedMoE
from methods.moe import MoE
from methods.ours import Ours
from methods.random import Random


METHOD_REGISTRY = {
    'bla-fedmoe': BLAFedMoE,
    'moe': MoE,
    'ours': Ours,
    'random': Random,
}


def build_method(args, device):
    try:
        cls = METHOD_REGISTRY[args.method]
    except KeyError as exc:
        raise ValueError(f'Unknown method: {args.method}') from exc
    return cls(args, device)
