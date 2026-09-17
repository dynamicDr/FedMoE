import torch


class MergeBase:
    """Aggregation interface. New policies only need to implement `merge`."""

    name = 'base'
    description = ''

    def __init__(self, args, device):
        self.args = args
        self.device = device

    def merge(self, global_model, payloads):
        """Return the next global state dict from client payloads."""
        raise NotImplementedError


def weighted_mean(tensors, weights, dtype):
    stacked = torch.stack(tensors)
    w = torch.tensor(weights, dtype=stacked.dtype)
    w = w / w.sum().clamp_min(1e-12)
    w = w.view(-1, *([1] * (stacked.ndim - 1)))
    return (stacked * w).sum(0).to(dtype)


def payload_sample_weight(payload):
    return max(float(payload['meta'].get('n_samples', 1)), 1.0)
