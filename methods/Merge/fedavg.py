import torch

from .base import MergeBase, payload_sample_weight, weighted_mean


class FedAvg(MergeBase):
    """Treat missing expert weights as zeros, then average all clients."""

    name = 'fedavg'
    description = 'zero-fill missing experts, then n_i-weighted average'

    def merge(self, global_model, payloads):
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            tensors = []
            weights = []
            zero = torch.zeros_like(g_param.detach().cpu().float())
            for payload in payloads:
                if key in payload['state']:
                    tensors.append(payload['state'][key].float())
                else:
                    tensors.append(zero.clone())
                weights.append(payload_sample_weight(payload))
            new_state[key] = weighted_mean(tensors, weights, g_param.dtype)
        return new_state
