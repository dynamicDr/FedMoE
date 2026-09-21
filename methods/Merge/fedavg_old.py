from .base import MergeBase, payload_sample_weight, weighted_mean


class FedAvgOld(MergeBase):
    """Skip missing tensors; average only uploaded expert weights."""

    name = 'fedavg-old'
    description = 'sample-size-weighted sparse FedAvg'

    def merge(self, global_model, payloads):
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            present = []
            weights = []
            for payload in payloads:
                if key not in payload['state']:
                    continue
                present.append(payload['state'][key].float())
                weights.append(payload_sample_weight(payload))
            if not present:
                new_state[key] = g_param.detach().cpu().clone().to(g_param.dtype)
                continue
            new_state[key] = weighted_mean(present, weights, g_param.dtype)
        return new_state
