from upload_budget import expert_id_from_key

from .base import MergeBase, payload_sample_weight, weighted_mean


class Ours(MergeBase):
    """Skip missing experts; weight expert tensors by routed-token counts."""

    name = 'ours'
    description = 'token-weighted sparse expert aggregate'

    def merge(self, global_model, payloads):
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            present = []
            weights = []
            for payload in payloads:
                if key not in payload['state']:
                    continue
                present.append(payload['state'][key].float())
                weights.append(_expert_or_sample_weight(payload, key))
            if not present:
                new_state[key] = g_param.detach().cpu().clone().to(g_param.dtype)
                continue
            new_state[key] = weighted_mean(present, weights, g_param.dtype)
        return new_state


def _expert_or_sample_weight(payload, key):
    eid = expert_id_from_key(key)
    if eid is None:
        return payload_sample_weight(payload)
    tokens = payload['meta'].get('expert_tokens')
    if tokens is None:
        return payload_sample_weight(payload)
    if isinstance(tokens, dict):
        value = tokens.get(eid, tokens.get(str(eid), 0.0))
    elif eid < len(tokens):
        value = tokens[eid]
    else:
        value = 0.0
    return max(float(value), 1e-6)
