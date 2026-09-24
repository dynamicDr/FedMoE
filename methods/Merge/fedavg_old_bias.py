import torch

from upload_budget import EXPERT_PREFIX, expert_id_from_key

from .base import payload_sample_weight, weighted_mean
from .fedavg_old import FedAvgOld


class FedAvgOldBias(FedAvgOld):
    """FedAvg-old, but each uploaded expert is aggregated with a random weight.

    Missing experts are still skipped. Shared parameters stay sample-size
    weighted. For every expert tensor, each client that uploaded it gets
    weight n_samples * U(0, 1), redrawn every round. A single shared random
    number per expert would cancel inside the average, so the draw is
    per client and per expert.
    """

    name = 'fedavg-old-bias'
    description = 'skip-missing FedAvg with a random per-client expert weight'

    def merge(self, global_model, payloads):
        self._round = int(getattr(self, '_round', 0)) + 1
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            present = []
            weights = []
            expert_id = expert_id_from_key(key) if key.startswith(EXPERT_PREFIX) else None
            for payload in payloads:
                if key not in payload['state']:
                    continue
                present.append(payload['state'][key].float())
                base = payload_sample_weight(payload)
                if expert_id is None:
                    weights.append(base)
                    continue
                cid = int(payload['meta'].get('client_id', 0))
                u = self._uniform(cid, expert_id)
                weights.append(base * u)
            if not present:
                new_state[key] = g_param.detach().cpu().clone().to(g_param.dtype)
                continue
            new_state[key] = weighted_mean(present, weights, g_param.dtype)
        return new_state

    def _uniform(self, client_id, expert_id):
        seed = int(getattr(self.args, 'seed', 0))
        # Independent of the bandwidth RNG. One draw per client, expert, round.
        g = torch.Generator()
        g.manual_seed(seed + self._round * 100003 + client_id * 1009 + int(expert_id))
        return float(torch.rand(1, generator=g).item())
