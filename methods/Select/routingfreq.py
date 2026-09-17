import numpy as np
import torch

from .base import SelectBase
from .ours import select_topk_experts


class RoutingFreq(SelectBase):
    """Probe hard Top-K dispatch counts, then train the most-used experts."""

    name = 'routingfreq'
    description = 'probe routing frequency, then train selected experts'

    def select(self, model, loader, k, client_id=None, round_id=None, **kwargs):
        utilities = self.probe_routing_freq(model, loader)
        return select_topk_experts(utilities, k)

    @torch.no_grad()
    def probe_routing_freq(self, model, loader):
        num_experts = int(self.args.num_experts)
        n_probe = max(1, int(getattr(self.args, 'probe_batches', 2)))
        topk = int(getattr(model.moe_head.gating, 'topk', self.args.topk))
        was_training = model.training
        model.eval()
        scores = torch.zeros(num_experts, device=self.device)
        n = 0
        for b, (x, _) in enumerate(loader):
            if b >= n_probe:
                break
            x = x.to(self.device)
            feat = model.backbone(x)
            logits = model.moe_head.gating.gate(feat)
            probs = torch.softmax(logits.float(), dim=-1)
            topk_idx = probs.topk(topk, dim=-1).indices
            for i in range(num_experts):
                scores[i] += (topk_idx == i).any(dim=-1).sum()
            n += int(x.size(0))
        if was_training:
            model.train()
        if n == 0:
            return np.ones(num_experts, dtype=np.float64) / max(num_experts, 1)
        return (scores / n).detach().cpu().numpy()
