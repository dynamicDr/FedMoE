import numpy as np
import torch

from .base import SelectBase


class Ours(SelectBase):
    """Probe gate probabilities, then keep the top-K experts."""

    name = 'ours'
    description = 'probe gate scores, then train selected experts'

    def select(self, model, loader, k, client_id=None, round_id=None, **kwargs):
        utilities = self.probe_utilities(model, loader)
        return select_topk_experts(utilities, k)

    @torch.no_grad()
    def probe_utilities(self, model, loader):
        num_experts = int(self.args.num_experts)
        n_probe = max(1, int(getattr(self.args, 'probe_batches', 2)))
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
            scores += probs.sum(dim=0)
            n += int(x.size(0))
        if was_training:
            model.train()
        if n == 0:
            return np.ones(num_experts, dtype=np.float64) / max(num_experts, 1)
        return (scores / n).detach().cpu().numpy()


def select_topk_experts(utilities, k):
    m = len(utilities)
    k = int(max(0, min(k, m)))
    if k <= 0:
        return []
    if k >= m:
        return list(range(m))
    order = sorted(range(m), key=lambda j: (-float(utilities[j]), j))
    return sorted(order[:k])
