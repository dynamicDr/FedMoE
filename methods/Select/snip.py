import copy

import numpy as np
import torch
import torch.nn as nn

from .base import SelectBase
from .ours import select_topk_experts


class SNIP(SelectBase):
    """Probe connection sensitivity with a few backward passes, then train Top-K."""

    name = 'snip'
    description = 'probe SNIP scores, then train selected experts'

    def select(self, model, loader, k, client_id=None, round_id=None, **kwargs):
        utilities = self.probe_snip(model, loader)
        return select_topk_experts(utilities, k)

    def probe_snip(self, model, loader):
        num_experts = int(self.args.num_experts)
        n_probe = max(1, int(getattr(self.args, 'probe_batches', 2)))
        label_smooth = float(getattr(self.args, 'label_smooth', 0.0))
        probe = copy.deepcopy(model).to(self.device)
        probe.eval()
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        scores = np.zeros(num_experts, dtype=np.float64)
        n = 0
        try:
            for b, (x, y) in enumerate(loader):
                if b >= n_probe:
                    break
                x = x.to(self.device)
                y = y.to(self.device)
                probe.zero_grad(set_to_none=True)
                loss = criterion(probe(x), y)
                loss.backward()
                for eid, expert in enumerate(probe.moe_head.experts):
                    saliency = 0.0
                    for param in expert.parameters():
                        if param.grad is None:
                            continue
                        saliency += float(
                            (param.detach() * param.grad.detach()).abs().sum().cpu()
                        )
                    scores[eid] += saliency
                n += int(x.size(0))
        finally:
            probe.zero_grad(set_to_none=True)
            del probe
        if n == 0 or not np.any(scores):
            return np.ones(num_experts, dtype=np.float64) / max(num_experts, 1)
        return scores
