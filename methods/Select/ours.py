import numpy as np
import torch

from .base import SelectBase


class Ours(SelectBase):
    """Probe gate probabilities, refresh stale experts, then keep Top-K."""

    name = 'ours'
    description = 'probe gate scores, refresh stale experts, then train selected'

    def __init__(self, args, device):
        super().__init__(args, device)
        self.last_trained = {}

    def select(self, model, loader, k, client_id=None, round_id=None, **kwargs):
        utilities = self.probe_utilities(model, loader)
        tau = self._client_tau(client_id)
        expert_ids = select_with_refresh(
            utilities, k, tau, round_id, resolve_n_ref(self.args),
        )
        self._mark_trained(tau, expert_ids, round_id)
        return expert_ids

    def _client_tau(self, client_id):
        cid = 0 if client_id is None else int(client_id)
        num_experts = int(self.args.num_experts)
        tau = self.last_trained.get(cid)
        if tau is None or len(tau) != num_experts:
            tau = np.zeros(num_experts, dtype=np.int64)
            self.last_trained[cid] = tau
        return tau

    @staticmethod
    def _mark_trained(tau, expert_ids, round_id):
        if round_id is None:
            return
        rnd = int(round_id)
        for eid in expert_ids:
            tau[int(eid)] = rnd

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


def resolve_n_ref(args):
    raw = int(getattr(args, 'n_ref', 0))
    if raw < 0:
        return -1
    if raw == 0:
        return max(int(args.num_experts), 1)
    return raw


def select_topk_experts(utilities, k):
    m = len(utilities)
    k = int(max(0, min(k, m)))
    if k <= 0:
        return []
    if k >= m:
        return list(range(m))
    order = sorted(range(m), key=lambda j: (-float(utilities[j]), j))
    return sorted(order[:k])


def select_with_refresh(utilities, k, last_trained, round_id, n_ref):
    """Force-select experts stale for n_ref rounds, then fill by utility."""
    if n_ref <= 0 or round_id is None:
        return select_topk_experts(utilities, k)
    m = len(utilities)
    k = int(max(0, min(k, m)))
    if k <= 0:
        return []
    if k >= m:
        return list(range(m))
    rnd = int(round_id)
    mandatory = [
        j for j in range(m)
        if rnd - int(last_trained[j]) >= int(n_ref)
    ]
    if len(mandatory) > k:
        mandatory.sort(key=lambda j: (int(last_trained[j]), j))
        return sorted(mandatory[:k])
    chosen = set(mandatory)
    remaining = k - len(chosen)
    if remaining > 0:
        rest = [j for j in range(m) if j not in chosen]
        rest.sort(key=lambda j: (-float(utilities[j]), j))
        chosen.update(rest[:remaining])
    return sorted(chosen)
