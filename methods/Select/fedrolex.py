from .base import SelectBase


class FedRolex(SelectBase):
    """Rolling window of K experts, advanced after each client round."""

    name = 'fedrolex'
    description = 'rolling-window expert selection'

    def __init__(self, args, device):
        super().__init__(args, device)
        self.cursors = {}

    def select(self, model, loader, k, client_id=0, round_id=None, **kwargs):
        num_experts = int(self.args.num_experts)
        k = int(max(0, min(k, num_experts)))
        if k <= 0:
            return []
        if k >= num_experts:
            return list(range(num_experts))
        cid = 0 if client_id is None else int(client_id)
        start = self.cursors.get(cid, cid % num_experts)
        expert_ids = [(start + offset) % num_experts for offset in range(k)]
        self.cursors[cid] = (start + k) % num_experts
        return sorted(expert_ids)
