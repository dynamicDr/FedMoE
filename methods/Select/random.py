from upload_budget import random_select_experts

from .base import SelectBase


class Random(SelectBase):
    """Before local SGD, uniformly sample K=floor(bw/expert_bw) experts."""

    name = 'random'
    description = 'random K experts from bandwidth, then train selected'

    def select(self, model, loader, k, client_id=0, round_id=0, **kwargs):
        return random_select_experts(
            self.args.num_experts, k, self.args.seed,
            client_id, round_id, salt='select-random',
        )
