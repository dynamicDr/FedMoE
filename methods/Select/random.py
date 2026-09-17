from upload_budget import random_select_experts

from .base import SelectBase


class Random(SelectBase):
    """Bandwidth-limited uniform random expert selection."""

    name = 'random'
    description = 'random experts'

    def select(self, model, loader, k, client_id=0, round_id=0, **kwargs):
        return random_select_experts(
            self.args.num_experts, k, self.args.seed,
            client_id, round_id, salt='select-random',
        )
