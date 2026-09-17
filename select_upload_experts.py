"""BLA-FedMoE 上传专家选择。

TODO: 还没设计。当前按端云带宽随机选 k 个专家，与普通 MoE 相同。
"""
from upload_budget import filter_expert_payload, random_select_experts


class SelectUploadExperts:
    """Placeholder selector. Replace `select` with the designed policy later."""

    def __init__(self, seed=0):
        self.seed = seed

    def select(self, num_experts, k, client_id, round_id,
               model=None, state=None, kron=None, loader=None, **_):
        # TODO: 还没设计。此处仅为占位，按带宽随机选择要上传的专家。
        return random_select_experts(
            num_experts, k, seed=self.seed,
            client_id=client_id, round_id=round_id, salt='bla-fedmoe',
        )

    def filter_upload(self, state, extra, expert_ids):
        return filter_expert_payload(state, extra, expert_ids)
