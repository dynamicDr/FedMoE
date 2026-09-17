from methods.base import FederatedMethod
from upload_budget import filter_expert_state, plan_upload
from utils import cpu_state_dict, release_cuda


class MoE(FederatedMethod):
    """Federated MoE: bandwidth-limited random expert upload, mean aggregation."""

    name = 'MoE'

    def local_update(self, global_model, loader, client_id, round_id, lr):
        model, loss = self.local_train(global_model, loader, lr)
        state = cpu_state_dict(model)
        del model
        release_cuda()
        bandwidth, upload_k, expert_ids = plan_upload(
            self.args, client_id, round_id, salt='moe',
        )
        state = filter_expert_state(state, expert_ids)
        return self.payload(
            state, {}, loss, client_id, expert_ids,
            bandwidth=bandwidth, upload_k=upload_k,
        )

    def aggregate(self, global_model, payloads):
        return self.mean_aggregate(global_model, payloads)
