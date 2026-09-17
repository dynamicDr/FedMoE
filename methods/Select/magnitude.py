import numpy as np

from upload_budget import expert_id_from_key

from .base import SelectBase
from .ours import select_topk_experts


class Magnitude(SelectBase):
    """Train all experts, then upload those with the largest update norms."""

    name = 'magnitude'
    description = 'train all experts, then upload top-k by update magnitude'
    after_train = True

    def select(self, model, loader, k, client_id=None, round_id=None,
               trained_model=None, global_model=None, **kwargs):
        trained = trained_model if trained_model is not None else model
        reference = global_model if global_model is not None else model
        t_state = trained.state_dict()
        g_state = reference.state_dict()
        num_experts = int(self.args.num_experts)
        scores = np.zeros(num_experts, dtype=np.float64)
        for key, t_param in t_state.items():
            eid = expert_id_from_key(key)
            if eid is None or key not in g_state:
                continue
            delta = t_param.detach().float().cpu() - g_state[key].detach().float().cpu()
            scores[eid] += float(delta.pow(2).sum())
        scores = np.sqrt(np.maximum(scores, 0.0))
        return select_topk_experts(scores, k)
