from .ours import Ours


class Posthoc(Ours):
    """Train all experts, then upload top-K by the same gate-score utility."""

    name = 'posthoc'
    description = 'train all experts, then upload top-k by gate scores'
    after_train = True
    dense_experts = True
