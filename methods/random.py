from methods.base import FederatedMethod


class Random(FederatedMethod):
    """Bandwidth-limited random K experts, train selected, then token-weighted merge."""

    name = 'Random'
    select_name = 'random'
    merge_name = 'ours'
