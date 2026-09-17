from methods.base import FederatedMethod


class Ours(FederatedMethod):
    """Probe-select experts, train the chosen ones, then token-weighted merge."""

    name = 'Ours'
    select_name = 'ours'
    merge_name = 'ours'
