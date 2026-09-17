class SelectBase:
    """Expert-selection interface. New policies only need to implement `select`."""

    name = 'base'
    description = ''
    after_train = False

    def __init__(self, args, device):
        self.args = args
        self.device = device

    def select(self, model, loader, k, client_id=None, round_id=None, **kwargs):
        """Return expert ids to train and/or upload this round."""
        raise NotImplementedError
