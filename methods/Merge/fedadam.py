import torch

from .fedavg import FedAvg


class FedAdam(FedAvg):
    """Skip-missing FedAvg, then FedAdam on uploaded tensors only."""

    name = 'fedadam'
    description = 'skip-missing FedAvg with server Adam'

    def __init__(self, args, device):
        super().__init__(args, device)
        self.m = {}
        self.v = {}
        self.beta1 = float(getattr(args, 'fedadam_beta1', 0.9))
        self.beta2 = float(getattr(args, 'fedadam_beta2', 0.99))
        self.server_lr = float(getattr(args, 'fedadam_lr', 0.01))
        self.tau = float(getattr(args, 'fedadam_tau', 1e-3))

    def merge(self, global_model, payloads):
        averaged = super().merge(global_model, payloads)
        present_keys = set()
        for payload in payloads:
            present_keys.update(payload['state'].keys())
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            avg = averaged[key]
            if key not in present_keys:
                new_state[key] = avg
                continue
            g_cpu = g_param.detach().cpu().float()
            delta = avg.float() - g_cpu
            m = self.m.get(key)
            v = self.v.get(key)
            if m is None:
                m = torch.zeros_like(delta)
            if v is None:
                v = torch.zeros_like(delta)
            m = self.beta1 * m + (1.0 - self.beta1) * delta
            v = self.beta2 * v + (1.0 - self.beta2) * delta.square()
            self.m[key] = m
            self.v[key] = v
            update = m / (v.sqrt() + self.tau)
            new_state[key] = (g_cpu + self.server_lr * update).to(g_param.dtype)
        return new_state
