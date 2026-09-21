import torch

from .base import collect_present_keys, trainable_param_names
from .fedavg_old import FedAvgOld


class FedAdam(FedAvgOld):
    """Skip-missing FedAvg, then FedAdam on trainable parameters."""

    name = 'fedadam'
    description = 'skip-missing FedAvg with server Adam on trainable params'

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
        present_keys = collect_present_keys(payloads)
        trainable = trainable_param_names(global_model)
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            avg = averaged[key]
            apply_adam = key in trainable and key in present_keys
            if not apply_adam:
                new_state[key] = avg
                if key in self.m:
                    self.m[key] = self.beta1 * self.m[key]
                if key in self.v:
                    self.v[key] = self.beta2 * self.v[key]
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
