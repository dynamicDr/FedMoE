import torch

from .base import collect_present_keys, trainable_param_names
from .fedavg_old import FedAvgOld


class FedAvgM(FedAvgOld):
    """Skip-missing FedAvg, then server momentum on trainable parameters."""

    name = 'fedavgm'
    description = 'skip-missing FedAvg with server momentum on trainable params'

    def __init__(self, args, device):
        super().__init__(args, device)
        self.velocity = {}
        self.momentum = float(getattr(args, 'server_momentum', 0.9))
        self.server_lr = float(getattr(args, 'server_lr', 1.0))

    def merge(self, global_model, payloads):
        averaged = super().merge(global_model, payloads)
        present_keys = collect_present_keys(payloads)
        trainable = trainable_param_names(global_model)
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            avg = averaged[key]
            apply_momentum = key in trainable and key in present_keys
            if not apply_momentum:
                new_state[key] = avg
                if key in self.velocity:
                    self.velocity[key] = self.momentum * self.velocity[key]
                continue
            g_cpu = g_param.detach().cpu().float()
            delta = avg.float() - g_cpu
            velocity = self.velocity.get(key)
            if velocity is None:
                velocity = torch.zeros_like(delta)
            velocity = self.momentum * velocity + delta
            self.velocity[key] = velocity
            new_state[key] = (g_cpu + self.server_lr * velocity).to(g_param.dtype)
        return new_state
