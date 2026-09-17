import torch

from .fedavg import FedAvg


class FedAvgM(FedAvg):
    """Skip-missing FedAvg, then server-side momentum on uploaded tensors only."""

    name = 'fedavgm'
    description = 'skip-missing FedAvg with server momentum'

    def __init__(self, args, device):
        super().__init__(args, device)
        self.velocity = {}
        self.momentum = float(getattr(args, 'server_momentum', 0.9))
        self.server_lr = float(getattr(args, 'server_lr', 1.0))

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
            velocity = self.velocity.get(key)
            if velocity is None:
                velocity = torch.zeros_like(delta)
            velocity = self.momentum * velocity + delta
            self.velocity[key] = velocity
            new_state[key] = (g_cpu + self.server_lr * velocity).to(g_param.dtype)
        return new_state
