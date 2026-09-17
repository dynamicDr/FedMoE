import torch
import torch.nn as nn
import torch.optim as optim

from methods.base import FederatedMethod
from select_upload_experts import SelectUploadExperts
from upload_budget import sample_bandwidth, upload_expert_count
from utils import cpu_state_dict, release_cuda


def compute_kronecker_factors(model, loader, device, lam, max_samples=256):
    model.eval()
    _acts  = {}
    _grads = {}
    _hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        def _fwd(n, has_bias):
            def hook(mod, inp, out):
                a = inp[0].detach().clone()
                if has_bias:
                    ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
                    a = torch.cat([a, ones], dim=1)
                _acts.setdefault(n, []).append(a)
            return hook

        def _bwd(n):
            def hook(mod, grad_in, grad_out):
                _grads.setdefault(n, []).append(grad_out[0].detach().clone())
            return hook

        _hooks.append(module.register_forward_hook(_fwd(name, module.bias is not None)))
        _hooks.append(module.register_full_backward_hook(_bwd(name)))

    count = 0
    for x, y in loader:
        if count >= max_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        nn.CrossEntropyLoss()(logits, y).backward()
        del logits
        count += x.size(0)

    for h in _hooks:
        h.remove()
    _hooks.clear()
    model.zero_grad()

    kron = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in _acts or name not in _grads:
            continue
        a_cat = torch.cat(_acts[name],  dim=0).float()
        g_cat = torch.cat(_grads[name], dim=0).float()
        del _acts[name], _grads[name]

        A = (a_cat.T @ a_cat) / a_cat.size(0)
        A = A + lam * torch.eye(A.size(0), device=device)
        B = (g_cat.T @ g_cat) / g_cat.size(0)
        B = B + lam * torch.eye(B.size(0), device=device)
        del a_cat, g_cat

        W = module.weight.detach().float()
        if module.bias is not None:
            M = torch.cat([W, module.bias.detach().float().unsqueeze(1)], dim=1)
        else:
            M = W.clone()
        del W
        kron[name] = {'A': A.cpu(), 'B': B.cpu(), 'M': M.cpu()}
        del A, B

    _acts.clear()
    _grads.clear()
    return kron


def fedlpa_layer_agg(Ms, As, Bs, agg_iters, agg_lr, device):
    K  = len(Ms)
    Ms = [m.float().to(device) for m in Ms]
    As = [a.float().to(device) for a in As]
    Bs = [b.float().to(device) for b in Bs]
    with torch.no_grad():
        z_bar = sum(Bs[k] @ Ms[k] @ As[k] for k in range(K))
    M_bar = torch.stack(Ms).mean(0).clone().detach().requires_grad_(True)
    optimizer = optim.Adam([M_bar], lr=agg_lr)
    for _ in range(agg_iters):
        optimizer.zero_grad()
        pred = sum(Bs[k] @ M_bar @ As[k] for k in range(K))
        loss = 0.5 * (pred - z_bar).pow(2).sum()
        loss.backward()
        optimizer.step()
    result = M_bar.detach().cpu()
    del Ms, As, Bs, z_bar, M_bar, optimizer
    release_cuda()
    return result


class BLAFedMoE(FederatedMethod):
    """Bandwidth-limited expert upload + FedLPA expert aggregation.

    Expert selection is not designed yet; currently random, same as MoE.
    """

    name = 'BLA-FedMoE'

    def __init__(self, args, device):
        super().__init__(args, device)
        self.expert_selector = SelectUploadExperts(seed=args.seed)

    def pack_upload(self, state, extra, client_id, round_id, model, loader):
        bandwidth = sample_bandwidth(self.args, client_id, round_id)
        upload_k = upload_expert_count(
            bandwidth, self.args.expert_bw, self.args.num_experts,
        )
        # TODO: 还没设计。当前按带宽随机选 upload_k 个专家。
        expert_ids = self.expert_selector.select(
            num_experts=self.args.num_experts,
            k=upload_k,
            client_id=client_id,
            round_id=round_id,
            model=model,
            state=state,
            kron=extra,
            loader=loader,
        )
        state, extra = self.expert_selector.filter_upload(state, extra, expert_ids)
        return state, extra, expert_ids, bandwidth, upload_k

    def local_update(self, global_model, loader, client_id, round_id, lr):
        args = self.args
        model, loss = self.local_train(global_model, loader, lr)
        extra = compute_kronecker_factors(
            model, loader, self.device, args.lam, args.kron_samples,
        )
        state = cpu_state_dict(model)
        del model
        release_cuda()
        state, extra, expert_ids, bandwidth, upload_k = self.pack_upload(
            state, extra, client_id, round_id, global_model, loader,
        )
        return self.payload(
            state, extra, loss, client_id, expert_ids,
            bandwidth=bandwidth, upload_k=upload_k,
        )

    def aggregate(self, global_model, payloads):
        args = self.args
        client_states = [p['state'] for p in payloads]
        client_krons = [p['extra'] for p in payloads]
        new_state = {}
        linear_modules = {
            name: mod for name, mod in global_model.named_modules()
            if isinstance(mod, nn.Linear)
        }
        for key, g_param in global_model.state_dict().items():
            present = [i for i, cs in enumerate(client_states) if key in cs]
            if not present:
                new_state[key] = g_param.detach().cpu().clone().to(g_param.dtype)
                continue
            stacked = torch.stack([client_states[i][key].float() for i in present])
            matched_linear = None
            for ln in linear_modules:
                if key == ln + '.weight' or key == ln + '.bias':
                    matched_linear = ln
                    break
            is_expert_linear = (
                matched_linear is not None
                and 'moe_head.experts.' in matched_linear
            )
            if not is_expert_linear:
                new_state[key] = stacked.mean(0).to(g_param.dtype)
                continue

            kron_ids = [i for i in present if matched_linear in client_krons[i]]
            if not kron_ids:
                new_state[key] = stacked.mean(0).to(g_param.dtype)
                continue

            As = [client_krons[i][matched_linear]['A'] for i in kron_ids]
            Bs = [client_krons[i][matched_linear]['B'] for i in kron_ids]
            Ms = [client_krons[i][matched_linear]['M'] for i in kron_ids]
            M_opt    = fedlpa_layer_agg(Ms, As, Bs, args.agg_iters, args.agg_lr, self.device)
            has_bias = linear_modules[matched_linear].bias is not None

            if key.endswith('.weight'):
                new_state[key] = (M_opt[:, :-1] if has_bias else M_opt).to(g_param.dtype)
            elif key.endswith('.bias') and has_bias:
                new_state[key] = M_opt[:, -1].to(g_param.dtype)
            else:
                new_state[key] = stacked.mean(0).to(g_param.dtype)
        return new_state
