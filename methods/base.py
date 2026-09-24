import torch

from methods.Merge import build_merger
from methods.Select import build_selector
from model import MoEFedModel
from upload_budget import (
    apply_drop_expert, filter_expert_state, sample_bandwidth, upload_expert_count,
)
from utils import cpu_state_dict, evaluate, release_cuda, run_local_sgd


class FederatedMethod:
    """Shared federated loop: Select experts, local SGD, then Merge uploads."""

    name = 'base'
    select_name = None
    merge_name = None

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.selector = build_selector(
            getattr(args, 'select', None) or self.select_name, args, device,
        )
        self.merger = build_merger(
            getattr(args, 'merge', None) or self.merge_name, args, device,
        )

    def build_model(self, cfg):
        return MoEFedModel(
            in_channels=cfg['in_channels'],
            num_classes=cfg['num_classes'],
            img_size=cfg['img_size'],
            num_experts=self.args.num_experts,
            topk=self.args.topk,
            backbone=getattr(self.args, 'backbone', 'resnet'),
            width_mult=getattr(self.args, 'width_mult', 1.0),
        ).to(self.device)

    def local_train(self, global_model, loader, lr, trainable_expert_ids=None,
                    expert_usage_out=None, dense_experts=False):
        args = self.args
        model, loss = run_local_sgd(
            global_model, loader, self.device,
            args.local_epochs, lr, args.momentum,
            args.weight_decay, args.label_smooth,
            trainable_expert_ids=trainable_expert_ids,
            expert_usage_out=expert_usage_out,
            dense_experts=dense_experts,
        )
        return model, loss

    def local_update(self, global_model, loader, client_id, round_id, lr):
        if self.selector is None:
            raise NotImplementedError
        args = self.args
        bandwidth = sample_bandwidth(args, client_id, round_id)
        upload_k = upload_expert_count(
            bandwidth, args.expert_bw, args.num_experts,
        )
        upload_k = apply_drop_expert(
            upload_k, getattr(args, 'drop_expert_prob', 0.0),
            args.seed, client_id, round_id,
        )
        n_samples = len(loader.dataset)
        usage = {}
        after_train = bool(getattr(self.selector, 'after_train', False))
        if after_train:
            model, loss = self.local_train(
                global_model, loader, lr, expert_usage_out=usage,
                dense_experts=bool(getattr(self.selector, 'dense_experts', False)),
            )
            expert_ids = self.selector.select(
                model=model,
                loader=loader,
                k=upload_k,
                client_id=client_id,
                round_id=round_id,
                trained_model=model,
                global_model=global_model,
            )
        else:
            expert_ids = self.selector.select(
                model=global_model,
                loader=loader,
                k=upload_k,
                client_id=client_id,
                round_id=round_id,
            )
            model, loss = self.local_train(
                global_model, loader, lr,
                trainable_expert_ids=expert_ids,
                expert_usage_out=usage,
            )
        expert_tokens = [
            float(usage.get(i, 0.0)) for i in range(int(args.num_experts))
        ]
        state = filter_expert_state(cpu_state_dict(model), expert_ids)
        del model
        release_cuda()
        return self.payload(
            state, {}, loss, client_id, expert_ids,
            bandwidth=bandwidth, upload_k=upload_k,
            n_samples=n_samples,
            expert_tokens=expert_tokens,
        )

    def aggregate(self, global_model, payloads):
        if self.merger is None:
            raise NotImplementedError
        return self.merger.merge(global_model, payloads)

    def evaluate(self, model, loader):
        return evaluate(model, loader, self.device)

    def mean_aggregate(self, global_model, payloads):
        client_states = [p['state'] for p in payloads]
        new_state = {}
        for key, g_param in global_model.state_dict().items():
            present = [cs[key] for cs in client_states if key in cs]
            if not present:
                new_state[key] = g_param.detach().cpu().clone().to(g_param.dtype)
                continue
            stacked = torch.stack([t.float() for t in present])
            new_state[key] = stacked.mean(0).to(g_param.dtype)
        return new_state

    def payload(self, state, extra, loss, client_id, expert_ids, **meta):
        return {
            'state': state,
            'extra': extra,
            'loss': float(loss),
            'meta': {
                'client_id': client_id,
                'expert_ids': list(expert_ids),
                **meta,
            },
        }
