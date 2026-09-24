import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


# Base stage widths. width_mult=1 keeps 64-128-256-512;
# width_mult=0.5 becomes 32-64-128-256. Experts are not scaled.
RESNET_STAGE_WIDTHS = (64, 128, 256, 512)
EXPERT_HIDDEN = 512


def scale_channels(channels, width_mult):
    mult = float(width_mult)
    if mult <= 0:
        raise ValueError(f'width_mult must be positive, got {width_mult}')
    return tuple(max(1, int(round(float(c) * mult))) for c in channels)


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, img_size, width_mult=1.0):
        super().__init__()
        c1, c2, c3, c4 = scale_channels(RESNET_STAGE_WIDTHS, width_mult)
        stem_stride = 1 if img_size <= 32 else 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(c1, c1, stride=1)
        self.layer2 = self._make_layer(c1, c2, stride=2)
        self.layer3 = self._make_layer(c2, c3, stride=2)
        self.layer4 = self._make_layer(c3, c4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = c4
        self.width_mult = float(width_mult)

    @staticmethod
    def _make_layer(in_ch, out_ch, stride):
        return nn.Sequential(
            BasicBlock(in_ch,  out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


# VGG11: 64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M
VGG11_CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG11Backbone(nn.Module):
    def __init__(self, in_channels, img_size, width_mult=1.0):
        super().__init__()
        base = [v for v in VGG11_CFG if v != 'M']
        scaled = {
            old: new for old, new in zip(base, scale_channels(base, width_mult))
        }
        layers = []
        ch = in_channels
        for v in VGG11_CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            out_ch = scaled[v]
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = ch
        self.width_mult = float(width_mult)
        # img_size kept for the same constructor signature as ResNetBackbone.

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


BACKBONES = {
    'resnet': ResNetBackbone,
    'vgg11': VGG11Backbone,
}


def build_backbone(name, in_channels, img_size, width_mult=1.0):
    key = str(name).lower()
    try:
        cls = BACKBONES[key]
    except KeyError as exc:
        raise ValueError(
            f'Unknown backbone: {name}. Available: {sorted(BACKBONES)}'
        ) from exc
    return cls(in_channels, img_size, width_mult=width_mult)


class ExpertFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TopKGating(nn.Module):
    def __init__(self, in_dim, num_experts, topk):
        super().__init__()
        self.topk = topk
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)

        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)

        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_vals)
        weights = weights.to(x.dtype)

        return weights, topk_idx


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, topk):
        super().__init__()
        self.gating  = TopKGating(in_dim, num_experts, topk)
        self.experts = nn.ModuleList([
            ExpertFFN(in_dim, hidden_dim, out_dim) for _ in range(num_experts)
        ])
        # Post-hoc sets this so every expert is forwarded and backpropped.
        self.dense_experts = False

    def forward(self, x):
        weights, topk_idx = self.gating(x)
        out = torch.zeros(x.size(0), self.experts[0].fc2.out_features, device=x.device, dtype=x.dtype)

        if self.dense_experts:
            for i, expert in enumerate(self.experts):
                expert_output = expert(x)
                out = out + expert_output * weights[:, i].unsqueeze(-1)
            return out

        for i, expert in enumerate(self.experts):
            batch_mask = (topk_idx == i).any(dim=-1)
            if batch_mask.any():
                expert_output = expert(x[batch_mask])
                gate_scores = weights[batch_mask, i].unsqueeze(-1)
                out[batch_mask] += expert_output * gate_scores

        return out


class MoEFedModel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, num_experts, topk,
                 backbone='resnet', width_mult=1.0):
        super().__init__()
        self.backbone_name = str(backbone).lower()
        self.width_mult = float(width_mult)
        self.backbone = build_backbone(
            self.backbone_name, in_channels, img_size, width_mult=self.width_mult,
        )
        feat_dim = self.backbone.feat_dim
        # Expert FFN hidden size stays fixed; only the backbone is scaled.
        self.moe_head = MoELayer(
            feat_dim, EXPERT_HIDDEN, num_classes, num_experts, topk,
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.moe_head(feat)
        return logits
