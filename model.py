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


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, img_size):
        super().__init__()
        stem_stride = 1 if img_size <= 32 else 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

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
    def __init__(self, in_channels, img_size):
        super().__init__()
        layers = []
        ch = in_channels
        for v in VGG11_CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            layers.append(nn.Conv2d(ch, v, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            ch = v
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512
        # img_size kept for the same constructor signature as ResNetBackbone.

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


BACKBONES = {
    'resnet': ResNetBackbone,
    'vgg11': VGG11Backbone,
}


def build_backbone(name, in_channels, img_size):
    key = str(name).lower()
    try:
        cls = BACKBONES[key]
    except KeyError as exc:
        raise ValueError(
            f'Unknown backbone: {name}. Available: {sorted(BACKBONES)}'
        ) from exc
    return cls(in_channels, img_size)


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

    def forward(self, x):
        weights, topk_idx = self.gating(x)
        out = torch.zeros(x.size(0), self.experts[0].fc2.out_features, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts):
            batch_mask = (topk_idx == i).any(dim=-1)
            if batch_mask.any():
                expert_output = expert(x[batch_mask])
                gate_scores = weights[batch_mask, i].unsqueeze(-1)
                out[batch_mask] += expert_output * gate_scores

        return out


class MoEFedModel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, num_experts, topk,
                 backbone='resnet'):
        super().__init__()
        self.backbone_name = str(backbone).lower()
        self.backbone = build_backbone(self.backbone_name, in_channels, img_size)
        feat_dim = self.backbone.feat_dim
        self.moe_head = MoELayer(feat_dim, 512, num_classes, num_experts, topk)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.moe_head(feat)
        return logits
