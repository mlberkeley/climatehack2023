import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import keys as keys

import torch
import torch.nn as nn
import torchvision.models as models


class SingleChannelBackbone(nn.Module):

        def __init__(self, channel):
            super(SingleChannelBackbone, self).__init__()

            self.channel = channel
            self.backbone = models.resnet18()
            self.droupout = nn.Dropout(0.5)
            self.head = nn.Sequential(
                nn.Linear(12 + 512, 256), nn.GELU(),
            )
            self.confidence_head = nn.Sequential(
                nn.Linear(512, 1), nn.Sigmoid()
            )
            self.backbone.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.backbone.fc = nn.Identity()

        def forward(self, pv, meta, nonhrv, weather):
            latent = self.backbone(nonhrv[self.channel])
            latent = self.droupout(latent)
            x = torch.cat([latent, pv], dim=1)
            x = self.head(x)
            confidence = self.confidence_head(latent)
            return x, confidence


class Model(nn.Module):

    REQUIRED_META = []
    REQUIRED_NONHRV = [
            keys.NONHRV.IR_016,
            keys.NONHRV.VIS006,
            keys.NONHRV.VIS008,
    ]
    REQUIRED_WEATHER = []

    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config

        self.backbones = nn.ModuleDict({
            k.name: SingleChannelBackbone(k) for k in self.REQUIRED_NONHRV
        })
        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Sequential(
                # nn.Linear(12 + 256, 256), nn.GELU(),
                nn.Linear(12 + 256, 48), nn.Sigmoid()
        )

    def forward(self, pv, meta, nonhrv, weather):
        nonhrv = [b(pv, meta, nonhrv, weather) for b in self.backbones.values()]
        nonhrv_latent = torch.stack([x for x, _ in nonhrv], dim=1)
        nonhrv_confidence = torch.stack([x for _, x in nonhrv], dim=1)
        # softmax the confidence
        nonhrv_weights = self.softmax(nonhrv_confidence)
        # weighted sum of the nonhrv latents
        x = torch.sum(nonhrv_latent * nonhrv_weights, dim=1)
        x = torch.cat([x, pv], dim=1)
        x = self.head(x)
        return x
