import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from keys import META, COMPUTED, HRV, NONHRV, WEATHER, AEROSOLS
import util as util


class ResNetPV(nn.Module):

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.channel = NONHRV.from_str(config['channel'])

        self.resnet_backbone = models.resnext50_32x4d()
        # torch.load('./resnext50_32x4d-1a0047aa.pth', map_location='cpu')

        self.head = nn.Sequential(
                nn.Linear(self.resnet_backbone.fc.in_features + 12, 256), nn.LeakyReLU(0.1),
                nn.Linear(256, 256), nn.LeakyReLU(0.1),
                nn.Linear(256, 48),
                nn.Sigmoid(),
        )

        self.resnet_backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_backbone.fc = nn.Identity()

    @property
    def required_features(self):
        return [self.channel]

    def forward(self, pv, features):
        feature = self.resnet_backbone(features[self.channel]) ## [:, [-5, -3, -1]]) if trying 3 channels
        x = torch.concat((feature, pv), dim=-1)
        x = self.head(x)
        return x


class MetaAndPv(nn.Module):

    output_dim = 48

    def __init__(self) -> None:
        super().__init__()

        self.lin1 = nn.Linear(12 + 5 + 12, self.output_dim)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, features):
        meta = util.site_normalize(features.copy())
        # pv = 3*pv - 1.5      #pv is 0-1, lets make it -1.5 to 1.5
        solar = features[COMPUTED.SOLAR_ANGLES]
        solar[solar[:, :, 0] < 0] = 0 # zero out angles at night (zenith_horizontal < 0)
        solar = solar.view(-1, 12)
        features = torch.concat((pv, solar,
            meta[META.LATITUDE].view(-1, 1),
            meta[META.LONGITUDE].view(-1, 1),
            meta[META.ORIENTATION].view(-1, 1),
            meta[META.TILT].view(-1, 1),
            meta[META.KWP].view(-1, 1)
        ), dim=-1)

        x = self.r(self.lin1(features))

        return x


class MainModel2(nn.Module):


    def __init__(self, config) -> None:
        super().__init__()

        self.nonhrv_channels = config.get('nonhrv_channels', []) or []
        self.nonhrv_channels = [NONHRV.from_str(channel) for channel in self.nonhrv_channels]
        self.weather_channels = config.get('weather_channels', []) or []
        self.weather_channels = [WEATHER.from_str(channel) for channel in self.weather_channels]

        self.meta_head = config['meta_head']

        self.meta_and_pv = MetaAndPv()

        self.nonhrv_backbones = nn.ModuleList([models.resnet18() for i in range(len(self.nonhrv_channels))])
        for bone in self.nonhrv_backbones:
            bone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bone.fc = nn.Identity()

        self.weather_backbones = nn.ModuleList([models.resnet18() for i in range(len(self.weather_channels))])
        for bone in self.weather_backbones:
            bone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bone.fc = nn.Identity()

        if self.meta_head:
            self.linear1 = nn.Linear(
                    len(self.nonhrv_channels) * 512 +
                    len(self.weather_channels) * 512 +
                    self.meta_and_pv.output_dim,
                256)
        else:
            self.linear1 = nn.Linear(
                    len(self.nonhrv_channels) * 512 +
                    len(self.weather_channels) * 512 +
                    12 + 12,
                256)

        self.linear2 = nn.Linear(256, 256, bias=True)
        self.linear3 = nn.Linear(256, 48)
        self.r = nn.ReLU(inplace=True)

    @property
    def required_features(self):
        return list(META) + [COMPUTED.SOLAR_ANGLES] + self.nonhrv_channels + self.weather_channels

    def forward(self, pv, features):
        if self.nonhrv_channels:
            feat1 = torch.concat([self.nonhrv_backbones[i](features[key]) for i, key in enumerate(self.nonhrv_channels)], dim=-1)
        else:
            feat1 = torch.Tensor([]).to("cuda")

        if self.weather_channels:
            feat2 = torch.concat([self.weather_backbones[i](features[key]) for i, key in enumerate(self.weather_channels)], dim=-1)
        else:
            feat2 = torch.Tensor([]).to("cuda")

        if self.meta_head:
            feat3 = self.meta_and_pv(pv, features)
        else:
            solar = features[COMPUTED.SOLAR_ANGLES]
            solar[solar[:, :, 0] < 0] = 0 # zero out angles at night (zenith_horizontal < 0)
            solar = solar.view(-1, 12)
            feat3 = torch.concat((pv, solar), dim=-1)

        all_feat = torch.concat([feat1, feat2, feat3], dim=-1)

        x = self.r(self.linear1(all_feat))
        x = self.r(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))

        return x
