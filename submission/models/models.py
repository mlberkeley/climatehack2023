import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from modules.solar import solar_pos

from keys import FUTURE, META, HRV, NONHRV, WEATHER, AEROSOLS
import util as util



class ResNetPV(nn.Module):
    REQUIRED_META = []
    REQUIRED_NONHRV = []
    REQUIRED_WEATHER = []
    # REQUIRED_FUTURE = [FUTURE.NONHRV] this works, makes each nonhrv have 60 channels

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.channel = NONHRV.from_str(config['channel'])
        ResNetPV.REQUIRED_NONHRV.append(self.channel)

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

    def forward(self, pv, meta, nonhrv, weather):
        feature = self.resnet_backbone(nonhrv[self.channel]) ## [:, [-5, -3, -1]]) if trying 3 channels
        x = torch.concat((feature, pv), dim=-1)
        x = self.head(x)
        return x


# NOTE  WIP
class FutureModel(nn.Module):

    REQUIRED_META = []
    REQUIRED_NONHRV = []
    REQUIRED_WEATHER = []
    REQUIRED_FUTURE = [FUTURE.NONHRV]

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.channel = NONHRV.from_str(config['channel'])
        FutureModel.REQUIRED_NONHRV.append(self.channel)

        act = nn.ReLU()

        self.head = nn.Sequential(
                nn.Linear(12 * 16 + 12 + 16, 256), act,
                nn.Linear(256, 256), act,
                nn.Linear(256, 256), act,
                # nn.Linear(256, 256), act,
        )
        self.head2 = nn.Sequential(
                nn.Linear(256 + 12 + 16, 256), act,
                nn.Linear(256, 128), act,
                nn.Linear(128, 1),
                nn.Sigmoid(),
        )

    def forward(self, pv, meta, nonhrv, weather):
        nonhrv = nonhrv[self.channel]
        bs = nonhrv.shape[0]

        inps = torch.concat((
            pv.unsqueeze(1).repeat(1, 48, 1),
            nonhrv[:, :12, 62:66, 62:66].flatten(start_dim=1).unsqueeze(1).repeat(1,48,1),
            nonhrv[:, 12:, 62:66, 62:66].flatten(start_dim=2),
        ), dim=-1)

        inps = inps.view(bs * 48, 12 * 16 + 12 + 16)
        x = self.head(inps)
        x = torch.concat((x, pv.unsqueeze(1).repeat(1, 48, 1).view(bs*48, 12), nonhrv[:, 12:, 62:66, 62:66].flatten(start_dim=2).view(bs*48, 16)), dim=-1)
        x = self.head2(x)
        x = x.view(bs, 48)

        return x


class NonHRVMeta(nn.Module):

    REQUIRED_META = [
            META.TIME,
            META.LATITUDE,
            META.LONGITUDE,
            META.ORIENTATION,
            META.TILT,
            META.KWP,
    ]
    REQUIRED_NONHRV = [
            NONHRV.VIS006,
    ]
    REQUIRED_WEATHER = []

    def __init__(self) -> None:
        super().__init__()

        self.resnet_backbone = models.resnet18()
        self.resnet_backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_backbone.fc = nn.Identity()

        self.linear1 = nn.Linear(self.resnet_backbone.fc.in_features + 12 + 5, 48 * 4)
        self.linear2 = nn.Linear(48 * 4, 48)
        self.lin0 = nn.Linear(12 + 5, 12 + 5)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, meta, nonhrv, weather):
        feature = self.resnet_backbone(nonhrv[NONHRV.VIS006])
        meta = util.site_normalize(meta)
        # print(meta)
        lin0_feat = self.r(self.lin0(torch.concat((
            pv,
            meta[META.LATITUDE].unsqueeze(-1),
            meta[META.LONGITUDE].unsqueeze(-1),
            meta[META.ORIENTATION].unsqueeze(-1),
            meta[META.TILT].unsqueeze(-1),
            meta[META.KWP].unsqueeze(-1),
        ), dim=-1)))

        x = torch.concat((feature, lin0_feat), dim=-1)

        x = self.r(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x


class NonHRVBackbone(nn.Module):
    output_dim = models.resnet18().fc.in_features
    def __init__(self) -> None:
        super().__init__()

        self.resnet_backbone = models.resnet18()
        self.resnet_backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_backbone.fc = nn.Identity()
        #self.r = nn.LeakyReLU(inplace=True)

    def forward(self, nonhrv):
        x = self.resnet_backbone(nonhrv)
        #x = self.r(x)
        return x

class MetaAndPv(nn.Module):

    output_dim = 48

    def __init__(self) -> None:
        super().__init__()

        self.lin1 = nn.Linear(12 + 12, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, self.output_dim)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, meta):
        # meta = util.site_normalize(meta)
        # pv = 3*pv - 1.5      #pv is 0-1, lets make it -1.5 to 1.5
        solar = meta[META.SOLAR_ANGLES]
        solar[solar[:, :, 0] < 0] = 0 # zero out angles at night (zenith_horizontal < 0)
        solar = solar.view(-1, 12)
        features = torch.concat((pv, solar), dim=-1)

        x = self.r(self.lin1(features))
        x = self.r(self.lin2(x))
        x = self.r(self.lin3(x))

        return x

class WeatherBackbone(nn.Module):
    output_dim = models.resnet18().fc.in_features
    def __init__(self, channels=1) -> None:
        super().__init__()

        #self.input_dim = (6*channels, 128, 128)
        self.input_dim = (128, 128)

        self.weather_bone = models.resnet18()
        self.resnet_backbone.fc = nn.Identity()

        self.weather_bone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #6 because 6 hours (default is 12 in resnet), 64 because inplanes (weather_bone.inplanes was breaking batchnorm for some reason)

        #self.batch_norm = nn.BatchNorm2d(6)
        self.layer_norm = nn.LayerNorm(self.input_dim)
        #self.r = nn.LeakyReLU(inplace=True)

    def forward(self, weather):
        #x = weather
        #x = self.batch_norm(weather)
        x = self.layer_norm(weather)
        x = self.weather_bone(x)
        #x = self.r(x)
        return x

class MetaAndPv5(nn.Module):

    output_dim = 30

    def __init__(self) -> None:
        super().__init__()

        self.meta_keys = [
            META.LATITUDE,
            META.LONGITUDE,
            META.ORIENTATION,
            META.TILT,
            META.KWP,
        ]

        self.r = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(17, 30)

    def forward(self, pv, meta):
        meta = util.site_normalize(meta)
        meta = torch.stack([meta[key] for key in self.meta_keys], dim=1)
        x = self.linear1(torch.concat([meta, pv], dim=-1))
        x = self.r(x)
        return x




class MainModel2(nn.Module):

    REQUIRED_META = [
        META.TIME,
        META.LATITUDE,
        META.LONGITUDE,
        META.ORIENTATION,
        META.TILT,
        META.KWP,
        META.SOLAR_ANGLES,
    ]

    REQUIRED_HRV = [
        # HRV.HRV,
    ]

    REQUIRED_NONHRV = [
        NONHRV.VIS008,
    ]

    REQUIRED_WEATHER = [
        # WEATHER.CLCH,
        # WEATHER.CLCL
    ]

    REQUIRED_AEROSOLS = [
        # AEROSOLS.DUST,
    ]

    def __init__(self, config) -> None:
        super().__init__()

        self.meta_head = True
        self.MetaAndPv = MetaAndPv()

        #self.WeatherBackbones = nn.ModuleList([WeatherBackbone() for i in range(len(self.REQUIRED_WEATHER))])
        #self.NonHRVBackbones = nn.ModuleList([NonHRVBackbone() for i in range(len(self.REQUIRED_NONHRV))])

        self.WeatherBackbones = nn.ModuleList([models.resnet18() for i in range(len(self.REQUIRED_WEATHER))])
        for bone in self.WeatherBackbones:
            bone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bone.fc = nn.Identity()

        self.NonHRVBackbones = nn.ModuleList([models.resnet18() for i in range(len(self.REQUIRED_NONHRV))])
        for bone in self.NonHRVBackbones:
            bone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bone.fc = nn.Identity()


        if self.meta_head:
            self.linear1 = nn.Linear(len(self.REQUIRED_WEATHER) * 512 + len(self.REQUIRED_NONHRV) * 512 + self.MetaAndPv.output_dim + 12, 256)
        else:
            self.linear1 = nn.Linear(len(self.REQUIRED_WEATHER) * 512 + len(self.REQUIRED_NONHRV) * 512 + 12 + 12, 256)


        self.linear2 = nn.Linear(256, 48)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, meta, hrv, nonhrv, weather, aerosols):
        if len(self.REQUIRED_NONHRV):
            feat1 = torch.concat([self.NonHRVBackbones[i](nonhrv[key]) for i, key in enumerate(self.REQUIRED_NONHRV)], dim=-1)
        else:
            feat1 = torch.Tensor([]).to("cuda")

        if len(self.REQUIRED_WEATHER):
            feat2 = torch.concat([self.WeatherBackbones[i](weather[key]) for i, key in enumerate(self.REQUIRED_WEATHER)], dim=-1)
        else:
            feat2 = torch.Tensor([]).to("cuda")

        if self.meta_head:
            feat3 = self.MetaAndPv(pv, meta)
        else:
            feat3 = pv

        all_feat = torch.concat([feat1, feat2, feat3], dim=-1)

        x = self.r(self.linear1(all_feat))
        x = torch.sigmoid(self.linear2(x))

        return x
