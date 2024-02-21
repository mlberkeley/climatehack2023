import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import keys as keys
import util

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models


nonhrv_means = np.array([ 64.32478999, 214.4245939 , 184.27316934, 192.57402931,
        149.51705036, 215.46701489, 231.37654464,  48.9898303,
         55.36402512, 155.32697878, 133.30849206]) / 255
nonhrv_stds = np.array([42.93865932,  9.29965694, 15.06005658,  8.77191701, 38.12814695,
        13.52211255,  9.89656887, 35.50711088, 36.63654748, 25.75839582,
        25.25472843]) / 255


class ResNetPV(nn.Module):

    REQUIRED_META = []
    REQUIRED_NONHRV = []
    REQUIRED_WEATHER = []
    # REQUIRED_FUTURE = [keys.FUTURE.NONHRV] this works, makes each nonhrv have 60 channels

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.channel = keys.NONHRV.from_str(config['channel'])
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

    def forward(self, pv, site_features, nonhrv, weather):
        # nonhrv = nonhrv.copy()
        # for k, v in nonhrv.items():
        #     nonhrv[k] = (v - nonhrv_means[k.value]) / nonhrv_stds[k.value]

        feature = self.resnet_backbone(nonhrv[self.channel]) ## [:, [-5, -3, -1]]) if trying 3 channels
        x = torch.concat((feature, pv), dim=-1)
        x = self.head(x)
        return x


# NOTE  WIP
class FutureModel(nn.Module):

    REQUIRED_META = []
    REQUIRED_NONHRV = []
    REQUIRED_WEATHER = []
    REQUIRED_FUTURE = [keys.FUTURE.NONHRV]

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.channel = keys.NONHRV.from_str(config['channel'])
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

    def forward(self, pv, site_features, nonhrv, weather):
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

        # preds = []
        # for i in range(48):
        #     # concat 12 pv points, middle 4x4 pixels (from 128x128) of input 12 nonhrv channels, and middle 4x4 of each of future 48 nonhrv channels
        #     x = torch.concat((pv, nonhrv[:, :12, 62:66, 62:66].flatten(start_dim=1), nonhrv[:, i, 62:66, 62:66].flatten(start_dim=1)), dim=-1)
        #     x = self.head(x)
        #     x = torch.concat((x, pv, nonhrv[:, i, 62:66, 62:66].flatten(start_dim=1)), dim=-1)
        #     x = self.head2(x)
        #     preds.append(x)

        # return torch.stack(preds, dim=1).squeeze(-1)


#class Model(nn.Module):

#    def __init__(self) -> None:
#        super().__init__()

#        self.nonhrvbone = resnet18()

#        self.cloudbone = resnet18()
#        self.snowbone = resnet18()
#        self.tempbone = resnet18()
#        self.rainbone = resnet18()
#        self.backbones = [self.cloudbone, self.snowbone, self.tempbone, self.rainbone]
#        for i, bone in enumerate(self.backbones):
#            bone.conv1 = nn.Conv2d(6 * [4, 2, 2, 1][i], 64, kernel_size=7, stride=2, padding=3, bias=False)

#        self.linear1 = nn.Linear((len(self.backbones) + 1) * 512 * BasicBlock.expansion + 12, 512)
#        self.r = nn.LeakyReLU(0.1)
#        self.linear2 = nn.Linear(512, 48)

#    def forward(self, pv, nonhrv, nwp):
#        last = 0
#        features = []

#        for i, num in enumerate([4, 2, 2, 1]):
#            features.append(self.backbones[i](nwp[:, last : last + 6 * num]))
#            last = last + 6 * num

#        feature_nwp = torch.cat(features, dim=-1)

#        feature_nonhrv = self.nonhrvbone(nonhrv)
#        x = torch.concat((feature_nonhrv, feature_nwp, pv), dim=-1)

#        x = self.r(self.linear1(x))
#        x = torch.sigmoid(self.linear2(x))

#        return x


## TODO  fix
#class NonHRVMeta(nn.Module):

#    REQUIRED_META = [
#            keys.META.TIME,
#            keys.META.LATITUDE,
#            keys.META.LONGITUDE,
#            keys.META.ORIENTATION,
#            keys.META.TILT,
#            keys.META.KWP,
#    ]
#    REQUIRED_NONHRV = [
#            keys.NONHRV.VIS006,
#    ]
#    REQUIRED_WEATHER = []

#    def __init__(self) -> None:
#        super().__init__()

#        self.resnet_backbone = resnet18()
#        self.linear1 = nn.Linear(512 * BasicBlock.expansion + 12 + 5, 48 * 4)
#        self.linear2 = nn.Linear(48 * 4, 48)
#        self.lin0 = nn.Linear(12 + 5, 12 + 5)
#        self.r = nn.ReLU(inplace=True)

#    def forward(self, pv, meta, nonhrv, weather):
#        feature = self.resnet_backbone(nonhrv[keys.NONHRV.VIS006])
#        meta = util.site_normalize(meta)
#        # print(meta)
#        lin0_feat = self.r(self.lin0(torch.concat((
#            pv,
#            meta[keys.META.LATITUDE].unsqueeze(-1),
#            meta[keys.META.LONGITUDE].unsqueeze(-1),
#            meta[keys.META.ORIENTATION].unsqueeze(-1),
#            meta[keys.META.TILT].unsqueeze(-1),
#            meta[keys.META.KWP].unsqueeze(-1),
#        ), dim=-1)))

#        x = torch.concat((feature, lin0_feat), dim=-1)

#        x = self.r(self.linear1(x))
#        x = torch.sigmoid(self.linear2(x))

#        return x

#class NoImage(nn.Module):

#    def __init__(self) -> None:
#        super().__init__()

#        self.lin1 = nn.Linear(17, 30)
#        self.lin2 = nn.Linear(30, 90)
#        self.lin3 = nn.Linear(90, 80)
#        self.lin4 = nn.Linear(80, 48)
#        self.r = nn.LeakyReLU(inplace=True)

#    def forward(self, pv, nonhrv, site_features):
#        site_features = util.site_normalize(site_features)

#        x = self.r(self.lin1(torch.concat((pv, site_features), dim=-1)))

#        x = self.r(self.lin2(x))
#        x = self.r(self.lin3(x))
#        x = torch.sigmoid(self.lin4(x))

#        return x


#class NonHRVBackbone(nn.Module):
#    block_type = BasicBlock
#    output_dim = 512 * block_type.expansion
#    def __init__(self) -> None:
#        super().__init__()

#        self.resnet_backbone = _resnet(BasicBlock, [3, 4, 6, 3], None)
#        #self.r = nn.LeakyReLU(inplace=True)

#    def forward(self, nonhrv):
#        x = self.resnet_backbone(nonhrv)
#        #x = self.r(x)
#        return x

#class MetaAndPv(nn.Module):
#    output_dim = 40
#    def __init__(self) -> None:
#        super().__init__()

#        self.lin1 = nn.Linear(17, 30)
#        self.lin2 = nn.Linear(30, 50)
#        self.lin3 = nn.Linear(50, self.output_dim)
#        self.r = nn.ReLU(inplace=True)

#    def forward(self, pv, site_features):
#        site_features = util.site_normalize(site_features)
#        pv = 3*pv - 1.5      #pv is 0-1, lets make it -1.5 to 1.5
#        features = torch.concat((pv, site_features), dim=-1)

#        x = self.r(self.lin1(features))
#        x = self.r(self.lin2(x))
#        x = self.lin3(x)

#        return x

#class WeatherBackbone(nn.Module):
#    block_type = BasicBlock
#    output_dim = 512 * block_type.expansion
#    def __init__(self, channels=1) -> None:
#        super().__init__()

#        #self.input_dim = (6*channels, 128, 128)
#        self.input_dim = (128, 128)

#        self.weather_bone = _resnet(BasicBlock, [3, 4, 6, 3], None)
#        self.weather_bone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
#        #6 because 6 hours (default is 12 in resnet), 64 because inplanes (weather_bone.inplanes was breaking batchnorm for some reason)

#        #self.batch_norm = nn.BatchNorm2d(6)
#        self.layer_norm = nn.LayerNorm(self.input_dim)
#        #self.r = nn.LeakyReLU(inplace=True)

#    def forward(self, weather):
#        #x = weather
#        #x = self.batch_norm(weather)
#        x = self.layer_norm(weather)
#        x = self.weather_bone(x)
#        #x = self.r(x)
#        return x

#class MainModel(nn.Module):
#    def __init__(self) -> None:
#        super().__init__()

#        self.weather_channels = 4
#        self.nonhrv_channels = 3
#        meta_channels = 5
#        pv_channels = 12

#        self.MetaAndPv = MetaAndPv()
#        self.WeatherBackbones = nn.ModuleList([WeatherBackbone() for i in range(self.weather_channels)])
#        self.NonHRVBackbones = nn.ModuleList([NonHRVBackbone() for i in range(self.nonhrv_channels)])


#        self.linear1 = nn.Linear(MetaAndPv.output_dim + self.nonhrv_channels * NonHRVBackbone.output_dim + self.weather_channels * WeatherBackbone.output_dim, 48 * 10)
#        #self.linear1 = nn.Linear(MetaAndPv.output_dim + self.nonhrv_channels * NonHRVBackbone.output_dim, 48 * 10)
#        self.linear2 = nn.Linear(48 * 10 + 17, 48)
#        #self.linear3 = nn.Linear(48 * 4, 48)
#        self.r = nn.ReLU(inplace=True)

#    def forward(self, pv, site_features, nonhrv, weather):
#        #for now dims are [(batch, 12), (batch, 5), (batch, 3, 12, 128, 128), (batch, 3, 6, 128, 128)]
#        feat1 = self.MetaAndPv(pv, site_features)
#        feat2 = torch.concat([self.NonHRVBackbones[channel](nonhrv[:,channel]) for channel in range(self.nonhrv_channels)], dim=-1)
#        feat3 = torch.concat([self.WeatherBackbones[channel](weather[:,channel]) for channel in range(self.weather_channels)], dim=-1)

#        #all_feat = torch.concat([feat1, feat2], dim=-1)
#        all_feat = torch.concat([feat1, feat2, feat3], dim=-1)
#        pv_site_raw = torch.concat([pv, site_features], dim=-1)

#        x = self.r(self.linear1(all_feat))

#        x = torch.concat([x, pv_site_raw], dim=-1)
#        #x = self.r(self.linear2(x))
#        #x = torch.sigmoid(self.linear3(x))
#        x = torch.sigmoid(self.linear2(x))

#        return x

