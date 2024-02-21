import util
from modules.solar import solar_pos
from submission.resnet import *
import torchvision.models as models


def _resnet(block, layers, weights, progress, inchannels, **kwargs):
    model = ResNet(block, layers, inchannels=inchannels, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


# TODO  fix
class NonHRVMeta(nn.Module):

    REQUIRED_META = [
            keys.META.TIME,
            keys.META.LATITUDE,
            keys.META.LONGITUDE,
            keys.META.ORIENTATION,
            keys.META.TILT,
            keys.META.KWP,
    ]
    REQUIRED_NONHRV = [
            keys.NONHRV.VIS006,
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
        feature = self.resnet_backbone(nonhrv[keys.NONHRV.VIS006])
        meta = util.site_normalize(meta)
        # print(meta)
        lin0_feat = self.r(self.lin0(torch.concat((
            pv,
            meta[keys.META.LATITUDE].unsqueeze(-1),
            meta[keys.META.LONGITUDE].unsqueeze(-1),
            meta[keys.META.ORIENTATION].unsqueeze(-1),
            meta[keys.META.TILT].unsqueeze(-1),
            meta[keys.META.KWP].unsqueeze(-1),
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
    output_dim = 40
    def __init__(self) -> None:
        super().__init__()

        self.lin1 = nn.Linear(17, 30)
        self.lin2 = nn.Linear(30, 50)
        self.lin3 = nn.Linear(50, self.output_dim)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, site_features):
        site_features = util.site_normalize(site_features)
        pv = 3*pv - 1.5      #pv is 0-1, lets make it -1.5 to 1.5
        features = torch.concat((pv, site_features), dim=-1)

        x = self.r(self.lin1(features))
        x = self.r(self.lin2(x))
        x = self.lin3(x)

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
            keys.META.LATITUDE,
            keys.META.LONGITUDE,
            keys.META.ORIENTATION,
            keys.META.TILT,
            keys.META.KWP,
        ]

        self.r = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(17, 30)

    def forward(self, pv, site_features):
        site_features = util.site_normalize(site_features)
        meta = torch.stack([site_features[key] for key in self.meta_keys], dim=1)
        x = self.linear1(torch.concat([meta, pv], dim=-1))
        x = self.r(x)
        return x




class MainModel2(nn.Module):

    REQUIRED_META = [
        keys.META.TIME,
        keys.META.LATITUDE,
        keys.META.LONGITUDE,
        keys.META.ORIENTATION,
        keys.META.TILT,
        keys.META.KWP
    ]

    REQUIRED_NONHRV = [
        keys.NONHRV.VIS008,
    ]

    REQUIRED_WEATHER = [
        #keys.WEATHER.CLCH,
        #keys.WEATHER.CLCL
    ]

    def __init__(self, config) -> None:
        super().__init__()

        self.MetaAndPv = MetaAndPv5()

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


        if len(self.REQUIRED_META):
            self.linear1 = nn.Linear(len(self.REQUIRED_WEATHER) * 512 + len(self.REQUIRED_NONHRV) * 512 + self.MetaAndPv.output_dim + 2 , 256)
        else:
            self.linear1 = nn.Linear(len(self.REQUIRED_WEATHER) * 512 + len(self.REQUIRED_NONHRV) * 512 + 12 + 2, 256)


        self.linear2 = nn.Linear(256, 48)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, site_features, nonhrv, weather):
        if len(self.REQUIRED_NONHRV):
            feat1 = torch.concat([self.NonHRVBackbones[i](nonhrv[key]) for i, key in enumerate(self.REQUIRED_NONHRV)], dim=-1)
        else:
            feat1 = torch.Tensor([]).to("cuda")
        if len(self.REQUIRED_WEATHER):
            feat2 = torch.concat([self.WeatherBackbones[i](weather[key]) for i, key in enumerate(self.REQUIRED_WEATHER)], dim=-1)
        else:
            feat2 = torch.Tensor([]).to("cuda")
        if len(self.REQUIRED_META):
            feat3 = self.MetaAndPv(pv, site_features)
        else:
            feat3 = pv

        solar_data = solar_pos(site_features, device="cuda")

        all_feat = torch.concat([feat1, feat2, feat3, solar_data], dim=-1)

        x = self.r(self.linear1(all_feat))
        x = torch.sigmoid(self.linear2(x))

        return x
