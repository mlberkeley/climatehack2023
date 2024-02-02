import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from util import util
import keys as keys

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        inchannels: int = 3,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.inchannels = inchannels
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.inchannels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional,
    progress: bool,
    inchannels: int = 12,
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, inchannels=inchannels, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


class ResNet18(nn.Module):

    REQUIRED_META = [
    ]
    REQUIRED_NONHRV = [
        keys.NONHRV.VIS006,
    ]
    REQUIRED_WEATHER = []

    def __init__(self) -> None:
        super().__init__()

        self.resnet_backbone = _resnet(BasicBlock, [2, 2, 2, 2], None, True)
        self.linear1 = nn.Linear(512 * BasicBlock.expansion + 12, 256)
        self.linear2 = nn.Linear(256, 48)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, site_features, nonhrv, weather):
        feature = self.resnet_backbone(nonhrv[keys.NONHRV.VIS006])
        x = torch.concat((feature, pv), dim=-1)

        x = self.r(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x


class ResNet34(nn.Module):

    REQUIRED_META = [
    ]
    REQUIRED_NONHRV = [
        keys.NONHRV.VIS006,
    ]
    REQUIRED_WEATHER = []

    def __init__(self) -> None:
        super().__init__()

        self.resnet_backbone = _resnet(BasicBlock, [3, 4, 6, 3], None, True)
        self.linear1 = nn.Linear(512 * BasicBlock.expansion + 12, 256)
        self.linear2 = nn.Linear(256, 48)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, site_features, nonhrv, weather):
        feature = self.resnet_backbone(nonhrv[keys.NONHRV.VIS006])
        x = torch.concat((feature, pv), dim=-1)

        x = self.r(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x


class ResNet50(nn.Module):

    REQUIRED_META = [
    ]
    REQUIRED_NONHRV = [
        keys.NONHRV.VIS006,
    ]
    REQUIRED_WEATHER = []

    def __init__(self) -> None:
        super().__init__()

        self.resnet_backbone = _resnet(Bottleneck, [3, 4, 6, 3], None, True)
        self.linear1 = nn.Linear(512 * Bottleneck.expansion + 12, 48)

    def forward(self, pv, meta, nonhrv, weather):
        feature = self.resnet_backbone(nonhrv[keys.NONHRV.VIS006])
        x = torch.concat((feature, pv), dim=-1)

        x = torch.sigmoid(self.linear1(x))

        return x


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.nonhrvbone = _resnet(BasicBlock, [2, 2, 2, 2], None, True)

        self.cloudbone = _resnet(BasicBlock, [2, 2, 2, 2], None, True)
        self.snowbone = _resnet(BasicBlock, [2, 2, 2, 2], None, True)
        self.tempbone = _resnet(BasicBlock, [2, 2, 2, 2], None, True)
        self.rainbone = _resnet(BasicBlock, [2, 2, 2, 2], None, True)
        self.backbones = [self.cloudbone, self.snowbone, self.tempbone, self.rainbone]
        for i, bone in enumerate(self.backbones):
            bone.conv1 = nn.Conv2d(6 * [4, 2, 2, 1][i], 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.linear1 = nn.Linear((len(self.backbones) + 1) * 512 * BasicBlock.expansion + 12, 512)
        self.r = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(512, 48)

    def forward(self, pv, nonhrv, nwp):
        last = 0
        features = []

        for i, num in enumerate([4, 2, 2, 1]):
            features.append(self.backbones[i](nwp[:, last : last + 6 * num]))
            last = last + 6 * num

        feature_nwp = torch.cat(features, dim=-1)

        feature_nonhrv = self.nonhrvbone(nonhrv)
        x = torch.concat((feature_nonhrv, feature_nwp, pv), dim=-1)

        x = self.r(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x


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

        self.resnet_backbone = _resnet(BasicBlock, [3, 4, 6, 3], None, True)
        self.linear1 = nn.Linear(512 * BasicBlock.expansion + 12 + 5, 48 * 4)
        self.linear2 = nn.Linear(48 * 4, 48)
        self.lin0 = nn.Linear(12 + 5, 12 + 5)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, nonhrv, site_features):
        feature = self.resnet_backbone(nonhrv)
        site_features = util.site_normalize(site_features)
        lin0_feat = self.r(self.lin0(torch.concat((pv, site_features), dim=-1)))

        x = torch.concat((feature, lin0_feat), dim=-1)

        x = self.r(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x

class NoImage(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.lin1 = nn.Linear(17, 30)
        self.lin2 = nn.Linear(30, 90)
        self.lin3 = nn.Linear(90, 80)
        self.lin4 = nn.Linear(80, 48)
        self.r = nn.LeakyReLU(inplace=True)

    def forward(self, pv, nonhrv, site_features):
        site_features = util.site_normalize(site_features)

        x = self.r(self.lin1(torch.concat((pv, site_features), dim=-1)))

        x = self.r(self.lin2(x))
        x = self.r(self.lin3(x))
        x = torch.sigmoid(self.lin4(x))

        return x


class NonHRVBackbone(nn.Module):
    block_type = BasicBlock
    output_dim = 512 * block_type.expansion
    def __init__(self) -> None:
        super().__init__()

        self.resnet_backbone = _resnet(BasicBlock, [3, 4, 6, 3], None, True)
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
    block_type = BasicBlock
    output_dim = 512 * block_type.expansion
    def __init__(self, channels=1) -> None:
        super().__init__()

        #self.input_dim = (6*channels, 128, 128)
        self.input_dim = (128, 128)

        self.weather_bone = _resnet(BasicBlock, [3, 4, 6, 3], None, True)
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

class MainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weather_channels = 4
        self.nonhrv_channels = 3
        meta_channels = 5
        pv_channels = 12

        self.MetaAndPv = MetaAndPv()
        self.WeatherBackbones = nn.ModuleList([WeatherBackbone() for i in range(self.weather_channels)])
        self.NonHRVBackbones = nn.ModuleList([NonHRVBackbone() for i in range(self.nonhrv_channels)])


        self.linear1 = nn.Linear(MetaAndPv.output_dim + self.nonhrv_channels * NonHRVBackbone.output_dim + self.weather_channels * WeatherBackbone.output_dim, 48 * 10)
        #self.linear1 = nn.Linear(MetaAndPv.output_dim + self.nonhrv_channels * NonHRVBackbone.output_dim, 48 * 10)
        self.linear2 = nn.Linear(48 * 10 + 17, 48)
        #self.linear3 = nn.Linear(48 * 4, 48)
        self.r = nn.ReLU(inplace=True)

    def forward(self, pv, site_features, nonhrv, weather):
        #for now dims are [(batch, 12), (batch, 5), (batch, 3, 12, 128, 128), (batch, 3, 6, 128, 128)]
        feat1 = self.MetaAndPv(pv, site_features)
        feat2 = torch.concat([self.NonHRVBackbones[channel](nonhrv[:,channel]) for channel in range(self.nonhrv_channels)], dim=-1)
        feat3 = torch.concat([self.WeatherBackbones[channel](weather[:,channel]) for channel in range(self.weather_channels)], dim=-1)

        #all_feat = torch.concat([feat1, feat2], dim=-1)
        all_feat = torch.concat([feat1, feat2, feat3], dim=-1)
        pv_site_raw = torch.concat([pv, site_features], dim=-1)

        x = self.r(self.linear1(all_feat))

        x = torch.concat([x, pv_site_raw], dim=-1)
        #x = self.r(self.linear2(x))
        #x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear2(x))

        return x

