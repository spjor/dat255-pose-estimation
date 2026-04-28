from typing import Optional

import torch
from fiftyone.core.annotation.constants import DEFAULT
from torch import nn
from torchvision.models import ResNet34_Weights, WeightsEnum, Weights, ResNet50_Weights, ResNet101_Weights, \
    ResNet152_Weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):
    ""

    def __init__(self, block, layers, name="resnet"):
        super(ResNetBackbone, self).__init__()
        self.name = name
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.Sequential(
            block(self.inplanes, planes, stride, downsample)
        )

        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(
                block(self.inplanes, planes)
            )

        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


def init_resnet_backbone(block, layers, weights: Optional[WeightsEnum] = None):
    model = ResNetBackbone(block, layers)

    if weights is not None:

        model.load_state_dict(weights.get_state_dict(), strict=False)

    return model

def get_resnet18_backbone(pretrained=False, **kwargs):
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = init_resnet_backbone(BasicBlock, [2, 2, 2, 2], weights)
    return model

def get_resnet34_backbone(pretrained=False, **kwargs):
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = init_resnet_backbone(BasicBlock, [3, 4, 6, 3], weights)
    return model

def get_resnet50_backbone(pretrained=False, **kwargs):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = init_resnet_backbone(Bottleneck, [3, 4, 6, 3], weights)
    return model

def get_resnet101_backbone(pretrained=False, **kwargs):
    weights = ResNet101_Weights.DEFAULT if pretrained else None
    model = init_resnet_backbone(Bottleneck, [3, 4, 23, 3], weights)
    return model

def get_resnet152_backbone(pretrained=False, **kwargs):
    weights = ResNet152_Weights.DEFAULT if pretrained else None
    model = init_resnet_backbone(Bottleneck, [3, 4, 23, 3], weights)
    return model

if __name__ == "__main__":
    model = get_resnet18_backbone(pretrained=True)
    print(model)

    model = get_resnet34_backbone(pretrained=True)
    print(model)

    model = get_resnet50_backbone(pretrained=True)
    print(model)
