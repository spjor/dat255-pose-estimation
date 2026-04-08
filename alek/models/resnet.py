import torch
from torch import nn



class ResNetBlock(nn.Module):
    def __init__(self, filters, kernel_size, stride, padding):
        super(ResNetBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding=padding)
        self.bn2 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(4 * filters, filters, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(filters, eps=1.001e-5)

        self.downsample = nn.Sequential(
            nn.Conv2d(4 * filters, filters, kernel_size=1),
            nn.BatchNorm2d(filters, eps=1.001e-5)
        )

        self.act3 = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        out = out + residual

        out = self.act3(out)

        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, filters, kernel_size, stride, padding):
        super(ResNet, self).__init__()

        self.input_layer = torch.randn(input_channels)
        self.conv1 = nn.Conv2d(input_channels, filters, kernel_size=(3, 3), stride=stride, padding=padding)

        self.blocks = nn.Sequential(
            ResNetBlock(filters, kernel_size, 2, padding),
            ResNetBlock(filters, kernel_size, 1, padding),
            ResNetBlock(filters, kernel_size, 1, padding),
        )

        self.pool = nn.AvgPool2d(filters)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(self.input_layer)
        out = self.blocks(out)
        out = self.pool(out)
        out = self.dropout(out)

        return out

