from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, planes, kernel_size=1, padding=0, stride=1):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.append(ConvLayer(in_channels, in_channels + i * growth_rate, kernel_size=3, padding=1))

    def forward(self, x):
        return self.layers(x)


class ConvNetBackbone(nn.Module):

    def __init__(self, blocks = None):
        super(ConvNetBackbone, self).__init__()
        self.conv1 = ConvLayer(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = ConvBlock(2, 64, 64)
        self.conv3 = ConvBlock(2, 128, 128)
        self.conv4 = ConvBlock(2, 256, 256)
        self.conv5 = ConvBlock(2, 512, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

if __name__ == "__main__":
    model = ConvNetBackbone()
    print(model)



