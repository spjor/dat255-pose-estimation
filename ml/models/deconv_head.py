from torch import nn


class DeconvHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv_layers = nn.Sequential()

        channels = in_channels
        for i in range(3):
            self.deconv_layers.append(
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1)
            )
            self.deconv_layers.append(
                nn.BatchNorm2d(channels // 2)
            )
            self.deconv_layers.append(
                nn.ReLU(inplace=True)
            )
            channels = channels // 2

        self.final_layer = nn.Conv2d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.deconv_layers(x)
        out = self.final_layer(out)

        return out

if __name__ == "__main__":

    in_channels = 2024
    out_channels = 17

    model = DeconvHead(2048)