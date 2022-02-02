import torch
import torch.nn as nn

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, relu=0.1):
        super(CBR2d, self).__init__()
        layers = list()
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=kernel_size,
                             padding=padding)]
        layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.ReLU(relu) if relu == 0 else nn.LeakyReLU(0.1)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_repeats=2):
        super(ResidualBlock, self).__init__()
        self.layers = list()
        for repeat in range(num_repeats):
            self.layers += [nn.Sequential(
                                CBR2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1),
                                CBR2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=3, padding=1)
                            )]

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)

        return x



