import torch
import torch.nn as nn

from layer import CBR2d, ResidualBlock

class YOLOv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOv3, self).__init__()
        self.conv1 = CBR2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.pool1 = CBR2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.res1 = ResidualBlock(in_channels=64, num_repeats=1)
        self.pool2 = CBR2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.res2 = ResidualBlock(in_channels=128, num_repeats=2)
        self.pool3 = CBR2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.res3 = ResidualBlock(in_channels=256, num_repeats=8)
        self.pool4 = CBR2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.res4 = ResidualBlock(in_channels=512, num_repeats=8)
        self.pool5 = CBR2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.res5 = ResidualBlock(in_channels=1024, num_repeats=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.pool2(x)
        x = self.res2(x)
        x = self.pool3(x)
        x = self.res3(x)
        x = self.pool4(x)
        x = self.res4(x)
        x = self.pool5(x)
        x = self.res5(x)


        return x

