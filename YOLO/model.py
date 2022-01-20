import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        layers = list()
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=1,
                             padding=1)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Detect(nn.Module):
    def __init__(self):
        super(Detect, self).__init__()
        self.conv_block1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block3_1 = ConvBlock(in_channels=192, out_channels=256)
        self.conv_block3_2 = ConvBlock(in_channels=256, out_channels=512)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # *4
        conv_block4 = [ConvBlock(in_channels=512, out_channels=512) for i in range(4)]
        self.conv_block4_1 = nn.Sequential(*conv_block4)
        self.conv_block4_2 = ConvBlock(in_channels=512, out_channels=1024)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # *2
        conv_block5 = [ConvBlock(in_channels=1024, out_channels=1024) for i in range(2)]
        self.conv_block5_1 = nn.Sequential(*conv_block5)
        self.conv_block5_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv_block5_3 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.conv_block6_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv_block6_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv_block1 = self.conv_block1(x)
        conv_block1 = self.max_pool1(conv_block1)

        conv_block2 = self.conv_block2(conv_block1)
        conv_block2 = self.max_pool2(conv_block2)

        conv_block3 = self.conv_block3_1(conv_block2)
        conv_block3 = self.conv_block3_2(conv_block3)
        conv_block3 = self.max_pool3(conv_block3)

        conv_block4 = self.conv_block4_1(conv_block3)
        conv_block4 = self.conv_block4_2(conv_block4)
        conv_block4 = self.max_pool4(conv_block4)

        conv_block5 = self.conv_block5_1(conv_block4)
        conv_block5 = self.conv_block5_2(conv_block5)
        conv_block5 = self.conv_block5_3(conv_block5)

        conv_block6 = self.conv_block6_1(conv_block5)
        conv_block6 = self.conv_block6_2(conv_block6)

        return conv_block6



