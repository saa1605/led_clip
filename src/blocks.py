import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),                                     # (Mohit): argh... forgot to remove this batchnorm
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)