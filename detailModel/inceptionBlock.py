import torch
import torch.nn as nn
import torch.nn.functional as F
from Resnet6 import detail_atten




# Inception v1 + BN
class InceptionBlock_v1(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_v1, self).__init__()
        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 5x5
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        # 3x3 pool -> 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch3(x)
        out3 = self.branch5(x)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)  # 输出通道为 480


class InceptionBlock_v2(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_v2, self).__init__()
        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> 3x3 ->3* 3相当于 5*5
        self.branch5 = nn.Sequential(

            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        # 3x3 pool -> 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch3(x)
        out3 = self.branch5(x)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)  # 输出通道为 480


class InceptionBlock_v3(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_v3, self).__init__()
        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> (1x3 -> 3x1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> (1x3 -> 3x1) -> (1x3 -> 3x1) ≈ 5x5
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        # 3x3 pool -> 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch3(x)
        out3 = self.branch5(x)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)  # 输出通道为 128+192+96+64 = 480


# Inception v1 + BN + atten
class InceptionBlock_v1_atten(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_v1_atten, self).__init__()
        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            detail_atten(dim=128)
        )

        # 1x1 -> 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            detail_atten(dim=192)
        )

        # 1x1 -> 5x5
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            detail_atten(dim=96)
        )

        # 3x3 pool -> 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            detail_atten(dim=64)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch3(x)
        out3 = self.branch5(x)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)  # 输出通道为 480


# Inception v2 + BN + atten
class InceptionBlock_v2_atten(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_v2_atten, self).__init__()
        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            detail_atten(dim=128)
        )

        # 1x1 -> 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            detail_atten(dim=192)
        )

        # 1x1 -> 3x3 ->3* 3相当于 5*5
        self.branch5 = nn.Sequential(

            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            detail_atten(dim=96)
        )

        # 3x3 pool -> 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            detail_atten(dim=64)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch3(x)
        out3 = self.branch5(x)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)  # 输出通道为 480


# Inception v3 + BN + atten
class InceptionBlock_v3_atten(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock_v3_atten, self).__init__()
        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            detail_atten(dim=128)
        )

        # 1x1 -> (1x3 -> 3x1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            detail_atten(dim=192)
        )

        # 1x1 -> (1x3 -> 3x1) -> (1x3 -> 3x1) ≈ 5x5
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            detail_atten(dim=96)
        )

        # 3x3 pool -> 1x1
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            detail_atten(dim=64)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch3(x)
        out3 = self.branch5(x)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)  # 输出通道为 128+192+96+64 = 480


# 整体网络结构
class Inception_atten(nn.Module):
    def __init__(self,v1=False,v2=False,v3=False):
        super(Inception_atten, self).__init__()

        # 前置卷积层：从 3 通道提升为 64 通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )


        # Inception 模块
        if v1==True:

            self.inception = InceptionBlock_v1(in_channels=128)  # 输出通道 480
        if(v2==True):
            self.inception = InceptionBlock_v2(in_channels=128) # 输出通道 480
        if v3==True:
            self.inception = InceptionBlock_v3(in_channels=128)  # 输出通道 480
        self.attn = detail_atten(dim=480)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出 (B, 480, 1, 1)
        self.fc = nn.Linear(480, 1)  # 二分类
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)                # -> (B, 64, 32, 32)
        x = self.conv2(x)                 # -> (B, 128, 32, 32)
        x = self.inception(x)             # -> (B, 480, 32, 32)
        x = self.attn(x)                  # -> (B, 480, 32, 32)
        x = self.gap(x)                   # -> (B, 480, 1, 1)
        x = x.view(x.size(0), -1)         # -> (B, 480)
        x = self.fc(x)                    # -> (B, 1)
        x = self.sigmoid(x)               # -> (B, 1), 概率值
        return x

# atten in Branch v1-v3
class Inception_atten_conv(nn.Module):
    def __init__(self, v1=False, v2=False, v3=False):
        super(Inception_atten_conv, self).__init__()

        # 前置卷积层：从 3 通道提升为 64 通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )


        # Inception 模块
        if v1:
            self.inception = InceptionBlock_v1_atten(in_channels=128)  # 输出通道 480
        if v2:
            self.inception = InceptionBlock_v2_atten(in_channels=128)  # 输出通道 480
        if v3:
            self.inception = InceptionBlock_v3_atten(in_channels=128)  # 输出通道 480
        self.conv3 = nn.Sequential(
            nn.Conv2d(480, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),                                     # 输出通道 256

        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出 (B, 480, 1, 1)
        self.fc = nn.Linear(256, 1)  # 二分类
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)  # -> (B, 64, 32, 32)
        x = self.conv2(x)  # -> (B, 128, 32, 32)
        x = self.inception(x)  # -> (B, 480, 32, 32)
        x = self.conv3(x)    # -> (B, 256, 32, 32)
        x = self.gap(x)  # -> (B, 480, 1, 1)
        x = x.view(x.size(0), -1)  # -> (B, 480)
        x = self.fc(x)  # -> (B, 1)
        x = self.sigmoid(x)  # -> (B, 1), 概率值
        return x


import torch
import torch.nn as nn

class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=256):
        super(InceptionResidualBlock, self).__init__()

        # 分支1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 分支2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 分支3: 1x1 -> 3x3 -> 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 1x1 线性融合（降维或对齐输出通道）
        self.conv_linear = nn.Sequential(
            nn.Conv2d(32 * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        # 残差连接（需对齐输入输出通道）
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv_linear(out)

        # 残差相加 + 激活
        out = self.relu(out + self.shortcut(x))
        return out

class InceptionResidual_atten(nn.Module):
    def __init__(self,v1=False,v2=False,v3=False):
        super(InceptionResidual_atten, self).__init__()

        # 前置卷积层：从 3 通道提升为 64 通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )


        self.inception = InceptionResidualBlock(in_channels=128)  # 输出通道 480
        self.attn = detail_atten(dim=256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出 (B, 480, 1, 1)
        self.fc = nn.Linear(256, 1)  # 二分类
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)                # -> (B, 64, 32, 32)
        x = self.conv2(x)                 # -> (B, 128, 32, 32)
        x = self.inception(x)             # -> (B, 480, 32, 32)
        x = self.attn(x)                  # -> (B, 480, 32, 32)
        x = self.gap(x)                   # -> (B, 480, 1, 1)
        x = x.view(x.size(0), -1)         # -> (B, 480)
        x = self.fc(x)                    # -> (B, 1)
        x = self.sigmoid(x)               # -> (B, 1), 概率值
        return x
