import math

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class DSConv(nn.Module):
    """Depthwise Separable Convolution: Depthwise conv + Pointwise conv."""
    def __init__(self, c1, c2, k=3, s=1, d=1, act=True):
        super().__init__()
        # Depthwise convolution: groups = c1 实现通道分离
        self.depthwise = Conv(c1, c1, k=k, s=s, p=None, g=c1, d=d, act=act)
        # Pointwise convolution: 1x1卷积整合通道
        self.pointwise = Conv(c1, c2, k=1, s=1, p=0, g=1, d=1, act=act)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

