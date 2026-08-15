import torch
import torch.nn as nn
import torch.nn.functional as F
from triton.ops.blocksparse import softmax


class ResNet6(nn.Module):
    def __init__(self):
        super(ResNet6, self).__init__()
        # 输入层：64x64 图片，3通道（RGB）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()


        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 特征图尺寸为 4x4
        self.fc2 = nn.Linear(512, 1)  # 输出1，代表是否是玻璃（0 或 1）

    def forward(self, x):
        # 第一层
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)  # 32x32

        # 第二层
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # 16x16

        # 第三层
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # 8x8

        # 残差连接
        residual = self.shortcut(x)  # 调整尺寸和通道数
        x = self.relu4(self.bn4(self.conv4(x)))
        x = x + residual  # 残差连接

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.relu5(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 二分类输出

        return x

class Detail_Net(nn.Module):
    def __init__(self):
        super(Detail_Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.large_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.shortcut_L_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.large_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 to 32x32
        self.large_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.shortcut_L_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(32)
        )
        self.large_route_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 to 16x16
        self.large_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),)
        self.shortcut_L_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(64)
        )
        self.large_route_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.large_route_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.shortcut_L_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )



        self.medium_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=6, padding=1, stride=4),
            # 64 x 64 to 16 x 16
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.shortcut_M_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.medium_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        self.shortcut_M_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.medium_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.medium_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.shortcut_M_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )


        self.small_route_l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=13, padding=3, stride=8),
            # 64 x 64 to 8 x 8
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        self.shortcut_S_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.small_route_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.shortcut_S_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )

        # 输入层：64x64 图片，3通道（RGB）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()


        # 全连接层
        self.fc0 = nn.Linear(2 * 128 * 8 * 2, 512)  # 特征图尺寸为 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 特征图尺寸为 4x4
        self.fc2 = nn.Linear(512, 1)  # 输出1，代表是否是玻璃（0 或 1）

    def forward(self, x):
        x = self.pool(x)
        # Large Route
        short_cut = self.shortcut_L_l1(x)
        x_L = self.large_route_l1(x) + short_cut
        x_L = self.large_route_pool_1(x_L)
        short_cut = self.shortcut_L_l2(x_L)
        x_L = self.large_route_l2(x_L) + short_cut
        x_L = self.large_route_pool_2(x_L)
        short_cut = self.shortcut_L_l3(x_L)
        x_L = self.large_route_l3(x_L) + short_cut
        x_L = self.large_route_pool_3(x_L)
        short_cut = self.shortcut_L_l4(x_L)
        x_L = self.large_route_l4(x_L) + short_cut

        # Medium Route
        short_cut = self.shortcut_M_l1(x)
        x_M = self.medium_route_l1(x) + short_cut
        short_cut = self.shortcut_M_l2(x_M)
        x_M = self.medium_route_l2(x_M) + short_cut
        x_M = self.medium_route_pool_1(x_M)
        short_cut = self.shortcut_M_l3(x_M)
        x_M = self.medium_route_l3(x_M) + short_cut

        # Small Route
        short_cut = self.shortcut_S_l1(x)
        x_S = self.small_route_l1(x) + short_cut
        short_cut = self.shortcut_S_l2(x_S)
        x_S = self.small_route_l2(x_S) + short_cut


        #y_medium = self.medium_route(x)
        #y_small = self.small_route(x)
        y_large_flattened = torch.flatten(x_L, start_dim=1)  # 从第1维（通道维）开始展平
        y_medium_flattened = torch.flatten(x_M, start_dim=1)
        y_small_flattened = torch.flatten(x_S, start_dim=1)

        # 拼接操作
        concatenated = torch.cat([y_large_flattened, y_medium_flattened, y_small_flattened], dim=1)
        # 展平

        # 全连接层
        x = self.relu4(self.fc0(concatenated))
        x = torch.sigmoid(self.fc2(x))  # 二分类输出

        return x



# 注意力模块
class detail_atten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,dilation=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=3,dilation=3)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=5,dilation=5)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=7,dilation=7)
        self.conv_merge = nn.Conv2d(4*dim, dim, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat([x1,x2,x3,x4],dim=1)
        out = F.sigmoid(self.conv_merge(out)) * x


        return out



class Detail_Net_attn(nn.Module):
    def __init__(self):
        super(Detail_Net_attn, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.large_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            )
        self.shortcut_L_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.large_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 to 32x32
        self.large_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            )
        self.shortcut_L_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(32)
        )
        self.large_route_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 to 16x16
        self.large_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
           )
        self.shortcut_L_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(64)
        )
        self.large_route_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.large_route_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.shortcut_L_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )



        self.medium_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=6, padding=1, stride=4),
            # 64 x 64 to 16 x 16
            nn.BatchNorm2d(16),
            )
        self.shortcut_M_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.medium_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
           )
        self.shortcut_M_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.medium_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.medium_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

        )
        self.shortcut_M_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )


        self.small_route_l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=13, padding=3, stride=8),
            # 64 x 64 to 8 x 8
            nn.BatchNorm2d(32),
           )
        self.shortcut_S_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.small_route_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

        )
        self.shortcut_S_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )

        # 输入层：64x64 图片，3通道（RGB）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        #ATTN
        self.atten_large1 = detail_atten(dim=32)
        self.atten_large = detail_atten(dim=128)
        self.atten_medium = detail_atten(dim=64)
        self.atten_medium1 = detail_atten(dim=32)
        self.atten_small = detail_atten(dim=64)


        # 全连接层
        self.fc0 = nn.Linear(2 * 128 * 8 * 2 , 512 )  # 特征图尺寸为 4x4
        self.fc1 = nn.Linear(128 * 4 * 4 , 512)  # 特征图尺寸为 4x4
        self.fc2 = nn.Linear(512, 1)  # 输出1，代表是否是玻璃（0 或 1）

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x)
        # Large Route
        short_cut = self.shortcut_L_l1(x)
        x_L = self.large_route_l1(x) + short_cut
        x_L = self.relu(x_L)
        x_L = self.large_route_pool_1(x_L)
        short_cut = self.shortcut_L_l2(x_L)
        x_L = self.large_route_l2(x_L) + short_cut
        x_L = self.relu(x_L)
        x_L = self.large_route_pool_2(x_L)
        short_cut = self.shortcut_L_l3(x_L)
        x_L = self.large_route_l3(x_L) + short_cut
        x_L = self.relu(x_L)
        x_L = self.large_route_pool_3(x_L)
        short_cut = self.shortcut_L_l4(x_L)
        x_L = self.large_route_l4(x_L) + short_cut
        x_L = self.relu(x_L)

        # Medium Route
        short_cut = self.shortcut_M_l1(x)
        x_M = self.medium_route_l1(x) + short_cut
        x_M = self.relu(x_M)
        short_cut = self.shortcut_M_l2(x_M)
        x_M = self.medium_route_l2(x_M) + short_cut
        x_M = self.relu(x_M)
        x_M = self.medium_route_pool_1(x_M)
        short_cut = self.shortcut_M_l3(x_M)
        x_M = self.medium_route_l3(x_M) + short_cut
        x_M = self.relu(x_M)

        # Small Route
        short_cut = self.shortcut_S_l1(x)
        x_S = self.small_route_l1(x) + short_cut
        x_S = self.relu(x_S)
        short_cut = self.shortcut_S_l2(x_S)
        x_S = self.small_route_l2(x_S) + short_cut
        x_S = self.relu(x_S)

        # 加入 attention 模块
        x_L = self.atten_large(x_L)  # shape: [B, 128, 8, 8]
        x_M = self.atten_medium(x_M)  # shape: [B, 64, 8, 8]
        x_S = self.atten_small(x_S)  # shape: [B, 64, 8, 8]

        #y_medium = self.medium_route(x)
        #y_small = self.small_route(x)
        y_large_flattened = torch.flatten(x_L, start_dim=1)  # 从第1维（通道维）开始展平
        y_medium_flattened = torch.flatten(x_M, start_dim=1)
        y_small_flattened = torch.flatten(x_S, start_dim=1)

        # 拼接操作
        concatenated = torch.cat([y_large_flattened, y_medium_flattened, y_small_flattened], dim=1)
        # 展平

        # 全连接层
        x = self.relu4(self.fc0(concatenated))
        x = self.fc2(x)

        # x= self.sigmoid(x)


        return x


class Detail_Net_attn_dialation(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # --- 1. Large Route (保持原有下采样逻辑，作为主干并提供尺寸对齐基准) ---
        # 结果为 8x8x128
        self.large_route_l1 = self._make_layer(3, 16, stride=1)
        self.large_route_pool1 = nn.MaxPool2d(2, 2)                  # 32x32
        self.large_route_l2 = self._make_layer(16, 32, stride=1)
        self.large_route_pool2 = nn.MaxPool2d(2, 2)                  # 16x16
        self.large_route_l3 = self._make_layer(32, 64, stride=1)
        self.large_route_pool3 = nn.MaxPool2d(2, 2)                  # 8x8
        self.large_route_l4 = self._make_layer(64, 128, stride=1)

        # --- 2. Medium Route (取消 Stride 4，改用 Dilation) ---
        # 原 RF=6。现用 kernel=3, dilation=3, stride=1 -> RF=7。
        self.medium_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Shortcut 也要取消 stride
        self.shortcut_M1 = nn.Conv2d(3, 16, kernel_size=1)

        self.medium_route_l2 = self._make_layer(16, 32)
        # 将 64x64 强制池化到 8x8 以匹配 Large Route
        self.medium_route_pool = nn.AdaptiveAvgPool2d(8)
        self.medium_route_l3 = self._make_layer(32, 64)

        # --- 3. Small Route (取消 Stride 8，改用大 Dilation) ---
        # 原 RF=13。现用 kernel=3, dilation=6, stride=1 -> RF=13。
        self.small_route_l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=6, stride=1, dilation=6),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.shortcut_S1 = nn.Conv2d(3, 32, kernel_size=1)

        self.small_route_l2 = self._make_layer(32, 64)
        # 将 64x64 强制池化到 8x8
        self.small_route_pool = nn.AdaptiveAvgPool2d(8)

        # --- 4. Attention ---
        self.atten_large = detail_atten(dim=128)
        self.atten_medium = detail_atten(dim=64)
        self.atten_small = detail_atten(dim=64)

        # --- 5. 全连接层 ---
        self.flatten_dim = (128 + 64 + 64) * 8 * 8  # 16384
        self.fc0 = nn.Linear(self.flatten_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_ch, out_ch, stride=1):
        return nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ),
            'shortcut': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        })

    def forward(self, x):
        # 1. Large Route
        res_L = self.large_route_l1['shortcut'](x)
        x_L = self.large_route_l1['conv'](x) + res_L
        x_L = self.large_route_pool1(x_L)
        res_L = self.large_route_l2['shortcut'](x_L)
        x_L = self.large_route_l2['conv'](x_L) + res_L
        x_L = self.large_route_pool2(x_L)
        res_L = self.large_route_l3['shortcut'](x_L)
        x_L = self.large_route_l3['conv'](x_L) + res_L
        x_L = self.large_route_pool3(x_L)
        res_L = self.large_route_l4['shortcut'](x_L)
        x_L = self.large_route_l4['conv'](x_L) + res_L
        x_L = self.atten_large(x_L)

        # 2. Medium Route (High-Res Dilated)
        x_M = self.medium_route_l1(x) + self.shortcut_M1(x)
        res_M = self.medium_route_l2['shortcut'](x_M)
        x_M = self.medium_route_l2['conv'](x_M) + res_M
        # 在特征提取后再进行池化，保证了卷积是在 64x64 的高分图上完成的
        x_M = self.medium_route_pool(x_M)
        res_M = self.medium_route_l3['shortcut'](x_M)
        x_M = self.medium_route_l3['conv'](x_M) + res_M
        x_M = self.atten_medium(x_M)

        # 3. Small Route (High-Res Dilated)
        x_S = self.small_route_l1(x) + self.shortcut_S1(x)
        res_S = self.small_route_l2['shortcut'](x_S)
        x_S = self.small_route_l2['conv'](x_S) + res_S
        x_S = self.small_route_pool(x_S)
        x_S = self.atten_small(x_S)

        #融合与分类
        y_L = torch.flatten(x_L, 1)
        y_M = torch.flatten(x_M, 1)
        y_S = torch.flatten(x_S, 1)
        combined = torch.cat([y_L, y_M, y_S], dim=1)
        out = F.relu(self.fc0(combined))
        out = self.fc2(out)
        return self.sigmoid(out)

class Detail_Net_attn_block(nn.Module):
    def __init__(self):
        super(Detail_Net_attn_block, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.large_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            )
        self.shortcut_L_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.large_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 to 32x32
        self.large_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
           )
        self.shortcut_L_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(32)
        )
        self.large_route_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 to 16x16
        self.large_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            )
        self.shortcut_L_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(64)
        )
        self.large_route_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.large_route_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),

        )
        self.shortcut_L_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )



        self.medium_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=6, padding=1, stride=4),
            # 64 x 64 to 16 x 16
            nn.BatchNorm2d(16),
      )
        self.shortcut_M_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.medium_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
     )
        self.shortcut_M_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.medium_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.medium_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )
        self.shortcut_M_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )


        self.small_route_l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=13, padding=3, stride=8),
            # 64 x 64 to 8 x 8
            nn.BatchNorm2d(32),)
        self.shortcut_S_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.small_route_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),

        )
        self.shortcut_S_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )

        # 输入层：64x64 图片，3通道（RGB）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )
        self.relu = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        #ATTN
        self.atten_large1 = detail_atten(dim=32)
        self.atten_large = detail_atten(dim=128)
        self.atten_medium = detail_atten(dim=64)
        self.atten_medium1 = detail_atten(dim=32)
        self.atten_small = detail_atten(dim=64)

        self.conv_l = nn.Conv2d(128, 1, kernel_size=1)
        self.conv_m = nn.Conv2d(64, 1, kernel_size=1)
        self.conv_s = nn.Conv2d(64, 1, kernel_size=1)
        self.bn_cat=  nn.BatchNorm2d(3)

        self.conv_for_yolo_mid = nn.Conv2d(64+3, 64, kernel_size=3,padding=1)

        self.bn_for_yolo_mid = nn.BatchNorm2d(64)
        self.relu_for_yolo_mid = nn.ReLU()

        self.conv_for_yolo_low = nn.Conv2d(64 + 3, 64, kernel_size=3, padding=1)
        
        self.bn_for_yolo_low = nn.BatchNorm2d(64)
        self.relu_for_yolo_low = nn.ReLU()

    def forward(self, x):
        x = self.pool(x)
        # Large Route
        short_cut = self.shortcut_L_l1(x)
        x_L = self.large_route_l1(x) + short_cut
        x_L = self.relu(x_L)
        x_L = self.large_route_pool_1(x_L)
        short_cut = self.shortcut_L_l2(x_L)
        x_L = self.large_route_l2(x_L) + short_cut
        x_L = self.relu(x_L)
        x_L = self.large_route_pool_2(x_L)
        short_cut = self.shortcut_L_l3(x_L)
        x_L = self.large_route_l3(x_L) + short_cut
        x_L = self.relu(x_L)
        x_L = self.large_route_pool_3(x_L)
        short_cut = self.shortcut_L_l4(x_L)
        x_L = self.large_route_l4(x_L) + short_cut
        x_L = self.relu(x_L)

        # Medium Route
        short_cut = self.shortcut_M_l1(x)
        x_M = self.medium_route_l1(x) + short_cut
        x_M = self.relu(x_M)
        short_cut = self.shortcut_M_l2(x_M)
        x_M = self.medium_route_l2(x_M) + short_cut
        x_M = self.relu(x_M)
        x_M = self.medium_route_pool_1(x_M)
        short_cut = self.shortcut_M_l3(x_M)
        x_M = self.medium_route_l3(x_M) + short_cut
        x_M = self.relu(x_M)

        # Small Route
        short_cut = self.shortcut_S_l1(x)
        x_S = self.small_route_l1(x) + short_cut
        x_M = self.relu(x_M)
        short_cut = self.shortcut_S_l2(x_S)
        x_S = self.small_route_l2(x_S) + short_cut
        x_M = self.relu(x_M)

        # 加入 attention 模块
        x_L = self.atten_large(x_L)  # shape: [B, 128, 8, 8]
        x_M = self.atten_medium(x_M)  # shape: [B, 64, 8, 8]
        x_S = self.atten_small(x_S)  # shape: [B, 64, 8, 8]

        x_L = self.conv_l(x_L)  # shape: [B, 128, 8, 8]
        x_M = self.conv_m(x_M)  # shape: [B, 64, 8, 8]
        x_S = self.conv_s(x_S)  # shape: [B, 64, 8, 8]



        # 拼接操作
        concatenated = self.relu4(self.bn_cat(torch.cat([x_L, x_M, x_S], dim=1)))
        # 展平




        return concatenated


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1,),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

#attn后先进行 gap 再进行 concat -> conv -> fc
class Detail_Net_attn_2(nn.Module):
    def __init__(self):
        super(Detail_Net_attn_2, self).__init__()

        self.conv1=nn.Conv2d(3, 32, kernel_size=3, padding=1,stride=2)
        # Large
        self.large = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(in_c=128, out_c=128, stride=1),
            ResidualBlock(in_c=128, out_c=128, stride=1),
            detail_atten(dim=128)
        )

        #Medium
        self.medium = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(in_c=128, out_c=128, stride=1),
            detail_atten(dim=128)
        )

        # Small
        self.small = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(in_c=64, out_c=64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            detail_atten(dim=64)
        )
        #concat后的卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(128 + 128 + 64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear( 128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.conv1(x)
        x_L = self.large(x)
        x_M = self.medium(x)
        x_S = self.small(x)

        x_cat = torch.cat([x_L, x_M, x_S], dim=1)
        x_fused = self.conv2(x_cat)
        x_gap = self.gap(x_fused).squeeze(-1).squeeze(-1)  # [B,128]
        out = self.classifier(x_gap)  # [B,1]
        return out

# 将最开始的 maxpooling 改成 conv stride 2
# 后面concat 改成 conv +gap +fc
class Detail_Net_attn_1(nn.Module):
    def __init__(self):
        super(Detail_Net_attn_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.large_route_l1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.shortcut_L_l1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.large_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 to 32x32
        self.large_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.shortcut_L_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(32)
        )
        self.large_route_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 to 16x16
        self.large_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),)
        self.shortcut_L_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(64)
        )
        self.large_route_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.large_route_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.shortcut_L_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )



        self.medium_route_l1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=6, padding=1, stride=4),
            # 64 x 64 to 16 x 16
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.shortcut_M_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.medium_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        self.shortcut_M_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.medium_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.medium_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.shortcut_M_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )


        self.small_route_l1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=13, padding=3, stride=8),
            # 64 x 64 to 8 x 8
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        self.shortcut_S_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.small_route_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.shortcut_S_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )

        # 输入层：64x64 图片，3通道（RGB）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        #ATTN
        self.atten_large1 = detail_atten(dim=32)
        self.atten_large = detail_atten(dim=128)
        self.atten_medium = detail_atten(dim=64)
        self.atten_medium1 = detail_atten(dim=32)
        self.atten_small = detail_atten(dim=64)

        # 全连接层
        self.fc0 = nn.Linear(2 * 128 * 8 * 2, 512)  # 特征图尺寸为 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 特征图尺寸为 4x4
        self.fc2 = nn.Linear(512, 1)  # 输出1，代表是否是玻璃（0 或 1）

        # concat后的卷积
        self.conv_final = nn.Sequential(
            nn.Conv2d(128 + 64 + 64, 128, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        # Large Route
        short_cut = self.shortcut_L_l1(x)
        x_L = self.large_route_l1(x) + short_cut
        x_L = self.large_route_pool_1(x_L)
        short_cut = self.shortcut_L_l2(x_L)
        x_L = self.large_route_l2(x_L) + short_cut
        x_L = self.large_route_pool_2(x_L)
        short_cut = self.shortcut_L_l3(x_L)
        x_L = self.large_route_l3(x_L) + short_cut
        x_L = self.large_route_pool_3(x_L)
        short_cut = self.shortcut_L_l4(x_L)
        x_L = self.large_route_l4(x_L) + short_cut

        # Medium Route
        short_cut = self.shortcut_M_l1(x)
        x_M = self.medium_route_l1(x) + short_cut
        short_cut = self.shortcut_M_l2(x_M)
        x_M = self.medium_route_l2(x_M) + short_cut
        x_M = self.medium_route_pool_1(x_M)
        short_cut = self.shortcut_M_l3(x_M)
        x_M = self.medium_route_l3(x_M) + short_cut

        # Small Route
        short_cut = self.shortcut_S_l1(x)
        x_S = self.small_route_l1(x) + short_cut
        short_cut = self.shortcut_S_l2(x_S)
        x_S = self.small_route_l2(x_S) + short_cut

        # # 加入 attention 模块
        # x_L = self.atten_large(x_L)  # shape: [B, 128, 8, 8]
        # x_M = self.atten_medium(x_M)  # shape: [B, 64, 8, 8]
        # x_S = self.atten_small(x_S)  # shape: [B, 64, 8, 8]


        # 拼接操作
        concatenated = torch.cat([x_L, x_M, x_S], dim=1)
        x = self.conv_final(concatenated)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)


        return x



class demo1(nn.Module):
    def __init__(self):
        super(demo1, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.large_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.shortcut_L_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.large_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 to 32x32
        self.large_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.shortcut_L_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(32)
        )
        self.large_route_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 to 16x16
        self.large_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),)
        self.shortcut_L_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(64)
        )
        self.large_route_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.large_route_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.shortcut_L_l4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # 1x1 卷积调整通道数和尺寸
            nn.BatchNorm2d(128)
        )



        self.medium_route_l1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=6, padding=1, stride=4),
            # 64 x 64 to 16 x 16
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.shortcut_M_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )
        self.medium_route_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        self.shortcut_M_l2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.medium_route_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 to 8x8
        self.medium_route_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.shortcut_M_l3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )


        self.small_route_l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=13, padding=3, stride=8),
            # 64 x 64 to 8 x 8
            nn.BatchNorm2d(32),
            nn.ReLU(),)
        self.shortcut_S_l1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )
        self.small_route_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.shortcut_S_l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )

        # 输入层：64x64 图片，3通道（RGB）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.gap= nn.AdaptiveAvgPool2d((1, 1))

        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        #ATTN
        self.atten_large1 = detail_atten(dim=32)
        self.atten_large = detail_atten(dim=128)
        self.atten_medium = detail_atten(dim=64)
        self.atten_medium1 = detail_atten(dim=32)
        self.atten_small = detail_atten(dim=64)
        self.atten = detail_atten(dim=256)

        # 全连接层
        self.fc = nn.Linear(256, 512)  # 特征图尺寸为 4x4
        self.fc0 = nn.Linear(2 * 128 * 8 * 2, 512)  # 特征图尺寸为 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 特征图尺寸为 4x4
        self.fc2 = nn.Linear(512, 1)  # 输出1，代表是否是玻璃（0 或 1）

    def forward(self, x):
        x = self.pool(x)
        # Large Route
        short_cut = self.shortcut_L_l1(x)
        x_L = self.large_route_l1(x) + short_cut
        x_L = self.large_route_pool_1(x_L)
        short_cut = self.shortcut_L_l2(x_L)
        x_L = self.large_route_l2(x_L) + short_cut
        x_L = self.large_route_pool_2(x_L)
        short_cut = self.shortcut_L_l3(x_L)
        x_L = self.large_route_l3(x_L) + short_cut
        x_L = self.large_route_pool_3(x_L)
        short_cut = self.shortcut_L_l4(x_L)
        x_L = self.large_route_l4(x_L) + short_cut

        # Medium Route
        short_cut = self.shortcut_M_l1(x)
        x_M = self.medium_route_l1(x) + short_cut
        short_cut = self.shortcut_M_l2(x_M)
        x_M = self.medium_route_l2(x_M) + short_cut
        x_M = self.medium_route_pool_1(x_M)
        short_cut = self.shortcut_M_l3(x_M)
        x_M = self.medium_route_l3(x_M) + short_cut

        # Small Route
        short_cut = self.shortcut_S_l1(x)
        x_S = self.small_route_l1(x) + short_cut
        short_cut = self.shortcut_S_l2(x_S)
        x_S = self.small_route_l2(x_S) + short_cut
        # # 加入 attention 模块
        x_L = self.atten_large(x_L)  # shape: [B, 128, 8, 8]
        x_M = self.atten_medium(x_M)  # shape: [B, 64, 8, 8]
        x_S = self.atten_small(x_S)  # shape: [B, 64, 8, 8]

        x=torch.cat([x_L, x_M, x_S], dim=1)
        x=self.atten(x)
        x =self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc(x))



        #y_medium = self.medium_route(x)
        #y_small = self.small_route(x)
        # y_large_flattened = torch.flatten(x_L, start_dim=1)  # 从第1维（通道维）开始展平
        # y_medium_flattened = torch.flatten(x_M, start_dim=1)
        # y_small_flattened = torch.flatten(x_S, start_dim=1)

        #
        # x=self.atten(concatenated)
        # # 全连接层
        # x = self.relu4(self.fc0(concatenated))
        x = torch.sigmoid(self.fc2(x))  # 二分类输出

        return x
