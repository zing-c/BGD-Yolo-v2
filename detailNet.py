import torch
import torch.nn as nn
import torch.nn.functional as F
from Resnet6 import detail_atten


class Detail_Net_attn_block(nn.Module):
    def __init__(self):
        super(Detail_Net_attn_block, self).__init__()

        # =======================================================
        # 1. Large Route (保留最多细节，只做 stride=1 的卷积)
        # =======================================================
        # 输入 3 -> 16
        self.conv_L1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.bn_L1 = nn.BatchNorm2d(16)

        # ResBlock 1 (16 -> 32)
        self.conv_L2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn_L2 = nn.BatchNorm2d(32)
        self.sc_L2 = nn.Sequential(nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32))  # 1x1 升维

        # ResBlock 2 (32 -> 64)
        self.conv_L3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn_L3 = nn.BatchNorm2d(64)
        self.sc_L3 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64))

        # ResBlock 3 (64 -> 128)
        self.conv_L4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn_L4 = nn.BatchNorm2d(128)
        self.sc_L4 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128))

        # =======================================================
        # 2. Medium Route (中等感受野，总 stride=4)
        # =======================================================
        # 使用两个 stride=2 的 3x3 卷积代替原来的大卷积，效果更细腻
        self.conv_M1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(32)
        )
        # 这里的 Shortcut 需要用 stride=4 的 1x1 卷积或者 Pool 配合
        self.sc_M1 = nn.Sequential(
            nn.Conv2d(3, 32, 1, stride=4),
            nn.BatchNorm2d(32)
        )

        # ResBlock 2 (32 -> 64)
        self.conv_M2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_M2 = nn.BatchNorm2d(64)
        self.sc_M2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64))

        # =======================================================
        # 3. Small Route (大感受野，总 stride=8)
        # =======================================================
        # 连续下采样 3 次: /2 -> /4 -> /8
        self.conv_S1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 最终 /8
            nn.BatchNorm2d(64)
        )
        self.sc_S1 = nn.Sequential(
            nn.Conv2d(3, 64, 1, stride=8),
            nn.BatchNorm2d(64)
        )

        # ResBlock 2 (64 -> 64)
        self.conv_S2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_S2 = nn.BatchNorm2d(64)
        # 输入输出都是64，直接相加即可，不需要 Conv 1x1，但为了对齐代码结构还是写一个
        self.sc_S2 = nn.Identity()

        # =======================================================
        # 4. Attention & Compression (保持你的原逻辑)
        # =======================================================
        # 假设 detail_atten 是你在外部定义的类
        self.atten_large = detail_atten(dim=128)
        self.atten_medium = detail_atten(dim=64)
        self.atten_small = detail_atten(dim=64)

        # 压缩到 1 通道
        self.compress_l = nn.Conv2d(128, 1, kernel_size=1)
        self.compress_m = nn.Conv2d(64, 1, kernel_size=1)
        self.compress_s = nn.Conv2d(64, 1, kernel_size=1)

        # 最终融合
        self.bn_cat = nn.BatchNorm2d(3)  # 1+1+1 = 3 通道
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, 3, H, W]

        # === 1. Large Route (保持原分辨率，提取细节) ===
        # Layer 1 (Conv Only)
        x_L = self.relu(self.bn_L1(self.conv_L1(x)))

        # Layer 2 (ResBlock)
        sc = self.sc_L2(x_L)
        feat = self.bn_L2(self.conv_L2(x_L))
        x_L = self.relu(feat + sc)  # 先加后 ReLU

        # Layer 3 (ResBlock)
        sc = self.sc_L3(x_L)
        feat = self.bn_L3(self.conv_L3(x_L))
        x_L = self.relu(feat + sc)

        # Layer 4 (ResBlock)
        sc = self.sc_L4(x_L)
        feat = self.bn_L4(self.conv_L4(x_L))
        x_L = self.relu(feat + sc)

        # === 2. Medium Route (下采样到 1/4) ===
        # Layer 1
        sc = self.sc_M1(x)
        feat = self.conv_M1(x)  # 已经在 Sequential 里包含了 BN
        x_M = self.relu(feat + sc)

        # Layer 2
        sc = self.sc_M2(x_M)
        feat = self.bn_M2(self.conv_M2(x_M))
        x_M = self.relu(feat + sc)

        # === 3. Small Route (下采样到 1/8) ===
        # Layer 1
        sc = self.sc_S1(x)
        feat = self.conv_S1(x)
        x_S = self.relu(feat + sc)

        # Layer 2
        feat = self.bn_S2(self.conv_S2(x_S))
        x_S = self.relu(feat + x_S)  # Identity shortcut

        # === 4. Attention ===
        x_L = self.atten_large(x_L)  # [B, 128, H, W]
        x_M = self.atten_medium(x_M)  # [B, 64, H/4, W/4]
        x_S = self.atten_small(x_S)  # [B, 64, H/8, W/8]

        # === 5. Compression (压缩到 1 通道) ===
        x_L = self.compress_l(x_L)  # [B, 1, H, W]
        x_M = self.compress_m(x_M)  # [B, 1, H/4, W/4]
        x_S = self.compress_s(x_S)  # [B, 1, H/8, W/8]

        # === 6. Upsampling & Concat (关键步骤) ===
        # 因为 x_M 和 x_S 变小了，必须放大回 x_L 的尺寸才能拼接
        target_size = x_L.shape[2:]  # (H, W)

        x_M_up = F.interpolate(x_M, size=target_size, mode='bilinear', align_corners=False)
        x_S_up = F.interpolate(x_S, size=target_size, mode='bilinear', align_corners=False)

        # 拼接: 1+1+1 -> 3 通道
        # output shape: [B, 3, H, W]
        concatenated = torch.cat([x_L, x_M_up, x_S_up], dim=1)

        out = self.relu(self.bn_cat(concatenated))

        return out




class Detail_Net_attn(nn.Module):
    def __init__(self):
        super(Detail_Net_attn, self).__init__()

        # =======================================================
        # 1. Large Route (保留最多细节，只做 stride=1 的卷积)
        # =======================================================
        # 输入 3 -> 16
        self.conv_L1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.bn_L1 = nn.BatchNorm2d(16)

        # ResBlock 1 (16 -> 32)
        self.conv_L2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn_L2 = nn.BatchNorm2d(32)
        self.sc_L2 = nn.Sequential(nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32))

        # ResBlock 2 (32 -> 64)
        self.conv_L3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn_L3 = nn.BatchNorm2d(64)
        self.sc_L3 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64))

        # ResBlock 3 (64 -> 128)
        self.conv_L4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn_L4 = nn.BatchNorm2d(128)
        self.sc_L4 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128))

        # =======================================================
        # 2. Medium Route (中等感受野，总 stride=4)
        # =======================================================
        self.conv_M1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(32)
        )
        self.sc_M1 = nn.Sequential(
            nn.Conv2d(3, 32, 1, stride=4),
            nn.BatchNorm2d(32)
        )

        # ResBlock 2 (32 -> 64)
        self.conv_M2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn_M2 = nn.BatchNorm2d(64)
        self.sc_M2 = nn.Sequential(nn.Conv2d(32, 64, 1), nn.BatchNorm2d(64))

        # =======================================================
        # 3. Small Route (大感受野，总 stride=8)
        # =======================================================
        self.conv_S1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 最终 /8
            nn.BatchNorm2d(64)
        )
        self.sc_S1 = nn.Sequential(
            nn.Conv2d(3, 64, 1, stride=8),
            nn.BatchNorm2d(64)
        )

        # ResBlock 2 (64 -> 64)
        self.conv_S2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_S2 = nn.BatchNorm2d(64)
        self.sc_S2 = nn.Identity()

        # =======================================================
        # 4. Attention & Compression
        # =======================================================
        self.atten_large = detail_atten(dim=128)
        self.atten_medium = detail_atten(dim=64)
        self.atten_small = detail_atten(dim=64)

        # 压缩到 1 通道 (关键步骤，大大减少 FC 参数量)
        self.compress_l = nn.Conv2d(128, 1, kernel_size=1)
        self.compress_m = nn.Conv2d(64, 1, kernel_size=1)
        self.compress_s = nn.Conv2d(64, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        # =======================================================
        # 5. Classification Head (全连接层)
        # =======================================================
        # 假设输入图片尺寸为 64x64:
        # Large:  64x64 * 1 channel = 4096
        # Medium: 16x16 * 1 channel = 256
        # Small:  8x8   * 1 channel = 64
        # Total:  4096 + 256 + 64 = 4416

        self.fc0 = nn.Linear(4416, 512)
        self.fc_final = nn.Linear(512, 1)  # 输出一个概率值

    def forward(self, x):
        # x: [B, 3, 64, 64]

        # === 1. Large Route ===
        x_L = self.relu(self.bn_L1(self.conv_L1(x)))

        sc = self.sc_L2(x_L)
        feat = self.bn_L2(self.conv_L2(x_L))
        x_L = self.relu(feat + sc)

        sc = self.sc_L3(x_L)
        feat = self.bn_L3(self.conv_L3(x_L))
        x_L = self.relu(feat + sc)

        sc = self.sc_L4(x_L)
        feat = self.bn_L4(self.conv_L4(x_L))
        x_L = self.relu(feat + sc)

        # === 2. Medium Route ===
        sc = self.sc_M1(x)
        feat = self.conv_M1(x)
        x_M = self.relu(feat + sc)

        sc = self.sc_M2(x_M)
        feat = self.bn_M2(self.conv_M2(x_M))
        x_M = self.relu(feat + sc)

        # === 3. Small Route ===
        sc = self.sc_S1(x)
        feat = self.conv_S1(x)
        x_S = self.relu(feat + sc)

        feat = self.bn_S2(self.conv_S2(x_S))
        x_S = self.relu(feat + x_S)

        # === 4. Attention ===
        x_L = self.atten_large(x_L)  # [B, 128, 64, 64]
        x_M = self.atten_medium(x_M)  # [B, 64, 16, 16]
        x_S = self.atten_small(x_S)  # [B, 64, 8, 8]

        # === 5. Compression (变回 1 通道) ===
        x_L = self.compress_l(x_L)  # [B, 1, 64, 64]
        x_M = self.compress_m(x_M)  # [B, 1, 16, 16]
        x_S = self.compress_s(x_S)  # [B, 1, 8, 8]

        # === 6. Flatten & Concat ===
        # 展平所有特征图
        flat_L = torch.flatten(x_L, start_dim=1)  # [B, 4096]
        flat_M = torch.flatten(x_M, start_dim=1)  # [B, 256]
        flat_S = torch.flatten(x_S, start_dim=1)  # [B, 64]

        # 拼接
        concatenated = torch.cat([flat_L, flat_M, flat_S], dim=1)  # [B, 4416]

        # === 7. FC Layers ===
        out = self.relu(self.fc0(concatenated))  # [B, 512]
        out = self.fc_final(out)  # [B, 1]
        out = torch.sigmoid(out)  # 归一化到 0~1 之间

        return out