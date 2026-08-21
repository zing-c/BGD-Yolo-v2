"""ZIP-revision Detail networks used by the active BGD-YOLO pipeline.

The public architecture keeps the ZIP revision's compressed channels and
post-addition ReLU residual blocks.  Its spatial attention is repaired for the
actual 4x4 route outputs: dilation 1 and 2 are used instead of 1/3/5/7, whose
5/7 branches collapse to center-only sampling on a 4x4 map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


DETAIL_ARCH_VERSION = "zip_compressed_d12_v1"


class CompactSpatialAttention(nn.Module):
    """Two valid spatial scales for a 4x4 feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv_merge = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        context = torch.cat((self.conv1(inputs), self.conv2(inputs)), dim=1)
        return torch.sigmoid(self.conv_merge(context)) * inputs


class _CompressedDetailBackbone(nn.Module):
    """Shared three-route backbone from the ZIP revision."""

    architecture_id = DETAIL_ARCH_VERSION
    route_channels = (32, 32, 16)

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.large_route_l1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8))
        self.shortcut_L_l1 = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8))
        self.large_route_pool_1 = nn.MaxPool2d(2)
        self.large_route_l2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16))
        self.shortcut_L_l2 = nn.Sequential(nn.Conv2d(8, 16, 1), nn.BatchNorm2d(16))
        self.large_route_pool_2 = nn.MaxPool2d(2)
        self.large_route_l3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32))
        self.shortcut_L_l3 = nn.Sequential(nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32))
        self.large_route_pool_3 = nn.MaxPool2d(2)
        self.large_route_l4 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32))
        self.shortcut_L_l4 = nn.Sequential(nn.Conv2d(32, 32, 1), nn.BatchNorm2d(32))

        self.medium_route_l1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=6, padding=1, stride=4), nn.BatchNorm2d(8)
        )
        self.shortcut_M_l1 = nn.Sequential(nn.MaxPool2d(4), nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8))
        self.medium_route_l2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16))
        self.shortcut_M_l2 = nn.Sequential(nn.Conv2d(8, 16, 1), nn.BatchNorm2d(16))
        self.medium_route_pool_1 = nn.MaxPool2d(2)
        self.medium_route_l3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32))
        self.shortcut_M_l3 = nn.Sequential(nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32))

        self.small_route_l1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=13, padding=3, stride=8), nn.BatchNorm2d(8)
        )
        self.shortcut_S_l1 = nn.Sequential(nn.MaxPool2d(8), nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8))
        self.small_route_l2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16))
        self.shortcut_S_l2 = nn.Sequential(nn.Conv2d(8, 16, 1), nn.BatchNorm2d(16))

        self.atten_large = CompactSpatialAttention(32)
        self.atten_medium = CompactSpatialAttention(32)
        self.atten_small = CompactSpatialAttention(16)

    def _forward_routes(self, inputs: torch.Tensor):
        inputs = self.pool(inputs)

        # Full-model checkpoints created before the ZIP migration pickle this
        # class by name, but carry the historical wide (16/32/64/128) module
        # layout.  That layout has ``relu1``...``relu5`` instead of the shared
        # ``relu`` below and already embeds ReLU in each main convolutional
        # route.  Preserve its exact forward semantics so such checkpoints can
        # be evaluated without silently transplanting current architecture
        # rules onto historical weights.
        if not hasattr(self, "relu"):
            large = self.large_route_l1(inputs) + self.shortcut_L_l1(inputs)
            large = self.large_route_pool_1(large)
            large = self.large_route_l2(large) + self.shortcut_L_l2(large)
            large = self.large_route_pool_2(large)
            large = self.large_route_l3(large) + self.shortcut_L_l3(large)
            large = self.large_route_pool_3(large)
            large = self.large_route_l4(large) + self.shortcut_L_l4(large)

            medium = self.medium_route_l1(inputs) + self.shortcut_M_l1(inputs)
            medium = self.medium_route_l2(medium) + self.shortcut_M_l2(medium)
            medium = self.medium_route_pool_1(medium)
            medium = self.medium_route_l3(medium) + self.shortcut_M_l3(medium)

            small = self.small_route_l1(inputs) + self.shortcut_S_l1(inputs)
            small = self.small_route_l2(small) + self.shortcut_S_l2(small)

            return (
                self.atten_large(large),
                self.atten_medium(medium),
                self.atten_small(small),
            )

        large = self.relu(self.large_route_l1(inputs) + self.shortcut_L_l1(inputs))
        large = self.large_route_pool_1(large)
        large = self.relu(self.large_route_l2(large) + self.shortcut_L_l2(large))
        large = self.large_route_pool_2(large)
        large = self.relu(self.large_route_l3(large) + self.shortcut_L_l3(large))
        large = self.large_route_pool_3(large)
        large = self.relu(self.large_route_l4(large) + self.shortcut_L_l4(large))

        medium = self.relu(self.medium_route_l1(inputs) + self.shortcut_M_l1(inputs))
        medium = self.relu(self.medium_route_l2(medium) + self.shortcut_M_l2(medium))
        medium = self.medium_route_pool_1(medium)
        medium = self.relu(self.medium_route_l3(medium) + self.shortcut_M_l3(medium))

        small = self.relu(self.small_route_l1(inputs) + self.shortcut_S_l1(inputs))
        small = self.relu(self.small_route_l2(small) + self.shortcut_S_l2(small))

        return (
            self.atten_large(large),
            self.atten_medium(medium),
            self.atten_small(small),
        )

    def pretrained_state_keys(self):
        """State entries that must come from Detail classification pretraining."""
        excluded = ("fc0.", "fc2.", "conv_l.", "conv_m.", "conv_s.", "bn_cat.", "semantic_head.")
        return {key for key in self.state_dict() if not key.startswith(excluded)}


class Detail_Net_attn(_CompressedDetailBackbone):
    """ZIP-revision binary Detail classifier used for standalone pretraining."""

    def __init__(self):
        super().__init__()
        flattened_features = sum(self.route_channels) * 4 * 4
        self.fc0 = nn.Linear(flattened_features, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        routes = self._forward_routes(inputs)
        features = torch.cat([route.flatten(1) for route in routes], dim=1)
        return torch.sigmoid(self.fc2(self.relu(self.fc0(features))))


class Detail_Net_attn_block(_CompressedDetailBackbone):
    """Feature-only ZIP Detail encoder returning [N, 3, 4, 4]."""

    _LEGACY_UNUSED_MODULES = (
        "conv1", "bn1", "pool1", "conv2", "bn2", "pool2", "conv3", "bn3", "pool3",
        "shortcut", "conv4", "bn4", "relu1", "relu2", "relu3", "relu4", "relu5",
        "atten_large1", "atten_medium1", "conv_for_yolo", "bn_for_yolo", "relu_for_yolo",
        "conv_for_yolo_mid", "bn_for_yolo_mid", "relu_for_yolo_mid",
        "conv_for_yolo_low", "bn_for_yolo_low", "relu_for_yolo_low",
    )

    def __init__(self):
        super().__init__()
        large_channels, medium_channels, small_channels = self.route_channels
        self.conv_l = nn.Conv2d(large_channels, 1, kernel_size=1)
        self.conv_m = nn.Conv2d(medium_channels, 1, kernel_size=1)
        self.conv_s = nn.Conv2d(small_channels, 1, kernel_size=1)
        self.bn_cat = nn.BatchNorm2d(3)

    def prune_legacy_unused_modules(self):
        for name in self._LEGACY_UNUSED_MODULES:
            if hasattr(self, name):
                delattr(self, name)
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        large, medium, small = self._forward_routes(inputs)
        compressed = torch.cat((self.conv_l(large), self.conv_m(medium), self.conv_s(small)), dim=1)
        activation = self.relu if hasattr(self, "relu") else self.relu4
        return activation(self.bn_cat(compressed))


class Detail_Net_attn_semantic(Detail_Net_attn_block):
    """Semantic-channel variant used by the unchanged V2/V3 fusion pipeline."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.out_channels = int(out_channels)
        groups = 8 if self.out_channels % 8 == 0 else 1
        self.semantic_head = nn.Sequential(
            nn.Conv2d(sum(self.route_channels), self.out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, self.out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.semantic_head(torch.cat(self._forward_routes(inputs), dim=1))
