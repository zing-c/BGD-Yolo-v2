"""Repaired, checkpoint-safe implementation of the complete ZIP BGD-YOLO method.

This is intentionally different from ``gradcam_fusion.py``.  It follows the
newer ZIP method end to end: every raw candidate above the internal threshold
contributes a crop location, one aggregate classic Grad-CAM is used per image,
Detail features are added at fixed 4x4 locations, and only the selected
Detect classification branch is recomputed.
"""

from __future__ import annotations

import contextlib
import os
import weakref
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detail_model import DETAIL_ARCH_VERSION, Detail_Net_attn, Detail_Net_attn_block
from global_local_fusion import Candidate, crop_original_patches, crop_tensor_patches, xywh_to_xyxy
from gradcam_fusion import _detach_output_tree, _features_before_detect, load_detail_encoder_pretrained
from ultralytics import YOLO, yolo
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, ops, yaml_load
from ultralytics.yolo.utils.metrics import box_iou
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors


ZIP_BGD_METHOD = "zip_bgd_complete_d12_v1"
ZIP_BGD_RESIDUAL_METHOD = "zip_bgd_local_residual_d12_v2"
ZIP_BGD_PRE_DETECT_METHOD = "zip_bgd_pre_detect_p5_d12_v3"
ZIP_BGD_SEMANTIC_RESIDUAL_METHOD = "zip_bgd_pretrained_semantic_local_residual_d12_v4"
ZIP_BGD_SEMANTIC_PYRAMID_METHOD = "zip_bgd_pretrained_semantic_pyramid_residual_d12_v5"
ZIP_BGD_SEMANTIC_SCORE_METHOD = "zip_bgd_pretrained_semantic_score_residual_d12_v6"
ZIP_BGD_SEMANTIC_SCORE_AUX_METHOD = "zip_bgd_pretrained_semantic_score_aux_d12_v7"
ZIP_BGD_ONE_SIDED_RESCUE_METHOD = "zip_bgd_one_sided_quality_rescue_d12_v8"
ZIP_BGD_GROUPED_RESCUE_METHOD = "zip_bgd_dense_distilled_grouped_rescue_d12_v9"
ZIP_BGD_BOUNDARY_RESCUE_METHOD = "zip_bgd_boundary_calibrated_grouped_rescue_d12_v10"
ZIP_BGD_HARD_RESCUE_METHOD = "zip_bgd_deployment_hard_negative_rescue_d12_v11"
ZIP_BGD_RANK_RESCUE_METHOD = "zip_bgd_global_bce_hard_pairwise_rescue_d12_v12"


_ZIP_BGD_TEACHERS = weakref.WeakKeyDictionary()


def register_zip_bgd_teacher(model: nn.Module, teacher: nn.Module | None) -> None:
    if teacher is None:
        _ZIP_BGD_TEACHERS.pop(model, None)
    else:
        _ZIP_BGD_TEACHERS[model] = teacher


def get_zip_bgd_teacher(model: nn.Module) -> nn.Module | None:
    return _ZIP_BGD_TEACHERS.get(model)


class ZipPretrainedSemanticDetail(Detail_Net_attn):
    """Convert the pretrained classifier evidence into three spatial maps.

    The historical ZIP feature encoder discarded ``fc0/fc2`` and randomly
    initialized all three 1x1 output projections.  Here each route contributes
    a parameter-free RMS energy map, while the pretrained classifier logit
    supplies its signed semantic direction.  Thus background-like crops can
    provide negative evidence instead of an unrelated random feature.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        routes = self._forward_routes(inputs)
        flattened = torch.cat([route.flatten(1) for route in routes], dim=1)
        semantic_logit = self.fc2(self.relu(self.fc0(flattened)))
        semantic_direction = torch.tanh(semantic_logit).view(-1, 1, 1, 1)
        maps = []
        for route in routes:
            energy = route.float().square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
            energy = energy / energy.mean((2, 3), keepdim=True).clamp_min(1e-6)
            maps.append(energy.to(dtype=route.dtype) * semantic_direction.to(dtype=route.dtype))
        return torch.cat(maps, dim=1)


class ZipCandidateScoreFusion(nn.Module):
    """Tiny Detail-conditioned residual on candidate classification logits.

    The final layer starts at zero, so a fresh module is exactly equivalent to
    the pretrained detector.  Every input feature contains Detail evidence;
    the adapter therefore cannot become an unrelated YOLO-only recalibrator.
    Box-regression logits and anchors outside the candidate set are untouched.
    """

    def __init__(self, hidden_features: int = 8, max_delta: float = 2.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.adapter = nn.Sequential(
            nn.Linear(5, int(hidden_features)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_features), 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(
        self,
        base_scores: torch.Tensor,
        detail_probabilities: torch.Tensor,
        pyramid_levels: torch.Tensor,
    ) -> torch.Tensor:
        base = base_scores.float().clamp(1e-5, 1 - 1e-5)
        detail = detail_probabilities.float().reshape(-1).clamp(1e-5, 1 - 1e-5)
        evidence = (torch.logit(detail).clamp(-6, 6) / 6).to(dtype=base.dtype)
        centered_base = 2 * base - 1
        level = F.one_hot(pyramid_levels.long(), num_classes=3).to(dtype=base.dtype)
        features = torch.cat((
            evidence[:, None],
            (evidence * centered_base)[:, None],
            evidence[:, None] * level,
        ), dim=1)
        return self.max_delta * torch.tanh(self.adapter(features).squeeze(1)).to(
            dtype=base_scores.dtype
        )


class ZipOneSidedRescueFusion(nn.Module):
    """Predict candidate box quality and only promote sub-threshold scores."""

    def __init__(
        self,
        hidden_features: int = 8,
        max_delta: float = 3.0,
        rescue_scale_init: float = -4.0,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.quality_head = nn.Sequential(
            nn.Linear(9, int(hidden_features)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_features), 1),
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)
        # A tiny positive start keeps gradients alive without materially
        # changing the pretrained detector before candidate-quality learning.
        self.rescue_scale_raw = nn.Parameter(torch.tensor(float(rescue_scale_init)))

    def forward(
        self,
        base_scores: torch.Tensor,
        detail_probabilities: torch.Tensor,
        pyramid_levels: torch.Tensor,
        geometry: torch.Tensor,
        deployment_conf: float = 0.5,
        quality_threshold: float = 0.5,
        bidirectional_alpha: float = 0.0,
        suppression_alpha: float = 0.0,
        suppression_threshold: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = base_scores.float().clamp(1e-5, 1 - 1e-5)
        detail = detail_probabilities.float().reshape(-1).clamp(1e-5, 1 - 1e-5)
        evidence = torch.logit(detail).clamp(-6, 6) / 6
        level = F.one_hot(pyramid_levels.long(), num_classes=3).to(dtype=base.dtype)
        features = torch.cat((
            evidence[:, None],
            (2 * base - 1)[:, None],
            geometry.float(),
            level,
        ), dim=1)
        quality_logits = self.quality_head(features).squeeze(1)
        rescue_delta = (
            self.max_delta
            * torch.sigmoid(self.rescue_scale_raw)
            * torch.sigmoid(quality_logits)
        )
        rescue_delta = torch.where(
            base < float(deployment_conf), rescue_delta, torch.zeros_like(rescue_delta)
        )
        return rescue_delta.to(dtype=base_scores.dtype), quality_logits


class ZipBoundaryCalibratedRescueFusion(ZipOneSidedRescueFusion):
    """Turn candidate-quality ordering into an explicit deployment-boundary decision."""

    def __init__(self, hidden_features: int = 8, logit_gain: float = 8.0, margin: float = 0.25):
        super().__init__(hidden_features=hidden_features, max_delta=3.0, rescue_scale_init=0.0)
        self.logit_gain = float(logit_gain)
        self.margin = float(margin)
        # Rescue and false-positive filtering are deliberately separate
        # decisions.  Sharing one scalar forced the previous versions to
        # trade recall against precision.  This head adds only 89 parameters.
        self.filter_head = nn.Sequential(
            nn.Linear(9, int(hidden_features)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_features), 1),
        )
        nn.init.zeros_(self.filter_head[-1].weight)
        nn.init.zeros_(self.filter_head[-1].bias)
        self.local_filter_head = nn.Sequential(
            nn.Linear(64, int(hidden_features)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_features), 1),
        )
        nn.init.zeros_(self.local_filter_head[-1].weight)
        nn.init.zeros_(self.local_filter_head[-1].bias)

    @staticmethod
    def candidate_features(
        base_scores: torch.Tensor,
        detail_probabilities: torch.Tensor,
        pyramid_levels: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        base = base_scores.float().clamp(1e-5, 1 - 1e-5)
        detail = detail_probabilities.float().reshape(-1).clamp(1e-5, 1 - 1e-5)
        evidence = torch.logit(detail).clamp(-6, 6) / 6
        level = F.one_hot(pyramid_levels.long(), num_classes=3).to(dtype=base.dtype)
        return torch.cat((
            evidence[:, None], (2 * base - 1)[:, None], geometry.float(), level,
        ), dim=1)

    def candidate_filter_logits(
        self,
        base_scores: torch.Tensor,
        detail_probabilities: torch.Tensor,
        pyramid_levels: torch.Tensor,
        geometry: torch.Tensor,
        local_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.candidate_features(
            base_scores, detail_probabilities, pyramid_levels, geometry
        )
        logits = self.filter_head(features).squeeze(1)
        if local_features is not None:
            logits = logits + self.local_filter_head(local_features.float()).squeeze(1)
        return self.logit_gain * logits

    def forward(
        self,
        base_scores: torch.Tensor,
        detail_probabilities: torch.Tensor,
        pyramid_levels: torch.Tensor,
        geometry: torch.Tensor,
        deployment_conf: float = 0.5,
        quality_threshold: float = 0.5,
        bidirectional_alpha: float = 0.0,
        suppression_alpha: float = 0.0,
        suppression_threshold: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = base_scores.float().clamp(1e-5, 1 - 1e-5)
        features = self.candidate_features(
            base_scores, detail_probabilities, pyramid_levels, geometry
        )
        quality_logits = self.logit_gain * self.quality_head(features).squeeze(1)
        quality = quality_logits.sigmoid()
        if float(bidirectional_alpha) > 0:
            score_delta = float(bidirectional_alpha) * torch.logit(quality.clamp(1e-4, 1 - 1e-4)).clamp(-4, 4)
            return score_delta.to(dtype=base_scores.dtype), quality_logits
        quality_threshold = min(max(float(quality_threshold), 0.0), 1.0 - 1e-5)
        boundary = torch.logit(base.new_tensor(float(deployment_conf)).clamp(1e-5, 1 - 1e-5))
        rescue_strength = ((quality - quality_threshold) / (1 - quality_threshold)).clamp(0, 1)
        desired_logit = boundary + self.margin * rescue_strength
        rescue_delta = (desired_logit - torch.logit(base)).clamp(0, self.max_delta)
        rescue_delta = torch.where(
            (base < float(deployment_conf)) & (quality > quality_threshold),
            rescue_delta,
            torch.zeros_like(rescue_delta),
        )
        if float(suppression_alpha) > 0:
            suppression_threshold = min(max(float(suppression_threshold), 1e-5), 1.0)
            suppression_strength = (
                (suppression_threshold - quality) / suppression_threshold
            ).clamp(0, 1)
            suppression_delta = -float(suppression_alpha) * suppression_strength
            suppress = (
                (base >= float(deployment_conf)) & (quality < suppression_threshold)
            )
            rescue_delta = torch.where(suppress, suppression_delta, rescue_delta)
        return rescue_delta.to(dtype=base_scores.dtype), quality_logits


def select_novel_low_candidates(
    candidates: Sequence[Candidate],
    base_scores: torch.Tensor,
    deployment_conf: float = 0.5,
    overlap_iou: float = 0.5,
    min_conf: float = 0.0,
) -> torch.Tensor:
    """Return boundary hypotheses that are not redundant with a deployed box."""
    if len(candidates) != int(base_scores.numel()):
        raise ValueError("Candidate and score counts must match")
    selected = torch.zeros_like(base_scores, dtype=torch.bool)
    if not candidates:
        return selected
    with torch.no_grad():
        boxes = torch.stack([candidate.box_xyxy for candidate in candidates]).to(
            device=base_scores.device, dtype=torch.float32
        )
        scores = base_scores.detach().float()
        keys = sorted({(int(item.batch_index), int(item.class_index)) for item in candidates})
        for batch_index, class_index in keys:
            members = torch.tensor(
                [
                    index for index, item in enumerate(candidates)
                    if int(item.batch_index) == batch_index and int(item.class_index) == class_index
                ],
                device=base_scores.device,
                dtype=torch.long,
            )
            high = members[scores[members] >= float(deployment_conf)]
            low = members[
                (scores[members] >= float(min_conf))
                & (scores[members] < float(deployment_conf))
            ]
            if not len(low):
                continue
            if len(high):
                redundant = box_iou(boxes[low], boxes[high]).amax(1) >= float(overlap_iou)
                low = low[~redundant]
            selected[low] = True
    return selected


def select_grouped_rescue_candidates(
    candidates: Sequence[Candidate],
    base_scores: torch.Tensor,
    quality_logits: torch.Tensor,
    deployment_conf: float = 0.5,
    overlap_iou: float = 0.5,
    min_conf: float = 0.0,
) -> torch.Tensor:
    """Select at most one quality leader per novel overlapping hypothesis."""
    if len(candidates) != int(quality_logits.numel()):
        raise ValueError("Candidate and quality counts must match")
    selected = torch.zeros_like(base_scores, dtype=torch.bool)
    novel = select_novel_low_candidates(
        candidates, base_scores, deployment_conf=deployment_conf,
        overlap_iou=overlap_iou, min_conf=min_conf,
    )
    if not bool(novel.any()):
        return selected
    with torch.no_grad():
        boxes = torch.stack([candidate.box_xyxy for candidate in candidates]).to(
            device=base_scores.device, dtype=torch.float32
        )
        qualities = quality_logits.detach().float()
        keys = sorted({(int(item.batch_index), int(item.class_index)) for item in candidates})
        for batch_index, class_index in keys:
            members = torch.tensor([
                index for index, item in enumerate(candidates)
                if int(item.batch_index) == batch_index and int(item.class_index) == class_index
            ], device=base_scores.device, dtype=torch.long)
            low = members[novel[members]]
            if not len(low):
                continue

            adjacency = box_iou(boxes[low], boxes[low]) >= float(overlap_iou)
            unseen = set(range(len(low)))
            while unseen:
                seed = unseen.pop()
                component = {seed}
                frontier = [seed]
                while frontier:
                    current = frontier.pop()
                    neighbors = torch.where(adjacency[current])[0].tolist()
                    new_members = [item for item in neighbors if item in unseen]
                    unseen.difference_update(new_members)
                    component.update(new_members)
                    frontier.extend(new_members)
                component_local = torch.tensor(
                    sorted(component), device=base_scores.device, dtype=torch.long
                )
                component_global = low[component_local]
                leader = component_global[qualities[component_global].argmax()]
                selected[leader] = True
    return selected


def load_full_detail_classifier(encoder: Detail_Net_attn, weight: Path) -> Dict[str, object]:
    """Load and verify the complete pretrained Detail classifier."""
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    source = checkpoint
    if isinstance(source, dict):
        for key in ("detail_model", "detail_encoder", "state_dict", "model_state_dict"):
            if isinstance(source.get(key), dict):
                source = source[key]
                break
    if not isinstance(source, dict):
        raise TypeError(f"Detail checkpoint does not contain a state_dict: {weight}")
    target = encoder.state_dict()
    compatible = {
        name: tensor for name, tensor in source.items()
        if name in target and target[name].shape == tensor.shape
    }
    loaded_numel = sum(target[name].numel() for name in compatible)
    total_numel = sum(tensor.numel() for tensor in target.values())
    coverage = loaded_numel / max(total_numel, 1)
    if coverage < 0.99:
        raise ValueError(
            f"Complete Detail classifier mismatch for {weight}: coverage={coverage:.1%}"
        )
    result = encoder.load_state_dict(compatible, strict=False)
    return {
        "architecture": DETAIL_ARCH_VERSION,
        "loaded": sorted(compatible),
        "missing": sorted(result.missing_keys),
        "pretrained_coverage": coverage,
        "semantic_classifier_preserved": True,
    }


class ZipDirectFusion(nn.Module):
    """The ZIP 67->64 direct fusion, generalized to the selected head width."""

    def __init__(self, global_channels: int, local_channels: int = 3):
        super().__init__()
        self.global_channels = int(global_channels)
        self.local_channels = int(local_channels)
        self.conv = nn.Conv2d(self.global_channels + self.local_channels, self.global_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(self.global_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, global_feature: torch.Tensor, local_canvas: torch.Tensor) -> torch.Tensor:
        if global_feature.shape[0] != local_canvas.shape[0] or global_feature.shape[-2:] != local_canvas.shape[-2:]:
            raise ValueError("Global and Detail feature maps must share batch/spatial dimensions")
        return self.relu(self.bn(self.conv(torch.cat((local_canvas, global_feature), dim=1))))


class ZipLocalResidualFusion(nn.Module):
    """Candidate-local Detail residual with an exact YOLO identity start.

    Unlike the ZIP direct 67->64 convolution, this module cannot rewrite the
    full P5 activation.  The learnable residual is masked to Detail locations,
    while ``alpha_raw=0`` makes a freshly assembled model bitwise-equivalent
    to the pretrained YOLO classification branch.
    """

    def __init__(self, global_channels: int, local_channels: int = 3, alpha_max: float = 0.5):
        super().__init__()
        self.global_channels = int(global_channels)
        self.alpha_max = float(alpha_max)
        self.local_proj = nn.Sequential(
            nn.Conv2d(local_channels, self.global_channels, 1, bias=False),
            nn.GroupNorm(1, self.global_channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Conv2d(2 * self.global_channels + 1, self.global_channels, 3, padding=1)
        self.alpha_raw = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.tanh(self.alpha_raw)

    def forward(
        self, global_feature: torch.Tensor, local_canvas: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if global_feature.shape[0] != local_canvas.shape[0] or global_feature.shape[-2:] != local_canvas.shape[-2:]:
            raise ValueError("Global and Detail feature maps must share batch/spatial dimensions")
        projected = self.local_proj(local_canvas)
        gate = torch.sigmoid(self.gate(torch.cat((global_feature, projected, mask), dim=1)))
        return global_feature + self.alpha * gate * projected * mask, gate


def _class_feature_channels(detect_head, level: int) -> int:
    if not 0 <= level < detect_head.nl:
        raise ValueError(f"head_level={level} is outside 0..{detect_head.nl - 1}")
    final = detect_head.cv3[level][-1]
    channels = getattr(final, "in_channels", None)
    if channels is None:
        raise TypeError("Could not infer selected Detect classification feature width")
    return int(channels)


def _detect_input_channels(detect_head, level: int) -> int:
    """Infer the neck feature width entering one Detect pyramid level."""
    if not 0 <= level < detect_head.nl:
        raise ValueError(f"head_level={level} is outside 0..{detect_head.nl - 1}")
    first = detect_head.cv2[level][0]
    convolution = getattr(first, "conv", first)
    channels = getattr(convolution, "in_channels", None)
    if channels is None:
        raise TypeError("Could not infer selected Detect input feature width")
    return int(channels)


def select_all_raw_candidates(decoded: torch.Tensor, conf_threshold: float) -> List[Candidate]:
    """Keep every raw anchor above confidence, exactly without NMS or top-k."""
    if decoded.ndim != 3 or decoded.shape[1] <= 4:
        raise ValueError(f"Expected [B,4+nc,N], got {tuple(decoded.shape)}")
    candidates: List[Candidate] = []
    for batch_index in range(decoded.shape[0]):
        scores, classes = decoded[batch_index, 4:].max(dim=0)
        anchors = torch.nonzero(scores.detach() >= float(conf_threshold), as_tuple=False).flatten()
        if not anchors.numel():
            continue
        boxes = xywh_to_xyxy(decoded[batch_index, :4, anchors].transpose(0, 1)).detach()
        image_candidates = [
            Candidate(
                batch_index=batch_index,
                anchor_index=int(anchor),
                class_index=int(classes[anchor]),
                score=decoded[batch_index, 4 + int(classes[anchor]), int(anchor)],
                box_xyxy=box,
            )
            for anchor, box in zip(anchors.tolist(), boxes)
        ]
        candidates.extend(image_candidates)
    return candidates


def aggregate_class_gradcam(decoded: torch.Tensor, activation: torch.Tensor) -> Tuple[torch.Tensor | None, str]:
    """Classic Grad-CAM from the sum of all decoded class scores (ZIP target)."""
    if not decoded.requires_grad or not activation.requires_grad:
        return None, "no_grad"
    gradient = torch.autograd.grad(
        decoded[:, 4:, :].sum(), activation, retain_graph=True, create_graph=False, allow_unused=True
    )[0]
    if gradient is None:
        return None, "unused_grad"
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True)).detach()
    if not torch.isfinite(cam).all():
        return None, "non_finite"
    ranges = cam.flatten(1).amax(1) - cam.flatten(1).amin(1)
    minimum = cam.flatten(1).amin(1).view(-1, 1, 1, 1)
    maximum = cam.flatten(1).amax(1).view(-1, 1, 1, 1)
    normalized = (cam - minimum) / (maximum - minimum).clamp_min(1e-12)
    # Classic Grad-CAM can be identically zero when the selected class head's
    # signed channel combination is negative everywhere.  The ZIP source then
    # silently chose the first equal-valued window.  Preserve all detections,
    # but mark this case so the locator can use the geometrically sound box
    # center fallback instead.
    return normalized, "gradcam" if bool((ranges > 1e-12).any()) else "flat_cam"


def aggregate_pyramid_class_gradcam(
    decoded: torch.Tensor, activations: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor | None, str]:
    """Average normalized classic Grad-CAM evidence from every Detect level."""
    if not activations or not decoded.requires_grad or any(not item.requires_grad for item in activations):
        return None, "no_grad"
    gradients = torch.autograd.grad(
        decoded[:, 4:, :].sum(), tuple(activations), retain_graph=True,
        create_graph=False, allow_unused=True,
    )
    output_size = activations[0].shape[-2:]
    normalized_cams = []
    non_flat = False
    for activation, gradient in zip(activations, gradients):
        if gradient is None:
            continue
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True)).detach()
        if not torch.isfinite(cam).all():
            continue
        minimum = cam.flatten(1).amin(1).view(-1, 1, 1, 1)
        maximum = cam.flatten(1).amax(1).view(-1, 1, 1, 1)
        ranges = maximum - minimum
        non_flat = non_flat or bool((ranges > 1e-12).any())
        normalized = (cam - minimum) / ranges.clamp_min(1e-12)
        normalized_cams.append(
            F.interpolate(normalized, size=output_size, mode="bilinear", align_corners=False)
        )
    if not normalized_cams:
        return None, "unused_grad"
    return torch.stack(normalized_cams).mean(0), "gradcam" if non_flat else "flat_cam"


def candidate_pyramid_level(candidate: Candidate, raw: Sequence[torch.Tensor]) -> int:
    """Map a decoded global anchor index back to its source Detect level."""
    anchor = int(candidate.anchor_index)
    offset = 0
    for level, prediction in enumerate(raw):
        count = int(prediction.shape[-2] * prediction.shape[-1])
        if offset <= anchor < offset + count:
            return level
        offset += count
    # Ground-truth-only candidates have no decoded anchor. Keep the historical
    # selected level as a safe fallback; pyramid experiments disable them.
    return len(raw) - 1


def candidate_pyramid_address(
    candidate: Candidate, raw: Sequence[torch.Tensor]
) -> Tuple[int, int]:
    """Return Detect level and flattened local anchor for a decoded candidate."""
    anchor = int(candidate.anchor_index)
    offset = 0
    for level, prediction in enumerate(raw):
        count = int(prediction.shape[-2] * prediction.shape[-1])
        if offset <= anchor < offset + count:
            return level, anchor - offset
        offset += count
    raise ValueError("Score fusion requires candidates backed by decoded anchors")


def _window_size_for_image(model, batch_index: int, default: int = 64) -> int:
    """Map the original 64px ZIP crop window into YOLO-input coordinates."""
    batch = getattr(model, "batch", None)
    if not isinstance(batch, dict) or "ratio_pad" not in batch:
        return int(default)
    try:
        gain = batch["ratio_pad"][batch_index][0]
        gain_y, gain_x = float(gain[0]), float(gain[1])
        return max(1, int(round(default * (gain_x + gain_y) / 2)))
    except (IndexError, TypeError, ValueError):
        return int(default)


def locate_zip_centers(
    cam: torch.Tensor,
    candidates: Sequence[Candidate],
    image_size: Tuple[int, int],
    model=None,
    crop_size: int = 64,
    deduplicate: bool = True,
) -> Tuple[List[Candidate], List[torch.Tensor]]:
    """Find maximum CAM-window centers inside boxes and remove duplicates.

    The ZIP source mixed x/y while pasting and used an exclusive bound that
    produced 3x3 edge inserts.  This implementation keeps coordinates as
    (x,y) throughout and always creates an exact 4x4 Detail placement later.
    """
    image_h, image_w = image_size
    full_cam = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False).squeeze(1)
    kept_candidates: List[Candidate] = []
    kept_centers: List[torch.Tensor] = []
    seen = [set() for _ in range(full_cam.shape[0])]
    pooled_by_image: Dict[int, Tuple[torch.Tensor, int]] = {}

    for candidate in candidates:
        b = candidate.batch_index
        window = min(_window_size_for_image(model, b, crop_size), image_h, image_w)
        if b not in pooled_by_image or pooled_by_image[b][1] != window:
            pooled = F.avg_pool2d(full_cam[b:b + 1, None], kernel_size=window, stride=1).squeeze(0).squeeze(0)
            pooled_by_image[b] = pooled, window
        pooled, window = pooled_by_image[b]

        box = candidate.box_xyxy.float()
        x1 = max(0, min(image_w - 1, int(torch.floor(box[0]).item())))
        y1 = max(0, min(image_h - 1, int(torch.floor(box[1]).item())))
        x2 = max(x1 + 1, min(image_w, int(torch.ceil(box[2]).item())))
        y2 = max(y1 + 1, min(image_h, int(torch.ceil(box[3]).item())))
        cam_is_flat = float(full_cam[b].max() - full_cam[b].min()) <= 1e-12
        if candidate.source == "gt":
            region = pooled.new_empty((0,))
        elif not cam_is_flat and x2 - x1 >= window and y2 - y1 >= window:
            right = x2 - window + 1
            bottom = y2 - window + 1
            region = pooled[y1:bottom, x1:right]
        else:
            region = pooled.new_empty((0,))
        if region.numel():
            flat = int(region.argmax())
            top = y1 + flat // region.shape[1]
            left = x1 + flat % region.shape[1]
            center_x, center_y = left + window // 2, top + window // 2
        else:
            center_x = int(round((x1 + x2 - 1) / 2))
            center_y = int(round((y1 + y2 - 1) / 2))
        key = (center_x, center_y)
        if deduplicate and key in seen[b]:
            continue
        seen[b].add(key)
        kept_candidates.append(candidate)
        kept_centers.append(cam.new_tensor((center_x, center_y)))
    return kept_candidates, kept_centers


def paste_fixed_detail(
    local_features: torch.Tensor,
    candidates: Sequence[Candidate],
    centers: Sequence[torch.Tensor],
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    """Add (not average) exact 4x4 Detail maps at their CAM centers."""
    if len(local_features) != len(candidates) or len(candidates) != len(centers):
        raise ValueError("Detail features, candidates and centers must have matching lengths")
    image_h, image_w = image_size
    feature_h, feature_w = feature_size
    if local_features.shape[-2:] != (4, 4):
        raise ValueError(f"ZIP Detail output must be 4x4, got {tuple(local_features.shape[-2:])}")
    if feature_h < 4 or feature_w < 4:
        raise ValueError("Selected detection feature must be at least 4x4")

    per_image: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    zero = local_features.new_zeros((local_features.shape[1], feature_h, feature_w))
    for feature, candidate, center in zip(local_features, candidates, centers):
        feature_x = int(torch.floor(center[0] * feature_w / image_w).item())
        feature_y = int(torch.floor(center[1] * feature_h / image_h).item())
        left = max(0, min(feature_w - 4, feature_x - 2))
        top = max(0, min(feature_h - 4, feature_y - 2))
        per_image[candidate.batch_index].append(
            F.pad(feature, (left, feature_w - left - 4, top, feature_h - top - 4))
        )
    return torch.stack([
        torch.stack(items).sum(0) if items else zero for items in per_image
    ])


def paste_average_fixed_detail(
    local_features: torch.Tensor,
    candidates: Sequence[Candidate],
    centers: Sequence[torch.Tensor],
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paste exact 4x4 Detail maps and average candidate overlaps."""
    if len(local_features) != len(candidates) or len(candidates) != len(centers):
        raise ValueError("Detail features, candidates and centers must have matching lengths")
    image_h, image_w = image_size
    feature_h, feature_w = feature_size
    if local_features.shape[-2:] != (4, 4):
        raise ValueError(f"ZIP Detail output must be 4x4, got {tuple(local_features.shape[-2:])}")
    contributions: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    counts: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    zero = local_features.new_zeros((local_features.shape[1], feature_h, feature_w))
    zero_count = local_features.new_zeros((1, feature_h, feature_w))
    for feature, candidate, center in zip(local_features, candidates, centers):
        feature_x = int(torch.floor(center[0] * feature_w / image_w).item())
        feature_y = int(torch.floor(center[1] * feature_h / image_h).item())
        left = max(0, min(feature_w - 4, feature_x - 2))
        top = max(0, min(feature_h - 4, feature_y - 2))
        padding = (left, feature_w - left - 4, top, feature_h - top - 4)
        contributions[candidate.batch_index].append(F.pad(feature, padding))
        counts[candidate.batch_index].append(F.pad(feature.new_ones((1, 4, 4)), padding))
    canvases, masks = [], []
    for items, item_counts in zip(contributions, counts):
        summed = torch.stack(items).sum(0) if items else zero
        count = torch.stack(item_counts).sum(0) if item_counts else zero_count
        canvases.append(summed / count.clamp_min(1))
        masks.append((count > 0).to(local_features.dtype))
    return torch.stack(canvases), torch.stack(masks)


def _box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if not boxes.numel():
        return boxes.new_empty((0,))
    intersection = (
        torch.minimum(box[2:], boxes[:, 2:]) - torch.maximum(box[:2], boxes[:, :2])
    ).clamp_min(0).prod(1)
    box_area = (box[2:] - box[:2]).clamp_min(0).prod()
    boxes_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    return intersection / (box_area + boxes_area - intersection).clamp_min(1e-9)


def _unmatched_ground_truth_candidates(model, images, predicted: Sequence[Candidate], match_iou: float):
    """Add GT-centered training crops only for targets missed by raw candidates."""
    batch = getattr(model, "batch", None)
    required = ("bboxes", "batch_idx", "cls")
    if not isinstance(batch, dict) or not all(key in batch for key in required):
        return []
    image_h, image_w = images.shape[-2:]
    normalized = batch["bboxes"].to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].to(device=images.device, dtype=torch.long)
    classes = batch["cls"].reshape(-1).to(device=images.device, dtype=torch.long)
    xywh = normalized * normalized.new_tensor((image_w, image_h, image_w, image_h))
    boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    boxes[:, 0::2].clamp_(0, image_w)
    boxes[:, 1::2].clamp_(0, image_h)
    predictions_by_image = [
        [candidate.box_xyxy for candidate in predicted if candidate.batch_index == index]
        for index in range(images.shape[0])
    ]
    missing = []
    for box, batch_index, class_index in zip(boxes, batch_indices, classes):
        predicted_boxes = predictions_by_image[int(batch_index)]
        if predicted_boxes and bool((_box_iou_one_to_many(box, torch.stack(predicted_boxes)) >= match_iou).any()):
            continue
        missing.append(Candidate(
            batch_index=int(batch_index), anchor_index=-1, class_index=int(class_index),
            score=images.new_zeros(()), box_xyxy=box.detach(), source="gt",
        ))
    return missing


def _detail_presence_auxiliary_loss(
    model,
    images: torch.Tensor,
    candidates: Sequence[Candidate],
    centers: Sequence[torch.Tensor],
    probabilities: torch.Tensor,
    route: str = "gradcam",
) -> Tuple[torch.Tensor | None, Dict[str, int]]:
    """Supervise whether the exact Detail screenshot contains a GT object.

    Candidate-box IoU is deliberately not used as the sole label: a poorly
    localized candidate can still yield a useful screenshot containing the
    object.  Positive crops contain a GT centre or at least half of a GT box;
    crops with at most 5% GT coverage are reliable negatives, and ambiguous
    crops are ignored.
    """
    batch = getattr(model, "batch", None)
    required = ("bboxes", "batch_idx")
    if not isinstance(batch, dict) or not all(key in batch for key in required):
        return None, {"aux_positive": 0, "aux_negative": 0, "aux_ignored": len(candidates)}
    image_h, image_w = images.shape[-2:]
    normalized = batch["bboxes"].to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].to(device=images.device, dtype=torch.long)
    xywh = normalized * normalized.new_tensor((image_w, image_h, image_w, image_h))
    gt_boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)

    labels, reliable = [], []
    for candidate, center in zip(candidates, centers):
        boxes = gt_boxes[batch_indices == int(candidate.batch_index)]
        if not len(boxes):
            labels.append(False)
            reliable.append(True)
            continue
        center = center.to(device=boxes.device, dtype=boxes.dtype)
        inside = (
            (center[0] >= boxes[:, 0]) & (center[0] <= boxes[:, 2])
            & (center[1] >= boxes[:, 1]) & (center[1] <= boxes[:, 3])
        ).any()
        window = float(_window_size_for_image(
            model, int(candidate.batch_index), int(model.zip_bgd_config.get("crop_size", 64))
        ))
        half = window / 2
        crop = torch.stack((center[0] - half, center[1] - half, center[0] + half, center[1] + half))
        intersection = (
            torch.minimum(crop[2:], boxes[:, 2:]) - torch.maximum(crop[:2], boxes[:, :2])
        ).clamp_min(0).prod(1)
        gt_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6).prod(1)
        coverage = (intersection / gt_area).amax()
        positive = bool(inside) or float(coverage) >= 0.5
        labels.append(positive)
        reliable.append(positive or float(coverage) <= 0.05)

    labels_tensor = probabilities.new_tensor(labels, dtype=torch.float32)
    reliable_tensor = torch.tensor(reliable, device=probabilities.device, dtype=torch.bool)
    positive = reliable_tensor & (labels_tensor > 0.5)
    negative = reliable_tensor & ~positive
    weights = probabilities.new_zeros(len(candidates), dtype=torch.float32)
    if bool(positive.any()):
        weights[positive] = (0.5 if bool(negative.any()) else 1.0) / positive.sum()
    if bool(negative.any()):
        weights[negative] = (0.5 if bool(positive.any()) else 1.0) / negative.sum()
    if not bool(reliable_tensor.any()):
        loss = probabilities.sum() * 0
    else:
        loss = F.binary_cross_entropy(
            probabilities.float().reshape(-1).clamp(1e-5, 1 - 1e-5),
            labels_tensor,
            weight=weights,
            reduction="sum",
        )
    if os.environ.get("BGD_ZIP_ROUTE_DIAGNOSTICS") == "1":
        records = getattr(model, "zip_detail_route_records", {})
        entry = records.setdefault(route, {"probabilities": [], "labels": [], "reliable": []})
        entry["probabilities"].append(probabilities.detach().float().reshape(-1).cpu())
        entry["labels"].append(labels_tensor.detach().cpu())
        entry["reliable"].append(reliable_tensor.detach().cpu())
        model.zip_detail_route_records = records
    return loss, {
        "aux_positive": int(positive.sum()),
        "aux_negative": int(negative.sum()),
        "aux_ignored": int((~reliable_tensor).sum()),
    }


def summarize_zip_detail_routes(model) -> Dict[str, Dict[str, float]]:
    """Summarize screenshot-content discrimination collected during validation."""
    summary = {}
    for route, entry in getattr(model, "zip_detail_route_records", {}).items():
        probability = torch.cat(entry["probabilities"])
        label = torch.cat(entry["labels"]).bool()
        reliable = torch.cat(entry["reliable"]).bool()
        probability, label = probability[reliable], label[reliable]
        positive, negative = probability[label], probability[~label]
        if len(positive) and len(negative):
            comparisons = positive[:, None] - negative[None, :]
            auc = float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())
        else:
            auc = float("nan")
        if len(positive):
            order = torch.argsort(probability, descending=True)
            ordered = label[order].float()
            precision = ordered.cumsum(0) / torch.arange(1, len(ordered) + 1)
            average_precision = float((precision * ordered).sum() / len(positive))
        else:
            average_precision = float("nan")
        summary[route] = {
            "count": int(len(probability)),
            "positive": int(len(positive)),
            "negative": int(len(negative)),
            "auc": auc,
            "average_precision": average_precision,
            "positive_mean": float(positive.mean()) if len(positive) else float("nan"),
            "negative_mean": float(negative.mean()) if len(negative) else float("nan"),
            "accuracy_at_0.5": float(((probability >= 0.5) == label).float().mean()) if len(label) else float("nan"),
        }
    return summary


def _candidate_quality_auxiliary_loss(
    model,
    images: torch.Tensor,
    candidates: Sequence[Candidate],
    quality_logits: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
    negative_cost: float = 1.0,
    ranking_mask: torch.Tensor | None = None,
    ranking_weight: float = 0.0,
    base_detections: Sequence[torch.Tensor] | None = None,
) -> Tuple[torch.Tensor | None, Dict[str, int]]:
    """Balanced candidate IoU supervision for the one-sided rescue decision."""
    batch = getattr(model, "batch", None)
    required = ("bboxes", "batch_idx")
    if not isinstance(batch, dict) or not all(key in batch for key in required):
        return None, {"quality_positive": 0, "quality_negative": 0, "quality_ignored": len(candidates)}
    image_h, image_w = images.shape[-2:]
    normalized = batch["bboxes"].to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].to(device=images.device, dtype=torch.long)
    xywh = normalized * normalized.new_tensor((image_w, image_h, image_w, image_h))
    gt_boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    qualities = []
    matched_gt_indices = []
    for candidate in candidates:
        image_gt_indices = torch.where(batch_indices == int(candidate.batch_index))[0]
        boxes = gt_boxes[image_gt_indices]
        if len(boxes):
            candidate_ious = _box_iou_one_to_many(candidate.box_xyxy, boxes)
            best_iou, best_local_index = candidate_ious.max(0)
            qualities.append(best_iou)
            matched_gt_indices.append(image_gt_indices[best_local_index])
        else:
            qualities.append(images.new_zeros(()))
            matched_gt_indices.append(torch.tensor(-1, device=images.device, dtype=torch.long))
    quality = torch.stack(qualities).to(dtype=quality_logits.dtype)
    matched_gt_indices = torch.stack(matched_gt_indices)
    all_positive = quality >= 0.5
    all_negative = quality < 0.3
    reliable = all_positive | all_negative
    supervised = reliable
    if sample_mask is not None:
        supervised = supervised & sample_mask.to(device=reliable.device, dtype=torch.bool)
    positive = all_positive & supervised
    negative = all_negative & supervised
    target = positive.to(dtype=quality_logits.dtype)
    weights = quality_logits.new_zeros(len(candidates))
    negative_cost = max(float(negative_cost), 0.0)
    if bool(positive.any()) and bool(negative.any()):
        normalizer = 1.0 + negative_cost
        weights[positive] = (1.0 / normalizer) / positive.sum()
        weights[negative] = (negative_cost / normalizer) / negative.sum()
    elif bool(positive.any()):
        weights[positive] = 1.0 / positive.sum()
    elif bool(negative.any()):
        weights[negative] = 1.0 / negative.sum()
    loss = F.binary_cross_entropy_with_logits(
        quality_logits, target, weight=weights, reduction="sum"
    ) if bool(supervised.any()) else quality_logits.sum() * 0
    rank_positive = quality_logits.new_zeros(len(candidates), dtype=torch.bool)
    rank_negative = quality_logits.new_zeros(len(candidates), dtype=torch.bool)
    if ranking_mask is not None and float(ranking_weight) > 0:
        ranking_mask = ranking_mask.to(device=reliable.device, dtype=torch.bool)
        rank_positive = all_positive & ranking_mask
        rank_negative = all_negative & ranking_mask
        if bool(rank_positive.any()) and bool(rank_negative.any()):
            pairwise_margin = (
                quality_logits[rank_positive, None] - quality_logits[None, rank_negative]
            )
            loss = loss + float(ranking_weight) * F.softplus(-pairwise_margin).mean()
    novel_positive_count = 0
    duplicate_positive_count = 0
    if os.environ.get("BGD_ZIP_ROUTE_DIAGNOSTICS") == "1":
        model.zip_bgd_last_quality_targets = quality.detach()
        records = getattr(model, "zip_detail_route_records", {})
        entry = records.setdefault(
            "candidate_quality", {"probabilities": [], "labels": [], "reliable": []}
        )
        entry["probabilities"].append(quality_logits.detach().float().sigmoid().cpu())
        entry["labels"].append(target.detach().bool().cpu())
        entry["reliable"].append(supervised.detach().cpu())
        hard_entry = records.setdefault(
            "candidate_quality_hard", {"probabilities": [], "labels": [], "reliable": []}
        )
        hard_entry["probabilities"].append(quality_logits.detach().float().sigmoid().cpu())
        hard_entry["labels"].append(all_positive.detach().bool().cpu())
        hard_entry["reliable"].append((rank_positive | rank_negative).detach().cpu())

        # IoU-positive candidates are not necessarily useful rescues: many
        # simply rediscover a GT that the original detector already recalled.
        # Compare against the detector's actual post-NMS outputs so this label
        # measures potential *new recall*, rather than box quality alone.
        gt_covered = torch.zeros(len(gt_boxes), device=images.device, dtype=torch.bool)
        if base_detections is not None:
            for batch_index, detections in enumerate(base_detections):
                image_gt_indices = torch.where(batch_indices == int(batch_index))[0]
                if not len(image_gt_indices) or not len(detections):
                    continue
                detection_boxes = detections[:, :4].to(
                    device=images.device, dtype=gt_boxes.dtype
                )
                covered = box_iou(detection_boxes, gt_boxes[image_gt_indices]).amax(0) >= 0.5
                gt_covered[image_gt_indices] = covered
        valid_match = matched_gt_indices >= 0
        matched_covered = torch.zeros_like(valid_match)
        matched_covered[valid_match] = gt_covered[matched_gt_indices[valid_match]]
        novel_positive = all_positive & ~matched_covered
        duplicate_positive = all_positive & matched_covered
        novel_reliable = novel_positive | duplicate_positive | all_negative
        novel_positive_count = int(novel_positive.sum())
        duplicate_positive_count = int(duplicate_positive.sum())
        model.zip_bgd_last_novel_targets = novel_positive.detach()
        model.zip_bgd_last_duplicate_targets = duplicate_positive.detach()

        novel_entry = records.setdefault(
            "candidate_novel_recall", {"probabilities": [], "labels": [], "reliable": []}
        )
        novel_entry["probabilities"].append(
            quality_logits.detach().float().sigmoid().cpu()
        )
        novel_entry["labels"].append(novel_positive.detach().cpu())
        novel_entry["reliable"].append(novel_reliable.detach().cpu())
        novel_hard_entry = records.setdefault(
            "candidate_novel_recall_hard",
            {"probabilities": [], "labels": [], "reliable": []},
        )
        hard_mask = (
            ranking_mask.to(device=images.device, dtype=torch.bool)
            if ranking_mask is not None
            else torch.ones_like(novel_reliable)
        )
        novel_hard_entry["probabilities"].append(
            quality_logits.detach().float().sigmoid().cpu()
        )
        novel_hard_entry["labels"].append(novel_positive.detach().cpu())
        novel_hard_entry["reliable"].append((novel_reliable & hard_mask).detach().cpu())
        model.zip_detail_route_records = records
    return loss, {
        "quality_positive": int(positive.sum()),
        "quality_negative": int(negative.sum()),
        "quality_ignored": int((~supervised).sum()),
        "quality_rank_positive": int(rank_positive.sum()),
        "quality_rank_negative": int(rank_negative.sum()),
        "quality_novel_positive": novel_positive_count,
        "quality_duplicate_positive": duplicate_positive_count,
    }


def _candidate_filter_auxiliary_loss(
    model,
    images: torch.Tensor,
    candidates: Sequence[Candidate],
    filter_logits: torch.Tensor,
    base_detections: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Train the filtering head on actual post-NMS one-to-one TP/FP labels.

    The rescue head answers whether a low-confidence anchor could cover an
    object.  That is not the same as deciding whether a final detection is a
    unique true positive: duplicate and background detections must be labelled
    with the validator's greedy one-to-one rule.
    """
    batch = getattr(model, "batch", None)
    required = ("bboxes", "batch_idx")
    if not isinstance(batch, dict) or not all(key in batch for key in required):
        return filter_logits.sum() * 0, {
            "filter_positive": 0, "filter_negative": 0, "filter_ignored": len(candidates)
        }
    image_h, image_w = images.shape[-2:]
    normalized = batch["bboxes"].to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].to(device=images.device, dtype=torch.long)
    xywh = normalized * normalized.new_tensor((image_w, image_h, image_w, image_h))
    gt_boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)

    selected_indices, selected_labels = [], []
    for image_index, detections in enumerate(base_detections):
        member_indices = [
            index for index, candidate in enumerate(candidates)
            if int(candidate.batch_index) == image_index
        ]
        if not len(detections) or not member_indices:
            continue
        candidate_boxes = torch.stack([
            candidates[index].box_xyxy for index in member_indices
        ]).to(device=images.device, dtype=gt_boxes.dtype)
        image_gt = gt_boxes[batch_indices == image_index]
        unmatched_gt = torch.ones(len(image_gt), device=images.device, dtype=torch.bool)
        used_candidates = torch.zeros(len(member_indices), device=images.device, dtype=torch.bool)
        for detection in detections:  # NMS output is already confidence sorted.
            candidate_iou = box_iou(
                detection[None, :4].to(dtype=candidate_boxes.dtype), candidate_boxes
            )[0].masked_fill(used_candidates, -1)
            local_candidate = int(candidate_iou.argmax())
            if float(candidate_iou[local_candidate]) < 0.95:
                continue
            used_candidates[local_candidate] = True
            is_true_positive = False
            if bool(unmatched_gt.any()):
                gt_iou = box_iou(
                    detection[None, :4].to(dtype=image_gt.dtype), image_gt
                )[0].masked_fill(~unmatched_gt, -1)
                gt_index = int(gt_iou.argmax())
                if float(gt_iou[gt_index]) >= 0.5:
                    unmatched_gt[gt_index] = False
                    is_true_positive = True
            selected_indices.append(member_indices[local_candidate])
            selected_labels.append(is_true_positive)

    if not selected_indices:
        return filter_logits.sum() * 0, {
            "filter_positive": 0, "filter_negative": 0, "filter_ignored": len(candidates)
        }
    indices = torch.tensor(selected_indices, device=filter_logits.device, dtype=torch.long)
    labels = torch.tensor(selected_labels, device=filter_logits.device, dtype=filter_logits.dtype)
    positive = labels >= 0.5
    negative = ~positive
    weights = torch.zeros_like(labels)
    if bool(positive.any()):
        weights[positive] = (0.5 if bool(negative.any()) else 1.0) / positive.sum()
    if bool(negative.any()):
        weights[negative] = (0.5 if bool(positive.any()) else 1.0) / negative.sum()
    selected_logits = filter_logits[indices]
    loss = F.binary_cross_entropy_with_logits(
        selected_logits, labels, weight=weights, reduction="sum"
    )
    if os.environ.get("BGD_ZIP_ROUTE_DIAGNOSTICS") == "1":
        records = getattr(model, "zip_detail_route_records", {})
        entry = records.setdefault(
            "final_detection_filter", {"probabilities": [], "labels": [], "reliable": []}
        )
        entry["probabilities"].append(selected_logits.detach().float().sigmoid().cpu())
        entry["labels"].append(labels.detach().bool().cpu())
        entry["reliable"].append(torch.ones_like(labels, dtype=torch.bool).cpu())
        model.zip_detail_route_records = records
    return loss, {
        "filter_positive": int(positive.sum()),
        "filter_negative": int(negative.sum()),
        "filter_ignored": int(len(candidates) - len(indices)),
    }


def _post_nms_recall_diagnostics(
    model,
    images: torch.Tensor,
    base_decoded: torch.Tensor,
    fused_decoded: torch.Tensor,
    conf: float = 0.5,
    nms_iou: float = 0.5,
    match_iou: float = 0.5,
) -> Dict[str, int]:
    """Count GT coverage gained or lost by fusion after the real final NMS."""
    batch = getattr(model, "batch", None)
    required = ("bboxes", "batch_idx")
    if not isinstance(batch, dict) or not all(key in batch for key in required):
        return {}
    image_h, image_w = images.shape[-2:]
    normalized = batch["bboxes"].to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].to(device=images.device, dtype=torch.long)
    xywh = normalized * normalized.new_tensor((image_w, image_h, image_w, image_h))
    gt_boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    with torch.no_grad():
        base_detections = ops.non_max_suppression(
            base_decoded.detach().clone(), float(conf), float(nms_iou),
            agnostic=True, max_det=300,
        )
        fused_detections = ops.non_max_suppression(
            fused_decoded.detach().clone(), float(conf), float(nms_iou),
            agnostic=True, max_det=300,
        )
        base_covered_total = 0
        fused_covered_total = 0
        gained_total = 0
        lost_total = 0
        base_detection_total = 0
        fused_detection_total = 0
        for batch_index, (base, fused) in enumerate(zip(base_detections, fused_detections)):
            boxes = gt_boxes[batch_indices == int(batch_index)]
            base_detection_total += int(len(base))
            fused_detection_total += int(len(fused))
            if not len(boxes):
                continue
            base_covered = torch.zeros(len(boxes), device=images.device, dtype=torch.bool)
            fused_covered = torch.zeros_like(base_covered)
            if len(base):
                base_covered = box_iou(
                    base[:, :4].to(device=images.device, dtype=boxes.dtype), boxes
                ).amax(0) >= float(match_iou)
            if len(fused):
                fused_covered = box_iou(
                    fused[:, :4].to(device=images.device, dtype=boxes.dtype), boxes
                ).amax(0) >= float(match_iou)
            base_covered_total += int(base_covered.sum())
            fused_covered_total += int(fused_covered.sum())
            gained_total += int((~base_covered & fused_covered).sum())
            lost_total += int((base_covered & ~fused_covered).sum())
    return {
        "post_nms_base_detection_count": base_detection_total,
        "post_nms_fused_detection_count": fused_detection_total,
        "post_nms_base_gt_covered": base_covered_total,
        "post_nms_fused_gt_covered": fused_covered_total,
        "post_nms_gt_gained": gained_total,
        "post_nms_gt_lost": lost_total,
    }


def _crop_patches(model, images, candidates, centers):
    config = model.zip_bgd_config
    if config["crop_source"] == "tensor":
        return crop_tensor_patches(images, candidates, centers, crop_size=config["crop_size"])[0]
    batch = getattr(model, "batch", None)
    if not isinstance(batch, dict) or "ori_img" not in batch or "ratio_pad" not in batch:
        raise RuntimeError(
            "crop_source='original' requires ori_img/ratio_pad metadata from this repository's dataloader; "
            "use crop_source='tensor' only for bare-tensor diagnostics."
        )
    return crop_original_patches(
        images, batch["ori_img"], batch["ratio_pad"], candidates, centers, crop_size=config["crop_size"]
    )[0]


def _decode_raw(detect_head, raw: List[torch.Tensor]) -> torch.Tensor:
    shape = raw[0].shape
    detect_head.anchors, detect_head.strides = (
        value.transpose(0, 1) for value in make_anchors(raw, detect_head.stride, 0.5)
    )
    detect_head.shape = shape
    joined = torch.cat([value.view(shape[0], detect_head.no, -1) for value in raw], dim=2)
    box, cls = joined.split((detect_head.reg_max * 4, detect_head.nc), dim=1)
    decoded_box = dist2bbox(
        detect_head.dfl(box), detect_head.anchors.unsqueeze(0), xywh=True, dim=1
    ) * detect_head.strides
    return torch.cat((decoded_box, cls.sigmoid()), dim=1)


def _detection_consensus(
    teacher_decoded: torch.Tensor,
    student_decoded: torch.Tensor,
    match_iou: float,
    student_gain: float,
    teacher_unmatched_gain: float,
    student_unmatched_gain: float,
    internal_conf: float = 0.01,
    teacher_rescue_min: float = -1.0,
    teacher_rescue_max: float = -1.0,
) -> torch.Tensor:
    """Late two-route fusion after independent NMS and spatial matching."""
    teacher_detections = ops.non_max_suppression(
        teacher_decoded, internal_conf, 0.7, agnostic=True, max_det=300
    )
    student_detections = ops.non_max_suppression(
        student_decoded, internal_conf, 0.7, agnostic=True, max_det=300
    )
    per_image = []
    for teacher, student in zip(teacher_detections, student_detections):
        pairs = []
        if len(teacher) and len(student):
            similarities = box_iou(teacher[:, :4], student[:, :4])
            available_teacher = torch.ones(len(teacher), dtype=torch.bool, device=teacher.device)
            available_student = torch.ones(len(student), dtype=torch.bool, device=student.device)
            while True:
                eligible = similarities.masked_fill(
                    ~(available_teacher[:, None] & available_student[None, :]), -1
                )
                flat = int(eligible.argmax())
                quality = float(eligible.flatten()[flat])
                if quality < float(match_iou):
                    break
                teacher_index = flat // len(student)
                student_index = flat % len(student)
                pairs.append((teacher_index, student_index))
                available_teacher[teacher_index] = False
                available_student[student_index] = False
        else:
            available_teacher = torch.ones(len(teacher), dtype=torch.bool, device=teacher.device)
            available_student = torch.ones(len(student), dtype=torch.bool, device=student.device)

        candidates = []
        for teacher_index, student_index in pairs:
            teacher_item, student_item = teacher[teacher_index], student[student_index]
            adjusted_student_score = student_item[4] * float(student_gain)
            use_teacher = teacher_item[4] >= adjusted_student_score
            box = teacher_item[:4] if bool(use_teacher) else student_item[:4]
            score = torch.maximum(teacher_item[4], adjusted_student_score)
            candidates.append(torch.cat((box, score[None], teacher_item[5:6])))
        for item in teacher[available_teacher]:
            unmatched_score = item[4] * float(teacher_unmatched_gain)
            if float(teacher_rescue_min) >= 0 and teacher_rescue_min <= float(item[4]) <= teacher_rescue_max:
                unmatched_score = item[4]
            candidates.append(torch.cat((item[:4], unmatched_score[None], item[5:6])))
        for item in student[available_student]:
            candidates.append(torch.cat((item[:4], (item[4] * float(student_unmatched_gain))[None], item[5:6])))
        per_image.append(torch.stack(candidates) if candidates else teacher.new_zeros((0, 6)))

    max_candidates = max((len(items) for items in per_image), default=0)
    output = teacher_decoded.new_zeros((teacher_decoded.shape[0], teacher_decoded.shape[1], max_candidates))
    for batch_index, items in enumerate(per_image):
        if not len(items):
            continue
        xyxy = items[:, :4]
        xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1)
        output[batch_index, :4, :len(items)] = xywh.transpose(0, 1)
        output[batch_index, 4, :len(items)] = items[:, 4]
    return output


def _zip_bgd_impl(model, images, profile=False, visualize=False):
    config = model.zip_bgd_config
    if model.training:
        model.zip_bgd_aux_loss = None
        model.zip_bgd_quality_aux_loss = None
        model.zip_bgd_filter_aux_loss = None
    features, detect_head = _features_before_detect(model, images, profile, visualize)
    level = int(config["head_level"])
    fusion_position = config.get("fusion_position", "class_branch")
    quality_rescue = config.get("method") in (
        ZIP_BGD_ONE_SIDED_RESCUE_METHOD, ZIP_BGD_GROUPED_RESCUE_METHOD,
        ZIP_BGD_BOUNDARY_RESCUE_METHOD, ZIP_BGD_HARD_RESCUE_METHOD,
        ZIP_BGD_RANK_RESCUE_METHOD,
    )
    score_fusion = config.get("method") in (
        ZIP_BGD_SEMANTIC_SCORE_METHOD, ZIP_BGD_SEMANTIC_SCORE_AUX_METHOD,
        ZIP_BGD_ONE_SIDED_RESCUE_METHOD, ZIP_BGD_GROUPED_RESCUE_METHOD,
        ZIP_BGD_BOUNDARY_RESCUE_METHOD, ZIP_BGD_HARD_RESCUE_METHOD,
        ZIP_BGD_RANK_RESCUE_METHOD,
    )
    pyramid_fusion = config.get("fusion_scope", "selected") == "pyramid"
    capture_levels = list(range(detect_head.nl)) if pyramid_fusion else [level]
    captured = {capture_level: [] for capture_level in capture_levels}
    handles = []
    if fusion_position == "class_branch":
        for capture_level in capture_levels:
            handles.append(detect_head.cv3[capture_level][1].register_forward_hook(
                lambda _module, _inputs, output, capture_level=capture_level:
                captured[capture_level].append(output)
            ))
    head_was_training = detect_head.training
    try:
        detect_head.eval()
        first_output = detect_head([feature.clone() for feature in features])
    finally:
        for handle in handles:
            handle.remove()
        detect_head.train(head_was_training)
    if fusion_position == "class_branch" and any(not captured[item] for item in capture_levels):
        raise RuntimeError("Selected Detect classification activation was not captured")
    activations = (
        [captured[item][-1] for item in capture_levels]
        if fusion_position == "class_branch" else []
    )
    activation = features[level] if fusion_position == "pre_detect" else captured[level][-1]
    decoded, raw = first_output if isinstance(first_output, tuple) else (first_output, None)
    if raw is None:
        raise RuntimeError("ZIP BGD requires standard non-export Detect outputs")
    if model.training:
        # Dense distillation constrains the detector before Detail correction,
        # leaving the sparse rescue branch free to improve selected anchors.
        model.zip_bgd_base_raw = raw

    candidates = select_all_raw_candidates(decoded, config["candidate_conf"])
    raw_candidate_count = len(candidates)
    gt_candidate_count = 0
    if model.training and config.get("use_gt_train", False):
        gt_candidates = _unmatched_ground_truth_candidates(
            model, images, candidates, float(config.get("gt_match_iou", 0.5))
        )
        candidates.extend(gt_candidates)
        gt_candidate_count = len(gt_candidates)
    if pyramid_fusion:
        cam, cam_reason = aggregate_pyramid_class_gradcam(decoded, activations)
    else:
        cam, cam_reason = aggregate_class_gradcam(decoded, activation)
    if not candidates or cam is None:
        output = raw if model.training else (decoded, raw)
        if model.training:
            model.zip_bgd_student_raw = raw
        model.zip_bgd_runtime = {
            "candidate_count": raw_candidate_count, "unique_crop_count": 0, "cam_success": 0,
            f"cam_{cam_reason}": 1,
        }
        return output if model.training else _detach_output_tree(output)

    candidates, centers = locate_zip_centers(
        cam, candidates, images.shape[-2:], model=model, crop_size=int(config["crop_size"]),
        deduplicate=not score_fusion,
    )
    patches = _crop_patches(model, images, candidates, centers)
    detail_features = model.detail_model(patches.to(dtype=next(model.detail_model.parameters()).dtype))
    aux_stats = {}
    route_diagnostics = os.environ.get("BGD_ZIP_ROUTE_DIAGNOSTICS") == "1"
    detail_auxiliary = model.training and float(config.get("detail_aux_weight", 0.0)) > 0
    if score_fusion and (detail_auxiliary or route_diagnostics):
        detail_loss, aux_stats = _detail_presence_auxiliary_loss(
            model, images, candidates, centers, detail_features, route="gradcam"
        )
        if detail_auxiliary:
            model.zip_bgd_aux_loss = detail_loss
    box_probabilities = None
    if score_fusion and route_diagnostics:
        box_centers = [
            ((candidate.box_xyxy[:2] + candidate.box_xyxy[2:]) / 2).detach()
            for candidate in candidates
        ]
        box_patches = _crop_patches(model, images, candidates, box_centers)
        with torch.no_grad():
            box_probabilities = model.detail_model(
                box_patches.to(dtype=next(model.detail_model.parameters()).dtype)
            )
        _detail_presence_auxiliary_loss(
            model, images, candidates, box_centers, box_probabilities, route="box_center"
        )
    gate = None
    gate_maps, mask_maps = [], []
    score_delta = None
    quality_logits = None
    filter_logits = None
    rescue_eligible = None
    if score_fusion:
        addresses = [candidate_pyramid_address(candidate, raw) for candidate in candidates]
        source_levels = torch.tensor(
            [address[0] for address in addresses], device=detail_features.device, dtype=torch.long
        )
        base_scores = torch.stack([candidate.score for candidate in candidates])
        if quality_rescue:
            boxes = torch.stack([candidate.box_xyxy for candidate in candidates]).to(
                device=detail_features.device, dtype=torch.float32
            )
            center_tensor = torch.stack(centers).to(device=boxes.device, dtype=boxes.dtype)
            box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            box_sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1)
            image_h, image_w = images.shape[-2:]
            geometry = torch.cat((
                ((center_tensor - box_centers) / box_sizes).clamp(-1, 1),
                torch.log((box_sizes / boxes.new_tensor((image_w, image_h))).clamp_min(1e-4)),
            ), dim=1)
            score_delta, quality_logits = model.zip_rescue_fusion(
                base_scores, detail_features, source_levels, geometry,
                deployment_conf=float(config.get("deployment_conf", 0.5)),
                quality_threshold=float(config.get("rescue_quality_threshold", 0.5)),
                bidirectional_alpha=float(config.get("quality_bidirectional_alpha", 0.0)),
                suppression_alpha=float(config.get("quality_suppression_alpha", 0.0)),
                suppression_threshold=float(config.get("quality_suppression_threshold", 0.2)),
            )
            if hasattr(model.zip_rescue_fusion, "candidate_filter_logits"):
                local_features = None
                if fusion_position == "class_branch" and pyramid_fusion:
                    local_vectors = []
                    for candidate, (source_level, local_anchor) in zip(candidates, addresses):
                        level_activation = captured[source_level][-1]
                        feature_width = int(level_activation.shape[-1])
                        feature_row, feature_col = divmod(int(local_anchor), feature_width)
                        local_vectors.append(
                            level_activation[
                                int(candidate.batch_index), :, feature_row, feature_col
                            ]
                        )
                    if local_vectors:
                        local_features = torch.stack(local_vectors)
                filter_logits = model.zip_rescue_fusion.candidate_filter_logits(
                    base_scores, detail_features, source_levels, geometry, local_features
                )
            novel_low = select_novel_low_candidates(
                candidates,
                base_scores,
                deployment_conf=float(config.get("deployment_conf", 0.5)),
                overlap_iou=float(config.get("rescue_group_iou", 0.5)),
                min_conf=float(config.get("rescue_min_conf", 0.0)),
            )
            if config.get("rescue_grouping", False):
                rescue_eligible = select_grouped_rescue_candidates(
                    candidates,
                    base_scores,
                    quality_logits,
                    deployment_conf=float(config.get("deployment_conf", 0.5)),
                    overlap_iou=float(config.get("rescue_group_iou", 0.5)),
                    min_conf=float(config.get("rescue_min_conf", 0.0)),
                )
                if (
                    float(config.get("quality_bidirectional_alpha", 0.0)) > 0
                    or float(config.get("quality_suppression_alpha", 0.0)) > 0
                ):
                    # Suppression is safe for every candidate. Positive changes
                    # below the deployment boundary remain one-per-group.
                    allowed = (
                        (score_delta <= 0)
                        | (base_scores >= float(config.get("deployment_conf", 0.5)))
                        | rescue_eligible
                    )
                    score_delta = torch.where(allowed, score_delta, torch.zeros_like(score_delta))
                else:
                    score_delta = torch.where(
                        rescue_eligible, score_delta, torch.zeros_like(score_delta)
                    )
            filter_threshold = float(config.get("filter_threshold", 0.0))
            if filter_logits is not None and filter_threshold > 0:
                filter_probability = filter_logits.sigmoid()
                filter_threshold = min(max(filter_threshold, 1e-5), 1 - 1e-5)
                filter_strength = (
                    (filter_threshold - filter_probability) / filter_threshold
                ).clamp(0, 1)
                boundary = torch.logit(base_scores.new_tensor(
                    float(config.get("deployment_conf", 0.5))
                ).clamp(1e-5, 1 - 1e-5))
                desired_logit = boundary - float(
                    config.get("filter_margin", 0.25)
                ) * filter_strength
                filter_delta = (
                    desired_logit - torch.logit(base_scores.float().clamp(1e-5, 1 - 1e-5))
                ).clamp(-3.0, 0.0)
                filter_mask = (
                    (base_scores >= float(config.get("deployment_conf", 0.5)))
                    & (filter_probability < filter_threshold)
                )
                score_delta = score_delta + torch.where(
                    filter_mask, filter_delta.to(dtype=score_delta.dtype),
                    torch.zeros_like(score_delta),
                )
            need_filter_supervision = (
                filter_logits is not None
                and (
                    (model.training and float(config.get("filter_aux_weight", 0.0)) > 0)
                    or route_diagnostics
                )
            )
            base_detections = None
            if need_filter_supervision or route_diagnostics:
                base_detections = ops.non_max_suppression(
                    decoded.detach().clone(),
                    float(config.get("deployment_conf", 0.5)),
                    float(config.get("deployment_nms_iou", 0.5)),
                    agnostic=True,
                    max_det=300,
                )
            if need_filter_supervision:
                filter_loss, filter_stats = _candidate_filter_auxiliary_loss(
                    model, images, candidates, filter_logits, base_detections
                )
                if model.training and float(config.get("filter_aux_weight", 0.0)) > 0:
                    model.zip_bgd_filter_aux_loss = filter_loss
                aux_stats.update(filter_stats)
            if (
                model.training and float(config.get("quality_aux_weight", 0.0)) > 0
            ) or route_diagnostics:
                quality_loss, quality_stats = _candidate_quality_auxiliary_loss(
                    model,
                    images,
                    candidates,
                    quality_logits,
                    sample_mask=novel_low if config.get("quality_hard_only", False) else None,
                    negative_cost=float(config.get("quality_negative_cost", 1.0)),
                    ranking_mask=(
                        novel_low if float(config.get("quality_ranking_weight", 0.0)) > 0 else None
                    ),
                    ranking_weight=float(config.get("quality_ranking_weight", 0.0)),
                    base_detections=base_detections,
                )
                if model.training and float(config.get("quality_aux_weight", 0.0)) > 0:
                    model.zip_bgd_quality_aux_loss = quality_loss
                aux_stats.update(quality_stats)
                if route_diagnostics and hasattr(model, "zip_bgd_last_quality_targets"):
                    adjusted = torch.sigmoid(
                        torch.logit(base_scores.detach().float().clamp(1e-5, 1 - 1e-5))
                        + score_delta.detach().float()
                    )
                    crossing = (
                        (base_scores.detach() < float(config.get("deployment_conf", 0.5)))
                        & (adjusted >= float(config.get("deployment_conf", 0.5)))
                    )
                    target_quality = model.zip_bgd_last_quality_targets.to(crossing.device)
                    novel_target = getattr(
                        model, "zip_bgd_last_novel_targets", torch.zeros_like(crossing)
                    ).to(crossing.device)
                    duplicate_target = getattr(
                        model, "zip_bgd_last_duplicate_targets", torch.zeros_like(crossing)
                    ).to(crossing.device)
                    reliable_quality = (target_quality >= 0.5) | (target_quality < 0.3)
                    hard_reliable = novel_low & reliable_quality
                    records = getattr(model, "zip_detail_route_records", {})
                    hard_routes = {
                        "gradcam_detail_iou_hard": detail_features.detach().float().reshape(-1),
                        "box_detail_iou_hard": box_probabilities.detach().float().reshape(-1),
                        "yolo_score_iou_hard": base_scores.detach().float().reshape(-1),
                    }
                    for route, probabilities in hard_routes.items():
                        entry = records.setdefault(
                            route, {"probabilities": [], "labels": [], "reliable": []}
                        )
                        entry["probabilities"].append(probabilities.cpu())
                        entry["labels"].append((target_quality >= 0.5).detach().cpu())
                        entry["reliable"].append(hard_reliable.detach().cpu())
                    model.zip_detail_route_records = records
                    aux_stats.update({
                        "crossing_iou_positive": int((crossing & (target_quality >= 0.5)).sum()),
                        "crossing_novel_positive": int((crossing & novel_target).sum()),
                        "crossing_duplicate_positive": int((crossing & duplicate_target).sum()),
                        "crossing_iou_negative": int((crossing & (target_quality < 0.3)).sum()),
                        "crossing_iou_ambiguous": int(
                            (crossing & (target_quality >= 0.3) & (target_quality < 0.5)).sum()
                        ),
                    })
        else:
            score_delta = model.zip_score_fusion(base_scores, detail_features, source_levels)
        fused_raw = list(raw)
        for fusion_level in range(detect_head.nl):
            member_indices = [
                index for index, address in enumerate(addresses) if address[0] == fusion_level
            ]
            if not member_indices:
                continue
            fused_level = raw[fusion_level].clone()
            flat_logits = fused_level.view(fused_level.shape[0], fused_level.shape[1], -1)
            for index in member_indices:
                candidate = candidates[index]
                local_anchor = addresses[index][1]
                class_channel = detect_head.reg_max * 4 + int(candidate.class_index)
                flat_logits[candidate.batch_index, class_channel, local_anchor] = (
                    flat_logits[candidate.batch_index, class_channel, local_anchor]
                    + score_delta[index]
                )
            fused_raw[fusion_level] = fused_level
        final_decoded = _decode_raw(detect_head, fused_raw)
    elif pyramid_fusion:
        fused_raw = list(raw)
        source_levels = [candidate_pyramid_level(candidate, raw) for candidate in candidates]
        for fusion_level in range(detect_head.nl):
            member_indices = [
                index for index, source_level in enumerate(source_levels)
                if source_level == fusion_level
            ]
            if not member_indices:
                continue
            member_tensor = torch.tensor(member_indices, device=detail_features.device, dtype=torch.long)
            member_candidates = [candidates[index] for index in member_indices]
            member_centers = [centers[index] for index in member_indices]
            level_activation = captured[fusion_level][-1]
            canvas, mask = paste_average_fixed_detail(
                detail_features[member_tensor], member_candidates, member_centers,
                images.shape[-2:], level_activation.shape[-2:], images.shape[0],
            )
            fused_feature, level_gate = model.zip_residual_fusion(
                level_activation,
                canvas.to(dtype=level_activation.dtype),
                mask.to(dtype=level_activation.dtype),
            )
            fused_logits = detect_head.cv3[fusion_level][2](fused_feature)
            fused_level = raw[fusion_level].clone()
            fused_level[:, detect_head.reg_max * 4:, :, :] = fused_logits
            fused_raw[fusion_level] = fused_level
            gate_maps.append(level_gate)
            mask_maps.append(mask)
        final_decoded = _decode_raw(detect_head, fused_raw)
    elif config["method"] in (ZIP_BGD_RESIDUAL_METHOD, ZIP_BGD_SEMANTIC_RESIDUAL_METHOD):
        canvas, mask = paste_average_fixed_detail(
            detail_features, candidates, centers, images.shape[-2:], activation.shape[-2:], images.shape[0]
        )
        fused_feature, gate = model.zip_residual_fusion(
            activation, canvas.to(dtype=activation.dtype), mask.to(dtype=activation.dtype)
        )
    else:
        canvas = paste_fixed_detail(
            detail_features, candidates, centers, images.shape[-2:], activation.shape[-2:], images.shape[0]
        )
        fusion_module = (
            model.zip_pre_detect_fusion if fusion_position == "pre_detect" else model.zip_direct_fusion
        )
        fused_feature = fusion_module(activation, canvas.to(dtype=activation.dtype))

    if score_fusion or pyramid_fusion:
        pass
    elif fusion_position == "pre_detect":
        fused_inputs = [feature.clone() for feature in features]
        fused_inputs[level] = fused_feature
        try:
            detect_head.train(head_was_training)
            second_output = detect_head(fused_inputs)
        finally:
            detect_head.train(head_was_training)
        if model.training:
            fused_raw = second_output
            final_decoded = None
        else:
            final_decoded, fused_raw = second_output
    else:
        fused_class_logits = detect_head.cv3[level][2](fused_feature)
        fused_raw = list(raw)
        # Avoid mutating the first-pass output retained for Grad-CAM autograd.
        fused_level = raw[level].clone()
        fused_level[:, detect_head.reg_max * 4:, :, :] = fused_class_logits
        fused_raw[level] = fused_level
        final_decoded = _decode_raw(detect_head, fused_raw)
    teacher_blend = config.get("teacher_blend", "none")
    if not model.training and teacher_blend != "none":
        teacher = getattr(model, "zip_teacher", None)
        if teacher is None:
            raise RuntimeError("teacher_blend requires a frozen zip_teacher model")
        with torch.no_grad():
            teacher_output = teacher(images)
            teacher_decoded = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output
        ratio = float(config.get("teacher_ratio", 0.5))
        if teacher_blend == "detection_consensus":
            final_decoded = _detection_consensus(
                teacher_decoded,
                final_decoded,
                match_iou=float(config.get("consensus_match_iou", 0.5)),
                student_gain=float(config.get("student_gain", 1.0)),
                teacher_unmatched_gain=float(config.get("teacher_unmatched_gain", 1.0)),
                student_unmatched_gain=float(config.get("student_unmatched_gain", 0.0)),
                internal_conf=float(config.get("consensus_internal_conf", 0.01)),
                teacher_rescue_min=float(config.get("teacher_rescue_min", -1.0)),
                teacher_rescue_max=float(config.get("teacher_rescue_max", -1.0)),
            )
        elif teacher_blend == "score_average":
            final_decoded[:, 4:] = ratio * teacher_decoded[:, 4:] + (1.0 - ratio) * final_decoded[:, 4:]
        elif teacher_blend == "anchor_winner":
            teacher_score = teacher_decoded[:, 4:].amax(1, keepdim=True)
            student_gain = float(config.get("student_gain", 1.0))
            adjusted_student_scores = final_decoded[:, 4:] * student_gain
            teacher_floor = float(config.get("student_teacher_floor", 0.0))
            if teacher_floor > 0:
                adjusted_student_scores = torch.where(
                    teacher_score >= teacher_floor,
                    adjusted_student_scores,
                    torch.zeros_like(adjusted_student_scores),
                )
            fused_score = adjusted_student_scores.amax(1, keepdim=True)
            use_teacher = teacher_score >= fused_score
            final_decoded[:, :4] = torch.where(use_teacher, teacher_decoded[:, :4], final_decoded[:, :4])
            final_decoded[:, 4:] = torch.maximum(teacher_decoded[:, 4:], adjusted_student_scores)
        else:
            raise ValueError(f"Unsupported teacher_blend mode: {teacher_blend}")
    if route_diagnostics and not model.training:
        aux_stats.update(_post_nms_recall_diagnostics(
            model,
            images,
            decoded,
            final_decoded,
            conf=float(config.get("deployment_conf", 0.5)),
            nms_iou=float(config.get("deployment_nms_iou", 0.5)),
            match_iou=0.5,
        ))

    model.zip_bgd_runtime = {
        "candidate_count": raw_candidate_count,
        "gt_candidate_count": gt_candidate_count,
        "unique_crop_count": len(candidates),
        "cam_success": int(cam_reason == "gradcam"),
        f"cam_{cam_reason}": 1,
        "head_level": level,
        "pre_detect_fusion": int(fusion_position == "pre_detect"),
        "pyramid_fusion": int(pyramid_fusion),
    }
    if gate_maps:
        model.zip_bgd_runtime.update({
            "residual_alpha": float(model.zip_residual_fusion.alpha.detach()),
            "gate_mean": float(torch.stack([item.detach().mean() for item in gate_maps]).mean()),
            "mask_coverage": float(torch.stack([item.detach().mean() for item in mask_maps]).mean()),
        })
    elif gate is not None:
        model.zip_bgd_runtime.update({
            "residual_alpha": float(model.zip_residual_fusion.alpha.detach()),
            "gate_mean": float(gate.detach().mean()),
            "mask_coverage": float(mask.detach().mean()),
        })
    if score_delta is not None:
        model.zip_bgd_runtime.update({
            "score_delta_abs_mean": float(score_delta.detach().abs().mean()),
            "score_delta_max": float(score_delta.detach().abs().max()),
            "detail_probability_mean": float(detail_features.detach().float().mean()),
        })
    if quality_logits is not None:
        deployment_conf = float(config.get("deployment_conf", 0.5))
        adjusted_scores = torch.sigmoid(
            torch.logit(base_scores.detach().float().clamp(1e-5, 1 - 1e-5))
            + score_delta.detach().float()
        )
        model.zip_bgd_runtime.update({
            "quality_probability_mean": float(quality_logits.detach().float().sigmoid().mean()),
            "rescue_scale": float(torch.sigmoid(model.zip_rescue_fusion.rescue_scale_raw.detach())),
            "rescue_eligible_count": int(
                rescue_eligible.sum() if rescue_eligible is not None
                else (base_scores.detach() < deployment_conf).sum()
            ),
            "rescue_crossing_count": int(
                ((base_scores.detach() < deployment_conf) & (adjusted_scores >= deployment_conf)).sum()
            ),
            "suppression_falling_count": int(
                ((base_scores.detach() >= deployment_conf) & (adjusted_scores < deployment_conf)).sum()
            ),
        })
    model.zip_bgd_runtime.update(aux_stats)
    totals = getattr(model, "zip_bgd_totals", {"forward_count": 0, "candidate_count": 0, "unique_crop_count": 0})
    totals["forward_count"] = totals.get("forward_count", 0) + 1
    totals["candidate_count"] = totals.get("candidate_count", 0) + model.zip_bgd_runtime["candidate_count"]
    totals["unique_crop_count"] = totals.get("unique_crop_count", 0) + len(candidates)
    for key in (
        "rescue_eligible_count", "rescue_crossing_count", "suppression_falling_count",
        "crossing_iou_positive", "crossing_iou_negative", "crossing_iou_ambiguous",
        "crossing_novel_positive", "crossing_duplicate_positive",
        "quality_novel_positive", "quality_duplicate_positive",
        "filter_positive", "filter_negative", "filter_ignored",
        "post_nms_base_detection_count", "post_nms_fused_detection_count",
        "post_nms_base_gt_covered", "post_nms_fused_gt_covered",
        "post_nms_gt_gained", "post_nms_gt_lost",
    ):
        if key in model.zip_bgd_runtime:
            totals[key] = totals.get(key, 0) + model.zip_bgd_runtime[key]
    model.zip_bgd_totals = totals
    output = fused_raw if model.training else (final_decoded, fused_raw)
    if model.training:
        model.zip_bgd_student_raw = fused_raw
    return output if model.training else _detach_output_tree(output)


def zip_bgd_predict_once(model, images, profile=False, visualize=False):
    """Pickle-stable DetectionModel forward replacement with CAM gradients enabled."""
    needs_grad = not torch.is_grad_enabled()
    inference_context = torch.inference_mode(False) if torch.is_inference_mode_enabled() else contextlib.nullcontext()
    grad_context = torch.enable_grad() if needs_grad else contextlib.nullcontext()
    with inference_context, grad_context:
        if torch.is_inference_mode_enabled() or needs_grad:
            images = images.detach().clone()
        return _zip_bgd_impl(model, images, profile, visualize)


class ZipBGDYOLO(YOLO):
    """Whole-model fine-tuning wrapper for the repaired complete ZIP method."""

    def __init__(
        self,
        model,
        detail_weight=None,
        candidate_conf: float = 0.10,
        head_select: str = "low",
        crop_size: int = 64,
        crop_source: str = "original",
        fusion_mode: str | None = None,
        fusion_position: str | None = "auto",
        detail_output: str | None = "auto",
        fusion_scope: str | None = "auto",
        detail_aux_weight: float | None = None,
        quality_aux_weight: float = 1.0,
        deployment_conf: float = 0.5,
        deployment_nms_iou: float = 0.5,
        rescue_min_conf: float | None = None,
        rescue_grouping: bool | None = None,
        rescue_group_iou: float = 0.5,
        preserve_teacher_strength: float = 0.0,
        dense_distill_strength: float = 0.0,
        quality_lr_multiplier: float = 1.0,
        quality_logit_gain: float = 8.0,
        rescue_margin: float = 0.25,
        rescue_quality_threshold: float | None = None,
        quality_hard_only: bool = False,
        quality_negative_cost: float = 1.0,
        quality_ranking_weight: float | None = None,
        quality_bidirectional_alpha: float | None = None,
        quality_suppression_alpha: float | None = None,
        quality_suppression_threshold: float | None = None,
        filter_aux_weight: float = 0.0,
        filter_threshold: float | None = None,
        filter_margin: float = 0.25,
        use_gt_train: bool = True,
        gt_match_iou: float = 0.50,
        residual_alpha_max: float = 0.50,
        teacher_weight=None,
        teacher_blend: str = "none",
        teacher_ratio: float = 0.50,
        student_gain: float = 1.0,
        student_teacher_floor: float = 0.0,
        consensus_match_iou: float = 0.50,
        teacher_unmatched_gain: float = 1.0,
        student_unmatched_gain: float = 0.0,
        consensus_internal_conf: float = 0.01,
        teacher_rescue_min: float = -1.0,
        teacher_rescue_max: float = -1.0,
        task: str = "detect",
    ):
        super().__init__(model=model, task=task)
        head_levels = {"high": 0, "mid": 1, "low": 2}
        if head_select not in head_levels:
            raise ValueError("head_select must be high, mid or low")
        if crop_source not in ("original", "tensor"):
            raise ValueError("crop_source must be original or tensor")
        detect_head = self.model.model[-1]
        level = head_levels[head_select]
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        loaded_config = getattr(self.model, "zip_bgd_config", {})
        loaded_method = loaded_config.get("method")
        if rescue_min_conf is None:
            rescue_min_conf = float(loaded_config.get("rescue_min_conf", candidate_conf))
        if not 0 <= rescue_min_conf < deployment_conf:
            raise ValueError("rescue_min_conf must be in [0, deployment_conf)")
        if not 0 < deployment_nms_iou < 1:
            raise ValueError("deployment_nms_iou must be in (0, 1)")
        if rescue_grouping is None:
            rescue_grouping = bool(loaded_config.get("rescue_grouping", False))
        if rescue_quality_threshold is None:
            rescue_quality_threshold = float(loaded_config.get("rescue_quality_threshold", 0.5))
        if not 0 <= rescue_quality_threshold < 1:
            raise ValueError("rescue_quality_threshold must be in [0, 1)")
        if quality_ranking_weight is None:
            quality_ranking_weight = float(loaded_config.get("quality_ranking_weight", 0.0))
        if quality_ranking_weight < 0:
            raise ValueError("quality_ranking_weight must be non-negative")
        if quality_bidirectional_alpha is None:
            quality_bidirectional_alpha = float(
                loaded_config.get("quality_bidirectional_alpha", 0.0)
            )
        if quality_bidirectional_alpha < 0:
            raise ValueError("quality_bidirectional_alpha must be non-negative")
        if quality_suppression_alpha is None:
            quality_suppression_alpha = float(
                loaded_config.get("quality_suppression_alpha", 0.0)
            )
        if quality_suppression_threshold is None:
            quality_suppression_threshold = float(
                loaded_config.get("quality_suppression_threshold", 0.2)
            )
        if quality_suppression_alpha < 0:
            raise ValueError("quality_suppression_alpha must be non-negative")
        if not 0 < quality_suppression_threshold <= 1:
            raise ValueError("quality_suppression_threshold must be in (0, 1]")
        if filter_threshold is None:
            filter_threshold = float(loaded_config.get("filter_threshold", 0.0))
        if filter_aux_weight < 0:
            raise ValueError("filter_aux_weight must be non-negative")
        if not 0 <= filter_threshold < 1:
            raise ValueError("filter_threshold must be in [0, 1)")
        if filter_margin <= 0:
            raise ValueError("filter_margin must be positive")
        if detail_aux_weight is None:
            detail_aux_weight = float(
                loaded_config.get(
                    "detail_aux_weight",
                    0.0 if loaded_method == ZIP_BGD_SEMANTIC_SCORE_METHOD else 0.5,
                )
            )
        if detail_aux_weight < 0:
            raise ValueError("detail_aux_weight must be non-negative")
        if detail_output is None or detail_output == "auto":
            detail_output = (
                "probability"
                if loaded_method in (
                    ZIP_BGD_SEMANTIC_SCORE_METHOD, ZIP_BGD_SEMANTIC_SCORE_AUX_METHOD,
                    ZIP_BGD_ONE_SIDED_RESCUE_METHOD, ZIP_BGD_GROUPED_RESCUE_METHOD,
                    ZIP_BGD_BOUNDARY_RESCUE_METHOD, ZIP_BGD_HARD_RESCUE_METHOD,
                    ZIP_BGD_RANK_RESCUE_METHOD,
                )
                else
                "semantic_maps"
                if loaded_method in (ZIP_BGD_SEMANTIC_RESIDUAL_METHOD, ZIP_BGD_SEMANTIC_PYRAMID_METHOD)
                else "feature"
            )
        if detail_output not in ("feature", "semantic_maps", "probability"):
            raise ValueError("detail_output must be auto, feature, semantic_maps or probability")
        if fusion_position is None or fusion_position == "auto":
            fusion_position = "pre_detect" if loaded_method == ZIP_BGD_PRE_DETECT_METHOD else "class_branch"
        if fusion_position not in ("class_branch", "pre_detect"):
            raise ValueError("fusion_position must be auto, class_branch or pre_detect")
        if fusion_mode is None or fusion_mode == "auto":
            fusion_mode = (
                "score"
                if loaded_method in (ZIP_BGD_SEMANTIC_SCORE_METHOD, ZIP_BGD_SEMANTIC_SCORE_AUX_METHOD)
                else "rescue"
                if loaded_method == ZIP_BGD_ONE_SIDED_RESCUE_METHOD
                else "grouped_rescue"
                if loaded_method == ZIP_BGD_GROUPED_RESCUE_METHOD
                else "boundary_rescue"
                if loaded_method == ZIP_BGD_BOUNDARY_RESCUE_METHOD
                else "hard_rescue"
                if loaded_method == ZIP_BGD_HARD_RESCUE_METHOD
                else "rank_rescue"
                if loaded_method == ZIP_BGD_RANK_RESCUE_METHOD
                else
                "residual"
                if loaded_method in (
                    ZIP_BGD_RESIDUAL_METHOD,
                    ZIP_BGD_SEMANTIC_RESIDUAL_METHOD,
                    ZIP_BGD_SEMANTIC_PYRAMID_METHOD,
                )
                else "direct"
            )
        rescue_modes = (
            "rescue", "grouped_rescue", "boundary_rescue", "hard_rescue", "rank_rescue"
        )
        if fusion_mode not in ("direct", "residual", "score", *rescue_modes):
            raise ValueError(
                "fusion_mode must be auto, direct, residual, score, rescue, grouped_rescue, "
                "boundary_rescue, hard_rescue or rank_rescue"
            )
        if fusion_position == "pre_detect" and fusion_mode != "direct":
            raise ValueError("The pre-Detect ablation preserves the ZIP direct-convolution fusion")
        if detail_output == "semantic_maps" and (
            fusion_position != "class_branch" or fusion_mode != "residual"
        ):
            raise ValueError(
                "Pretrained semantic Detail maps require class_branch residual fusion"
            )
        if detail_output == "probability" and (
            fusion_position != "class_branch" or fusion_mode not in ("score", *rescue_modes)
        ):
            raise ValueError(
                "Pretrained Detail probability requires class-branch score fusion"
            )
        if fusion_scope is None or fusion_scope == "auto":
            fusion_scope = loaded_config.get(
                "fusion_scope",
                "pyramid" if loaded_method == ZIP_BGD_SEMANTIC_PYRAMID_METHOD else "selected",
            )
        if fusion_scope not in ("selected", "pyramid"):
            raise ValueError("fusion_scope must be auto, selected or pyramid")
        if fusion_scope == "pyramid":
            valid_pyramid_feature = (
                detail_output == "semantic_maps"
                and fusion_position == "class_branch"
                and fusion_mode == "residual"
            )
            valid_pyramid_score = (
                detail_output == "probability"
                and fusion_position == "class_branch"
                and fusion_mode in ("score", *rescue_modes)
            )
            if not (valid_pyramid_feature or valid_pyramid_score):
                raise ValueError(
                    "Pyramid scope requires semantic-map residual fusion or Detail score fusion"
                )
        channels = (
            _detect_input_channels(detect_head, level)
            if fusion_position == "pre_detect" else _class_feature_channels(detect_head, level)
        )
        requested_method = (
            ZIP_BGD_PRE_DETECT_METHOD
            if fusion_position == "pre_detect"
            else ZIP_BGD_RANK_RESCUE_METHOD
            if fusion_mode == "rank_rescue"
            else ZIP_BGD_HARD_RESCUE_METHOD
            if fusion_mode == "hard_rescue"
            else ZIP_BGD_BOUNDARY_RESCUE_METHOD
            if fusion_mode == "boundary_rescue"
            else ZIP_BGD_GROUPED_RESCUE_METHOD
            if fusion_mode == "grouped_rescue"
            else ZIP_BGD_ONE_SIDED_RESCUE_METHOD
            if fusion_mode == "rescue"
            else ZIP_BGD_SEMANTIC_SCORE_AUX_METHOD
            if fusion_mode == "score" and detail_aux_weight > 0
            else ZIP_BGD_SEMANTIC_SCORE_METHOD
            if fusion_mode == "score"
            else ZIP_BGD_SEMANTIC_PYRAMID_METHOD
            if fusion_scope == "pyramid"
            else ZIP_BGD_SEMANTIC_RESIDUAL_METHOD
            if detail_output == "semantic_maps"
            else ZIP_BGD_RESIDUAL_METHOD if fusion_mode == "residual" else ZIP_BGD_METHOD
        )
        fusion_attribute = (
            "zip_pre_detect_fusion"
            if fusion_position == "pre_detect"
            else "zip_rescue_fusion" if fusion_mode in rescue_modes
            else "zip_score_fusion" if fusion_mode == "score"
            else "zip_residual_fusion" if fusion_mode == "residual" else "zip_direct_fusion"
        )
        owns_method = all(hasattr(self.model, name) for name in ("detail_model", fusion_attribute, "zip_bgd_config"))
        owns_method = owns_method and loaded_method == requested_method
        if owns_method:
            self.detail_load_report = {"source": "checkpoint", "preserved": True}
            self.model.detail_model.to(device=device, dtype=dtype)
            fusion_module = getattr(self.model, fusion_attribute)
            if (
                isinstance(fusion_module, ZipBoundaryCalibratedRescueFusion)
                and not hasattr(fusion_module, "filter_head")
            ):
                fusion_module.filter_head = nn.Sequential(
                    nn.Linear(9, 8), nn.SiLU(inplace=True), nn.Linear(8, 1)
                )
                nn.init.zeros_(fusion_module.filter_head[-1].weight)
                nn.init.zeros_(fusion_module.filter_head[-1].bias)
            if (
                isinstance(fusion_module, ZipBoundaryCalibratedRescueFusion)
                and not hasattr(fusion_module, "local_filter_head")
            ):
                fusion_module.local_filter_head = nn.Sequential(
                    nn.Linear(64, 8), nn.SiLU(inplace=True), nn.Linear(8, 1)
                )
                nn.init.zeros_(fusion_module.local_filter_head[-1].weight)
                nn.init.zeros_(fusion_module.local_filter_head[-1].bias)
            fusion_module.to(device=device, dtype=dtype)
        else:
            if detail_weight is None:
                raise ValueError("A compressed ZIP Detail checkpoint is required for a fresh model")
            if detail_output == "semantic_maps":
                detail = ZipPretrainedSemanticDetail()
                self.detail_load_report = load_full_detail_classifier(detail, Path(detail_weight))
            elif detail_output == "probability":
                detail = Detail_Net_attn()
                self.detail_load_report = load_full_detail_classifier(detail, Path(detail_weight))
            else:
                detail = Detail_Net_attn_block()
                self.detail_load_report = load_detail_encoder_pretrained(detail, Path(detail_weight))
            self.model.detail_model = detail.to(device=device, dtype=dtype)
            if fusion_position == "pre_detect":
                self.model.zip_pre_detect_fusion = ZipDirectFusion(channels).to(device=device, dtype=dtype)
            elif fusion_mode in rescue_modes:
                if fusion_mode in ("boundary_rescue", "hard_rescue", "rank_rescue"):
                    rescue = ZipBoundaryCalibratedRescueFusion(
                        logit_gain=quality_logit_gain, margin=rescue_margin
                    )
                else:
                    scale_init = -2.0 if fusion_mode == "grouped_rescue" else -4.0
                    rescue = ZipOneSidedRescueFusion(rescue_scale_init=scale_init)
                self.model.zip_rescue_fusion = rescue.to(device=device, dtype=dtype)
            elif fusion_mode == "score":
                self.model.zip_score_fusion = ZipCandidateScoreFusion().to(device=device, dtype=dtype)
            elif fusion_mode == "residual":
                self.model.zip_residual_fusion = ZipLocalResidualFusion(
                    channels, alpha_max=residual_alpha_max
                ).to(device=device, dtype=dtype)
            else:
                self.model.zip_direct_fusion = ZipDirectFusion(channels).to(device=device, dtype=dtype)

        self.model.zip_bgd_config = {
            "method": requested_method,
            "detail_architecture": DETAIL_ARCH_VERSION,
            "candidate_conf": float(candidate_conf),
            "candidate_selection": "all_raw_no_nms_no_topk",
            "cam_method": "gradcam_aggregate_all_scores",
            "head_select": head_select,
            "head_level": level,
            "fusion_position": fusion_position,
            "detail_output": detail_output,
            "fusion_scope": fusion_scope,
            "detail_aux_weight": float(detail_aux_weight),
            "quality_aux_weight": float(quality_aux_weight),
            "deployment_conf": float(deployment_conf),
            "deployment_nms_iou": float(deployment_nms_iou),
            "rescue_min_conf": float(rescue_min_conf),
            "rescue_grouping": bool(
                fusion_mode in (
                    "grouped_rescue", "boundary_rescue", "hard_rescue", "rank_rescue"
                )
                or (rescue_grouping and fusion_mode == "rescue")
            ),
            "rescue_group_iou": float(rescue_group_iou),
            "preserve_teacher_strength": float(preserve_teacher_strength),
            "dense_distill_strength": float(dense_distill_strength),
            "quality_lr_multiplier": float(quality_lr_multiplier),
            "quality_logit_gain": float(quality_logit_gain),
            "rescue_margin": float(rescue_margin),
            "rescue_quality_threshold": float(rescue_quality_threshold),
            "quality_hard_only": bool(quality_hard_only or fusion_mode == "hard_rescue"),
            "quality_negative_cost": float(quality_negative_cost),
            "quality_ranking_weight": float(
                quality_ranking_weight if fusion_mode == "rank_rescue" else 0.0
            ),
            "quality_bidirectional_alpha": float(quality_bidirectional_alpha),
            "quality_suppression_alpha": float(quality_suppression_alpha),
            "quality_suppression_threshold": float(quality_suppression_threshold),
            "filter_aux_weight": float(filter_aux_weight),
            "filter_threshold": float(filter_threshold),
            "filter_margin": float(filter_margin),
            "crop_size": int(crop_size),
            "crop_source": crop_source,
            "fusion": (
                "fixed_4x4_add_to_p5_neck_then_direct_conv_before_detect"
                if fusion_position == "pre_detect"
                else "one_sided_low_confidence_candidate_quality_rescue"
                if fusion_mode in rescue_modes
                else "detail_conditioned_zero_start_candidate_score_residual"
                if fusion_mode == "score"
                else "fixed_4x4_average_masked_zero_start_residual"
                if fusion_mode == "residual"
                else "fixed_4x4_add_then_direct_conv"
            ),
            "use_gt_train": bool(use_gt_train and fusion_mode == "residual"),
            "gt_match_iou": float(gt_match_iou),
            "residual_alpha_max": float(residual_alpha_max),
            "teacher_blend": str(teacher_blend),
            "teacher_ratio": float(teacher_ratio),
            "student_gain": float(student_gain),
            "student_teacher_floor": float(student_teacher_floor),
            "consensus_match_iou": float(consensus_match_iou),
            "teacher_unmatched_gain": float(teacher_unmatched_gain),
            "student_unmatched_gain": float(student_unmatched_gain),
            "consensus_internal_conf": float(consensus_internal_conf),
            "teacher_rescue_min": float(teacher_rescue_min),
            "teacher_rescue_max": float(teacher_rescue_max),
        }
        if teacher_blend != "none":
            if teacher_blend not in ("score_average", "anchor_winner", "detection_consensus"):
                raise ValueError("teacher_blend must be none, score_average, anchor_winner or detection_consensus")
            if teacher_weight is None:
                raise ValueError("teacher_weight is required when teacher_blend is enabled")
            teacher = YOLO(str(teacher_weight), task=task).model.to(device=device, dtype=dtype).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            self.model.zip_teacher = teacher
        if preserve_teacher_strength > 0 or dense_distill_strength > 0:
            if teacher_weight is None:
                raise ValueError("teacher_weight is required for training-time preservation/distillation")
            preservation_teacher = YOLO(str(teacher_weight), task=task).model.to(
                device=device, dtype=dtype
            ).eval()
            for parameter in preservation_teacher.parameters():
                parameter.requires_grad_(False)
            register_zip_bgd_teacher(self.model, preservation_teacher)
        else:
            register_zip_bgd_teacher(self.model, None)
        self.model._predict_once = self.model.zip_bgd_predict_once
        self.model.is_zip_bgd = True
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        if hasattr(self.model, "zip_teacher"):
            self.model.zip_teacher.eval()
            for parameter in self.model.zip_teacher.parameters():
                parameter.requires_grad_(False)
        dfl_conv = getattr(getattr(detect_head, "dfl", None), "conv", None)
        if dfl_conv is not None:
            for parameter in dfl_conv.parameters():
                parameter.requires_grad_(False)
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        LOGGER.info(
            f"Complete ZIP BGD: head={head_select}/P{level + 3}, position={fusion_position}, channels={channels}, "
            f"candidate_conf={candidate_conf}, no NMS/top-k, crop={crop_source}:{crop_size}, "
            f"fusion={fusion_mode}, scope={fusion_scope}, detail_output={detail_output}, "
            f"rescue_grouping={self.model.zip_bgd_config['rescue_grouping']}, "
            f"gt_train={self.model.zip_bgd_config['use_gt_train']}, "
            f"whole-model trainable={trainable:,}"
        )

    def train(self, **kwargs):
        """Train the assembled custom model instead of rebuilding plain YOLO."""
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        if kwargs.get("cfg"):
            overrides = yaml_load(check_yaml(kwargs["cfg"]))
        overrides.update(kwargs)
        overrides["mode"] = "train"
        overrides["pretrained"] = False
        if not overrides.get("data"):
            raise AttributeError("Dataset is required")
        self.task = overrides.get("task") or self.task
        self.trainer = yolo.v8.detect.DetectionTrainer(overrides=overrides, _callbacks=self.callbacks)
        if os.environ.get("WANDB_MODE", "").lower() == "disabled":
            for event, callbacks in self.trainer.callbacks.items():
                self.trainer.callbacks[event] = [
                    callback for callback in callbacks if callback.__module__ != "ultralytics.yolo.utils.callbacks.wb"
                ]
        self.trainer.model = self.model
        self.trainer.hub_session = self.session
        self.trainer.train()
        if RANK in (-1, 0) and Path(self.trainer.best).exists():
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)

    def val(self, data=None, **kwargs):
        """Validate with gradients temporarily available for aggregate Grad-CAM."""
        overrides = self.overrides.copy()
        overrides["rect"] = True
        overrides.update(kwargs)
        overrides["mode"] = "val"
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        args.task = self.task
        validator = yolo.v8.detect.DetectionValidator(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics
