"""Core operations for Grad-CAM-guided global/local feature fusion."""

from dataclasses import dataclass
from math import ceil, floor, sqrt
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Candidate:
    """A decoded YOLO candidate that retains its differentiable class score."""

    batch_index: int
    anchor_index: int
    class_index: int
    score: torch.Tensor
    box_xyxy: torch.Tensor
    source: str = "prediction"
    # Multiview methods may process every raw proposal through the local/global
    # evidence path while emitting only one consensus representative per graph
    # component.  These fields are inert for ordinary YOLO candidates.
    emit: bool = True
    fused_box_xyxy: Optional[torch.Tensor] = None
    consensus: bool = False
    # Number of independent augmented views supporting this hypothesis and
    # number of raw hypotheses merged into its emitted representative.  They
    # are diagnostic metadata; ordinary YOLO candidates retain the defaults.
    view_support: int = 1
    component_size: int = 1
    # Canonical input is view 0; the two independently transformed inputs are
    # views 1 and 2.  Keeping this identity lets the fusion module compare
    # visual embeddings across views instead of relying only on box geometry.
    # Training-only GT crops use -1 and never participate in view consensus.
    view_index: int = 0
    # Greedy one-to-one matches across transformed views share a pair id.
    # A negative value means that no explicit independent-view pair exists.
    pair_id: int = -1


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from center xywh to corner xyxy without changing gradients."""
    centers = boxes[..., :2]
    half_size = boxes[..., 2:] / 2
    return torch.cat((centers - half_size, centers + half_size), dim=-1)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Return kept indices using dependency-free class-aware NMS."""
    if boxes.numel() == 0:
        return torch.empty(0, device=boxes.device, dtype=torch.long)

    boxes = boxes.detach().float()
    scores = scores.detach().float()
    classes = classes.detach().float()
    max_coordinate = boxes.abs().max().clamp_min(1)
    offsets = classes[:, None] * (max_coordinate + 1)
    boxes = boxes + offsets

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    order = scores.argsort(descending=True)
    keep = []

    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        xx1 = torch.maximum(x1[current], x1[remaining])
        yy1 = torch.maximum(y1[current], y1[remaining])
        xx2 = torch.minimum(x2[current], x2[remaining])
        yy2 = torch.minimum(y2[current], y2[remaining])
        intersection = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        union = areas[current] + areas[remaining] - intersection
        iou = intersection / union.clamp_min(1e-9)
        order = remaining[iou <= iou_threshold]

    return torch.stack(keep).to(dtype=torch.long)


def select_candidates(
    decoded: torch.Tensor,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.60,
    topk: Optional[int] = 5,
    pre_nms_topk: Optional[int] = 300,
) -> List[Candidate]:
    """Select NMS-filtered candidates while preserving their score tensors."""
    if decoded.ndim != 3 or decoded.shape[1] <= 4:
        raise ValueError(f"Expected decoded predictions [B, 4+nc, N], got {tuple(decoded.shape)}")

    candidates: List[Candidate] = []
    for batch_index in range(decoded.shape[0]):
        class_scores = decoded[batch_index, 4:, :]
        scores, classes = class_scores.max(dim=0)
        anchor_indices = torch.nonzero(scores.detach() >= conf_threshold, as_tuple=False).flatten()
        if anchor_indices.numel() == 0:
            continue

        if pre_nms_topk is not None and pre_nms_topk > 0 and anchor_indices.numel() > pre_nms_topk:
            relative = scores.detach()[anchor_indices].topk(pre_nms_topk).indices
            anchor_indices = anchor_indices[relative]

        boxes = xywh_to_xyxy(decoded[batch_index, :4, anchor_indices].transpose(0, 1))
        selected_scores = scores[anchor_indices]
        selected_classes = classes[anchor_indices]
        keep = class_aware_nms(boxes, selected_scores, selected_classes, iou_threshold)
        if topk is not None and topk > 0:
            keep = keep[:topk]

        for relative_index in keep.tolist():
            anchor_index = int(anchor_indices[relative_index])
            class_index = int(classes[anchor_index])
            candidates.append(
                Candidate(
                    batch_index=batch_index,
                    anchor_index=anchor_index,
                    class_index=class_index,
                    score=decoded[batch_index, 4 + class_index, anchor_index],
                    box_xyxy=boxes[relative_index].detach(),
                )
            )
    return candidates


def box_center(candidate: Candidate, image_size: Tuple[int, int]) -> torch.Tensor:
    """Return a clamped candidate-box center in input-image coordinates."""
    image_h, image_w = image_size
    box = candidate.box_xyxy.float()
    center = (box[:2] + box[2:]) / 2
    center[0] = center[0].clamp(0, max(image_w - 1, 0))
    center[1] = center[1].clamp(0, max(image_h - 1, 0))
    return center.detach()


def gradcam_center(
    candidate: Candidate,
    activation: torch.Tensor,
    image_size: Tuple[int, int],
    return_reason: bool = False,
    cam_method: str = "layercam",
):
    """Find a gradient-localized maximum inside a candidate box.

    ``gradcam`` implements the original Grad-CAM channel weighting: gradients
    are globally averaged over space and the weighted activation sum is passed
    through ReLU. ``layercam`` preserves the legacy detector-oriented behavior.
    """
    if cam_method not in {"gradcam", "layercam"}:
        raise ValueError(f"Unsupported CAM method: {cam_method}")
    fallback = box_center(candidate, image_size)

    def result(center, valid, reason):
        return (center, valid, reason) if return_reason else (center, valid)

    def usable(cam):
        return bool(torch.isfinite(cam).all() and float(cam.max() - cam.min()) > 1e-12)

    if not candidate.score.requires_grad or not activation.requires_grad:
        return result(fallback, False, "no_grad")

    gradient = torch.autograd.grad(
        candidate.score,
        activation,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    if gradient is None:
        return result(fallback, False, "unused_grad")

    batch_index = candidate.batch_index
    feature = activation[batch_index]
    spatial_gradient = gradient[batch_index]

    if cam_method == "gradcam":
        weights = spatial_gradient.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * feature).sum(dim=0)).detach()
        method = "gradcam"
    else:
        cam = torch.relu((torch.relu(spatial_gradient) * feature).sum(dim=0)).detach()
        method = "layercam"
        if not usable(cam):
            cam = (spatial_gradient * feature).abs().sum(dim=0).detach()
            method = "abs_relevance"
    if not usable(cam):
        return result(fallback, False, "flat_cam")

    image_h, image_w = image_size
    feature_h, feature_w = cam.shape
    box = candidate.box_xyxy
    x1 = max(0, min(feature_w - 1, floor(float(box[0]) * feature_w / image_w)))
    y1 = max(0, min(feature_h - 1, floor(float(box[1]) * feature_h / image_h)))
    x2 = max(x1 + 1, min(feature_w, ceil(float(box[2]) * feature_w / image_w)))
    y2 = max(y1 + 1, min(feature_h, ceil(float(box[3]) * feature_h / image_h)))
    region = cam[y1:y2, x1:x2]
    if region.numel() == 0:
        return result(fallback, False, "empty_region")
    if not usable(region):
        return result(fallback, False, "flat_region")

    flat_index = int(region.argmax())
    region_w = region.shape[1]
    feature_y = y1 + flat_index // region_w
    feature_x = x1 + flat_index % region_w
    center = cam.new_tensor(
        [
            (feature_x + 0.5) * image_w / feature_w,
            (feature_y + 0.5) * image_h / feature_h,
        ]
    )
    return result(center.detach(), True, method)


def locate_centers(
    candidates: Sequence[Candidate],
    activation,
    image_size: Tuple[int, int],
    mode: str,
    return_reasons: bool = False,
    cam_method: str = "layercam",
):
    """Locate one crop center for each candidate."""
    if mode not in {"gradcam", "box_center"}:
        raise ValueError(f"Unsupported localization mode: {mode}")
    centers, used_cam, reasons = [], [], []
    for candidate in candidates:
        if mode == "gradcam" and candidate.source == "prediction":
            candidate_activation = activation
            if isinstance(activation, (list, tuple)):
                # YOLO concatenates flattened P3/P4/P5 predictions in order.
                # A candidate score only depends on the feature level that
                # produced its anchor, so requesting every CAM from P3 makes
                # P4/P5 gradients identically zero.
                anchor_offset = 0
                candidate_activation = None
                for level_activation in activation:
                    level_anchor_count = level_activation.shape[-2] * level_activation.shape[-1]
                    if candidate.anchor_index < anchor_offset + level_anchor_count:
                        candidate_activation = level_activation
                        break
                    anchor_offset += level_anchor_count
                if candidate_activation is None:
                    raise IndexError(
                        f"Candidate anchor {candidate.anchor_index} is outside "
                        f"the {anchor_offset} anchors represented by detection features"
                    )
            center, valid, reason = gradcam_center(
                candidate,
                candidate_activation,
                image_size,
                return_reason=True,
                cam_method=cam_method,
            )
        else:
            reason = "gt_center" if candidate.source == "gt" else "box_center"
            center, valid = box_center(candidate, image_size), False
        centers.append(center)
        used_cam.append(valid)
        reasons.append(reason)
    if return_reasons:
        return centers, used_cam, reasons
    return centers, used_cam


def crop_tensor_patches(
    images: torch.Tensor,
    candidates: Sequence[Candidate],
    centers: Sequence[torch.Tensor],
    crop_size: int = 64,
    source_scale: float = 0.0,
    min_source_size: int = 64,
    max_source_size: int = 192,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop padded square patches from the exact tensor received by YOLO.

    ``crop_size`` is the encoder input resolution. When ``source_scale`` is
    positive, the source window adapts to the candidate area and is resized to
    that resolution; otherwise the legacy fixed-size crop is preserved.
    """
    if len(candidates) != len(centers):
        raise ValueError("candidates and centers must have the same length")
    batch_size, channels, image_h, image_w = images.shape
    if not candidates:
        empty_patches = images.new_empty((0, channels, crop_size, crop_size))
        return empty_patches, images.new_empty((0, 4)), torch.empty(0, device=images.device, dtype=torch.long)

    patches, crop_boxes, batch_indices = [], [], []
    for candidate, center in zip(candidates, centers):
        batch_index = candidate.batch_index
        if not 0 <= batch_index < batch_size:
            raise IndexError(f"Candidate batch index {batch_index} is outside batch size {batch_size}")

        # Decoded detector boxes are allowed to extend beyond the letterboxed
        # image. A bare-tensor warm-up can therefore produce a box centre far
        # outside the valid sampling domain. Clamp the sampling centre just as
        # ``crop_original_patches`` does; otherwise the padding calculation can
        # create a patch hundreds of pixels tall instead of ``source_size``.
        center_x = min(max(float(center[0]), 0.0), max(image_w - 1.0, 0.0))
        center_y = min(max(float(center[1]), 0.0), max(image_h - 1.0, 0.0))

        source_size = crop_size
        if source_scale > 0:
            box = candidate.box_xyxy.float()
            box_area = float(((box[2] - box[0]).clamp_min(1) * (box[3] - box[1]).clamp_min(1)))
            source_size = int(round(sqrt(box_area) * source_scale))
            source_size = max(int(min_source_size), min(int(max_source_size), source_size))
        half = source_size / 2
        x1 = floor(center_x - half)
        y1 = floor(center_y - half)
        x2, y2 = x1 + source_size, y1 + source_size
        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(image_w, x2), min(image_h, y2)
        patch = images[batch_index:batch_index + 1, :, src_y1:src_y2, src_x1:src_x2]
        patch = F.pad(patch, (src_x1 - x1, x2 - src_x2, src_y1 - y1, y2 - src_y2))
        if source_size != crop_size:
            patch = F.interpolate(patch, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        patches.append(patch.squeeze(0))
        crop_boxes.append(images.new_tensor((src_x1, src_y1, src_x2, src_y2)))
        batch_indices.append(batch_index)

    patches = torch.stack(patches)
    mean = patches.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = patches.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    patches = (patches - mean) / std
    return patches, torch.stack(crop_boxes), torch.tensor(batch_indices, device=images.device, dtype=torch.long)


def _letterbox_parameters(ratio_pad) -> Tuple[float, float, float, float]:
    """Return ``gain_x, gain_y, pad_x, pad_y`` for one pure-LetterBox sample.

    Ultralytics stores the initial image-resize ratio as ``(gain_y, gain_x)``
    and LetterBox then nests it with ``(pad_x, pad_y)``.  A deeper nesting is
    produced by Mosaic/RandomPerspective and is deliberately rejected: one
    original image is not sufficient to invert a mosaic batch safely.
    """
    if not isinstance(ratio_pad, (tuple, list)) or len(ratio_pad) != 2:
        raise ValueError(f"Invalid ratio_pad metadata: {ratio_pad!r}")
    gain, pad = ratio_pad
    if not (
        isinstance(gain, (tuple, list))
        and len(gain) == 2
        and all(np.isscalar(value) for value in gain)
        and isinstance(pad, (tuple, list))
        and len(pad) == 2
        and all(np.isscalar(value) for value in pad)
    ):
        raise ValueError(
            "Original-image crops require pure LetterBox metadata. "
            "Disable mosaic, perspective/affine transforms and flips."
        )
    gain_y, gain_x = (float(value) for value in gain)
    pad_x, pad_y = (float(value) for value in pad)
    if gain_x <= 0 or gain_y <= 0:
        raise ValueError(f"LetterBox gains must be positive, got {(gain_x, gain_y)}")
    return gain_x, gain_y, pad_x, pad_y


def crop_original_patches(
    images: torch.Tensor,
    original_images: Sequence[np.ndarray],
    ratio_pads: Sequence,
    candidates: Sequence[Candidate],
    centers: Sequence[torch.Tensor],
    crop_size: int = 64,
    source_scale: float = 0.0,
    min_source_size: int = 64,
    max_source_size: int = 192,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map Grad-CAM centers to originals and crop high-resolution RGB ROIs.

    The crop window is measured in original-image pixels.  Its actual clipped
    extent is mapped forward to YOLO-input coordinates so feature pasting and
    original-image sampling remain spatially consistent.  CPU crops are
    stacked and transferred to the accelerator once per batch.
    """
    if len(candidates) != len(centers):
        raise ValueError("candidates and centers must have the same length")
    batch_size, channels, _, _ = images.shape
    if channels != 3:
        raise ValueError(f"Expected three-channel YOLO input, got {channels}")
    if len(original_images) != batch_size or len(ratio_pads) != batch_size:
        raise ValueError(
            "original_images and ratio_pads must contain one item per YOLO batch image"
        )
    if not candidates:
        return (
            images.new_empty((0, channels, crop_size, crop_size)),
            images.new_empty((0, 4)),
            torch.empty(0, device=images.device, dtype=torch.long),
        )

    patches, crop_boxes, batch_indices = [], [], []
    for candidate, center in zip(candidates, centers):
        batch_index = candidate.batch_index
        if not 0 <= batch_index < batch_size:
            raise IndexError(f"Candidate batch index {batch_index} is outside batch size {batch_size}")

        original = original_images[batch_index]
        if torch.is_tensor(original):
            original = original.detach().cpu().numpy()
        original = np.asarray(original)
        if original.ndim != 3 or original.shape[2] != 3:
            raise ValueError(f"Expected original RGB image [H,W,3], got {original.shape}")
        original_h, original_w = original.shape[:2]
        gain_x, gain_y, pad_x, pad_y = _letterbox_parameters(ratio_pads[batch_index])

        center_x = (float(center[0]) - pad_x) / gain_x
        center_y = (float(center[1]) - pad_y) / gain_y
        center_x = min(max(center_x, 0.0), max(original_w - 1.0, 0.0))
        center_y = min(max(center_y, 0.0), max(original_h - 1.0, 0.0))

        source_size = int(crop_size)
        if source_scale > 0:
            box = candidate.box_xyxy.float()
            box_w = float((box[2] - box[0]).clamp_min(1)) / gain_x
            box_h = float((box[3] - box[1]).clamp_min(1)) / gain_y
            source_size = int(round(sqrt(box_w * box_h) * source_scale))
            source_size = max(int(min_source_size), min(int(max_source_size), source_size))
        source_size = max(source_size, 1)

        x1 = floor(center_x - source_size / 2)
        y1 = floor(center_y - source_size / 2)
        x2, y2 = x1 + source_size, y1 + source_size
        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(original_w, x2), min(original_h, y2)

        patch = original[src_y1:src_y2, src_x1:src_x2]
        patch = cv2.copyMakeBorder(
            patch,
            src_y1 - y1,
            y2 - src_y2,
            src_x1 - x1,
            x2 - src_x2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        if patch.shape[:2] != (crop_size, crop_size):
            patch = cv2.resize(patch, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        patches.append(torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))))
        crop_boxes.append(
            (
                src_x1 * gain_x + pad_x,
                src_y1 * gain_y + pad_y,
                src_x2 * gain_x + pad_x,
                src_y2 * gain_y + pad_y,
            )
        )
        batch_indices.append(batch_index)

    patches = torch.stack(patches).to(device=images.device, dtype=images.dtype, non_blocking=True)
    patches = patches.div(255.0)
    mean = patches.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = patches.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    patches = (patches - mean) / std
    return (
        patches,
        images.new_tensor(crop_boxes),
        torch.tensor(batch_indices, device=images.device, dtype=torch.long),
    )


def paste_and_average(
    local_features: torch.Tensor,
    crop_boxes: torch.Tensor,
    batch_indices: torch.Tensor,
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paste local features into a full feature canvas and average overlaps."""
    if local_features.ndim != 4:
        raise ValueError(f"Expected local features [N,C,H,W], got {tuple(local_features.shape)}")
    if not (len(local_features) == len(crop_boxes) == len(batch_indices)):
        raise ValueError("local_features, crop_boxes and batch_indices must have matching lengths")

    image_h, image_w = image_size
    feature_h, feature_w = feature_size
    channels = local_features.shape[1]
    zero_canvas = local_features.new_zeros((channels, feature_h, feature_w))
    zero_count = local_features.new_zeros((1, feature_h, feature_w))
    contributions: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
    counts: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

    for feature, box, batch_index_tensor in zip(local_features, crop_boxes, batch_indices):
        batch_index = int(batch_index_tensor)
        x1 = max(0, min(feature_w - 1, floor(float(box[0]) * feature_w / image_w)))
        y1 = max(0, min(feature_h - 1, floor(float(box[1]) * feature_h / image_h)))
        x2 = max(x1 + 1, min(feature_w, ceil(float(box[2]) * feature_w / image_w)))
        y2 = max(y1 + 1, min(feature_h, ceil(float(box[3]) * feature_h / image_h)))
        resized = F.interpolate(feature.unsqueeze(0), size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
        contributions[batch_index].append(
            F.pad(resized.squeeze(0), (x1, feature_w - x2, y1, feature_h - y2))
        )
        count = local_features.new_ones((1, y2 - y1, x2 - x1))
        counts[batch_index].append(F.pad(count, (x1, feature_w - x2, y1, feature_h - y2)))

    canvas_batch, mask_batch = [], []
    for batch_index in range(batch_size):
        summed = torch.stack(contributions[batch_index]).sum(0) if contributions[batch_index] else zero_canvas
        count = torch.stack(counts[batch_index]).sum(0) if counts[batch_index] else zero_count
        canvas_batch.append(summed / count.clamp_min(1))
        mask_batch.append((count > 0).to(local_features.dtype))
    return torch.stack(canvas_batch), torch.stack(mask_batch)


def scale_and_clip_boxes(
    boxes: torch.Tensor,
    scale: float,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Scale xyxy boxes about their centers and clip them to the image."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)
    image_h, image_w = image_size
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    half_sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1) * (float(scale) / 2)
    scaled = torch.cat((centers - half_sizes, centers + half_sizes), dim=1)
    scaled[:, 0::2] = scaled[:, 0::2].clamp(0, image_w)
    scaled[:, 1::2] = scaled[:, 1::2].clamp(0, image_h)
    return scaled


def roi_align_tensor(
    feature_map: torch.Tensor,
    boxes: torch.Tensor,
    batch_indices: torch.Tensor,
    image_size: Tuple[int, int],
    output_size: Tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Differentiably sample fixed-size FPN RoIs without torchvision ops."""
    if len(boxes) != len(batch_indices):
        raise ValueError("boxes and batch_indices must have matching lengths")
    if not len(boxes):
        return feature_map.new_empty((0, feature_map.shape[1], *output_size))

    image_h, image_w = image_size
    output_h, output_w = output_size
    rois = []
    for box, batch_index_tensor in zip(boxes, batch_indices):
        batch_index = int(batch_index_tensor)
        # grid_sample with align_corners=False maps pixel centers using
        # normalized coordinates (2 * x + 1) / size - 1.
        x_fraction = (torch.arange(output_w, device=box.device, dtype=box.dtype) + 0.5) / output_w
        y_fraction = (torch.arange(output_h, device=box.device, dtype=box.dtype) + 0.5) / output_h
        xs = box[0] + x_fraction * (box[2] - box[0])
        ys = box[1] + y_fraction * (box[3] - box[1])
        xs = (2 * xs + 1) / image_w - 1
        ys = (2 * ys + 1) / image_h - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        rois.append(
            F.grid_sample(
                feature_map[batch_index:batch_index + 1],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
        )
    return torch.stack(rois)


class GlobalLocalFusion(nn.Module):
    """Masked gated residual fusion at one YOLO detection scale."""

    def __init__(self, global_channels: int, local_channels: int = 3, alpha_init: float = 0.01):
        super().__init__()
        self.local_proj = nn.Sequential(
            nn.Conv2d(local_channels, global_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(global_channels),
            nn.SiLU(inplace=True),
        )
        self.gate_conv = nn.Conv2d(2 * global_channels + 1, global_channels, kernel_size=3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        global_feature: torch.Tensor,
        local_canvas: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if global_feature.shape[0] != local_canvas.shape[0] or global_feature.shape[-2:] != local_canvas.shape[-2:]:
            raise ValueError("Global feature and local canvas must share batch and spatial dimensions")
        if mask.shape != (global_feature.shape[0], 1, *global_feature.shape[-2:]):
            raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")
        projected = self.local_proj(local_canvas) * mask
        gate = torch.sigmoid(self.gate_conv(torch.cat((global_feature, projected, mask), dim=1)))
        fused = global_feature + self.alpha * gate * projected * mask
        return fused, gate


def candidate_fpn_level(
    candidate: Candidate,
    image_size: Tuple[int, int],
    thresholds: Tuple[float, float] = (96.0, 256.0),
) -> int:
    """Assign a candidate to P3/P4/P5 from its scale in input pixels."""
    del image_size  # boxes are already expressed in input-image pixels
    box = candidate.box_xyxy.float()
    scale = float(torch.sqrt((box[2] - box[0]).clamp_min(1) * (box[3] - box[1]).clamp_min(1)))
    if scale < thresholds[0]:
        return 0
    if scale < thresholds[1]:
        return 1
    return 2


class PyramidGlobalLocalFusion(nn.Module):
    """Per-FPN-level local projection and gated residual fusion."""

    def __init__(self, global_channels, local_channels: int = 64, alpha_init: float = 0.05):
        super().__init__()
        self.local_proj = nn.ModuleList()
        self.gate_conv = nn.ModuleList()
        self.alpha = nn.Parameter(torch.full((len(global_channels),), float(alpha_init)))
        for channels in global_channels:
            groups = 8 if channels % 8 == 0 else 1
            self.local_proj.append(
                nn.Sequential(
                    nn.Conv2d(local_channels, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(groups, channels),
                    nn.SiLU(inplace=True),
                )
            )
            self.gate_conv.append(
                nn.Conv2d(2 * channels + 1, channels, kernel_size=3, padding=1)
            )

    def project(self, level: int, local_features: torch.Tensor) -> torch.Tensor:
        return self.local_proj[level](local_features)

    def enable_roi_context(self, beta_init: float = 0.05):
        """Attach V3 detection-context adapters, including to loaded V2 modules."""
        if hasattr(self, "roi_context_norm"):
            return
        channels = [projection[0].out_channels for projection in self.local_proj]
        self.roi_context_norm = nn.ModuleList()
        self.roi_context_gate = nn.ModuleList()
        self.roi_beta = nn.Parameter(torch.full((len(channels),), float(beta_init)))
        for channel_count in channels:
            groups = 8 if channel_count % 8 == 0 else 1
            self.roi_context_norm.append(nn.GroupNorm(groups, channel_count))
            self.roi_context_gate.append(
                nn.Conv2d(2 * channel_count, channel_count, kernel_size=1)
            )

    def project_with_context(
        self,
        level: int,
        local_features: torch.Tensor,
        roi_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modulate Detail features with aligned detection features from the same RoI."""
        if not hasattr(self, "roi_context_norm"):
            raise RuntimeError("ROI context adapters have not been enabled")
        projected = self.project(level, local_features)
        context = self.roi_context_norm[level](roi_context)
        gate = torch.sigmoid(self.roi_context_gate[level](torch.cat((projected, context), dim=1)))
        aligned = projected + self.roi_beta[level] * gate * (context - projected)
        return aligned, gate

    def fuse(self, level: int, global_feature, projected_canvas, mask):
        projected_canvas = projected_canvas * mask
        gate = torch.sigmoid(
            self.gate_conv[level](torch.cat((global_feature, projected_canvas, mask), dim=1))
        )
        fused = global_feature + self.alpha[level] * gate * projected_canvas * mask
        return fused, gate
