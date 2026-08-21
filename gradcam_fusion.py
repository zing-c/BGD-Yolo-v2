"""Grad-CAM-guided two-pass YOLO global/local feature fusion."""

import contextlib
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detail_model import DETAIL_ARCH_VERSION, Detail_Net_attn_block, Detail_Net_attn_semantic
from global_local_fusion import (
    Candidate,
    GlobalLocalFusion,
    PyramidGlobalLocalFusion,
    candidate_fpn_level,
    crop_original_patches,
    crop_tensor_patches,
    locate_centers,
    paste_and_average,
    roi_align_tensor,
    scale_and_clip_boxes,
    select_candidates,
)
from ultralytics import YOLO, yolo
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, yaml_load
from ultralytics.yolo.utils.checks import check_yaml


def _unwrap_detail_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("detail_model", "detail_encoder", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def load_detail_encoder_pretrained(encoder: Detail_Net_attn_block, weight: Path) -> Dict[str, object]:
    """Load only matching encoder tensors and report every mismatch explicitly."""
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    source = _unwrap_detail_state_dict(checkpoint)
    if not isinstance(source, dict):
        raise TypeError(f"Detail checkpoint does not contain a state_dict: {weight}")

    target = encoder.state_dict()
    compatible, unexpected, shape_mismatch = {}, [], []
    for name, tensor in source.items():
        if name not in target:
            unexpected.append(name)
        elif target[name].shape != tensor.shape:
            shape_mismatch.append((name, tuple(tensor.shape), tuple(target[name].shape)))
        else:
            compatible[name] = tensor

    required = encoder.pretrained_state_keys()
    loaded_required = required.intersection(compatible)
    required_numel = sum(target[name].numel() for name in required)
    loaded_required_numel = sum(target[name].numel() for name in loaded_required)
    pretrained_coverage = loaded_required_numel / max(required_numel, 1)
    missing_required = sorted(required - loaded_required)
    if pretrained_coverage < 0.95:
        raise ValueError(
            f"Detail checkpoint architecture mismatch for {weight}: "
            f"expected {DETAIL_ARCH_VERSION}, but only {pretrained_coverage:.1%} of the "
            "compressed backbone is compatible. Retrain/use the ZIP-revision Detail checkpoint; "
            "the historical wide exp3_4_1.pt is intentionally rejected."
        )
    result = encoder.load_state_dict(compatible, strict=False)
    report = {
        "architecture": DETAIL_ARCH_VERSION,
        "loaded": sorted(compatible),
        "missing": sorted(result.missing_keys),
        "missing_required": missing_required,
        "pretrained_coverage": pretrained_coverage,
        "unexpected": sorted(unexpected),
        "shape_mismatch": shape_mismatch,
    }
    LOGGER.info(
        "Detail encoder preload: "
        f"loaded={len(report['loaded'])}, missing={len(report['missing'])}, "
        f"coverage={report['pretrained_coverage']:.1%}, unexpected={len(report['unexpected'])}, "
        f"shape_mismatch={len(report['shape_mismatch'])}"
    )
    if report["missing"]:
        LOGGER.info("Detail encoder randomly initialized tensors: " + ", ".join(report["missing"]))
    if report["unexpected"]:
        LOGGER.info("Detail classifier-only tensors ignored: " + ", ".join(report["unexpected"]))
    if report["shape_mismatch"]:
        LOGGER.info("Detail tensors ignored due to shape mismatch: " + repr(report["shape_mismatch"]))
    return report


def _detect_input_channels(detect_head, level: int) -> int:
    if not 0 <= level < detect_head.nl:
        raise ValueError(f"fusion_level={level} is outside Detect levels 0..{detect_head.nl - 1}")
    first_conv = detect_head.cv2[level][0]
    conv = getattr(first_conv, "conv", first_conv)
    channels = getattr(conv, "in_channels", None)
    if channels is None:
        raise TypeError("Could not determine Detect input channels for the requested fusion level")
    return int(channels)


def _features_before_detect(model, images, profile=False, visualize=False):
    """Run the YOLO graph up to, but not including, the Detect head."""
    x = images
    saved, timings = [], []
    for module in model.model[:-1]:
        if module.f != -1:
            x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
        if profile:
            model._profile_one_layer(module, x, timings)
        if hasattr(module, "backbone"):
            x = module(x)
            for _ in range(5 - len(x)):
                x.insert(0, None)
            for index, feature in enumerate(x):
                saved.append(feature if index in model.save else None)
            x = x[-1]
        else:
            x = module(x)
            saved.append(x if module.i in model.save else None)
        if visualize:
            model.feature_visualization(x, module.type, module.i, save_dir=visualize)

    detect_head = model.model[-1]
    sources = detect_head.f if isinstance(detect_head.f, (list, tuple)) else [detect_head.f]
    detect_features = [x if index == -1 else saved[index] for index in sources]
    if any(feature is None for feature in detect_features):
        raise RuntimeError(f"Detect inputs {sources} were not retained by the YOLO graph")
    return detect_features, detect_head


def _apply_stage_runtime_modes(model):
    """Keep frozen YOLO BatchNorm statistics fixed during staged training."""
    if not model.training:
        return
    stage = int(model.gl_fusion_config["stage"])
    if stage == 3:
        return

    for module in model.model:
        module.eval()
    if stage in (0, 4):
        # Keep the frozen pretrained Detail backbone, including BatchNorm
        # running statistics, fixed while warming up only its new output heads.
        model.detail_encoder.eval()
        if stage == 0 and is_pyramid_method(model):
            model.detail_encoder.semantic_head.train()
        elif stage == 0:
            for name in ("conv_l", "conv_m", "conv_s", "bn_cat"):
                getattr(model.detail_encoder, name).train()
    else:
        model.detail_encoder.train()
    model.global_local_fusion.train()
    if hasattr(model, "detail_aux_head") and stage != 4:
        model.detail_aux_head.train()

    detect_head = model.model[-1]
    fusion_level = int(model.gl_fusion_config["fusion_level"])
    if stage == 2:
        fusion_levels = range(detect_head.nl) if is_pyramid_method(model) else (fusion_level,)
        for level in fusion_levels:
            producer = model.model[detect_head.f[level]]
            producer.train()
            # Batch size is intentionally small because Grad-CAM is a two-pass
            # path. Updating pretrained BN statistics here caused an immediate
            # validation collapse even though the convolution gradients were
            # healthy. Keep affine parameters trainable but running statistics
            # fixed, matching standard small-batch fine-tuning practice.
            for child in producer.modules():
                if isinstance(child, nn.modules.batchnorm._BatchNorm):
                    child.eval()
        detect_head.train()
        for child in detect_head.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()
    else:
        # Detect.forward uses this top-level flag to choose raw training output.
        # Its frozen child Conv/BN modules deliberately remain in eval mode.
        detect_head.training = True


def _detach_output_tree(output):
    """Detach inference outputs after CAM has consumed its temporary graph."""
    if torch.is_tensor(output):
        return output.detach()
    if isinstance(output, tuple):
        return tuple(_detach_output_tree(item) for item in output)
    if isinstance(output, list):
        return [_detach_output_tree(item) for item in output]
    if isinstance(output, dict):
        return {key: _detach_output_tree(value) for key, value in output.items()}
    return output


def config_method(model):
    return model.gl_fusion_config.get("method", "gradcam_global_local_v1")


def is_pyramid_method(model):
    return config_method(model) in ("gradcam_global_local_v2", "gradcam_global_local_v3")


def _crop_detail_patches(model, images, candidates, centers, adaptive=False):
    """Crop from the configured source while keeping one coordinate contract."""
    config = model.gl_fusion_config
    crop_kwargs = {
        "crop_size": int(config["crop_size"]),
        "source_scale": float(config["source_crop_scale"]) if adaptive else 0.0,
        "min_source_size": int(config["min_source_size"]),
        "max_source_size": int(config["max_source_size"]),
    }
    if config.get("crop_source", "tensor") == "tensor":
        return crop_tensor_patches(images, candidates, centers, **crop_kwargs)

    batch = getattr(model, "batch", None)
    if not isinstance(batch, dict) or "ori_img" not in batch or "ratio_pad" not in batch:
        raise RuntimeError(
            "crop_source='original' requires dataloader metadata 'ori_img' and 'ratio_pad'. "
            "Use this repository's train/val pipeline instead of a bare tensor predict call."
        )
    return crop_original_patches(
        images,
        batch["ori_img"],
        batch["ratio_pad"],
        candidates,
        centers,
        **crop_kwargs,
    )


def _box_iou_one_to_many(box, boxes):
    if boxes.numel() == 0:
        return boxes.new_empty((0,))
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (rb - lt).clamp_min(0).prod(dim=1)
    box_area = (box[2:] - box[:2]).clamp_min(0).prod()
    boxes_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (box_area + boxes_area - intersection).clamp_min(1e-9)


def _ground_truth_candidates(model, images):
    """Convert the current augmented training batch's normalized GT to candidates."""
    batch = getattr(model, "batch", None)
    if not isinstance(batch, dict) or not all(key in batch for key in ("bboxes", "batch_idx", "cls")):
        return [], [images.new_empty((0, 4)) for _ in range(images.shape[0])]
    image_h, image_w = images.shape[-2:]
    normalized = batch["bboxes"].to(device=images.device, dtype=images.dtype)
    batch_indices = batch["batch_idx"].to(device=images.device, dtype=torch.long)
    classes = batch["cls"].reshape(-1).to(device=images.device, dtype=torch.long)
    xywh = normalized * normalized.new_tensor((image_w, image_h, image_w, image_h))
    boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), dim=1)
    boxes[:, 0::2].clamp_(0, image_w)
    boxes[:, 1::2].clamp_(0, image_h)
    gt_by_image = [boxes[batch_indices == index].detach() for index in range(images.shape[0])]
    candidates = [
        Candidate(
            batch_index=int(batch_index),
            anchor_index=-1,
            class_index=int(class_index),
            score=images.new_zeros(()),
            box_xyxy=box.detach(),
            source="gt",
        )
        for box, batch_index, class_index in zip(boxes, batch_indices, classes)
    ]
    return candidates, gt_by_image


def _rescore_candidate_anchors(decoded, candidates, quality_logits, floor=0.25, power=1.0):
    """Use the locally supervised quality head to suppress unreliable anchors."""
    rescored = decoded.clone()
    quality = torch.sigmoid(quality_logits.detach())
    multipliers = (float(floor) + (1.0 - float(floor)) * quality).pow(float(power))
    for candidate, multiplier in zip(candidates, multipliers):
        if candidate.source != "prediction":
            continue
        rescored[
            candidate.batch_index,
            4 + candidate.class_index,
            candidate.anchor_index,
        ] = rescored[
            candidate.batch_index,
            4 + candidate.class_index,
            candidate.anchor_index,
        ] * multiplier
    return rescored, quality


def _gradcam_global_local_v2_impl(model, images, profile=False, visualize=False):
    config = model.gl_fusion_config
    _apply_stage_runtime_modes(model)
    features, detect_head = _features_before_detect(model, images, profile, visualize)
    for level, feature in enumerate(features):
        if not feature.requires_grad:
            features[level] = feature.detach().requires_grad_(True)

    head_was_training = detect_head.training
    detect_head.eval()
    first_output = detect_head([feature.clone() for feature in features])
    if head_was_training and int(config["stage"]) in (0, 1, 4):
        detect_head.training = True
    else:
        detect_head.train(head_was_training)
    first_decoded = first_output[0] if isinstance(first_output, tuple) else first_output
    predicted_candidates = select_candidates(
        first_decoded,
        conf_threshold=float(config["candidate_conf"]),
        iou_threshold=float(config["candidate_iou"]),
        topk=config["candidate_topk"],
        pre_nms_topk=config["pre_nms_topk"],
    )

    if model.training:
        gt_candidates, gt_by_image = _ground_truth_candidates(model, images)
    else:
        gt_candidates = []
        gt_by_image = [images.new_empty((0, 4)) for _ in range(images.shape[0])]
    candidates = list(predicted_candidates)
    if model.training and config.get("use_gt_train", True):
        # Add GT only when no prediction overlaps it. Matched predictions keep
        # inference-aligned LayerCAM crops; missed targets receive a GT-centered
        # teacher crop so the local branch can still learn from them.
        predictions_by_image = [
            [candidate.box_xyxy for candidate in predicted_candidates if candidate.batch_index == index]
            for index in range(images.shape[0])
        ]
        for candidate in gt_candidates:
            predicted_boxes = predictions_by_image[candidate.batch_index]
            if not predicted_boxes:
                candidates.append(candidate)
                continue
            overlaps = _box_iou_one_to_many(candidate.box_xyxy, torch.stack(predicted_boxes))
            if not bool((overlaps >= float(config["gt_match_iou"])).any()):
                candidates.append(candidate)

    image_size: Tuple[int, int] = tuple(images.shape[-2:])
    centers, used_cam, localization_reasons = locate_centers(
        candidates,
        features,
        image_size,
        "gradcam",
        return_reasons=True,
        cam_method=config.get("cam_method", "layercam"),
    )
    level_indices = [[] for _ in features]
    for index, candidate in enumerate(candidates):
        level = candidate_fpn_level(candidate, image_size, tuple(config["fpn_thresholds"]))
        level_indices[min(level, len(features) - 1)].append(index)

    if candidates:
        patches, crop_boxes, batch_indices = _crop_detail_patches(
            model, images, candidates, centers, adaptive=True
        )
        local_features = model.detail_encoder(patches)
        if local_features.shape[1] != int(config["detail_channels"]):
            raise RuntimeError(
                f"Detail encoder must return {config['detail_channels']} channels, got {tuple(local_features.shape)}"
            )
        quality_logits = model.detail_aux_head(local_features).flatten()
        if model.training:
            logits = quality_logits
            labels = []
            for candidate in candidates:
                if candidate.source == "gt":
                    labels.append(1.0)
                    continue
                overlaps = _box_iou_one_to_many(candidate.box_xyxy, gt_by_image[candidate.batch_index])
                labels.append(
                    float(
                        overlaps.numel() > 0
                        and float(overlaps.max()) >= float(config["aux_positive_iou"])
                    )
                )
            labels = logits.new_tensor(labels)
            positives = labels.sum()
            negatives = labels.numel() - positives
            positive_weight = (negatives / positives.clamp_min(1)).clamp(0.5, 5.0)
            model.gl_aux_loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=positive_weight
            )
        else:
            model.gl_aux_loss = images.sum() * 0.0
        candidate_boxes = torch.stack([candidate.box_xyxy for candidate in candidates]).to(
            device=images.device, dtype=images.dtype
        )
        if config.get("paste_region", "crop") == "candidate":
            destination_boxes = scale_and_clip_boxes(
                candidate_boxes,
                float(config.get("paste_box_scale", 0.5)),
                image_size,
            )
        else:
            destination_boxes = crop_boxes
    else:
        crop_boxes = images.new_empty((0, 4))
        candidate_boxes = images.new_empty((0, 4))
        destination_boxes = crop_boxes
        batch_indices = torch.empty(0, device=images.device, dtype=torch.long)
        local_features = images.new_empty((0, int(config["detail_channels"]), 4, 4))
        quality_logits = images.new_empty((0,))
        model.gl_aux_loss = images.sum() * 0.0

    second_features = list(features)
    gates, masks, context_gates = [], [], []
    for level, global_feature in enumerate(features):
        indices = level_indices[level]
        if indices:
            tensor_indices = torch.tensor(indices, device=images.device, dtype=torch.long)
            if config_method(model) == "gradcam_global_local_v3":
                roi_context = roi_align_tensor(
                    global_feature,
                    candidate_boxes[tensor_indices],
                    batch_indices[tensor_indices],
                    image_size=image_size,
                    output_size=tuple(local_features.shape[-2:]),
                )
                projected, context_gate = model.global_local_fusion.project_with_context(
                    level,
                    local_features[tensor_indices],
                    roi_context,
                )
                context_gates.append(context_gate)
            else:
                projected = model.global_local_fusion.project(level, local_features[tensor_indices])
            canvas, mask = paste_and_average(
                projected,
                destination_boxes[tensor_indices],
                batch_indices[tensor_indices],
                image_size=image_size,
                feature_size=tuple(global_feature.shape[-2:]),
                batch_size=images.shape[0],
            )
        else:
            canvas = global_feature.new_zeros(global_feature.shape)
            mask = global_feature.new_zeros((images.shape[0], 1, *global_feature.shape[-2:]))
        second_features[level], gate = model.global_local_fusion.fuse(level, global_feature, canvas, mask)
        gates.append(gate)
        masks.append(mask)
    second_output = detect_head(second_features)

    quality = torch.sigmoid(quality_logits.detach())
    if not model.training and config.get("quality_rescore", False) and candidates:
        decoded = second_output[0] if isinstance(second_output, tuple) else second_output
        rescored, quality = _rescore_candidate_anchors(
            decoded,
            candidates,
            quality_logits,
            floor=float(config.get("quality_floor", 0.25)),
            power=float(config.get("quality_power", 1.0)),
        )
        second_output = (rescored, *second_output[1:]) if isinstance(second_output, tuple) else rescored

    model.gl_fusion_runtime = {
        "candidate_count": len(candidates),
        "original_crop_count": len(candidates) if config.get("crop_source") == "original" else 0,
        "prediction_candidate_count": len(predicted_candidates),
        "gt_candidate_count": len(candidates) - len(predicted_candidates),
        "gradcam_count": sum(used_cam),
        "fallback_count": len(used_cam) - sum(used_cam),
        "mask_coverage": sum(float(mask.detach().mean()) for mask in masks) / len(masks),
        "gate_mean": sum(float(gate.detach().mean()) for gate in gates) / len(gates),
        "aux_loss": float(model.gl_aux_loss.detach()),
        "quality_mean": float(quality.mean()) if quality.numel() else 0.0,
    }
    if context_gates:
        model.gl_fusion_runtime["context_gate_mean"] = sum(
            float(gate.detach().mean()) for gate in context_gates
        ) / len(context_gates)
    for level, indices in enumerate(level_indices):
        model.gl_fusion_runtime[f"p{level + 3}_candidate_count"] = len(indices)
    for reason in set(localization_reasons):
        model.gl_fusion_runtime[f"localization_{reason}_count"] = localization_reasons.count(reason)
    _update_runtime_totals(model)
    return second_output if model.training else _detach_output_tree(second_output)


def _update_runtime_totals(model):
    totals = getattr(model, "gl_fusion_totals", {"forward_count": 0})
    totals["forward_count"] += 1
    for key, value in model.gl_fusion_runtime.items():
        if key.endswith("_count"):
            totals[key] = totals.get(key, 0) + int(value)
    totals["mask_coverage_sum"] = totals.get("mask_coverage_sum", 0.0) + model.gl_fusion_runtime["mask_coverage"]
    totals["gate_mean_sum"] = totals.get("gate_mean_sum", 0.0) + model.gl_fusion_runtime["gate_mean"]
    model.gl_fusion_totals = totals


def _gradcam_global_local_impl(model, images, profile=False, visualize=False):
    if is_pyramid_method(model):
        return _gradcam_global_local_v2_impl(model, images, profile, visualize)
    config = model.gl_fusion_config
    _apply_stage_runtime_modes(model)
    features, detect_head = _features_before_detect(model, images, profile, visualize)
    fusion_level = int(config["fusion_level"])

    # Grad-CAM still needs a differentiable activation when stage 1 freezes all
    # YOLO parameters or when validation is called under no_grad/inference_mode.
    if config["localization"] == "gradcam":
        for level, feature in enumerate(features):
            if not feature.requires_grad:
                features[level] = feature.detach().requires_grad_(True)
    activation = features[fusion_level]

    # Pass 1 is always run with the head in eval mode: it yields decoded boxes,
    # remains differentiable, and does not update Detect BN statistics twice.
    head_was_training = detect_head.training
    detect_head.eval()
    first_output = detect_head([feature.clone() for feature in features])
    if head_was_training and int(config["stage"]) in (0, 1, 4):
        # Restore raw-output behavior without re-enabling frozen child BN layers.
        detect_head.training = True
    else:
        detect_head.train(head_was_training)
    first_decoded = first_output[0] if isinstance(first_output, tuple) else first_output

    candidates = select_candidates(
        first_decoded,
        conf_threshold=float(config["candidate_conf"]),
        iou_threshold=float(config["candidate_iou"]),
        topk=config["candidate_topk"],
        pre_nms_topk=config["pre_nms_topk"],
    )
    image_size: Tuple[int, int] = tuple(images.shape[-2:])
    centers, used_cam, localization_reasons = locate_centers(
        candidates,
        features if config["localization"] == "gradcam" else activation,
        image_size,
        config["localization"],
        return_reasons=True,
        cam_method=config.get("cam_method", "layercam"),
    )

    if candidates:
        patches, crop_boxes, batch_indices = _crop_detail_patches(
            model, images, candidates, centers, adaptive=False
        )
        local_features = model.detail_encoder(patches)
        if local_features.shape[1] != 3:
            raise RuntimeError(f"Detail encoder must return three channels, got {tuple(local_features.shape)}")
        local_canvas, mask = paste_and_average(
            local_features,
            crop_boxes,
            batch_indices,
            image_size=image_size,
            feature_size=tuple(activation.shape[-2:]),
            batch_size=images.shape[0],
        )
    else:
        local_canvas = activation.new_zeros((images.shape[0], 3, *activation.shape[-2:]))
        mask = activation.new_zeros((images.shape[0], 1, *activation.shape[-2:]))

    fused_feature, gate = model.global_local_fusion(activation, local_canvas, mask)
    second_features = [feature.clone() for feature in features]
    second_features[fusion_level] = fused_feature
    second_output = detect_head(second_features)

    model.gl_fusion_runtime = {
        "candidate_count": len(candidates),
        "original_crop_count": len(candidates) if config.get("crop_source") == "original" else 0,
        "gradcam_count": sum(used_cam),
        "fallback_count": len(used_cam) - sum(used_cam),
        "mask_coverage": float(mask.detach().mean()),
        "gate_mean": float(gate.detach().mean()),
    }
    for reason in set(localization_reasons):
        model.gl_fusion_runtime[f"localization_{reason}_count"] = localization_reasons.count(reason)

    # Keep cumulative counters as well. Epoch-end callbacks otherwise see only
    # the last validation batch, which made the previous W&B diagnostics look
    # representative when they were not.
    _update_runtime_totals(model)
    # Validation temporarily enables autograd to obtain CAMs even though no
    # detector gradient is needed afterwards. Returning graph-connected tensors
    # lets the validator retain one graph per batch in its prediction/statistics
    # buffers and caused test-set memory to grow to tens of GB. Training must
    # keep the graph, while evaluation can safely release it here.
    return second_output if model.training else _detach_output_tree(second_output)


def gradcam_global_local_predict_once(model, images, profile=False, visualize=False):
    """DetectionModel._predict_once replacement with Grad-CAM enabled as needed."""
    needs_grad_context = model.gl_fusion_config["localization"] == "gradcam" and not torch.is_grad_enabled()
    inference_context = torch.inference_mode(False) if torch.is_inference_mode_enabled() else contextlib.nullcontext()
    grad_context = torch.enable_grad() if needs_grad_context else contextlib.nullcontext()
    with inference_context, grad_context:
        if torch.is_inference_mode_enabled() or needs_grad_context:
            images = images.detach().clone()
        return _gradcam_global_local_impl(model, images, profile, visualize)


class GradCAMGlobalLocalYOLO(YOLO):
    """YOLO wrapper implementing Grad-CAM-guided spatial feature fusion."""

    def __init__(
        self,
        model,
        detail_weight,
        localization="gradcam",
        cam_method="gradcam",
        fusion_level=0,
        candidate_conf=0.10,
        candidate_iou=0.70,
        candidate_topk=None,
        pre_nms_topk=None,
        crop_size=64,
        crop_source="original",
        alpha_init=0.01,
        method="v1",
        detail_channels=64,
        use_gt_train=True,
        gt_match_iou=0.50,
        aux_positive_iou=0.30,
        aux_weight=0.20,
        source_crop_scale=0.25,
        min_source_size=64,
        max_source_size=192,
        fpn_thresholds=(96.0, 256.0),
        paste_region="crop",
        paste_box_scale=0.5,
        roi_beta_init=0.05,
        quality_rescore=False,
        quality_floor=0.25,
        quality_power=1.0,
        stage=1,
        task="detect",
    ):
        super().__init__(model=model, task=task)
        detect_head = self.model.model[-1]
        if method not in ("v1", "v2", "v3"):
            raise ValueError("method must be v1, v2 or v3")
        if crop_source not in ("original", "tensor"):
            raise ValueError("crop_source must be 'original' or 'tensor'")
        if cam_method not in ("gradcam", "layercam"):
            raise ValueError("cam_method must be 'gradcam' or 'layercam'")
        method_name = f"gradcam_global_local_{method}"
        pyramid_method = method in ("v2", "v3")
        global_channels = (
            [_detect_input_channels(detect_head, level) for level in range(detect_head.nl)]
            if pyramid_method
            else _detect_input_channels(detect_head, fusion_level)
        )

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        owns_fusion_modules = all(
            hasattr(self.model, name)
            for name in ("detail_encoder", "global_local_fusion", "gl_fusion_config")
        )
        loaded_method = self.model.gl_fusion_config.get("method") if owns_fusion_modules else None
        has_trained_fusion = owns_fusion_modules and loaded_method == method_name
        upgrading_v2_to_v3 = (
            method == "v3"
            and owns_fusion_modules
            and loaded_method == "gradcam_global_local_v2"
        )
        if pyramid_method:
            has_trained_fusion = (
                (has_trained_fusion or upgrading_v2_to_v3)
                and hasattr(self.model, "detail_aux_head")
            )
        if has_trained_fusion:
            # A trained fusion checkpoint already owns these modules. Replacing
            # them here would silently discard the learned parameters during
            # validation, fine-tuning, or resume.
            self.detail_load_report = {"source": "checkpoint", "preserved": True}
            self.model.detail_encoder.to(device=device, dtype=dtype)
            self.model.global_local_fusion.to(device=device, dtype=dtype)
            if hasattr(self.model, "detail_aux_head"):
                self.model.detail_aux_head.to(device=device, dtype=dtype)
        else:
            detail_encoder = (
                Detail_Net_attn_semantic(out_channels=detail_channels)
                if pyramid_method
                else Detail_Net_attn_block()
            )
            self.detail_load_report = load_detail_encoder_pretrained(detail_encoder, Path(detail_weight))
            self.model.detail_encoder = detail_encoder.to(device=device, dtype=dtype)
            if pyramid_method:
                self.model.global_local_fusion = PyramidGlobalLocalFusion(
                    global_channels=global_channels,
                    local_channels=detail_channels,
                    alpha_init=alpha_init,
                ).to(device=device, dtype=dtype)
                self.model.detail_aux_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(detail_channels, 1),
                ).to(device=device, dtype=dtype)
            else:
                self.model.global_local_fusion = GlobalLocalFusion(
                    global_channels=global_channels,
                    local_channels=3,
                    alpha_init=alpha_init,
                ).to(device=device, dtype=dtype)

        # Old fused checkpoints were pickled with several Detail layers that
        # never participated in the V1/V2/V3 forward graph. Prune them after
        # loading so both fresh and resumed models contain only effective layers.
        self.model.detail_encoder.prune_legacy_unused_modules()

        if method == "v3":
            self.model.global_local_fusion.enable_roi_context(beta_init=roi_beta_init)
            self.model.global_local_fusion.to(device=device, dtype=dtype)

        requested_config = {
            "method": method_name,
            "localization": localization,
            "cam_method": cam_method,
            "fusion_level": int(fusion_level),
            "fusion_stride": float(detect_head.stride[fusion_level]),
            "global_channels": global_channels,
            "candidate_conf": float(candidate_conf),
            "candidate_iou": float(candidate_iou),
            "candidate_topk": None if candidate_topk is None or int(candidate_topk) <= 0 else int(candidate_topk),
            "pre_nms_topk": None if pre_nms_topk is None or int(pre_nms_topk) <= 0 else int(pre_nms_topk),
            "crop_size": int(crop_size),
            "crop_source": str(crop_source),
            "alpha_init": float(alpha_init),
            "detail_channels": int(detail_channels if pyramid_method else 3),
            "use_gt_train": bool(use_gt_train),
            "gt_match_iou": float(gt_match_iou),
            "aux_positive_iou": float(aux_positive_iou),
            "aux_weight": float(aux_weight),
            "source_crop_scale": float(source_crop_scale if pyramid_method else 0.0),
            "min_source_size": int(min_source_size),
            "max_source_size": int(max_source_size),
            "fpn_thresholds": tuple(float(value) for value in fpn_thresholds),
            "paste_region": str(paste_region),
            "paste_box_scale": float(paste_box_scale),
            "roi_beta_init": float(roi_beta_init),
            "quality_rescore": bool(quality_rescore),
            "quality_floor": float(quality_floor),
            "quality_power": float(quality_power),
            "stage": int(stage),
        }
        if has_trained_fusion:
            self.model.gl_fusion_config = {
                **self.model.gl_fusion_config,
                **requested_config,
            }
        else:
            self.model.gl_fusion_config = requested_config
        # Use a class-defined wrapper so EMA checkpoints can pickle and reload
        # this dynamically selected forward path by method name.
        self.model._predict_once = self.model.gradcam_global_local_predict_once
        self.model.is_gradcam_global_local = True
        self.configure_stage(stage)

        LOGGER.info(
            f"Grad-CAM Global-Local {method.upper()}: "
            f"levels={detect_head.nl}, fusion_level={fusion_level}, "
            f"channels={global_channels}, "
            f"localization={localization}, cam_method={cam_method}, "
            f"candidate_topk={self.model.gl_fusion_config['candidate_topk'] or 'all'}, "
            f"pre_nms_topk={self.model.gl_fusion_config['pre_nms_topk'] or 'all'}, "
            f"crop_source={crop_source}"
        )

    def configure_stage(self, stage: int):
        """Apply the documented staged-freezing policy and print trainable names."""
        stage = int(stage)
        for parameter in self.model.parameters():
            parameter.requires_grad_(stage == 3)

        if stage == 0:
            # Warm up only the randomly initialized Detail output adapters and
            # the new gated fusion module before fine-tuning the pretrained
            # Detail encoder at a lower learning rate.
            stage_zero_modules = (
                (self.model.detail_encoder.semantic_head, self.model.detail_aux_head)
                if is_pyramid_method(self.model)
                else tuple(getattr(self.model.detail_encoder, name) for name in ("conv_l", "conv_m", "conv_s", "bn_cat"))
            )
            for module in stage_zero_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.model.global_local_fusion.parameters():
                parameter.requires_grad_(True)
        if stage == 4:
            if config_method(self.model) != "gradcam_global_local_v3":
                raise ValueError("stage 4 is the V3 ROI-adapter warm-up stage")
            roi_modules = [
                self.model.global_local_fusion.roi_context_norm,
                self.model.global_local_fusion.roi_context_gate,
            ]
            for module in roi_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            self.model.global_local_fusion.roi_beta.requires_grad_(True)
        if stage in (1, 2):
            modules = [self.model.detail_encoder, self.model.global_local_fusion]
            if hasattr(self.model, "detail_aux_head"):
                modules.append(self.model.detail_aux_head)
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if stage == 2:
            detect_head = self.model.model[-1]
            levels = range(detect_head.nl) if is_pyramid_method(self.model) else (self.model.gl_fusion_config["fusion_level"],)
            modules = [self.model.model[detect_head.f[level]] for level in levels] + [detect_head]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if stage not in (0, 1, 2, 3, 4):
            raise ValueError("stage must be 0, 1, 2, 3 or 4")

        # DFL's 1x1 convolution stores the fixed [0, ..., reg_max - 1]
        # expectation vector. It is part of inference, but is not learnable.
        detect_head = self.model.model[-1]
        dfl_conv = getattr(getattr(detect_head, "dfl", None), "conv", None)
        if dfl_conv is not None:
            for parameter in dfl_conv.parameters():
                parameter.requires_grad_(False)

        self.model.gl_fusion_config["stage"] = stage
        trainable = [(name, parameter.numel()) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("The selected training stage did not enable any parameters")
        LOGGER.info(f"Stage {stage} trainable parameters: {sum(count for _, count in trainable):,}")
        LOGGER.info("Trainable tensors:\n  " + "\n  ".join(name for name, _ in trainable))

    def train(self, **kwargs):
        """Train the already assembled two-pass model without rebuilding plain YOLO from YAML."""
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
            for event, event_callbacks in self.trainer.callbacks.items():
                self.trainer.callbacks[event] = [
                    callback
                    for callback in event_callbacks
                    if callback.__module__ != "ultralytics.yolo.utils.callbacks.wb"
                ]
        self.trainer.model = self.model
        self.trainer.hub_session = self.session
        self.trainer.train()
        if RANK in (-1, 0) and Path(self.trainer.best).exists():
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)

    def val(self, data=None, **kwargs):
        """Use the standard detector validator with gradients available for Grad-CAM."""
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
