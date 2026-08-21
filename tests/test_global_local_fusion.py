"""Unit tests for Grad-CAM Global-Local Fusion (not an experiment runner)."""

import numpy as np
import pytest
import torch

from detail_model import (
    CompactSpatialAttention,
    Detail_Net_attn,
    Detail_Net_attn_block,
    Detail_Net_attn_semantic,
)
from gradcam_fusion import load_detail_encoder_pretrained
from global_local_fusion import (
    Candidate,
    GlobalLocalFusion,
    PyramidGlobalLocalFusion,
    candidate_fpn_level,
    crop_original_patches,
    crop_tensor_patches,
    gradcam_center,
    locate_centers,
    paste_and_average,
    roi_align_tensor,
    scale_and_clip_boxes,
    select_candidates,
)


def test_detail_encoder_shape_and_finite_values():
    encoder = Detail_Net_attn_block().eval()
    output = encoder(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 3, 4, 4)
    assert torch.isfinite(output).all()


def test_detail_encoder_contains_only_effective_parameters():
    encoder = Detail_Net_attn_block()
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 76_945
    assert not any(hasattr(encoder, name) for name in encoder._LEGACY_UNUSED_MODULES)

    encoder(torch.randn(2, 3, 64, 64)).sum().backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_compact_attention_uses_valid_dilations_for_four_by_four_features():
    attention = CompactSpatialAttention(8)
    assert attention.conv1.dilation == (1, 1)
    assert attention.conv2.dilation == (2, 2)
    assert not hasattr(attention, "conv3")
    assert not hasattr(attention, "conv4")


def test_zip_classifier_checkpoint_preloads_compressed_encoder(tmp_path):
    weight = tmp_path / "zip_detail.pt"
    torch.save({"state_dict": Detail_Net_attn().state_dict()}, weight)
    report = load_detail_encoder_pretrained(Detail_Net_attn_block(), weight)
    assert report["pretrained_coverage"] == 1.0
    assert not report["missing_required"]


def test_historical_wide_checkpoint_is_rejected():
    weight = "run/detail_net_atten/exp3_4_1.pt"
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_detail_encoder_pretrained(Detail_Net_attn_block(), weight)


def test_legacy_detail_modules_are_pruned_after_checkpoint_load():
    encoder = Detail_Net_attn_block()
    encoder.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
    assert hasattr(encoder, "conv1")
    assert encoder.prune_legacy_unused_modules() is encoder
    assert not hasattr(encoder, "conv1")


def test_empty_canvas_is_zero_and_mask_is_empty():
    local = torch.empty(0, 3, 4, 4)
    canvas, mask = paste_and_average(
        local,
        crop_boxes=torch.empty(0, 4),
        batch_indices=torch.empty(0, dtype=torch.long),
        image_size=(64, 64),
        feature_size=(8, 8),
        batch_size=2,
    )
    assert canvas.shape == (2, 3, 8, 8)
    assert mask.shape == (2, 1, 8, 8)
    assert torch.count_nonzero(canvas) == 0
    assert torch.count_nonzero(mask) == 0


def test_overlapping_local_features_are_averaged():
    local = torch.stack((torch.ones(3, 4, 4), torch.full((3, 4, 4), 3.0)))
    boxes = torch.tensor([[0.0, 0.0, 64.0, 64.0], [0.0, 0.0, 64.0, 64.0]])
    canvas, mask = paste_and_average(
        local,
        crop_boxes=boxes,
        batch_indices=torch.tensor([0, 0]),
        image_size=(64, 64),
        feature_size=(8, 8),
        batch_size=1,
    )
    assert torch.allclose(canvas, torch.full_like(canvas, 2.0))
    assert torch.all(mask == 1)


def test_boundary_crop_is_padded_to_requested_size():
    images = torch.rand(1, 3, 32, 32)
    candidate = Candidate(0, 0, 0, torch.tensor(0.8), torch.tensor([0.0, 0.0, 8.0, 8.0]))
    patches, boxes, batch_indices = crop_tensor_patches(
        images,
        candidates=[candidate],
        centers=[torch.tensor([1.0, 1.0])],
        crop_size=64,
    )
    assert patches.shape == (1, 3, 64, 64)
    assert boxes.shape == (1, 4)
    assert batch_indices.tolist() == [0]
    assert torch.isfinite(patches).all()


def test_zero_mask_keeps_global_feature_unchanged():
    module = GlobalLocalFusion(global_channels=8).eval()
    global_feature = torch.randn(2, 8, 10, 10)
    canvas = torch.randn(2, 3, 10, 10)
    mask = torch.zeros(2, 1, 10, 10)
    fused, gate = module(global_feature, canvas, mask)
    assert torch.equal(fused, global_feature)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_fusion_path_receives_finite_gradients():
    module = GlobalLocalFusion(global_channels=8)
    global_feature = torch.randn(2, 8, 10, 10)
    canvas = torch.randn(2, 3, 10, 10, requires_grad=True)
    mask = torch.ones(2, 1, 10, 10)
    fused, _ = module(global_feature, canvas, mask)
    fused.square().mean().backward()
    assert canvas.grad is not None and torch.isfinite(canvas.grad).all()
    assert module.local_proj[0].weight.grad is not None
    assert module.gate_conv.weight.grad is not None
    assert module.alpha.grad is not None


def test_layercam_localization_is_nonflat_and_inside_candidate_box():
    activation = torch.linspace(0.1, 1.0, 64).reshape(1, 1, 8, 8).requires_grad_(True)
    score = activation[0, 0, 2:6, 2:6].square().sum()
    candidate = Candidate(
        batch_index=0,
        anchor_index=0,
        class_index=0,
        score=score,
        box_xyxy=torch.tensor([16.0, 16.0, 48.0, 48.0]),
    )
    center, valid, reason = gradcam_center(
        candidate,
        activation,
        image_size=(64, 64),
        return_reason=True,
    )
    assert valid
    assert reason == "layercam"
    assert 16.0 <= float(center[0]) <= 48.0
    assert 16.0 <= float(center[1]) <= 48.0


def test_classic_gradcam_uses_global_average_gradient_weights():
    activation = torch.linspace(0.1, 1.0, 64).reshape(1, 1, 8, 8).requires_grad_(True)
    score = activation[0, 0, 2:6, 2:6].sum()
    candidate = Candidate(0, 0, 0, score, torch.tensor([16.0, 16.0, 48.0, 48.0]))
    center, valid, reason = gradcam_center(
        candidate,
        activation,
        image_size=(64, 64),
        return_reason=True,
        cam_method="gradcam",
    )
    assert valid
    assert reason == "gradcam"
    assert 16.0 <= float(center[0]) <= 48.0
    assert 16.0 <= float(center[1]) <= 48.0


def test_candidate_selection_can_keep_all_candidates_without_topk_limits():
    decoded = torch.zeros(1, 5, 3)
    decoded[0, :4] = torch.tensor(
        [[10.0, 30.0, 50.0], [10.0, 30.0, 50.0], [8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]
    )
    decoded[0, 4] = torch.tensor([0.9, 0.8, 0.7])
    candidates = select_candidates(decoded, conf_threshold=0.5, topk=None, pre_nms_topk=None)
    assert len(candidates) == 3


def test_multiscale_cam_uses_the_feature_level_that_produced_the_anchor():
    p3 = torch.ones(1, 1, 4, 4, requires_grad=True)
    p4 = torch.linspace(0.1, 1.0, 4).reshape(1, 1, 2, 2).requires_grad_(True)
    score = p4.square().sum()
    candidate = Candidate(
        batch_index=0,
        anchor_index=16,  # first anchor after the 4x4 P3 range
        class_index=0,
        score=score,
        box_xyxy=torch.tensor([0.0, 0.0, 64.0, 64.0]),
    )
    _, used_cam, reasons = locate_centers(
        [candidate],
        [p3, p4],
        image_size=(64, 64),
        mode="gradcam",
        return_reasons=True,
    )
    assert used_cam == [True]
    assert reasons == ["layercam"]


def test_gt_candidate_uses_box_center_without_requesting_gradcam():
    activation = torch.randn(1, 4, 8, 8, requires_grad=True)
    candidate = Candidate(
        0,
        -1,
        0,
        torch.tensor(0.0),
        torch.tensor([10.0, 20.0, 30.0, 60.0]),
        source="gt",
    )
    centers, used_cam, reasons = locate_centers(
        [candidate], activation, (64, 64), "gradcam", return_reasons=True
    )
    assert torch.allclose(centers[0], torch.tensor([20.0, 40.0]))
    assert used_cam == [False]
    assert reasons == ["gt_center"]


def test_adaptive_source_crop_is_resized_to_encoder_resolution():
    images = torch.rand(1, 3, 640, 640)
    candidate = Candidate(
        0,
        0,
        0,
        torch.tensor(0.8),
        torch.tensor([100.0, 100.0, 500.0, 500.0]),
    )
    patches, boxes, _ = crop_tensor_patches(
        images,
        [candidate],
        [torch.tensor([300.0, 300.0])],
        crop_size=64,
        source_scale=0.25,
        min_source_size=64,
        max_source_size=192,
    )
    assert patches.shape == (1, 3, 64, 64)
    assert torch.equal(boxes[0], torch.tensor([250.0, 250.0, 350.0, 350.0]))


def test_gradcam_center_maps_to_the_correct_original_image_roi():
    original = np.zeros((100, 200, 3), dtype=np.uint8)
    original[24:27, 149:152] = 255
    images = torch.zeros(1, 3, 640, 640)
    candidate = Candidate(
        0,
        0,
        0,
        torch.tensor(0.8),
        torch.tensor([440.0, 200.0, 520.0, 280.0]),
    )
    # Original (150, 25) -> resize by 3.2 and add 160 px top padding.
    patches, boxes, batch_indices = crop_original_patches(
        images,
        [original],
        [((3.2, 3.2), (0.0, 160.0))],
        [candidate],
        [torch.tensor([480.0, 240.0])],
        crop_size=20,
    )

    assert patches.shape == (1, 3, 20, 20)
    assert batch_indices.tolist() == [0]
    assert torch.allclose(boxes[0], torch.tensor([448.0, 208.0, 512.0, 272.0]))
    peak = patches[0].sum(dim=0).argmax()
    peak_y, peak_x = divmod(int(peak), 20)
    assert 9 <= peak_x <= 11
    assert 9 <= peak_y <= 11


def test_original_crop_rejects_noninvertible_mosaic_metadata():
    images = torch.zeros(1, 3, 640, 640)
    original = np.zeros((100, 200, 3), dtype=np.uint8)
    candidate = Candidate(0, 0, 0, torch.tensor(0.8), torch.tensor([0.0, 0.0, 20.0, 20.0]))
    with pytest.raises(ValueError, match="pure LetterBox"):
        crop_original_patches(
            images,
            [original],
            [(((3.2, 3.2), (0.0, 160.0)), (0.0, 0.0))],
            [candidate],
            [torch.tensor([10.0, 10.0])],
            crop_size=20,
        )


def test_candidates_are_assigned_to_scale_appropriate_fpn_levels():
    def candidate(size):
        return Candidate(0, 0, 0, torch.tensor(0.8), torch.tensor([0.0, 0.0, size, size]))

    assert candidate_fpn_level(candidate(64.0), (640, 640)) == 0
    assert candidate_fpn_level(candidate(160.0), (640, 640)) == 1
    assert candidate_fpn_level(candidate(320.0), (640, 640)) == 2


def test_semantic_detail_encoder_retains_feature_channels():
    encoder = Detail_Net_attn_semantic(out_channels=64).eval()
    output = encoder(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 64, 4, 4)
    assert torch.isfinite(output).all()


def test_pyramid_fusion_has_identity_mask_and_gradients_per_level():
    module = PyramidGlobalLocalFusion((16, 32, 64), local_channels=8)
    for level, channels in enumerate((16, 32, 64)):
        global_feature = torch.randn(1, channels, 8, 8)
        local_feature = torch.randn(1, 8, 4, 4, requires_grad=True)
        projected = module.project(level, local_feature)
        projected = torch.nn.functional.interpolate(projected, size=(8, 8), mode="bilinear")
        zero_mask = torch.zeros(1, 1, 8, 8)
        identity, _ = module.fuse(level, global_feature, projected, zero_mask)
        assert torch.equal(identity, global_feature)

        one_mask = torch.ones_like(zero_mask)
        fused, _ = module.fuse(level, global_feature, projected, one_mask)
        fused.square().mean().backward()
        assert local_feature.grad is not None and torch.isfinite(local_feature.grad).all()
        assert module.alpha.grad is not None and torch.isfinite(module.alpha.grad[level])


def test_candidate_destination_boxes_are_scaled_and_clipped():
    boxes = torch.tensor([[100.0, 100.0, 500.0, 500.0], [-20.0, -10.0, 100.0, 90.0]])
    scaled = scale_and_clip_boxes(boxes, scale=0.5, image_size=(640, 640))
    assert torch.allclose(scaled[0], torch.tensor([200.0, 200.0, 400.0, 400.0]))
    assert torch.allclose(scaled[1], torch.tensor([10.0, 15.0, 70.0, 65.0]))


def test_roi_align_tensor_preserves_batch_identity_and_gradients():
    features = torch.stack((torch.ones(3, 8, 8), torch.full((3, 8, 8), 2.0))).requires_grad_(True)
    boxes = torch.tensor([[0.0, 0.0, 64.0, 64.0], [16.0, 16.0, 48.0, 48.0]])
    rois = roi_align_tensor(
        features,
        boxes,
        torch.tensor([0, 1]),
        image_size=(64, 64),
        output_size=(4, 4),
    )
    assert rois.shape == (2, 3, 4, 4)
    assert torch.allclose(rois[0], torch.ones_like(rois[0]))
    assert torch.allclose(rois[1], torch.full_like(rois[1], 2.0))
    rois.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_roi_context_adapter_is_detection_aware_and_trainable():
    module = PyramidGlobalLocalFusion((16, 32, 64), local_channels=8)
    module.enable_roi_context(beta_init=0.05)
    local = torch.randn(2, 8, 4, 4, requires_grad=True)
    context = torch.randn(2, 32, 4, 4, requires_grad=True)
    aligned, gate = module.project_with_context(1, local, context)
    assert aligned.shape == context.shape
    assert torch.all((gate >= 0) & (gate <= 1))
    aligned.square().mean().backward()
    assert local.grad is not None and torch.isfinite(local.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert module.roi_beta.grad is not None and float(module.roi_beta.grad[1].abs()) > 0
    assert module.roi_context_gate[1].weight.grad is not None
