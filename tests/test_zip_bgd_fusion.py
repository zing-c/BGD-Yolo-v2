"""Small regression tests for the complete ZIP BGD method."""

import torch

from global_local_fusion import Candidate
from zip_bgd_fusion import (
    ZipCandidateScoreFusion,
    ZipBoundaryCalibratedRescueFusion,
    ZipDirectFusion,
    aggregate_class_gradcam,
    locate_zip_centers,
    paste_fixed_detail,
    select_all_raw_candidates,
    select_grouped_rescue_candidates,
    select_novel_low_candidates,
)


def _decoded_predictions():
    decoded = torch.zeros(2, 5, 6)
    decoded[:, :4] = torch.tensor((32.0, 32.0, 20.0, 20.0)).view(1, 4, 1)
    decoded[0, 4, 1:3] = torch.tensor((0.2, 0.3))
    decoded[1, 4, 4] = 0.4
    return decoded


def test_all_candidates_has_no_nms_or_topk():
    candidates = select_all_raw_candidates(_decoded_predictions(), 0.1)
    assert [(item.batch_index, item.anchor_index) for item in candidates] == [(0, 1), (0, 2), (1, 4)]


def test_duplicate_centers_are_removed_per_image():
    candidates = select_all_raw_candidates(_decoded_predictions(), 0.1)
    cam = torch.zeros(2, 1, 8, 8)
    cam[:, :, 4, 4] = 1
    unique, centers = locate_zip_centers(cam, candidates, (64, 64), crop_size=8)
    assert len(unique) == len(centers) == 2


def test_score_fusion_keeps_all_candidates_and_starts_at_identity():
    candidates = select_all_raw_candidates(_decoded_predictions(), 0.1)
    cam = torch.zeros(2, 1, 8, 8)
    all_candidates, centers = locate_zip_centers(
        cam, candidates, (64, 64), crop_size=8, deduplicate=False
    )
    fusion = ZipCandidateScoreFusion()
    detail = torch.tensor((0.2, 0.5, 0.8), requires_grad=True)
    delta = fusion(
        torch.stack([item.score for item in all_candidates]),
        detail,
        torch.tensor((0, 1, 2)),
    )
    assert len(all_candidates) == len(centers) == len(candidates)
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert fusion.adapter[-1].weight.grad is not None


def test_boundary_rescue_threshold_is_calibratable_without_changing_quality_order():
    fusion = ZipBoundaryCalibratedRescueFusion()
    with torch.no_grad():
        fusion.quality_head[-1].bias.fill_(-0.4)
    inputs = (
        torch.tensor((0.49,)), torch.tensor((0.8,)), torch.tensor((1,)),
        torch.zeros(1, 4),
    )
    strict_delta, strict_logits = fusion(*inputs, quality_threshold=0.5)
    calibrated_delta, calibrated_logits = fusion(*inputs, quality_threshold=0.01)
    assert strict_delta.item() == 0
    assert calibrated_delta.item() > 0
    assert torch.equal(strict_logits, calibrated_logits)


def test_grouped_rescue_keeps_only_best_novel_hypothesis():
    def candidate(index, box, score):
        return Candidate(
            batch_index=0, class_index=0, score=torch.tensor(score),
            box_xyxy=torch.tensor(box, dtype=torch.float32), anchor_index=index,
        )

    candidates = [
        candidate(0, (0, 0, 10, 10), 0.40),
        candidate(1, (1, 1, 11, 11), 0.45),
        candidate(2, (30, 30, 40, 40), 0.30),
        candidate(3, (31, 31, 41, 41), 0.80),
    ]
    selected = select_grouped_rescue_candidates(
        candidates,
        torch.tensor((0.40, 0.45, 0.30, 0.80)),
        torch.tensor((0.10, 0.90, 0.80, 0.20)),
    )
    # Candidate 1 wins its low-confidence group. Candidate 2 is redundant
    # because it already overlaps the untouched high-confidence candidate 3.
    assert selected.tolist() == [False, True, False, False]


def test_novel_low_candidates_exclude_boxes_covered_by_deployed_detection():
    def candidate(index, batch, class_index, box, score):
        return Candidate(
            batch_index=batch, class_index=class_index, score=torch.tensor(score),
            box_xyxy=torch.tensor(box, dtype=torch.float32), anchor_index=index,
        )

    candidates = [
        candidate(0, 0, 0, (0, 0, 10, 10), 0.40),
        candidate(1, 0, 0, (1, 1, 11, 11), 0.80),
        candidate(2, 0, 0, (30, 30, 40, 40), 0.45),
        candidate(3, 0, 1, (1, 1, 11, 11), 0.30),
        candidate(4, 1, 0, (1, 1, 11, 11), 0.20),
    ]
    selected = select_novel_low_candidates(
        candidates, torch.tensor((0.40, 0.80, 0.45, 0.30, 0.20))
    )
    # Candidate 0 is redundant only within the same image/class. Candidate 2
    # is spatially novel; candidates 3 and 4 belong to different groups.
    assert selected.tolist() == [False, False, True, True, True]


def test_novel_low_candidates_respect_rescue_confidence_floor():
    candidates = [
        Candidate(batch_index=0, anchor_index=0, class_index=0, score=torch.tensor(0.20), box_xyxy=torch.tensor((0, 0, 10, 10), dtype=torch.float32)),
        Candidate(batch_index=0, anchor_index=1, class_index=0, score=torch.tensor(0.35), box_xyxy=torch.tensor((20, 20, 30, 30), dtype=torch.float32)),
        Candidate(batch_index=0, anchor_index=2, class_index=0, score=torch.tensor(0.50), box_xyxy=torch.tensor((40, 40, 50, 50), dtype=torch.float32)),
    ]
    selected = select_novel_low_candidates(
        candidates, torch.tensor((0.20, 0.35, 0.50)), min_conf=0.25,
    )
    assert selected.tolist() == [False, True, False]


def test_fixed_paste_and_direct_fusion_are_differentiable():
    candidates = select_all_raw_candidates(_decoded_predictions(), 0.1)
    cam = torch.zeros(2, 1, 8, 8)
    unique, centers = locate_zip_centers(cam, candidates, (64, 64), crop_size=8)
    local = torch.randn(len(unique), 3, 4, 4, requires_grad=True)
    canvas = paste_fixed_detail(local, unique, centers, (64, 64), (8, 8), 2)
    fused = ZipDirectFusion(64)(torch.randn(2, 64, 8, 8), canvas)
    fused.mean().backward()
    assert canvas.shape == (2, 3, 8, 8)
    assert local.grad is not None and torch.isfinite(local.grad).all()


def test_classic_gradcam_shape_and_finiteness():
    activation = torch.randn(2, 64, 4, 4, requires_grad=True)
    projection = torch.randn(1, 64, 1, 1)
    scores = torch.sigmoid(torch.nn.functional.conv2d(activation, projection)).flatten(2)
    boxes = torch.ones(2, 4, scores.shape[-1])
    decoded = torch.cat((boxes, scores), dim=1)
    cam, reason = aggregate_class_gradcam(decoded, activation)
    assert reason in ("gradcam", "flat_cam")
    assert cam.shape == (2, 1, 4, 4)
    assert torch.isfinite(cam).all()
