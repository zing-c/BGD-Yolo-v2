"""Candidate-level Grad-CAM, Detail and YOLO ROI feature fusion.

The original ZIP implementation pastes Detail maps over a whole P5 feature
map and consequently changes many unrelated anchors.  This variant keeps the
pretrained detector prediction as the identity path.  Every raw candidate
above the internal threshold is localized with classic Grad-CAM, encoded by
the Detail network, aligned with its same-scale YOLO ROI, and corrected only
at that candidate's class logit and box.
"""

from __future__ import annotations

import contextlib
import math
import os
import weakref
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detail_model import DETAIL_ARCH_VERSION, Detail_Net_attn, Detail_Net_attn_block
from global_local_fusion import (
    Candidate,
    candidate_fpn_level,
    class_aware_nms,
    crop_original_patches,
    crop_tensor_patches,
    roi_align_tensor,
)
from gradcam_fusion import _detach_output_tree, _features_before_detect, load_detail_encoder_pretrained
from ultralytics import YOLO, yolo
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.metrics import bbox_iou, box_iou
from ultralytics.yolo.utils.torch_utils import scale_img
from zip_bgd_fusion import _decode_raw, select_all_raw_candidates


CANDIDATE_ROI_METHOD = "gradcam_detail_candidate_roi_v77_factorized_semantic_quality_fusion"


def _primary_recall_supported(candidate: Candidate) -> bool:
    """Return whether a primary proposal has independent geometric support.

    v53 required both transformed views to agree with the canonical proposal.
    That recovered only GTs already emitted by the secondary-pair branch.  A
    single independent transformed view is still genuine cross-view evidence,
    so v54 admits two-of-three support while retaining an environment switch
    for an exact three-view ablation.
    """
    if os.environ.get("BGD_REQUIRE_THREE_VIEW_RESCUE") == "1":
        return bool(candidate.consensus)
    return int(candidate.view_support) >= 2


def _rescue_quality_support(quality_logits: torch.Tensor) -> torch.Tensor:
    """Return continuous joint object/localization support for recall rescue.

    The two logits are already learned from the fused Grad-CAM Detail 4x4 and
    same-scale YOLO ROI representation.  Multiplying their probabilities is a
    parameter-free soft AND: a candidate receives a large positive residual
    only when the shared representation predicts both that it is an object
    and that its box is well localized.  This changes no confidence threshold
    and retains an exact v54 ablation for causal validation.
    """
    if os.environ.get("BGD_DISABLE_RESCUE_QUALITY_PRODUCT") == "1":
        return quality_logits.new_ones((len(quality_logits),))
    probabilities = quality_logits.float().sigmoid()
    return (probabilities[:, 0] * probabilities[:, 1]).to(
        dtype=quality_logits.dtype
    )


def _candidate_deployment_source_box(
    candidate: Candidate,
    index: int,
    cross_view: torch.Tensor,
    semantic_detail: torch.Tensor | None,
    config: Dict,
) -> torch.Tensor:
    """Return the source geometry shared by training and deployment.

    Low-confidence recall proposals need the complete multiview localization
    consensus.  For an already accepted primary detection, however, replacing
    a strong canonical box unconditionally can turn a baseline TP into a near
    miss when one transformed view is weak.  Blend canonical and consensus
    geometry continuously using two independent signals already produced by
    the method: Detail object probability and the weakest supported view
    score.  Their probabilistic union is square-root balanced so strong
    agreement still receives nearly the full multiview localization benefit.
    No new parameter or decision threshold is introduced.
    """
    canonical = candidate.box_xyxy
    fused = candidate.fused_box_xyxy
    if fused is None:
        return canonical
    deployment_conf = float(config.get("deployment_conf", 0.50))
    if candidate.source == "secondary":
        return fused if candidate.emit else canonical
    if candidate.source != "prediction":
        return fused if candidate.consensus else canonical
    if not _primary_recall_supported(candidate):
        return canonical
    if float(candidate.score.detach()) < deployment_conf:
        return fused
    if os.environ.get("BGD_V55_FULL_MULTIVIEW_GEOMETRY") == "1":
        return fused

    cluster_iou = float(config.get("consensus_cluster_iou", 0.50))
    supported_scores = [
        cross_view[index, offset].float().clamp(0, 1)
        for offset in (0, 6)
        if float(cross_view[index, offset + 1]) >= cluster_iou
    ]
    if not supported_scores:
        return canonical
    weakest_view = torch.stack(supported_scores).amin()
    detail_probability = (
        semantic_detail[index, -1].float().clamp(0, 1)
        if semantic_detail is not None else weakest_view.new_zeros(())
    )
    semantic_or_view = 1.0 - (
        1.0 - detail_probability
    ) * (1.0 - weakest_view)
    reliability = semantic_or_view.clamp(0, 1).sqrt().to(dtype=canonical.dtype)
    return canonical + reliability * (fused.to(canonical.dtype) - canonical)


# A distillation teacher is a training-only loss provider.  Keeping it as a
# Module attribute makes PyTorch pickle it into every student checkpoint even
# when it is deliberately not registered in ``model.parameters()``.  A weak
# external registry keeps the runtime association without inflating or
# changing the deployable student network.
_CANDIDATE_ROI_TEACHERS = weakref.WeakKeyDictionary()


def register_candidate_roi_teacher(model: nn.Module, teacher: nn.Module | None) -> None:
    if teacher is None:
        _CANDIDATE_ROI_TEACHERS.pop(model, None)
    else:
        _CANDIDATE_ROI_TEACHERS[model] = teacher


def get_candidate_roi_teacher(model: nn.Module) -> nn.Module | None:
    return _CANDIDATE_ROI_TEACHERS.get(model)


class LightweightCrossAttention(nn.Module):
    """Cross-attend YOLO ROI tokens to Detail tokens with an identity-safe residual."""

    def __init__(self, channels: int, attention_dim: int = 32, heads: int = 4):
        super().__init__()
        channels, attention_dim, heads = int(channels), int(attention_dim), int(heads)
        if attention_dim % heads:
            raise ValueError("attention_dim must be divisible by heads")
        self.query_projection = nn.Linear(channels, attention_dim)
        self.detail_projection = nn.Linear(channels, attention_dim)
        self.query_norm = nn.LayerNorm(attention_dim)
        self.detail_norm = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim, heads, dropout=0.0, batch_first=True
        )
        self.output_projection = nn.Linear(attention_dim, channels)
        # tanh(0) is exactly zero: adding this module cannot perturb the
        # inherited detector before supervised optimization moves the scale.
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        global_feature: torch.Tensor,
        local_feature: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = global_feature.shape
        global_tokens = global_feature.flatten(2).transpose(1, 2)
        local_tokens = local_feature.flatten(2).transpose(1, 2)
        query = self.query_norm(self.query_projection(global_tokens))
        detail = self.detail_norm(self.detail_projection(local_tokens))
        attended, _ = self.attention(
            query, detail, detail, need_weights=False
        )
        residual = self.output_projection(attended).transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return identity + self.residual_scale.tanh() * residual


class CandidateROIFusion(nn.Module):
    """Small same-scale fusion with spatial and multiview set alignment.

    A deployed primary detection and a low-confidence recall proposal have
    opposite residual constraints: the former may only be suppressed while
    the latter may only be promoted.  Sharing their final logit caused the
    positive rescue gradients to dominate hard-background rejection.  The
    common multimodal representation is therefore retained, but the two
    decisions use independent zero-initialized heads.  Mean/max pooling alone
    discarded where the Grad-CAM-centred Detail response agreed with the YOLO
    ROI.  A 4x4 cross-modal attention map now preserves that correspondence and
    pools fused, local, global and cosine-alignment maps at the same locations.

    Two shared quality logits learn objectness and localization IoU directly
    from every unambiguous raw candidate.  A second lightweight set encoder
    compares the candidate embedding with score/IoU-weighted embeddings from
    the two independent views.  Deployment requires the single-candidate and
    set branches to agree: a baseline detection is rejected only when both
    call it background, and a missed candidate is rescued only when both call
    it foreground.  This protects the strong pretrained path while retaining
    genuine multiview recall.  All output heads remain zero initialized,
    preserving exact YOLO identity at fresh initialization.
    """

    def __init__(
        self,
        global_channels: Sequence[int],
        hidden_channels: int = 16,
        local_channels: int = 3,
        semantic_features: int = 0,
        cross_view_features: int = 6,
        transformer_fusion: bool = False,
        transformer_dim: int = 32,
        transformer_heads: int = 4,
    ):
        super().__init__()
        hidden_channels = int(hidden_channels)
        groups = 4 if hidden_channels % 4 == 0 else 1
        self.local_projection = nn.Sequential(
            nn.Conv2d(int(local_channels), hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.global_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(int(channels), hidden_channels, 1, bias=False),
                nn.GroupNorm(groups, hidden_channels),
                nn.SiLU(inplace=True),
            )
            for channels in global_channels
        ])
        self.cross_gate = nn.ModuleList([
            nn.Conv2d(2 * hidden_channels, hidden_channels, 1) for _ in global_channels
        ])
        self.cross_attention_fusion = (
            LightweightCrossAttention(
                hidden_channels,
                attention_dim=transformer_dim,
                heads=transformer_heads,
            )
            if transformer_fusion else None
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(3 * hidden_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.semantic_projection = (
            nn.Sequential(
                nn.Linear(int(semantic_features), hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(inplace=True),
            )
            if semantic_features else None
        )
        # Mean/max pooled fused/local/global maps plus attention-pooled fused,
        # local, global and cosine-alignment maps, optional pretrained Detail
        # semantics, base logit, geometry and level one-hot.
        self.cross_view_features = int(cross_view_features)
        input_features = (
            hidden_channels * (10 + int(self.semantic_projection is not None))
            + 1 + 5 + len(global_channels) + self.cross_view_features
        )
        self.trunk = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 16),
            nn.SiLU(inplace=True),
        )
        # Context is intentionally outside the shared role trunk.  v27 showed
        # that scene signals useful for rescue can weaken the precise keep
        # rejection learned from object ROI/Detail evidence.  This independent
        # 16-D path is consumed only by the component rescue classifier.
        context_features = 16 + hidden_channels * 6
        self.context_trunk = nn.Sequential(
            nn.Linear(context_features, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 16),
            nn.SiLU(inplace=True),
        )
        # Independent scene/2x-context expert for conservative suppression.
        # It may veto a locally background-looking crop when the surrounding
        # detector representation remains object-like.  This is intentionally
        # a single 16->1 layer (17 parameters) and is zero initialized so an
        # upgraded checkpoint starts on the original identity path.
        self.keep_context_head = nn.Linear(16, 1)
        self.keep_head = nn.Linear(16, 1)
        self.rescue_head = nn.Linear(16, 1)
        self.box_head = nn.Linear(16, 4)
        # Pooling the 4x4 Detail/YOLO maps erased left/right and up/down
        # information, so the old box head could only learn a near-zero
        # dataset-average residual. Give localization an explicit coordinate
        # path: Grad-CAM centre offset, crop scale and the first two moments of
        # cross-modal spatial attention. This adds only 484 parameters.
        self.spatial_box_head = nn.Sequential(
            nn.Linear(16 + 5 + 4, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 4),
        )
        self.quality_head = nn.Linear(16, 2)
        # Learn how the candidate-level experts should be combined instead of
        # requiring a brittle unanimous hard gate.  The inputs are all model
        # evidence produced from the Grad-CAM screenshot, Detail route map,
        # spatially aligned YOLO ROI, scene context and independent views.
        # Role-specific outputs preserve the directional residual contract:
        # output 0 can only suppress a deployed primary candidate and output
        # 1 can only rescue a supported low-confidence candidate.
        deployment_features = 3 + 2 + 2 + 1 + self.cross_view_features + 1
        self.deployment_projection = nn.Sequential(
            nn.Linear(deployment_features, 16),
            nn.SiLU(inplace=True),
        )
        self.deployment_head = nn.Linear(16, 2)
        # Candidate-local evidence alone generalizes poorly to the test set's
        # many background-only images.  Compare every deployed candidate's
        # fused Detail/global token with its scene-conditioned context token.
        # The eight explicit measurements retain both Detail screenshots,
        # learned fused quality, independent-view support, and the existing
        # individual/relation background logits.  Exposing the latter two
        # avoids asking a tiny scene head to rediscover already learned
        # local-global relations.  This expert remains tiny (929 parameters)
        # and can suppress only when the semantic background expert agrees.
        image_relation_features = 16 * 3 + 8
        self.image_relation_projection = nn.Sequential(
            nn.Linear(image_relation_features, 16),
            nn.SiLU(inplace=True),
        )
        self.image_relation_head = nn.Linear(16, 1)
        # v71 deployment content expert.  The previous path used the absolute
        # probability of the jointly fine-tuned Detail classifier as a score
        # editor.  Its foreground probability shifted upward for both true
        # objects and hard backgrounds, so the native 0.5 classifier boundary
        # was not a stable detector decision.  Instead learn one signed
        # foreground margin from the *paired* CAM-centred and box-centred
        # Detail observations, their relative contrast, and the spatially
        # fused YOLO/Detail ROI token.  No detector confidence or NMS threshold
        # enters this head.  Positive means foreground, negative background.
        # The zero output initialization preserves the inherited detector
        # exactly until this new content expert has received supervision.
        dual_content_features = 16 * 3 + 13
        self.dual_content_projection = nn.Sequential(
            nn.Linear(dual_content_features, 24),
            nn.LayerNorm(24),
            nn.SiLU(inplace=True),
            nn.Linear(24, 12),
            nn.SiLU(inplace=True),
        )
        self.dual_content_head = nn.Linear(12, 1)
        # v75 uses one foreground/background metric memory for both score
        # directions.  Keeping and rescuing are the same semantic question;
        # splitting the sparse rescue examples into a separate v74 memory
        # made that memory classify every candidate as background.  Four
        # modes per class retain multimodality, while sharing them exposes the
        # memory to every strict-IoU example and halves this branch to 96
        # parameters.  The two classes start from identical modes, preserving
        # the inherited detector exactly before supervision.
        shared_semantic_prototypes = torch.empty(1, 4, 12)
        nn.init.normal_(shared_semantic_prototypes, std=0.02)
        self.dual_shared_prototypes = nn.Parameter(
            shared_semantic_prototypes.repeat(2, 1, 1)
        )
        # Background-only images are common at deployment, while a
        # candidate crop cannot by itself tell whether a glass-like texture
        # is the only response in an otherwise empty scene.  Pool mean/max
        # context from every projected FPN level and predict image-level
        # object presence.  One linear layer adds only 32*L+1 parameters and
        # is zero initialized for checkpoint-safe identity.
        self.image_presence_head = nn.Linear(
            2 * hidden_channels * len(global_channels), 1
        )
        # Keep the strongest, semantically meaningful measurements visible to
        # the role heads instead of forcing a 513-D classifier embedding and
        # all geometry through a 16-D bottleneck.  The final layer is zero
        # initialized, so this skip path is exactly inert before training.
        explicit_evidence_features = 4 if self.semantic_projection is not None else 3
        self.evidence_head = nn.Sequential(
            nn.Linear(explicit_evidence_features, 8),
            nn.SiLU(inplace=True),
            nn.Linear(8, 2),
        )
        # self embedding, set mean/max, agreement product/difference and the
        # explicit geometric cross-view evidence already computed for every
        # raw proposal.  This is permutation invariant with respect to the two
        # secondary views and adds only about three thousand parameters.
        relation_features = 16 * 5 + self.cross_view_features
        self.relation_projection = nn.Sequential(
            nn.Linear(relation_features, 32),
            nn.SiLU(inplace=True),
        )
        self.relation_head = nn.Linear(32, 2)
        # A transformed-view pair is one set-prediction component, not two
        # independent candidates. Pool both Detail/global embeddings and
        # their explicit evidence symmetrically, then emit one shared rescue
        # logit for the component. This removes view-order dependence and
        # prevents dense token aliases from voting multiple times.
        # The scalar evidence vector used by v25 left the MLP to rediscover
        # every useful agreement relation from only four values.  That was
        # especially brittle for hard backgrounds where Detail and YOLO were
        # both confident.  v26 exposes six calibrated modalities and their
        # within-view agreement/disagreement before symmetric pair pooling:
        # base score, cross-view score/IoU, Detail probability and the two
        # learned dense quality probabilities.  Each member descriptor has
        # 27 values; mean/min/max/range pooling keeps the pair permutation
        # invariant.  The hidden ROI token contributes the same four moments.
        component_member_features = 27
        component_features = 32 * 4 + component_member_features * 4
        self.component_projection = nn.Sequential(
            nn.Linear(component_features, 48),
            nn.SiLU(inplace=True),
            nn.Linear(48, 32),
            nn.SiLU(inplace=True),
        )
        # Explicit foreground/background evidence makes uncertainty visible.
        # The deployed rescue margin is foreground minus background, so both
        # zero logits preserve the exact pretrained detector identity.
        self.component_head = nn.Linear(32, 2)
        # Four compact modes per class cover heterogeneous glass/background
        # appearances with only 256 parameters.  A zero scalar keeps this
        # metric branch inert at initialization; the auxiliary prototype loss
        # below learns the prototypes before their margin affects deployment.
        self.component_prototypes = nn.Parameter(torch.empty(2, 4, 32))
        nn.init.normal_(self.component_prototypes, std=0.02)
        self.component_prototype_scale = nn.Parameter(torch.zeros(()))
        # Direct foreground/background memory for deployed primary boxes.
        # Initializing both classes from the same four vectors makes their
        # similarity margin exactly zero, so an upgraded checkpoint preserves
        # the inherited detector until supervised training separates them.
        shared_keep_prototypes = torch.empty(1, 4, 16)
        nn.init.normal_(shared_keep_prototypes, std=0.02)
        self.keep_prototypes = nn.Parameter(shared_keep_prototypes.repeat(2, 1, 1))
        # Start exactly at the pretrained YOLO identity for both role scores
        # and boxes. Shared quality is zero too, so it never edits a fresh model.
        for layer in (
            self.keep_head, self.rescue_head, self.box_head,
            self.quality_head, self.keep_context_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.spatial_box_head[-1].weight)
        nn.init.zeros_(self.spatial_box_head[-1].bias)
        nn.init.zeros_(self.evidence_head[-1].weight)
        nn.init.zeros_(self.evidence_head[-1].bias)
        nn.init.zeros_(self.relation_head.weight)
        nn.init.zeros_(self.relation_head.bias)
        nn.init.zeros_(self.component_head.weight)
        nn.init.zeros_(self.component_head.bias)
        nn.init.zeros_(self.deployment_head.weight)
        nn.init.zeros_(self.deployment_head.bias)
        nn.init.zeros_(self.image_relation_head.weight)
        nn.init.zeros_(self.image_relation_head.bias)
        nn.init.zeros_(self.dual_content_head.weight)
        nn.init.zeros_(self.dual_content_head.bias)
        nn.init.zeros_(self.image_presence_head.weight)
        nn.init.zeros_(self.image_presence_head.bias)

    @staticmethod
    def _pool(feature: torch.Tensor) -> torch.Tensor:
        return torch.cat((feature.mean((2, 3)), feature.amax((2, 3))), dim=1)

    def forward(
        self,
        level: int,
        global_roi: torch.Tensor,
        local_detail: torch.Tensor,
        base_logit: torch.Tensor,
        geometry: torch.Tensor,
        localization_geometry: torch.Tensor,
        semantic_context: torch.Tensor | None = None,
        cross_view: torch.Tensor | None = None,
        global_context_roi: torch.Tensor | None = None,
        scene_context: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.local_projection(local_detail)
        global_feature = self.global_projection[level](global_roi)
        if global_context_roi is None or scene_context is None:
            raise ValueError("global_context_roi and scene_context are required")
        context_feature = self.global_projection[level](global_context_roi)
        if scene_context.shape[1] != 2 * global_feature.shape[1]:
            raise ValueError("scene_context must contain projected FPN mean and max")
        gate = torch.sigmoid(self.cross_gate[level](torch.cat((global_feature, local), dim=1)))
        fused = global_feature + gate * local
        cross_attention_fusion = getattr(self, "cross_attention_fusion", None)
        if cross_attention_fusion is not None:
            fused = cross_attention_fusion(global_feature, local, fused)
        alignment = F.normalize(local.float(), dim=1).to(local.dtype) * F.normalize(
            global_feature.float(), dim=1
        ).to(global_feature.dtype)
        attention_logits = self.spatial_attention(
            torch.cat((local, global_feature, local * global_feature), dim=1)
        ).flatten(2)
        attention = attention_logits.softmax(2).view_as(attention_logits).reshape(
            len(local), 1, local.shape[-2], local.shape[-1]
        )

        def attentive_pool(feature: torch.Tensor) -> torch.Tensor:
            return (feature * attention).sum((2, 3))

        level_code = F.one_hot(
            torch.full((len(global_roi),), int(level), device=global_roi.device),
            num_classes=len(self.global_projection),
        ).to(dtype=global_roi.dtype)
        summary_parts = [
            self._pool(fused), self._pool(local), self._pool(global_feature),
            attentive_pool(fused), attentive_pool(local),
            attentive_pool(global_feature), attentive_pool(alignment),
        ]
        semantic_projection = getattr(self, "semantic_projection", None)
        if semantic_projection is not None:
            if semantic_context is None:
                raise ValueError("semantic_context is required by this fusion head")
            summary_parts.append(semantic_projection(semantic_context))
        if self.cross_view_features:
            if cross_view is None or cross_view.shape[1] != self.cross_view_features:
                raise ValueError(f"cross_view must have {self.cross_view_features} features")
            summary_parts.append(cross_view)
        summary_parts.extend((base_logit[:, None], geometry, level_code))
        summary = torch.cat(summary_parts, dim=1)
        hidden = self.trunk(summary)
        context_summary = torch.cat((
            hidden,
            self._pool(context_feature),
            self._pool((context_feature - global_feature).abs()),
            scene_context,
        ), dim=1).detach()
        context_hidden = self.context_trunk(context_summary)
        quality_logits = self.quality_head(hidden)
        if cross_view is None:
            raise ValueError("cross_view is required for explicit evidence fusion")
        view_score_columns = [0] + ([6] if cross_view.shape[1] > 7 else [])
        view_iou_columns = [1] + ([7] if cross_view.shape[1] > 7 else [])
        # Trainer validation casts the model to FP16. In half precision
        # ``1 - 1e-4`` rounds back to exactly one, so torch.logit produced
        # +inf for perfect cross-view IoUs and subsequently NaN role logits.
        # Clamp and transform in FP32, then cast the finite log-odds back.
        def stable_logit(probability: torch.Tensor) -> torch.Tensor:
            return torch.logit(
                probability.float().clamp(1e-4, 1 - 1e-4)
            ).to(dtype=base_logit.dtype)

        view_score = cross_view[:, view_score_columns].float().amax(1)
        view_iou = cross_view[:, view_iou_columns].float().amax(1)
        evidence_parts = [base_logit, stable_logit(view_score), stable_logit(view_iou)]
        if self.semantic_projection is not None:
            evidence_parts.append(stable_logit(semantic_context[:, -1]))
        evidence_logits = self.evidence_head(torch.stack(evidence_parts, dim=1))
        if localization_geometry.shape != (len(hidden), 5):
            raise ValueError("localization_geometry must have shape [N, 5]")
        axis_y = torch.linspace(
            -1.0, 1.0, attention.shape[-2],
            device=attention.device, dtype=attention.dtype,
        ).view(1, 1, -1, 1)
        axis_x = torch.linspace(
            -1.0, 1.0, attention.shape[-1],
            device=attention.device, dtype=attention.dtype,
        ).view(1, 1, 1, -1)
        attention_x = (attention * axis_x).sum((1, 2, 3))
        attention_y = (attention * axis_y).sum((1, 2, 3))
        attention_std_x = (
            (attention * (axis_x - attention_x[:, None, None, None]).square())
            .sum((1, 2, 3)).clamp_min(1e-8).sqrt()
        )
        attention_std_y = (
            (attention * (axis_y - attention_y[:, None, None, None]).square())
            .sum((1, 2, 3)).clamp_min(1e-8).sqrt()
        )
        spatial_geometry = torch.stack((
            attention_x, attention_y, attention_std_x, attention_std_y,
        ), 1)
        spatial_box = self.spatial_box_head(torch.cat((
            hidden, localization_geometry.to(dtype=hidden.dtype), spatial_geometry,
        ), 1))
        # Dense object/IoU quality is an auxiliary representation task only.
        # Adding it to both role logits made even clear false positives
        # positive and prevented the conservative keep conjunction from ever
        # rejecting them.
        return (
            self.keep_head(hidden).squeeze(1) + evidence_logits[:, 0],
            self.rescue_head(hidden).squeeze(1) + evidence_logits[:, 1],
            self.box_head(hidden) + spatial_box,
            quality_logits,
            gate,
            hidden,
            context_hidden,
        )

    def relation_forward(
        self,
        hidden: torch.Tensor,
        set_mean: torch.Tensor,
        set_max: torch.Tensor,
        cross_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pair evidence and its contrastively supervised embedding."""
        relation = torch.cat((
            hidden,
            set_mean,
            set_max,
            hidden * set_mean,
            (hidden - set_mean).abs(),
            cross_view,
        ), dim=1)
        embedding = self.relation_projection(relation)
        return self.relation_head(embedding), embedding

    def keep_prototype_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to compact foreground/background ROI memories."""
        feature = F.normalize(hidden.float(), dim=1)
        prototypes = F.normalize(self.keep_prototypes.float(), dim=2)
        return torch.einsum("nd,ckd->nck", feature, prototypes).amax(2)

    def deployment_forward(
        self,
        individual_logits: torch.Tensor,
        relation_logits: torch.Tensor,
        quality_logits: torch.Tensor,
        semantic_context: torch.Tensor | None,
        cross_view: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse every learned modality into two calibrated role logits."""
        if semantic_context is None:
            semantic_logit = quality_logits.new_zeros((len(quality_logits), 1))
        else:
            semantic_logit = torch.logit(
                semantic_context[:, -1:].float().clamp(1e-4, 1 - 1e-4)
            ).to(dtype=quality_logits.dtype)
        base_logit = torch.logit(
            base_scores[:, None].float().clamp(1e-4, 1 - 1e-4)
        ).to(dtype=quality_logits.dtype)
        evidence = torch.cat((
            individual_logits,
            relation_logits,
            quality_logits,
            semantic_logit,
            cross_view.to(dtype=quality_logits.dtype),
            base_logit,
        ), dim=1)
        return self.deployment_head(self.deployment_projection(evidence))

    def image_relation_forward(
        self,
        candidates: Sequence[Candidate],
        hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        base_scores: torch.Tensor,
        cam_semantic_context: torch.Tensor | None,
        box_semantic_context: torch.Tensor | None,
        quality_logits: torch.Tensor,
        individual_keep_logits: torch.Tensor,
        relation_keep_logits: torch.Tensor,
        cross_view: torch.Tensor,
        deployment_conf: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict background risk only where geometric views do not agree."""
        logits = hidden.new_zeros(len(candidates))
        selected = torch.zeros(len(candidates), device=hidden.device, dtype=torch.bool)
        if not candidates:
            return logits, selected
        indices = [
            index for index, candidate in enumerate(candidates)
            if candidate.source == "prediction"
            and float(base_scores[index].detach()) >= float(deployment_conf)
            and not candidate.consensus
        ]
        if not indices:
            return logits, selected
        index_tensor = torch.tensor(indices, device=hidden.device, dtype=torch.long)
        candidate_hidden = hidden[index_tensor]
        scene_hidden = context_hidden[index_tensor]

        def detail_probability(context: torch.Tensor | None) -> torch.Tensor:
            if context is None:
                return hidden.new_zeros(len(indices))
            return context[index_tensor, -1].float().clamp(0, 1).to(hidden.dtype)

        support = hidden.new_tensor([
            candidates[index].view_support / 3.0 for index in indices
        ])
        weakest_view_score = torch.minimum(
            cross_view[index_tensor, 0], cross_view[index_tensor, 6]
        ).to(hidden.dtype)
        numeric = torch.stack((
            base_scores[index_tensor].to(hidden.dtype),
            support,
            detail_probability(cam_semantic_context),
            detail_probability(box_semantic_context),
            quality_logits[index_tensor, 0].float().sigmoid().to(hidden.dtype),
            weakest_view_score,
            individual_keep_logits[index_tensor].to(hidden.dtype),
            relation_keep_logits[index_tensor].to(hidden.dtype),
        ), dim=1)
        representation = torch.cat((
            candidate_hidden,
            scene_hidden,
            (candidate_hidden - scene_hidden).abs(),
            numeric,
        ), dim=1)
        relation_hidden = self.image_relation_projection(representation)
        # index_copy preserves the gradient from the selected logits to the
        # tiny expert; an in-place assignment here made that contract opaque.
        logits = logits.index_copy(
            0, index_tensor, self.image_relation_head(relation_hidden).squeeze(1)
        )
        selected[index_tensor] = True
        return logits, selected

    def dual_content_forward(
        self,
        hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        cam_semantic_context: torch.Tensor | None,
        box_semantic_context: torch.Tensor | None,
        quality_logits: torch.Tensor,
        cross_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a threshold-free signed foreground margin for each candidate.

        Both Detail screenshots and the YOLO/global fusion token are required.
        Explicit mean/min/contrast terms make the decision depend on paired
        evidence rather than on a drifting absolute Detail probability.
        """
        if cam_semantic_context is None or box_semantic_context is None:
            return (
                hidden.new_zeros(len(hidden)),
                hidden.new_zeros((len(hidden), 2)),
            )

        def stable_logit(probability: torch.Tensor) -> torch.Tensor:
            return torch.logit(
                probability.float().clamp(1e-4, 1 - 1e-4)
            ).to(dtype=hidden.dtype)

        cam_logit = stable_logit(cam_semantic_context[:, -1])
        box_logit = stable_logit(box_semantic_context[:, -1])
        detail_mean = 0.5 * (cam_logit + box_logit)
        detail_min = torch.minimum(cam_logit, box_logit)
        detail_max = torch.maximum(cam_logit, box_logit)
        detail_contrast = (cam_logit - box_logit).abs()
        quality_object = quality_logits[:, 0].to(dtype=hidden.dtype)
        quality_iou = quality_logits[:, 1].to(dtype=hidden.dtype)
        view_score_columns = [0] + ([6] if cross_view.shape[1] > 7 else [])
        view_iou_columns = [1] + ([7] if cross_view.shape[1] > 7 else [])
        view_score = stable_logit(
            cross_view[:, view_score_columns].float().amax(1)
        )
        view_iou = stable_logit(
            cross_view[:, view_iou_columns].float().amax(1)
        )
        numeric = torch.stack((
            cam_logit,
            box_logit,
            detail_mean,
            detail_min,
            detail_max,
            detail_contrast,
            quality_object,
            quality_iou,
            detail_mean - quality_object,
            detail_min - quality_object,
            view_score,
            view_iou,
            detail_mean - view_score,
        ), dim=1)
        representation = torch.cat((
            hidden,
            context_hidden,
            (hidden - context_hidden).abs(),
            numeric,
        ), dim=1)
        content_hidden = self.dual_content_projection(representation)
        dense_margin = self.dual_content_head(content_hidden).squeeze(1)
        normalized_content = F.normalize(content_hidden.float(), dim=1)
        normalized_prototypes = F.normalize(
            self.dual_shared_prototypes.float(), dim=2
        )
        class_mode_similarity = torch.einsum(
            "nd,ckd->nck", normalized_content, normalized_prototypes
        )
        # Smooth mode pooling gives every prototype a learning signal.  v74's
        # hard maximum allowed one accidental background mode to dominate all
        # sparse rescue positives.  The common log(K) term cancels in the
        # foreground/background likelihood ratio.
        temperature = 0.20
        class_similarity = temperature * torch.logsumexp(
            class_mode_similarity / temperature, dim=2
        )
        shared_margin = (
            class_similarity[:, 1] - class_similarity[:, 0]
        ).to(dtype=content_hidden.dtype)
        role_margins = shared_margin[:, None].expand(-1, 2)
        return dense_margin, role_margins

    def component_forward(
        self,
        candidates: Sequence[Candidate],
        hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        semantic_context: torch.Tensor | None,
        cross_view: torch.Tensor,
        base_scores: torch.Tensor,
        quality_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one permutation-invariant rescue representation per pair."""
        scores = hidden.new_zeros(len(candidates))
        embeddings = hidden.new_zeros((len(candidates), 32))
        paired = torch.zeros(len(candidates), device=hidden.device, dtype=torch.bool)
        if semantic_context is None:
            return scores, embeddings, paired

        def stable_logit(probability: torch.Tensor) -> torch.Tensor:
            return torch.logit(
                probability.float().clamp(1e-4, 1 - 1e-4)
            ).to(dtype=hidden.dtype)

        view_score_columns = [0] + ([6] if cross_view.shape[1] > 7 else [])
        view_iou_columns = [1] + ([7] if cross_view.shape[1] > 7 else [])
        probabilities = torch.stack((
            base_scores.float().clamp(1e-4, 1 - 1e-4),
            cross_view[:, view_score_columns].float().amax(1).clamp(1e-4, 1 - 1e-4),
            cross_view[:, view_iou_columns].float().amax(1).clamp(1e-4, 1 - 1e-4),
            semantic_context[:, -1].float().clamp(1e-4, 1 - 1e-4),
            quality_logits[:, 0].float().sigmoid().clamp(1e-4, 1 - 1e-4),
            quality_logits[:, 1].float().sigmoid().clamp(1e-4, 1 - 1e-4),
        ), dim=1).to(dtype=hidden.dtype)
        logits = stable_logit(probabilities)
        base, cross_score, cross_iou, detail, quality_object, quality_iou = (
            probabilities[:, index] for index in range(probabilities.shape[1])
        )
        object_modalities = torch.stack((base, cross_score, detail, quality_object), dim=1)
        object_statistics = torch.stack((
            object_modalities.mean(1),
            object_modalities.amin(1),
            object_modalities.amax(1),
            object_modalities.prod(1),
            object_modalities.float().std(1, unbiased=False).to(hidden.dtype),
        ), dim=1)
        agreement = torch.stack((
            detail * quality_object,
            detail * cross_score,
            base * cross_score,
            torch.minimum(detail, quality_object),
            torch.minimum(detail, cross_score),
            (detail - cross_score).abs(),
            (detail - quality_object).abs(),
        ), dim=1)
        localization = torch.stack((
            torch.minimum(cross_iou, quality_iou),
            cross_iou * quality_iou,
            (cross_iou - quality_iou).abs(),
        ), dim=1)
        member_descriptor = torch.cat((
            probabilities, logits, object_statistics, agreement, localization,
        ), dim=1)
        groups = []
        for pair_id in sorted({candidate.pair_id for candidate in candidates if candidate.pair_id >= 0}):
            groups.append([
                index for index, candidate in enumerate(candidates)
                if candidate.pair_id == pair_id
            ])
        # Non-emitting secondary hypotheses are valuable hard examples.  They
        # cannot alter deployment output, but treating each as a singleton
        # component exposes the component classifier to the complete raw
        # background distribution instead of only the rare matched pairs.
        groups.extend([
            [index] for index, candidate in enumerate(candidates)
            if candidate.source == "secondary" and candidate.pair_id < 0
        ])
        for members in groups:
            if not members:
                continue
            member_tensor = torch.tensor(members, device=hidden.device, dtype=torch.long)
            member_hidden = torch.cat((
                hidden[member_tensor].detach(), context_hidden[member_tensor],
            ), dim=1)
            member_explicit = member_descriptor[member_tensor]
            pair_token = torch.cat((
                member_hidden.mean(0),
                member_hidden.amax(0),
                member_hidden.amin(0),
                member_hidden.amax(0) - member_hidden.amin(0),
                member_explicit.mean(0),
                member_explicit.amax(0),
                member_explicit.amin(0),
                member_explicit.amax(0) - member_explicit.amin(0),
            ), dim=0)
            embedding = self.component_projection(pair_token[None]).squeeze(0)
            class_logits = self.component_head(embedding).squeeze(0)
            normalized_embedding = F.normalize(embedding.float(), dim=0)
            normalized_prototypes = F.normalize(
                self.component_prototypes.float(), dim=2
            )
            prototype_similarity = torch.einsum(
                "d,ckd->ck", normalized_embedding, normalized_prototypes
            ).amax(1)
            prototype_margin = prototype_similarity[1] - prototype_similarity[0]
            score = (
                class_logits[1] - class_logits[0]
                + self.component_prototype_scale * prototype_margin.to(class_logits.dtype)
            )
            scores[member_tensor] = score
            embeddings[member_tensor] = embedding
            paired[member_tensor] = True
        return scores, embeddings, paired


class DetailSemanticEncoder(Detail_Net_attn):
    """Return both pretrained 4x4 route maps and classifier semantics."""

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        routes = self._forward_routes(inputs)
        route_map = torch.cat(routes, dim=1)
        flattened = torch.cat([route.flatten(1) for route in routes], dim=1)
        hidden = self.relu(self.fc0(flattened))
        probability = torch.sigmoid(self.fc2(hidden))
        return route_map, torch.cat((hidden, probability), dim=1)


def _load_full_detail_classifier(model: DetailSemanticEncoder, path: Path) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "detail_model"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Detail classifier checkpoint {path} does not contain a state dictionary")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Full Detail classifier load mismatch: missing={missing}, unexpected={unexpected}")
    return {"source": str(path), "loaded": len(state), "coverage": 1.0, "full_classifier": True}


def _candidate_level(candidate: Candidate, raw: Sequence[torch.Tensor]) -> Tuple[int, int]:
    offset = 0
    for level, output in enumerate(raw):
        count = int(output.shape[-2] * output.shape[-1])
        if candidate.anchor_index < offset + count:
            return level, candidate.anchor_index - offset
        offset += count
    raise IndexError(f"Anchor {candidate.anchor_index} is outside {offset} decoded anchors")


def _classic_gradcam(decoded: torch.Tensor, activation: torch.Tensor) -> torch.Tensor | None:
    """One aggregate classic Grad-CAM per detection scale."""
    gradient = torch.autograd.grad(
        decoded[:, 4:, :].sum(), activation, retain_graph=True, create_graph=False, allow_unused=True
    )[0]
    if gradient is None:
        return None
    # In-training validation is forced to FP16 by this Ultralytics version.
    # Aggregate CAMs in FP32: tiny Grad-CAM ranges otherwise underflow and the
    # normalization denominator can become zero in half precision.
    activation_fp32, gradient_fp32 = activation.float(), gradient.float()
    weights = gradient_fp32.mean((2, 3), keepdim=True)
    cam = torch.relu((weights * activation_fp32).sum(1, keepdim=True)).detach()
    if not torch.isfinite(cam).all():
        return None
    minimum = cam.flatten(1).amin(1).view(-1, 1, 1, 1)
    maximum = cam.flatten(1).amax(1).view(-1, 1, 1, 1)
    return (cam - minimum) / (maximum - minimum).clamp_min(torch.finfo(cam.dtype).eps)


def _center_from_cam(
    candidate: Candidate,
    cam: torch.Tensor | None,
    image_size: Tuple[int, int],
    window: int,
) -> Tuple[torch.Tensor, bool]:
    """Max-average CAM window inside a candidate, with box-center fallback."""
    image_h, image_w = image_size
    box = candidate.box_xyxy.float()
    x1 = max(0, min(image_w - 1, int(torch.floor(box[0]))))
    y1 = max(0, min(image_h - 1, int(torch.floor(box[1]))))
    x2 = max(x1 + 1, min(image_w, int(torch.ceil(box[2]))))
    y2 = max(y1 + 1, min(image_h, int(torch.ceil(box[3]))))
    fallback = box.new_tensor(((x1 + x2 - 1) / 2, (y1 + y2 - 1) / 2))
    if cam is None:
        return fallback, False
    full = F.interpolate(
        cam[candidate.batch_index:candidate.batch_index + 1],
        size=image_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    if not bool(torch.isfinite(full).all()) or float(full.max() - full.min()) <= 1e-12:
        return fallback, False
    window = max(1, min(int(window), image_h, image_w))
    if x2 - x1 < window or y2 - y1 < window:
        region = full[y1:y2, x1:x2]
        if not region.numel():
            return fallback, False
        flat = int(region.argmax())
        return full.new_tensor((x1 + flat % region.shape[1] + 0.5, y1 + flat // region.shape[1] + 0.5)), True
    pooled = F.avg_pool2d(full[None, None], kernel_size=window, stride=1).squeeze()
    region = pooled[y1:y2 - window + 1, x1:x2 - window + 1]
    if not region.numel():
        return fallback, False
    flat = int(region.argmax())
    left = x1 + flat % region.shape[1]
    top = y1 + flat // region.shape[1]
    return full.new_tensor((left + window / 2, top + window / 2)), True


def _crop_window(model, batch_index: int, default: int) -> int:
    batch = getattr(model, "batch", None)
    if not isinstance(batch, dict) or "ratio_pad" not in batch:
        return int(default)
    try:
        gain = batch["ratio_pad"][batch_index][0]
        return max(1, int(round(default * (float(gain[0]) + float(gain[1])) / 2)))
    except (IndexError, TypeError, ValueError):
        return int(default)


def _unique_crops(
    model,
    images: torch.Tensor,
    candidates: Sequence[Candidate],
    centers: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Encode each unique screenshot and retain its input-space support box.

    The returned support is the exact image region sampled by the Detail
    screenshot.  Reusing it for the YOLO ROI is what makes the two 4x4 feature
    grids spatially comparable rather than merely equal in tensor shape.
    """
    unique_candidates, unique_centers, inverse, keys = [], [], [], {}
    for candidate, center in zip(candidates, centers):
        key = (candidate.batch_index, int(round(float(center[0]))), int(round(float(center[1]))))
        if key not in keys:
            keys[key] = len(unique_candidates)
            unique_candidates.append(candidate)
            unique_centers.append(center)
        inverse.append(keys[key])
    config = model.candidate_roi_config
    if config["crop_source"] == "tensor":
        patches, crop_boxes, _crop_batch_indices = crop_tensor_patches(
            images, unique_candidates, unique_centers, crop_size=int(config["crop_size"]),
            source_scale=float(config.get("crop_source_scale", 0.0)),
            min_source_size=int(config.get("crop_min_source_size", 64)),
            max_source_size=int(config.get("crop_max_source_size", 256)),
        )
    else:
        batch = getattr(model, "batch", None)
        if not isinstance(batch, dict) or "ori_img" not in batch or "ratio_pad" not in batch:
            # AutoBackend warm-up and bare-tensor deployment do not carry
            # dataloader metadata.  They still use the same Grad-CAM centers,
            # but sample the already-letterboxed input tensor.  Dataset-backed
            # train/val batches always take the high-resolution branch below.
            patches, crop_boxes, _crop_batch_indices = crop_tensor_patches(
                images, unique_candidates, unique_centers, crop_size=int(config["crop_size"]),
                source_scale=float(config.get("crop_source_scale", 0.0)),
                min_source_size=int(config.get("crop_min_source_size", 64)),
                max_source_size=int(config.get("crop_max_source_size", 256)),
            )
        else:
            patches, crop_boxes, _crop_batch_indices = crop_original_patches(
                images, batch["ori_img"], batch["ratio_pad"], unique_candidates, unique_centers,
                crop_size=int(config["crop_size"]),
                source_scale=float(config.get("crop_source_scale", 0.0)),
                min_source_size=int(config.get("crop_min_source_size", 64)),
                max_source_size=int(config.get("crop_max_source_size", 256)),
            )
    encoded = model.detail_model(patches.to(dtype=next(model.detail_model.parameters()).dtype))
    if isinstance(encoded, tuple):
        detail, semantic = encoded
    else:
        detail, semantic = encoded, None
    inverse_tensor = torch.tensor(inverse, device=detail.device, dtype=torch.long)
    return (
        detail[inverse_tensor],
        semantic[inverse_tensor] if semantic is not None else None,
        inverse_tensor,
        crop_boxes[inverse_tensor],
    )


def _geometry(candidates: Sequence[Candidate], image_size: Tuple[int, int], dtype, device) -> torch.Tensor:
    image_h, image_w = image_size
    values = []
    for candidate in candidates:
        box = candidate.box_xyxy.float()
        width = (box[2] - box[0]).clamp_min(1) / image_w
        height = (box[3] - box[1]).clamp_min(1) / image_h
        center = (box[:2] + box[2:]) / 2
        values.append(torch.stack((center[0] / image_w, center[1] / image_h, width, height, width * height)))
    return torch.stack(values).to(device=device, dtype=dtype)


def _multiview_set_context(
    candidates: Sequence[Candidate], hidden: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build explicit pair tokens from independent-view candidate embeddings.

    Secondary proposals use the exact greedy one-to-one partner that created
    their consensus component.  Primary proposals select one best matching
    token from each transformed view.  Unlike v20, dense neighboring anchors
    are never averaged together, so a genuine paired Detail crop cannot be
    diluted by background features.  Mean/max over at most two tokens remains
    permutation invariant to the order of transformed views.
    """
    set_sum = torch.zeros_like(hidden)
    set_max = torch.full_like(hidden, -torch.inf)
    set_count = hidden.new_zeros((len(candidates), 1))
    if not candidates:
        return set_sum, set_sum

    views = (-1, 0, 1, 2)
    for batch_index in sorted({candidate.batch_index for candidate in candidates}):
        batch_indices = [
            index for index, candidate in enumerate(candidates)
            if candidate.batch_index == batch_index
        ]
        for query_view in views:
            query = [
                index for index in batch_indices
                if candidates[index].view_index == query_view
            ]
            if not query:
                continue
            query_tensor = torch.tensor(query, device=hidden.device, dtype=torch.long)
            query_boxes = torch.stack([
                (
                    candidates[index].fused_box_xyxy
                    if candidates[index].fused_box_xyxy is not None
                    else candidates[index].box_xyxy
                )
                for index in query
            ]).to(device=hidden.device, dtype=torch.float32)
            query_classes = torch.tensor(
                [candidates[index].class_index for index in query],
                device=hidden.device, dtype=torch.long,
            )
            for neighbor_view in (0, 1, 2):
                if neighbor_view == query_view:
                    continue
                neighbors = [
                    index for index in batch_indices
                    if candidates[index].view_index == neighbor_view
                ]
                if not neighbors:
                    continue
                neighbor_tensor = torch.tensor(neighbors, device=hidden.device, dtype=torch.long)
                neighbor_boxes = torch.stack([
                    (
                        candidates[index].fused_box_xyxy
                        if candidates[index].fused_box_xyxy is not None
                        else candidates[index].box_xyxy
                    )
                    for index in neighbors
                ]).to(device=hidden.device, dtype=torch.float32)
                neighbor_classes = torch.tensor(
                    [candidates[index].class_index for index in neighbors],
                    device=hidden.device, dtype=torch.long,
                )
                similarities = box_iou(query_boxes, neighbor_boxes)
                compatible = query_classes[:, None] == neighbor_classes[None]
                # An explicitly matched secondary pair must use its exact
                # partner.  Unpaired primary/GT candidates use the strongest
                # score-aware geometric match from this independent view.
                pair_compatible = torch.tensor([
                    [
                        candidates[query_index].pair_id >= 0
                        and candidates[query_index].pair_id == candidates[neighbor_index].pair_id
                        for neighbor_index in neighbors
                    ]
                    for query_index in query
                ], device=hidden.device, dtype=torch.bool)
                paired_query = torch.tensor(
                    [candidates[index].pair_id >= 0 for index in query],
                    device=hidden.device, dtype=torch.bool,
                )
                compatible[paired_query] = pair_compatible[paired_query]
                neighbor_scores = torch.stack([
                    candidates[index].score.detach().float() for index in neighbors
                ]).to(hidden.device)
                pair_quality = similarities * neighbor_scores[None].clamp_min(1e-6).sqrt()
                pair_quality = pair_quality.masked_fill(~compatible, -1)
                best_quality, best_index = pair_quality.max(1)
                supported = best_quality > 0
                if not bool(supported.any()):
                    continue
                supported_query = query_tensor[supported]
                supported_context = hidden[neighbor_tensor[best_index[supported]]]
                set_sum[supported_query] = set_sum[supported_query] + supported_context
                set_max[supported_query] = torch.maximum(
                    set_max[supported_query], supported_context
                )
                set_count[supported_query] = set_count[supported_query] + 1

    supported = set_count.squeeze(1) > 0
    set_mean = set_sum / set_count.clamp_min(1)
    set_max = torch.where(supported[:, None], set_max, torch.zeros_like(set_max))
    return set_mean, set_max


def _cross_view_evidence(
    model, images: torch.Tensor, detect_head, candidates: Sequence[Candidate]
) -> Tuple[torch.Tensor, List[Candidate], torch.Tensor]:
    """Build a three-view candidate graph without dropping raw candidates.

    Every raw secondary candidate remains available to Grad-CAM, Detail and
    global-ROI fusion.  Only mutual flip/scale matches may emit a new box, and
    one confidence-weighted box represents each consensus component.  Thus
    `no topk` still means all candidates are inspected, while duplicate raw
    anchors can no longer become independent false-positive detections.
    """
    image_h, image_w = images.shape[-2:]
    view_specs = tuple(model.candidate_roi_config.get(
        "cross_view_specs", ((0.83, 3), (0.67, None))
    ))
    secondary_candidates_by_view: List[List[Candidate]] = []
    with torch.no_grad():
        for scale, flip_dimension in view_specs:
            scale = float(scale)
            view_input = images.flip(int(flip_dimension)) if flip_dimension is not None else images
            secondary_input = scale_img(view_input, scale, gs=int(model.stride.max()))
            secondary_features, secondary_head = _features_before_detect(model, secondary_input)
            head_was_training = secondary_head.training
            try:
                secondary_head.eval()
                secondary_output = secondary_head([feature.clone() for feature in secondary_features])
            finally:
                secondary_head.train(head_was_training)
            secondary_decoded = secondary_output[0] if isinstance(secondary_output, tuple) else secondary_output
            secondary_decoded = secondary_decoded.clone()
            secondary_decoded[:, :4] /= scale
            if flip_dimension == 3:
                secondary_decoded[:, 0] = image_w - secondary_decoded[:, 0]
            elif flip_dimension == 2:
                secondary_decoded[:, 1] = image_h - secondary_decoded[:, 1]
            secondary_candidates_by_view.append(select_all_raw_candidates(
                secondary_decoded, float(model.candidate_roi_config["candidate_conf"])
            ))
    if len(secondary_candidates_by_view) != 2:
        raise RuntimeError("candidate graph currently requires exactly two secondary views")

    result = images.new_zeros((len(candidates), 12))
    cluster_iou = float(model.candidate_roi_config.get("consensus_cluster_iou", 0.50))
    proposals: List[Candidate] = []
    proposal_values: List[torch.Tensor] = []
    next_pair_id = 0

    def relative_values(reference_boxes, matched_boxes, matched_scores, matched_iou):
        reference_centers = (reference_boxes[:, :2] + reference_boxes[:, 2:]) / 2
        reference_sizes = (reference_boxes[:, 2:] - reference_boxes[:, :2]).clamp_min(1)
        matched_centers = (matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2
        matched_sizes = (matched_boxes[:, 2:] - matched_boxes[:, :2]).clamp_min(1)
        deltas = torch.cat((
            (matched_centers - reference_centers) / reference_sizes,
            torch.log(matched_sizes / reference_sizes),
        ), 1)
        return torch.cat((matched_scores[:, None], matched_iou[:, None], deltas), 1)

    for batch_index in range(images.shape[0]):
        primary_indices = [index for index, item in enumerate(candidates) if item.batch_index == batch_index]
        secondary_views = [
            [item for item in view if item.batch_index == batch_index]
            for view in secondary_candidates_by_view
        ]
        if not any(secondary_views):
            continue
        # Raw decoded boxes routinely have areas above 65,504.  FP16 area
        # products therefore overflow to inf and produce inf/inf = NaN IoUs
        # during the trainer's forced-half validation path.  All geometry is
        # deliberately computed in FP32 and cast only at the fusion boundary.
        if primary_indices:
            primary_boxes = torch.stack([candidates[index].box_xyxy for index in primary_indices]).to(
                device=images.device, dtype=torch.float32
            )
            primary_classes = torch.tensor(
                [candidates[index].class_index for index in primary_indices],
                device=images.device, dtype=torch.long,
            )
            primary_tensor = torch.tensor(primary_indices, device=images.device, dtype=torch.long)
            for view_index, secondary in enumerate(secondary_views):
                if not secondary:
                    continue
                secondary_boxes = torch.stack([item.box_xyxy for item in secondary]).to(
                    device=images.device, dtype=torch.float32
                )
                secondary_scores = torch.stack([item.score.detach() for item in secondary]).to(
                    device=images.device, dtype=torch.float32
                )
                secondary_classes = torch.tensor(
                    [item.class_index for item in secondary], device=images.device, dtype=torch.long
                )
                similarities = box_iou(primary_boxes, secondary_boxes)
                compatible = primary_classes[:, None] == secondary_classes[None]
                alignment = similarities * secondary_scores[None].clamp_min(1e-6).sqrt()
                alignment = alignment.masked_fill(~compatible, -1)
                matched = alignment.argmax(1)
                matched_iou = similarities[
                    torch.arange(len(primary_indices), device=images.device), matched
                ]
                values = relative_values(
                    primary_boxes, secondary_boxes[matched], secondary_scores[matched], matched_iou
                )
                values[~compatible.any(1)] = 0
                result[primary_tensor, view_index * 6:(view_index + 1) * 6] = values.to(result.dtype)
            for index in primary_indices:
                candidate = candidates[index]
                supported_offsets = [
                    offset for offset in (0, 6)
                    if float(result[index, offset + 1]) >= cluster_iou
                ]
                candidate.view_support = 1 + len(supported_offsets)
                candidate.consensus = len(supported_offsets) == 2
                if not supported_offsets:
                    continue

                # Reconstruct only the independently supported transformed
                # boxes and perform confidence-weighted box fusion.  Full
                # three-view agreement remains ``candidate.consensus``;
                # partial two-view support is recorded separately and is
                # eligible only for the recall path.
                reference_box = candidate.box_xyxy.detach().float()
                reference_center = (reference_box[:2] + reference_box[2:]) / 2
                reference_size = (reference_box[2:] - reference_box[:2]).clamp_min(1)
                fused_boxes = [reference_box]
                fused_weights = [candidate.score.detach().float().clamp_min(1e-6)]
                for offset in supported_offsets:
                    matched_score = result[index, offset].detach().float()
                    matched_delta = result[index, offset + 2:offset + 6].detach().float()
                    matched_center = reference_center + matched_delta[:2] * reference_size
                    matched_size = reference_size * matched_delta[2:].exp()
                    fused_boxes.append(torch.cat((
                        matched_center - matched_size / 2,
                        matched_center + matched_size / 2,
                    )))
                    fused_weights.append(matched_score.clamp_min(1e-6))
                weights = torch.stack(fused_weights)
                candidate.fused_box_xyxy = (
                    torch.stack(fused_boxes) * weights[:, None]
                ).sum(0) / weights.sum()
                candidate.view_support = 1 + len(supported_offsets)
                candidate.component_size = candidate.view_support

        if bool(model.candidate_roi_config.get("secondary_proposals", False)):
            local_proposals: List[List[Candidate]] = [[], []]
            local_values: List[List[torch.Tensor]] = [[], []]
            view_boxes, view_scores, view_classes = [], [], []
            for view_index, secondary in enumerate(secondary_views):
                if secondary:
                    view_boxes.append(torch.stack([item.box_xyxy for item in secondary]).float())
                    view_scores.append(torch.stack([item.score.detach() for item in secondary]).float())
                    view_classes.append(torch.tensor(
                        [item.class_index for item in secondary], device=images.device, dtype=torch.long
                    ))
                else:
                    view_boxes.append(images.new_empty((0, 4), dtype=torch.float32))
                    view_scores.append(images.new_empty((0,), dtype=torch.float32))
                    view_classes.append(torch.empty(0, device=images.device, dtype=torch.long))
                for item in secondary:
                    value = images.new_zeros(12, dtype=torch.float32)
                    value[view_index * 6] = item.score.detach().float()
                    value[view_index * 6 + 1] = 1.0
                    proposal = Candidate(
                        batch_index=batch_index,
                        anchor_index=-2,
                        class_index=item.class_index,
                        score=item.score.detach().clamp_max(0.499),
                        box_xyxy=item.box_xyxy.detach(),
                        source="secondary",
                        emit=False,
                        view_index=view_index + 1,
                    )
                    local_proposals[view_index].append(proposal)
                    local_values[view_index].append(value)

            if len(view_boxes[0]) and len(view_boxes[1]):
                pair_iou = box_iou(view_boxes[0], view_boxes[1])
                compatible = view_classes[0][:, None] == view_classes[1][None]
                pair_alignment = pair_iou * (
                    view_scores[0][:, None].clamp_min(1e-6)
                    * view_scores[1][None].clamp_min(1e-6)
                ).sqrt()
                pair_alignment = pair_alignment.masked_fill(~compatible, -1)
                best_1 = pair_alignment.argmax(1)
                best_0 = pair_alignment.argmax(0)
                rows = torch.arange(len(view_boxes[0]), device=images.device)
                matched_iou_0 = pair_iou[rows, best_1]
                values_0 = relative_values(
                    view_boxes[0], view_boxes[1][best_1], view_scores[1][best_1], matched_iou_0
                )
                for index, value in enumerate(values_0):
                    if bool(compatible[index].any()):
                        local_values[0][index][6:12] = value
                columns = torch.arange(len(view_boxes[1]), device=images.device)
                matched_iou_1 = pair_iou[best_0, columns]
                values_1 = relative_values(
                    view_boxes[1], view_boxes[0][best_0], view_scores[0][best_0], matched_iou_1
                )
                for index, value in enumerate(values_1):
                    if bool(compatible[:, index].any()):
                        local_values[1][index][0:6] = value

                # v16 required a pair to be the independent argmax in both
                # directions. Dense YOLO anchors frequently create ties, so a
                # valid object could disappear merely because one duplicate
                # anchor won either argmax.  Perform a global greedy one-to-one
                # matching over *all* compatible edges instead.  No proposal or
                # edge is truncated; each secondary hypothesis may support at
                # most one emitted representative.
                greedy_pairs = []
                valid_edges = torch.nonzero(
                    compatible & (pair_iou >= cluster_iou), as_tuple=False
                )
                if len(valid_edges):
                    edge_quality = pair_alignment[
                        valid_edges[:, 0], valid_edges[:, 1]
                    ]
                    edge_order = edge_quality.argsort(descending=True)
                else:
                    edge_order = torch.empty(0, device=images.device, dtype=torch.long)
                used_0, used_1 = set(), set()
                for edge_index in edge_order.tolist():
                    index_0, index_1 = valid_edges[edge_index].tolist()
                    if index_0 in used_0 or index_1 in used_1:
                        continue
                    used_0.add(index_0)
                    used_1.add(index_1)
                    proposal_0 = local_proposals[0][index_0]
                    proposal_1 = local_proposals[1][index_1]
                    proposal_0.pair_id = next_pair_id
                    proposal_1.pair_id = next_pair_id
                    proposal_0.view_support = proposal_1.view_support = 2
                    proposal_0.component_size = proposal_1.component_size = 2
                    next_pair_id += 1
                    score_0, score_1 = view_scores[0][index_0], view_scores[1][index_1]
                    weights = torch.stack((score_0, score_1)).clamp_min(1e-6)
                    fused_box = (
                        view_boxes[0][index_0] * weights[0]
                        + view_boxes[1][index_1] * weights[1]
                    ) / weights.sum()
                    pair_score = (score_0 * score_1).clamp_min(1e-12).sqrt()
                    representative = proposal_0 if float(score_0) >= float(score_1) else proposal_1
                    representative.fused_box_xyxy = fused_box.detach()
                    representative.score = pair_score.detach().clamp_max(0.499)
                    representative.consensus = True
                    greedy_pairs.append((representative, fused_box, pair_score))

                # A consensus proposal overlapping a deployed primary survivor
                # is redundant.  Use only the actual deployment NMS survivors,
                # never the dense raw primary anchors, for this graph edge.
                accepted_boxes = images.new_empty((0, 4), dtype=torch.float32)
                accepted_classes = torch.empty(0, device=images.device, dtype=torch.long)
                if primary_indices:
                    accepted = [
                        index for index in primary_indices
                        if candidates[index].source == "prediction"
                        and float(candidates[index].score.detach()) >= float(
                            model.candidate_roi_config.get("deployment_conf", 0.50)
                        )
                    ]
                    if accepted:
                        boxes = torch.stack([candidates[index].box_xyxy for index in accepted]).float()
                        scores = torch.stack([candidates[index].score.detach() for index in accepted]).float()
                        classes = torch.tensor(
                            [candidates[index].class_index for index in accepted],
                            device=images.device, dtype=torch.long,
                        )
                        keep = class_aware_nms(
                            boxes, scores, classes,
                            float(model.candidate_roi_config.get("deployment_nms_iou", 0.70)),
                        )
                        accepted_boxes, accepted_classes = boxes[keep], classes[keep]

                emit_pairs = []
                for representative, fused_box, pair_score in greedy_pairs:
                    redundant = False
                    if len(accepted_boxes):
                        same_class = accepted_classes == representative.class_index
                        if bool(same_class.any()):
                            redundant = bool(
                                (box_iou(fused_box[None], accepted_boxes[same_class]).amax() >= cluster_iou)
                            )
                    if redundant:
                        continue

                    # A two-view transformed proposal is not sufficient to
                    # create a new deployment detection.  Require a matching
                    # low-confidence candidate from the canonical image as a
                    # third independent observation, then include its box in
                    # the weighted localization consensus.  All canonical raw
                    # candidates above the internal 0.1 threshold remain
                    # eligible; no top-k pruning is introduced.
                    canonical_support = [
                        index for index in primary_indices
                        if candidates[index].source == "prediction"
                        and candidates[index].class_index == representative.class_index
                        and float(candidates[index].score.detach()) < float(
                            model.candidate_roi_config.get("deployment_conf", 0.50)
                        )
                    ]
                    if not canonical_support:
                        continue
                    canonical_boxes = torch.stack([
                        candidates[index].box_xyxy for index in canonical_support
                    ]).float()
                    canonical_iou = box_iou(fused_box[None], canonical_boxes).squeeze(0)
                    supported = canonical_iou >= cluster_iou
                    if not bool(supported.any()):
                        continue
                    support_tensor = torch.tensor(
                        canonical_support, device=images.device, dtype=torch.long
                    )[supported]
                    supported_iou = canonical_iou[supported]
                    supported_scores = torch.stack([
                        candidates[index].score.detach().float()
                        for index in support_tensor.tolist()
                    ])
                    best_support = (supported_iou * supported_scores.sqrt()).argmax()
                    canonical_index = int(support_tensor[best_support])
                    canonical_box = candidates[canonical_index].box_xyxy.detach().float()
                    canonical_score = candidates[canonical_index].score.detach().float().clamp_min(1e-6)
                    pair_weight = 2.0 * pair_score.detach().float().clamp_min(1e-6)
                    fused_box = (
                        pair_weight * fused_box + canonical_score * canonical_box
                    ) / (pair_weight + canonical_score)
                    representative.fused_box_xyxy = fused_box.detach()
                    representative.view_support = 3
                    representative.component_size = 3
                    emit_pairs.append((representative, fused_box, pair_score))
                if emit_pairs:
                    boxes = torch.stack([item[1] for item in emit_pairs]).float()
                    scores = torch.stack([item[2] for item in emit_pairs]).float()
                    classes = torch.tensor(
                        [item[0].class_index for item in emit_pairs],
                        device=images.device, dtype=torch.long,
                    )
                    keep = class_aware_nms(boxes, scores, classes, cluster_iou)
                    for keep_index in keep.tolist():
                        emit_pairs[keep_index][0].emit = True

            for view_index in range(2):
                proposals.extend(local_proposals[view_index])
                proposal_values.extend(local_values[view_index])
    proposal_tensor = (
        torch.stack(proposal_values).to(device=images.device, dtype=images.dtype)
        if proposal_values else images.new_zeros((0, 12))
    )
    return result, proposals, proposal_tensor


def _target_boxes(model, images: torch.Tensor) -> List[torch.Tensor]:
    batch = getattr(model, "batch", None)
    result = [images.new_empty((0, 4)) for _ in range(images.shape[0])]
    if not isinstance(batch, dict) or not all(key in batch for key in ("bboxes", "batch_idx")):
        return result
    image_h, image_w = images.shape[-2:]
    xywh = batch["bboxes"].to(device=images.device, dtype=images.dtype)
    xywh = xywh * xywh.new_tensor((image_w, image_h, image_w, image_h))
    boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    indices = batch["batch_idx"].to(device=images.device, dtype=torch.long)
    for batch_index in range(images.shape[0]):
        result[batch_index] = boxes[indices == batch_index]
    return result


def _image_presence_loss(
    model, image_presence_logits: torch.Tensor, images: torch.Tensor
) -> torch.Tensor:
    """Class-balanced signed-margin supervision for global object presence."""
    targets = image_presence_logits.new_tensor([
        float(len(boxes) > 0) for boxes in _target_boxes(model, images)
    ])
    object_mask = targets > 0.5
    background_mask = ~object_mask
    counts = getattr(
        model, "candidate_roi_presence_class_counts",
        targets.new_zeros(2),
    ).to(device=targets.device, dtype=targets.dtype)
    with torch.no_grad():
        counts = counts + torch.stack((object_mask.sum(), background_mask.sum())).to(counts)
    model.candidate_roi_presence_class_counts = counts.detach()
    total_seen = counts.sum().clamp_min(1)
    inverse_prior = total_seen / (2 * counts.clamp_min(1))
    weights = torch.where(object_mask, inverse_prior[0], inverse_prior[1])
    signed_target = 2.0 * targets - 1.0
    model.candidate_roi_presence_stats = {
        "object_images": int(object_mask.sum()),
        "background_images": int(background_mask.sum()),
    }
    return (
        F.softplus(1.0 - signed_target * image_presence_logits) * weights
    ).mean()


def _candidate_auxiliary_loss(
    model,
    candidates: Sequence[Candidate],
    keep_logits: torch.Tensor,
    rescue_logits: torch.Tensor,
    keep_decisions: torch.Tensor,
    rescue_decisions: torch.Tensor,
    individual_logits: torch.Tensor,
    relation_logits: torch.Tensor,
    deployment_logits: torch.Tensor,
    dual_content_logits: torch.Tensor,
    dual_role_logits: torch.Tensor,
    image_relation_logits: torch.Tensor,
    image_relation_mask: torch.Tensor,
    relation_embedding: torch.Tensor,
    keep_prototype_similarity: torch.Tensor,
    quality_logits: torch.Tensor,
    semantic_detail: torch.Tensor | None,
    cross_view: torch.Tensor,
    cam_semantic_detail: torch.Tensor | None,
    box_center_semantic_detail: torch.Tensor | None,
    dual_crop_role: torch.Tensor,
    cam_screenshot_boxes: torch.Tensor,
    box_center_screenshot_boxes: torch.Tensor,
    corrected_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    images: torch.Tensor,
) -> torch.Tensor:
    targets = _target_boxes(model, images)

    def crop_presence_targets(crop_boxes: torch.Tensor) -> torch.Tensor:
        """Continuous target-content coverage for each Detail screenshot.

        IoU is the correct label for a deployable detection but the wrong
        label for a local semantic crop: a crop fully inside a large target
        has low box IoU while containing unambiguous target texture.  The
        overlap coefficient (intersection divided by the smaller region)
        equals one when either the crop or target is contained in the other
        and falls continuously to zero for an unrelated screenshot.
        """
        result = images.new_zeros(len(candidates))
        for batch_index, gt in enumerate(targets):
            indices = [
                index for index, candidate in enumerate(candidates)
                if candidate.batch_index == batch_index
            ]
            if not indices or not len(gt):
                continue
            index_tensor = torch.tensor(
                indices, device=images.device, dtype=torch.long
            )
            crops = crop_boxes[index_tensor].to(
                device=images.device, dtype=images.dtype
            )
            left_top = torch.maximum(crops[:, None, :2], gt[None, :, :2])
            right_bottom = torch.minimum(crops[:, None, 2:], gt[None, :, 2:])
            intersection = (right_bottom - left_top).clamp_min(0).prod(2)
            crop_area = (crops[:, 2:] - crops[:, :2]).clamp_min(0).prod(1)
            gt_area = (gt[:, 2:] - gt[:, :2]).clamp_min(0).prod(1)
            denominator = torch.minimum(
                crop_area[:, None], gt_area[None, :]
            ).clamp_min(1e-6)
            result[index_tensor] = (intersection / denominator).amax(1).clamp(0, 1)
        return result.detach()

    cam_presence_targets = crop_presence_targets(cam_screenshot_boxes)
    box_presence_targets = crop_presence_targets(box_center_screenshot_boxes)
    candidate_presence_targets = torch.maximum(
        cam_presence_targets, box_presence_targets
    )
    # Dense quality targets initialize keep supervision. Rescue supervision is
    # subsequently converted to a deployment-aware one-to-one assignment: an
    # uncovered GT contributes at most one positive recall proposal, duplicate
    # proposals are ignored, and clear background proposals remain negative.
    labels = images.new_full((len(candidates),), -1.0)
    box_targets = images.new_zeros((len(candidates), 4))
    box_matched_targets = images.new_zeros((len(candidates), 4))
    box_mask = torch.zeros(len(candidates), device=images.device, dtype=torch.bool)
    box_source_iou = images.new_full((len(candidates),), -1.0)
    rescue_groups: List[List[int]] = []
    max_shift = float(model.candidate_roi_config["max_box_shift"])
    max_scale = float(model.candidate_roi_config["max_box_scale"])

    def candidate_source_box(index: int) -> torch.Tensor:
        """Use the same multiview source geometry as deployed refinement."""
        return _candidate_deployment_source_box(
            candidates[index], index, cross_view, semantic_detail,
            model.candidate_roi_config,
        )

    def assign(index: int, matched: torch.Tensor, source_iou: float):
        """Assign a bounded inference candidate to a target."""
        box = candidate_source_box(index).to(
            device=images.device, dtype=images.dtype
        )
        center = (box[:2] + box[2:]) / 2
        size = (box[2:] - box[:2]).clamp_min(1)
        target_center = (matched[:2] + matched[2:]) / 2
        target_size = (matched[2:] - matched[:2]).clamp_min(1)
        delta = torch.cat(((target_center - center) / size, torch.log(target_size / size)))
        delta[:2].clamp_(-max_shift, max_shift)
        delta[2:].clamp_(-max_scale, max_scale)
        labels[index] = 1
        box_targets[index] = delta.detach()
        box_matched_targets[index] = matched.detach()
        box_mask[index] = True
        box_source_iou[index] = float(source_iou)

    assignment_iou = float(model.candidate_roi_config.get("assignment_iou", 0.30))
    positive_iou = float(model.candidate_roi_config.get("positive_iou", 0.50))
    for batch_index, gt in enumerate(targets):
        predicted = [
            index for index, candidate in enumerate(candidates)
            if candidate.batch_index == batch_index
            and candidate.source != "gt"
            and (candidate.source != "secondary" or candidate.emit)
        ]
        if predicted and len(gt):
            boxes = torch.stack([
                candidate_source_box(index)
                for index in predicted
            ]).to(
                device=images.device, dtype=images.dtype
            )
            similarities = box_iou(boxes, gt)
            qualities, matches = similarities.max(1)
            for local_index, candidate_index in enumerate(predicted):
                quality = float(qualities[local_index])
                # A near miss is exactly where the multimodal box head is
                # useful.  Previous versions ignored IoU 0.3--0.5 candidates,
                # so high-confidence boxes just outside the metric boundary
                # could never learn the bounded correction already available
                # in the architecture.
                if quality >= assignment_iou:
                    assign(candidate_index, gt[int(matches[local_index])], quality)
                else:
                    labels[candidate_index] = 0
        elif predicted:
            labels[predicted] = 0

        # Train on the primary boxes that the deployed confidence filter and
        # NMS will actually expose. Dense raw anchors removed by NMS are not
        # precision errors and would otherwise swamp the few high-confidence
        # false positives the fusion head must learn to suppress. Keep every
        # secondary-view proposal supervised for the separate recall path.
        if predicted:
            predicted_base = torch.stack([candidates[index].score for index in predicted]).detach()
            predicted_boxes = torch.stack([
                candidate_source_box(index)
                for index in predicted
            ]).float()
            primary_local = torch.tensor(
                [
                    local_index for local_index, candidate_index in enumerate(predicted)
                    if candidates[candidate_index].source == "prediction"
                ],
                device=images.device,
                dtype=torch.long,
            )
            deployment_conf = float(model.candidate_roi_config.get("deployment_conf", 0.50))
            deployment_nms_iou = float(model.candidate_roi_config.get("deployment_nms_iou", 0.70))
            accepted_local = primary_local[predicted_base[primary_local] >= deployment_conf]
            low_local = primary_local[predicted_base[primary_local] < deployment_conf]
            survivor_local = accepted_local.new_empty((0,))
            if len(accepted_local):
                accepted_classes = torch.tensor(
                    [candidates[predicted[index]].class_index for index in accepted_local.tolist()],
                    device=images.device,
                    dtype=torch.long,
                )
                keep = class_aware_nms(
                    predicted_boxes[accepted_local], predicted_base[accepted_local],
                    accepted_classes, deployment_nms_iou,
                )
                survivor_local = accepted_local[keep]
            survivor_set = set(survivor_local.tolist())
            ignored_local = []
            for local_index in primary_local.tolist():
                candidate = candidates[predicted[local_index]]
                is_high = float(predicted_base[local_index]) >= deployment_conf
                if (is_high and local_index not in survivor_set) or (
                    not is_high and not _primary_recall_supported(candidate)
                ):
                    ignored_local.append(local_index)
            if len(survivor_local) and len(low_local):
                duplicate_low = box_iou(
                    predicted_boxes[low_local], predicted_boxes[survivor_local]
                ).amax(1) >= 0.5
                ignored_local.extend(low_local[duplicate_low].tolist())
            for local_index in ignored_local:
                candidate_index = predicted[local_index]
                labels[candidate_index] = -1
                box_mask[candidate_index] = False

            # Recall is a set prediction problem. Identify GTs already
            # recovered by a real deployment survivor and collect every valid
            # multiview hypothesis for each uncovered GT. v18 supervises these
            # groups with a multiple-instance objective: at least one member
            # must cross the deployment boundary, while inference still emits
            # only one learned representative per overlap component.
            rescue_local = torch.tensor(
                [
                    local_index for local_index, candidate_index in enumerate(predicted)
                    if (
                        candidates[candidate_index].source == "secondary"
                        and candidates[candidate_index].emit
                    ) or (
                        os.environ.get("BGD_DISABLE_PRIMARY_CONSENSUS_RESCUE") != "1"
                        and
                        candidates[candidate_index].source == "prediction"
                        and _primary_recall_supported(candidates[candidate_index])
                        and float(candidates[candidate_index].score.detach()) < deployment_conf
                    )
                ],
                device=images.device,
                dtype=torch.long,
            )
            if len(rescue_local) and len(gt):
                covered_gt = torch.zeros(len(gt), device=images.device, dtype=torch.bool)
                if len(survivor_local):
                    covered_gt = (
                        box_iou(predicted_boxes[survivor_local], gt.float()).amax(0)
                        >= positive_iou
                    )
                rescue_iou = box_iou(predicted_boxes[rescue_local], gt.float())
                for gt_index in torch.nonzero(~covered_gt, as_tuple=False).flatten().tolist():
                    members = []
                    for rescue_position in torch.nonzero(
                        rescue_iou[:, gt_index] >= positive_iou, as_tuple=False
                    ).flatten().tolist():
                        local_index = int(rescue_local[rescue_position])
                        candidate_index = predicted[local_index]
                        if bool(labels[candidate_index] >= 0):
                            members.append(candidate_index)
                    if members:
                        rescue_groups.append(members)

        # Optional teacher crops supervise Detail without competing with real
        # candidates in the one-to-one assignment or modifying detector logits.
        for index, candidate in enumerate(candidates):
            if candidate.batch_index != batch_index or candidate.source != "gt" or not len(gt):
                continue
            similarities = box_iou(candidate.box_xyxy[None].to(gt), gt).flatten()
            best_similarity, best_index = similarities.max(0)
            assign(index, gt[int(best_index)], float(best_similarity))

    # Shared quality supervision deliberately sees the complete raw candidate
    # distribution, including low/non-emitting anchors that must not edit YOLO
    # logits. This supplies abundant explicit background examples to the
    # Detail/global representation without turning dense anchors into output
    # boxes or overwhelming the role-specific decision losses.
    quality_labels = images.new_full((len(candidates),), -1.0)
    quality_iou_targets = images.new_zeros((len(candidates),))
    for batch_index, gt in enumerate(targets):
        dense_indices = [
            index for index, candidate in enumerate(candidates)
            if candidate.batch_index == batch_index
        ]
        if not dense_indices:
            continue
        if not len(gt):
            quality_labels[dense_indices] = 0
            continue
        dense_boxes = torch.stack([
            (
                candidates[index].fused_box_xyxy
                if candidates[index].fused_box_xyxy is not None
                else candidates[index].box_xyxy
            )
            for index in dense_indices
        ]).to(device=images.device, dtype=images.dtype)
        dense_quality = box_iou(dense_boxes, gt).amax(1)
        dense_tensor = torch.tensor(dense_indices, device=images.device, dtype=torch.long)
        quality_iou_targets[dense_tensor] = dense_quality.detach().clamp(0, 1)
        quality_labels[dense_tensor[dense_quality >= positive_iou]] = 1
        quality_labels[dense_tensor[dense_quality < assignment_iou]] = 0

    # Exact pair supervision.  A pair is positive when either transformed
    # member localizes an object, negative only when both are clear
    # background, and otherwise ignored.  Both screenshots inherit the same
    # pair label so the relation encoder learns the pair rather than merely
    # copying the representative candidate confidence.
    pair_labels = images.new_full((len(candidates),), -1.0)
    pair_representative = torch.zeros(
        len(candidates), device=images.device, dtype=torch.bool
    )
    pair_ids = sorted({candidate.pair_id for candidate in candidates if candidate.pair_id >= 0})
    for pair_id in pair_ids:
        members = [index for index, candidate in enumerate(candidates) if candidate.pair_id == pair_id]
        if len(members) < 2:
            continue
        pair_representative[members[0]] = True
        member_tensor = torch.tensor(members, device=images.device, dtype=torch.long)
        pair_quality = quality_iou_targets[member_tensor].amax()
        if float(pair_quality) >= positive_iou:
            pair_labels[member_tensor] = 1
        elif float(pair_quality) < assignment_iou:
            pair_labels[member_tensor] = 0

    # Train-time singleton components cover every secondary raw hypothesis
    # which failed the strict two-view geometric match.  They never emit at
    # inference, but their dense IoU labels provide the background diversity
    # missing from v26's 165-or-so matched negative components per epoch.
    for index, candidate in enumerate(candidates):
        if candidate.source != "secondary" or candidate.pair_id >= 0:
            continue
        quality = quality_iou_targets[index]
        pair_representative[index] = True
        if float(quality) >= positive_iou:
            pair_labels[index] = 1
        elif float(quality) < assignment_iou:
            pair_labels[index] = 0

    labels_tensor = labels.clamp_min(0)
    base_scores = torch.stack([candidate.score for candidate in candidates]).detach()
    positive = labels_tensor > 0
    negative = labels == 0
    valid = labels >= 0
    deployment_conf = float(model.candidate_roi_config.get("deployment_conf", 0.50))
    keep_role = torch.tensor(
        [
            candidate.source == "prediction"
            and float(candidate.score.detach()) >= deployment_conf
            for candidate in candidates
        ],
        device=images.device,
        dtype=torch.bool,
    ) & valid
    rescue_role = torch.tensor(
        [
            candidate.source == "gt"
            or (candidate.source == "secondary" and candidate.emit)
            or (
                os.environ.get("BGD_DISABLE_PRIMARY_CONSENSUS_RESCUE") != "1"
                and
                candidate.source == "prediction"
                and _primary_recall_supported(candidate)
                and float(candidate.score.detach()) < deployment_conf
            )
            for candidate in candidates
        ],
        device=images.device,
        dtype=torch.bool,
    ) & valid

    def balanced_weights(mask: torch.Tensor) -> torch.Tensor:
        """Give object/background equal mass inside one inference role."""
        weights = labels_tensor.new_zeros(len(labels_tensor))
        role_positive = mask & positive
        role_negative = mask & negative
        if bool(role_positive.any()):
            values = 1.0 + 4.0 * (0.5 - base_scores[role_positive]).clamp_min(0)
            mass = 0.5 if bool(role_negative.any()) else 1.0
            weights[role_positive] = mass * values / values.sum().clamp_min(1e-6)
        if bool(role_negative.any()):
            values = 1.0 + 2.0 * base_scores[role_negative]
            mass = 0.5 if bool(role_positive.any()) else 1.0
            weights[role_negative] = mass * values / values.sum().clamp_min(1e-6)
        return weights

    keep_weights = balanced_weights(keep_role)
    rescue_positive = rescue_role & positive
    rescue_negative = rescue_role & negative
    # Unlike v18, every valid recall candidate receives direct calibrated
    # supervision.  The MIL term below still chooses one representative per
    # missed-object set, while inference NMS prevents duplicate emissions.
    rescue_weights = balanced_weights(rescue_role)
    decision_weights = keep_weights + rescue_weights
    zero = corrected_logits.sum() * 0
    keep_loss = (
        0.5 * sum(
            F.binary_cross_entropy_with_logits(
                branch, 1.0 - labels_tensor, weight=keep_weights, reduction="sum"
            )
            for branch in (
                individual_logits[:, 0], relation_logits[:, 0], individual_logits[:, 2]
            )
        )
        if bool(keep_role.any()) else zero
    )
    # The individual view remains an auxiliary object classifier.  The shared
    # component margin is supervised once per component below; applying this
    # balanced emitted-candidate loss to it as well was the main source of the
    # positive prior shift in v25.
    rescue_loss = (
        F.binary_cross_entropy_with_logits(
            individual_logits[:, 1], labels_tensor,
            weight=rescue_weights, reduction="sum",
        )
        if bool(rescue_role.any()) else zero
    )
    # This is the deployed evidence combiner, so supervise it directly under
    # the same role-balanced labels used by the one-way score residuals.  The
    # scalar branches above remain representation auxiliaries; they no longer
    # need to agree unanimously at inference.
    # The deployed keep logit has a semantic zero boundary: background must
    # be positive so inference can turn it into a one-way negative residual,
    # while real objects must be negative.  BCE learned the ordering in prior
    # runs but left both classes on the foreground side of zero, making the
    # entire Grad-CAM/Detail/global score branch inert after clamp_min(0).
    # A fixed signed unit margin calibrates that native decision boundary.  It
    # is representation supervision, not a detector confidence threshold.
    deployment_keep_signed_target = 1.0 - 2.0 * labels_tensor
    deployment_keep_loss = (
        (
            F.softplus(
                1.0
                - deployment_keep_signed_target * deployment_logits[:, 0]
            ) * keep_weights
        ).sum()
        if bool(keep_role.any()) else zero
    )
    deployment_rescue_loss = (
        F.binary_cross_entropy_with_logits(
            deployment_logits[:, 1], labels_tensor,
            weight=rescue_weights, reduction="sum",
        )
        if bool(rescue_role.any()) else zero
    )
    image_relation_role = keep_role & image_relation_mask
    # The training split is object-heavy while the held-out test split is
    # background-heavy. Per-batch balancing still gives every object-only
    # batch unit mass and drove v57's background logit negative everywhere.
    # Track the actually observed deployed candidates across batches and use
    # their inverse empirical prior. This is data-adaptive class balancing,
    # not a deployment threshold, and keeps the loss scale normalized.
    scene_counts = getattr(
        model, "candidate_roi_scene_class_counts",
        labels_tensor.new_zeros(2),
    ).to(device=images.device, dtype=labels_tensor.dtype)
    scene_object = image_relation_role & positive
    scene_background_label = image_relation_role & negative
    with torch.no_grad():
        scene_counts = scene_counts + torch.stack((
            scene_object.sum(), scene_background_label.sum()
        )).to(scene_counts)
    model.candidate_roi_scene_class_counts = scene_counts.detach()
    image_relation_weights = labels_tensor.new_zeros(len(labels_tensor))
    if bool(image_relation_role.any()):
        total_seen = scene_counts.sum().clamp_min(1)
        inverse_prior = total_seen / (2 * scene_counts.clamp_min(1))
        image_relation_weights[scene_object] = inverse_prior[0]
        image_relation_weights[scene_background_label] = inverse_prior[1]
        # Do not renormalize the weights inside each mini-batch.  Background
        # candidates are sparse and concentrated in a small number of
        # batches; per-batch normalization cancels their inverse-frequency
        # weight, while every object-only batch still contributes unit mass.
        # Dividing the weighted BCE by the number of active candidates keeps
        # the ordinary BCE scale but preserves equal object/background mass
        # over the globally observed candidate stream.
        image_relation_normalizer = image_relation_role.sum().to(
            dtype=labels_tensor.dtype
        ).clamp_min(1)
    else:
        image_relation_normalizer = labels_tensor.new_tensor(1.0)
    # A signed unit-margin logistic objective aligns the learned zero point
    # with the deployment contract: clear background must lie above zero and
    # real objects below zero.  Ordinary BCE learned the ordering but left
    # both classes negative, so clamp_min(0) erased the expert at inference.
    # The fixed unit margin is a representation constraint, not a confidence
    # threshold; final detector evaluation remains conf=0.5, IoU=0.7.
    scene_signed_target = 1.0 - 2.0 * labels_tensor
    image_relation_loss = (
        (
            F.softplus(
                1.0 - scene_signed_target * image_relation_logits
            ) * image_relation_weights
        ).sum() / image_relation_normalizer
        if bool(image_relation_role.any()) else zero
    )
    # A deployed primary is classified directly in the fused Grad-CAM /
    # Detail / YOLO embedding space.  Four prototypes per class cover several
    # glass and hard-background modes without another MLP bottleneck.  Use the
    # same role-balanced mass as the deployment keep objective, while the
    # temperature supplies smooth gradients to all compact memories.
    keep_prototype_targets = negative.long()
    keep_prototype_loss = (
        (
            F.cross_entropy(
                keep_prototype_similarity / 0.2,
                keep_prototype_targets,
                reduction="none",
            ) * keep_weights
        ).sum()
        if bool(keep_role.any()) else zero
    )
    # The deployed heads must learn from the same complete raw distribution as
    # the representation heads.  Earlier versions only exposed the final keep
    # head to rare post-NMS false positives and the final rescue head to a few
    # emitted pairs (roughly 50 positives per run), while tens of thousands of
    # exact raw-candidate and component labels stopped at intermediate heads.
    # Balance each class inside the batch so duplicate foreground anchors do
    # not drown the hard background components.
    def balanced_dense_bce(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        hardness: torch.Tensor,
    ) -> torch.Tensor:
        weights = targets.new_zeros(len(targets))
        class_one = mask & (targets > 0.5)
        class_zero = mask & (targets <= 0.5)
        if bool(class_one.any()):
            values = 1.0 + hardness[class_one]
            mass = 0.5 if bool(class_zero.any()) else 1.0
            weights[class_one] = mass * values / values.sum().clamp_min(1e-6)
        if bool(class_zero.any()):
            values = 1.0 + hardness[class_zero]
            mass = 0.5 if bool(class_one.any()) else 1.0
            weights[class_zero] = mass * values / values.sum().clamp_min(1e-6)
        return (
            F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights, reduction="sum"
            )
            if bool(mask.any()) else zero
        )

    def balanced_dense_signed_margin(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        hardness: torch.Tensor,
    ) -> torch.Tensor:
        """Class-balanced unit-margin loss around the deployed zero sign."""
        weights = targets.new_zeros(len(targets))
        class_one = mask & (targets > 0.5)
        class_zero = mask & (targets <= 0.5)
        if bool(class_one.any()):
            values = 1.0 + hardness[class_one]
            mass = 0.5 if bool(class_zero.any()) else 1.0
            weights[class_one] = mass * values / values.sum().clamp_min(1e-6)
        if bool(class_zero.any()):
            values = 1.0 + hardness[class_zero]
            mass = 0.5 if bool(class_one.any()) else 1.0
            weights[class_zero] = mass * values / values.sum().clamp_min(1e-6)
        signed_targets = 2.0 * targets - 1.0
        return (
            (
                F.softplus(1.0 - signed_targets * logits) * weights
            ).sum()
            if bool(mask.any()) else zero
        )

    primary_raw = torch.tensor(
        [candidate.source == "prediction" for candidate in candidates],
        device=images.device, dtype=torch.bool,
    )
    dense_keep_mask = primary_raw & (quality_labels >= 0)
    dense_keep_targets = 1.0 - quality_labels.clamp_min(0)
    # High-score backgrounds and poorly localized foreground hypotheses are
    # the cases most likely to alter fixed-conf deployment metrics.
    dense_keep_hardness = torch.where(
        dense_keep_targets > 0.5,
        2.0 * base_scores,
        1.0 - quality_iou_targets,
    )
    deployment_allraw_keep_loss = balanced_dense_signed_margin(
        deployment_logits[:, 0], dense_keep_targets,
        dense_keep_mask, dense_keep_hardness,
    )
    # v71 directly supervises the score editor that is used at deployment.
    # Its sign is foreground-positive/background-negative and its training
    # population is every unambiguous primary raw candidate, not merely the
    # sparse post-NMS survivors.  The training manifest contains equal numbers
    # of object and background images; the additional per-image hardest-risk
    # terms stop thousands of easy foreground aliases from hiding one false
    # positive that would reduce precision at fixed evaluation confidence.
    dual_content_targets = quality_labels.clamp_min(0)
    dual_content_hardness = torch.where(
        dual_content_targets > 0.5,
        (1.0 - quality_iou_targets).clamp_min(0),
        2.0 * base_scores,
    )
    dual_content_loss = balanced_dense_signed_margin(
        dual_content_logits,
        dual_content_targets,
        dense_keep_mask,
        dual_content_hardness,
    )
    dual_content_foreground_risk = []
    dual_content_background_risk = []
    for batch_index in range(images.shape[0]):
        image_primary = torch.tensor(
            [candidate.batch_index == batch_index for candidate in candidates],
            device=images.device, dtype=torch.bool,
        ) & dense_keep_mask
        image_foreground = image_primary & (dual_content_targets > 0.5)
        image_background = image_primary & (dual_content_targets <= 0.5)
        if bool(image_foreground.any()):
            # At least one correctly localized content hypothesis must have a
            # positive unit margin.  Smooth max is stable when several dense
            # anchors describe the same object.
            values = dual_content_logits[image_foreground]
            temperature = 0.25
            smooth_max = (
                temperature * torch.logsumexp(values / temperature, dim=0)
                - temperature * math.log(len(values))
            )
            dual_content_foreground_risk.append(F.softplus(1.0 - smooth_max))
        if bool(image_background.any()):
            # A single high foreground margin can create a false detection;
            # optimize the smooth maximum background error per image.
            values = dual_content_logits[image_background]
            temperature = 0.25
            smooth_max = (
                temperature * torch.logsumexp(values / temperature, dim=0)
                - temperature * math.log(len(values))
            )
            dual_content_background_risk.append(F.softplus(1.0 + smooth_max))
    dual_content_image_risk = (
        (torch.stack(dual_content_foreground_risk).mean()
         if dual_content_foreground_risk else zero)
        + (torch.stack(dual_content_background_risk).mean()
           if dual_content_background_risk else zero)
    )

    # v74 uses the metric-valid IoU labels for score decisions.  v73 reused
    # the box-refinement assignment, where every IoU >= 0.30 near miss was
    # called foreground; that directly contradicted the detector metric's
    # IoU >= 0.50 contract and shifted both role heads positive.  Ambiguous
    # 0.30--0.50 candidates remain available to the box loss but are excluded
    # from the score memories.  The role-specific multi-prototype cosine
    # margins have a native, threshold-free zero boundary.
    strict_role_valid = quality_labels >= 0
    strict_role_targets = quality_labels.clamp_min(0)
    strict_role_positive = strict_role_targets > 0.5
    strict_role_negative = strict_role_valid & ~strict_role_positive
    strict_keep_role = keep_role & strict_role_valid
    strict_rescue_role = rescue_role & strict_role_valid

    def globally_balanced_role_loss(
        logits: torch.Tensor,
        role: torch.Tensor,
        count_name: str,
    ) -> torch.Tensor:
        foreground = role & strict_role_positive
        background = role & strict_role_negative
        counts = getattr(
            model, count_name, labels_tensor.new_zeros(2)
        ).to(device=images.device, dtype=labels_tensor.dtype)
        with torch.no_grad():
            counts = counts + torch.stack((
                foreground.sum(), background.sum()
            )).to(counts)
        setattr(model, count_name, counts.detach())
        if not bool(role.any()):
            return zero
        total_seen = counts.sum().clamp_min(1)
        inverse_prior = total_seen / (2 * counts.clamp_min(1))
        weights = labels_tensor.new_zeros(len(labels_tensor))
        weights[foreground] = inverse_prior[0]
        weights[background] = inverse_prior[1]
        signed_target = 2.0 * strict_role_targets - 1.0
        return (
            F.softplus(1.0 - signed_target * logits) * weights
        ).sum() / role.sum().to(labels_tensor.dtype).clamp_min(1)

    # The shared v75 memory is supervised by every unambiguous raw proposal,
    # so sparse recall candidates cannot move an isolated rescue classifier
    # to the all-background solution observed in v74.  Role-specific losses
    # remain as diagnostics, while the single globally balanced likelihood
    # loss is the optimization term.
    dual_shared_semantic_loss = globally_balanced_role_loss(
        dual_role_logits[:, 0], strict_role_valid,
        "candidate_roi_dual_shared_class_counts",
    )
    dual_role_keep_loss = globally_balanced_role_loss(
        dual_role_logits[:, 0], strict_keep_role,
        "candidate_roi_dual_keep_class_counts",
    )
    dual_role_rescue_loss = globally_balanced_role_loss(
        dual_role_logits[:, 1], strict_rescue_role,
        "candidate_roi_dual_rescue_class_counts",
    )

    def role_image_risk(
        logits: torch.Tensor,
        role: torch.Tensor,
        require_all_foreground: bool,
    ) -> Tuple[torch.Tensor, int, int]:
        foreground_terms, background_terms = [], []
        temperature = 0.25
        for batch_index in range(images.shape[0]):
            image_role = torch.tensor(
                [candidate.batch_index == batch_index for candidate in candidates],
                device=images.device, dtype=torch.bool,
            ) & role
            foreground_values = logits[image_role & strict_role_positive]
            background_values = logits[image_role & strict_role_negative]
            if len(foreground_values):
                if require_all_foreground:
                    summary = (
                        -temperature * torch.logsumexp(
                            -foreground_values / temperature, dim=0
                        )
                        + temperature * math.log(len(foreground_values))
                    )
                else:
                    summary = (
                        temperature * torch.logsumexp(
                            foreground_values / temperature, dim=0
                        )
                        - temperature * math.log(len(foreground_values))
                    )
                foreground_terms.append(F.softplus(1.0 - summary))
            if len(background_values):
                hardest_background = (
                    temperature * torch.logsumexp(
                        background_values / temperature, dim=0
                    )
                    - temperature * math.log(len(background_values))
                )
                background_terms.append(F.softplus(1.0 + hardest_background))
        risk = (
            (torch.stack(foreground_terms).mean() if foreground_terms else zero)
            + (torch.stack(background_terms).mean() if background_terms else zero)
        )
        return risk, len(foreground_terms), len(background_terms)

    dual_keep_image_risk, dual_keep_fg_images, dual_keep_bg_images = role_image_risk(
        dual_role_logits[:, 0], strict_keep_role, True
    )
    dual_rescue_image_risk, dual_rescue_fg_images, dual_rescue_bg_images = role_image_risk(
        dual_role_logits[:, 1], strict_rescue_role, False
    )
    dual_role_image_risk = dual_keep_image_risk + dual_rescue_image_risk

    component_mask_for_loss = pair_representative & (pair_labels >= 0)
    component_targets = pair_labels.clamp_min(0)
    component_hardness = torch.where(
        component_targets > 0.5,
        (1.0 - quality_iou_targets).clamp_min(0),
        2.0 * base_scores,
    )
    deployment_allraw_rescue_loss = balanced_dense_bce(
        deployment_logits[:, 1], component_targets,
        component_mask_for_loss, component_hardness,
    )
    # v50 encoded a second Detail screenshot around every unsupported primary
    # candidate, but the classifier was only reached indirectly through the
    # fused detector heads.  Consequently the candidate-centred view retained
    # the same mistakes as the CAM-centred view and the two-view inference
    # ablation was effectively identical.  Supervise both screenshots with
    # the exact candidate IoU labels used by the detector quality head.  The
    # loss is class-balanced inside each batch and ignores the ambiguous
    # IoU 0.3--0.5 interval; it adds no deployment threshold or parameters.
    semantic_candidate_mask = torch.tensor([
        candidate.source in {"prediction", "gt", "secondary"}
        for candidate in candidates
    ], device=images.device, dtype=torch.bool)
    dual_semantic_mask = semantic_candidate_mask
    dual_semantic_positive = dual_semantic_mask & (
        candidate_presence_targets >= 0.5
    )
    dual_semantic_negative = dual_semantic_mask & (
        candidate_presence_targets <= 0.05
    )

    def detail_crop_loss(
        semantic: torch.Tensor | None,
        presence_targets: torch.Tensor,
    ) -> torch.Tensor:
        # Ignore only partially clipped crops whose visual label is genuinely
        # ambiguous; retain the continuous overlap target for every clear
        # foreground/background crop.
        semantic_mask = dual_semantic_mask & (
            (presence_targets >= 0.5) | (presence_targets <= 0.05)
        )
        if semantic is None or not bool(semantic_mask.any()):
            return zero
        probability = semantic[:, -1].float().clamp(1e-4, 1 - 1e-4)
        semantic_targets = presence_targets.float()
        # Focus within each balanced class on the examples the current Detail
        # classifier finds difficult.  Base confidence only prioritizes
        # deployment-relevant background; it is not a decision threshold.
        hardness = torch.where(
            semantic_targets > 0.5,
            4.0 * (1.0 - probability.detach()),
            2.0 * base_scores.float() + 2.0 * probability.detach(),
        )
        return balanced_dense_bce(
            torch.logit(probability), semantic_targets,
            semantic_mask, hardness,
        )

    dual_crop_semantic_loss = 0.5 * (
        detail_crop_loss(cam_semantic_detail, cam_presence_targets)
        + detail_crop_loss(box_center_semantic_detail, box_presence_targets)
    )
    score_loss = F.binary_cross_entropy_with_logits(
        corrected_logits, labels_tensor, weight=decision_weights, reduction="sum"
    )
    # Multiple-instance recall loss.  A smooth normalized max represents the
    # one candidate that set-level inference will retain.  The effective logit
    # exactly mirrors the positive-only rescue residual, so the objective
    # trains a member to cross the real deployment boundary rather than merely
    # predicting an abstract object label.
    base_logits = torch.stack([
        torch.logit(candidate.score.clamp(1e-5, 1 - 1e-5)) for candidate in candidates
    ]).detach()
    rescue_quality_support = _rescue_quality_support(quality_logits)
    mil_effective_logits = base_logits + float(
        model.candidate_roi_config["max_score_delta"]
    ) * torch.tanh(rescue_decisions.clamp_min(0)) * rescue_quality_support
    group_scores = []
    for members in rescue_groups:
        member_tensor = torch.tensor(members, device=images.device, dtype=torch.long)
        member_logits = mil_effective_logits[member_tensor]
        temperature = 0.25
        group_scores.append(
            temperature * torch.logsumexp(member_logits / temperature, dim=0)
            - temperature * math.log(len(members))
        )
    if group_scores:
        group_scores_tensor = torch.stack(group_scores)
        rescue_group_loss = F.softplus(-group_scores_tensor).mean()
        if bool(rescue_negative.any()):
            hard_negative = mil_effective_logits[rescue_negative].amax()
            rescue_group_ranking = F.softplus(
                0.5 - group_scores_tensor + hard_negative
            ).mean()
        else:
            rescue_group_ranking = zero
    else:
        rescue_group_loss = zero
        rescue_group_ranking = zero

    # Dense multimodal quality learning.  The first logit classifies an object
    # candidate, and the second regresses continuous localization IoU through
    # BCE.  Balanced object/background mass prevents the many background raw
    # anchors from overwhelming the rare high-value recall proposals.
    quality_valid = quality_labels >= 0
    quality_positive = quality_labels > 0
    quality_negative = quality_labels == 0
    quality_weights = labels_tensor.new_zeros(len(labels_tensor))
    if bool(quality_positive.any()):
        mass = 0.5 if bool(quality_negative.any()) else 1.0
        positive_values = 1.0 + (1.0 - quality_iou_targets[quality_positive])
        quality_weights[quality_positive] = (
            mass * positive_values / positive_values.sum().clamp_min(1e-6)
        )
    if bool(quality_negative.any()):
        mass = 0.5 if bool(quality_positive.any()) else 1.0
        negative_values = 1.0 + 2.0 * base_scores[quality_negative]
        quality_weights[quality_negative] = (
            mass * negative_values / negative_values.sum().clamp_min(1e-6)
        )
    object_quality_loss = (
        (
            F.binary_cross_entropy_with_logits(
                quality_logits[:, 0], candidate_presence_targets, reduction="none"
            ) * (
                # Reuse the stable class-balanced dense weights, but do not
                # discard semantically positive near-miss boxes merely
                # because their detector IoU lies below 0.5.
                quality_weights + 0.25 * semantic_candidate_mask.to(quality_weights)
                / semantic_candidate_mask.sum().clamp_min(1)
            )
        ).sum()
        if bool(semantic_candidate_mask.any()) else zero
    )
    iou_quality_loss = (
        (
            F.binary_cross_entropy_with_logits(
                quality_logits[:, 1], quality_iou_targets, reduction="none"
            ) * quality_weights
        ).sum()
        if bool(quality_valid.any()) else zero
    )
    # One pair is one deployable hypothesis.  Supervise it once instead of
    # duplicating both transformed members, and optimize foreground coverage
    # and background rejection as separate risks.  The latter includes a
    # smooth maximum for every image, directly penalizing the single hardest
    # false rescue that would damage image-level precision in a background-
    # heavy test split.
    pair_positive = pair_representative & (pair_labels > 0)
    pair_negative = pair_representative & (pair_labels == 0)
    positive_pair_logits = relation_logits[pair_positive, 1]
    negative_pair_logits = relation_logits[pair_negative, 1]
    positive_component_loss = (
        F.softplus(1.0 - positive_pair_logits).mean()
        if len(positive_pair_logits) else zero
    )
    negative_component_loss = (
        F.softplus(1.0 + negative_pair_logits).mean()
        if len(negative_pair_logits) else zero
    )
    pair_classification_loss = positive_component_loss + negative_component_loss

    background_image_terms = []
    for batch_index in range(images.shape[0]):
        image_negative = torch.tensor(
            [candidate.batch_index == batch_index for candidate in candidates],
            device=images.device, dtype=torch.bool,
        ) & pair_negative
        image_logits = relation_logits[image_negative, 1]
        if len(image_logits):
            temperature = 0.25
            smooth_max = (
                temperature * torch.logsumexp(image_logits / temperature, dim=0)
                - temperature * math.log(len(image_logits))
            )
            background_image_terms.append(F.softplus(1.0 + smooth_max))
    background_image_risk = (
        torch.stack(background_image_terms).mean()
        if background_image_terms else zero
    )
    if bool(pair_positive.any()) and bool(pair_negative.any()):
        pair_ranking_loss = F.softplus(
            1.0 - positive_pair_logits[:, None] + negative_pair_logits[None, :]
        ).mean()
    else:
        pair_ranking_loss = zero

    # Supervised contrastive pair embedding.  Each pair member provides the
    # other transformed screenshot as a guaranteed same-label positive; all
    # opposite-label pairs in the mini-batch are negatives.  This creates a
    # genuine foreground/background pair geometry before the scalar head.
    pair_valid = pair_representative & (pair_labels >= 0)
    pair_contrastive_loss = zero
    if int(pair_valid.sum()) > 2 and bool(pair_positive.any()) and bool(pair_negative.any()):
        pair_features = F.normalize(relation_embedding[pair_valid].float(), dim=1)
        pair_targets = pair_labels[pair_valid]
        similarity = pair_features @ pair_features.transpose(0, 1) / 0.2
        eye = torch.eye(len(pair_features), device=images.device, dtype=torch.bool)
        same_label = pair_targets[:, None] == pair_targets[None, :]
        positive_mask = same_label & ~eye
        logits_masked = similarity.masked_fill(eye, -torch.inf)
        log_probability = similarity - torch.logsumexp(logits_masked, dim=1, keepdim=True)
        valid_anchor = positive_mask.any(1)
        pair_contrastive_loss = -(
            (log_probability.masked_fill(~positive_mask, 0).sum(1)
             / positive_mask.sum(1).clamp_min(1))[valid_anchor]
        ).mean()
    prototype_classification_loss = zero
    if bool(pair_valid.any()):
        prototype_features = F.normalize(
            relation_embedding[pair_valid].float(), dim=1
        )
        prototypes = F.normalize(
            model.candidate_roi_fusion.component_prototypes.float(), dim=2
        )
        prototype_logits = torch.einsum(
            "nd,ckd->nck", prototype_features, prototypes
        ).amax(2) / 0.2
        prototype_classification_loss = F.cross_entropy(
            prototype_logits, pair_labels[pair_valid].long()
        )
    ranking_terms = []
    for batch_index in range(images.shape[0]):
        image_mask = torch.tensor(
            [candidate.batch_index == batch_index for candidate in candidates],
            device=images.device,
            dtype=torch.bool,
        )
        positive_scores = keep_logits[image_mask & keep_role & positive]
        negative_scores = keep_logits[image_mask & keep_role & negative]
        if len(positive_scores) and len(negative_scores):
            ranking_terms.append(
                F.softplus(1.0 - positive_scores[:, None] + negative_scores[None, :]).mean()
            )
    ranking_loss = torch.stack(ranking_terms).mean() if ranking_terms else score_loss * 0
    if bool(box_mask.any()):
        # Optimize the geometry that is actually deployed.  Smooth-L1 on the
        # bounded center/scale parameters can decrease while leaving a box on
        # the wrong side of the IoU=0.5 matching boundary.  Decode the learned
        # residual from the same multiview source box used at inference and
        # train it with CIoU against its matched GT.  A small coordinate term
        # keeps the bounded parameterization numerically well conditioned.
        source_boxes = torch.stack([
            candidate_source_box(index) for index in range(len(candidates))
        ]).to(device=images.device, dtype=images.dtype)
        source_centers = (source_boxes[:, :2] + source_boxes[:, 2:]) / 2
        source_sizes = (source_boxes[:, 2:] - source_boxes[:, :2]).clamp_min(1)
        corrected_centers = source_centers + box_deltas[:, :2] * source_sizes
        corrected_sizes = source_sizes * torch.exp(box_deltas[:, 2:])
        corrected_boxes = torch.cat((
            corrected_centers - corrected_sizes / 2,
            corrected_centers + corrected_sizes / 2,
        ), 1)
        ciou = bbox_iou(
            corrected_boxes, box_matched_targets,
            xywh=False, CIoU=True,
        ).flatten()
        coordinate_loss = F.smooth_l1_loss(
            box_deltas, box_targets, reduction="none"
        ).mean(1)
        refinement_loss = 1.0 - ciou + 0.10 * coordinate_loss
        # v76 aligns geometric learning with the reported mAP75 target.
        # Correct boxes below IoU 0.75, but teach the deployed residual to be
        # exactly zero for already-stable IoU>=0.75 boxes.  v75 optimized a
        # GT residual even for excellent boxes; those tiny unnecessary moves
        # improved mAP75 on average but occasionally damaged mAP50.  The 0.75
        # boundary is the named evaluation metric, not a searched parameter.
        refinement = box_mask & (box_source_iou < 0.75)
        stable = box_mask & ~refinement
        stable_identity_loss = box_deltas.square().mean(1)
        per_box_loss = torch.where(
            refinement, refinement_loss, stable_identity_loss
        )
        box_weights = per_box_loss.new_zeros(len(per_box_loss))
        if bool(refinement.any()):
            refinement_values = 1.0 + (
                0.75 - box_source_iou[refinement]
            ).clamp_min(0) / max(0.75 - assignment_iou, 1e-6)
            refinement_mass = 0.5 if bool(stable.any()) else 1.0
            box_weights[refinement] = (
                refinement_mass
                * refinement_values
                / refinement_values.sum().clamp_min(1e-6)
            )
        if bool(stable.any()):
            stable_values = torch.ones_like(box_source_iou[stable])
            stable_mass = 0.5 if bool(refinement.any()) else 1.0
            box_weights[stable] = (
                stable_mass
                * stable_values
                / stable_values.sum().clamp_min(1e-6)
            )
        box_loss = (per_box_loss * box_weights).sum()
        near_miss = box_mask & (box_source_iou < positive_iou)
    else:
        near_miss = box_mask
        refinement = box_mask
        stable = box_mask
        box_loss = score_loss * 0
    # False candidates have no regression target; explicitly keep their
    # geometric residual at zero so shared head weights cannot move FPs while
    # learning corrections for matched objects.
    negative_box_loss = box_deltas[negative].square().mean() if bool(negative.any()) else score_loss * 0
    regularizer = (corrected_logits - base_logits).square().mean()
    anchor_loss = zero
    anchor_parameters = getattr(model, "candidate_roi_anchor_parameters", {})
    anchor_strength = float(model.candidate_roi_config.get("anchor_strength", 0.0))
    if anchor_strength > 0 and anchor_parameters:
        anchor_terms = []
        for name, parameter in model.named_parameters():
            reference = anchor_parameters.get(name)
            if reference is None or reference.shape != parameter.shape:
                continue
            if reference.device != parameter.device or reference.dtype != parameter.dtype:
                reference = reference.to(device=parameter.device, dtype=parameter.dtype)
                anchor_parameters[name] = reference
            anchor_terms.append((parameter.float() - reference.float()).square().sum())
        if anchor_terms:
            # Classical L2-SP: unlike ordinary weight decay, the attractor is
            # the original pretrained detector rather than zero.
            anchor_loss = torch.stack(anchor_terms).sum()
    current_stats = {
        "keep_positive": int((keep_role & positive).sum()),
        "keep_negative": int((keep_role & negative).sum()),
        "rescue_positive_candidates": int(rescue_positive.sum()),
        "rescue_negative": int(rescue_negative.sum()),
        "rescue_groups": len(rescue_groups),
        "quality_positive": int(quality_positive.sum()),
        "quality_negative": int(quality_negative.sum()),
        "pair_positive": int(pair_positive.sum()),
        "pair_negative": int(pair_negative.sum()),
        "background_risk_images": len(background_image_terms),
        "box_nearmiss": int(near_miss.sum()),
        "box_refinement_below_iou75": int(refinement.sum()),
        "box_stable_iou75_identity": int(stable.sum()),
        "dual_semantic_positive": int(dual_semantic_positive.sum()),
        "dual_semantic_negative": int(dual_semantic_negative.sum()),
        "cam_semantic_target_mean_x1000": int(
            1000 * cam_presence_targets.mean()
        ),
        "box_semantic_target_mean_x1000": int(
            1000 * box_presence_targets.mean()
        ),
        "dual_content_positive": int((dense_keep_mask & (dual_content_targets > 0.5)).sum()),
        "dual_content_negative": int((dense_keep_mask & (dual_content_targets <= 0.5)).sum()),
        "dual_content_foreground_risk_images": len(dual_content_foreground_risk),
        "dual_content_background_risk_images": len(dual_content_background_risk),
        "dual_keep_foreground_risk_images": dual_keep_fg_images,
        "dual_keep_background_risk_images": dual_keep_bg_images,
        "dual_rescue_foreground_risk_images": dual_rescue_fg_images,
        "dual_rescue_background_risk_images": dual_rescue_bg_images,
        "dual_keep_strict_positive": int(
            (strict_keep_role & strict_role_positive).sum()
        ),
        "dual_keep_strict_negative": int(
            (strict_keep_role & strict_role_negative).sum()
        ),
        "dual_rescue_strict_positive": int(
            (strict_rescue_role & strict_role_positive).sum()
        ),
        "dual_rescue_strict_negative": int(
            (strict_rescue_role & strict_role_negative).sum()
        ),
        "scene_background_candidates": int(image_relation_role.sum()),
        "scene_background_negative": int((image_relation_role & negative).sum()),
        "scene_background_positive": int((image_relation_role & positive).sum()),
    }
    model.candidate_roi_aux_stats = current_stats
    accumulated = getattr(model, "candidate_roi_aux_stats_accum", {})
    model.candidate_roi_aux_stats_accum = {
        key: int(accumulated.get(key, 0)) + value for key, value in current_stats.items()
    }
    return (
        keep_loss + rescue_loss + deployment_keep_loss
        + deployment_rescue_loss + image_relation_loss + deployment_allraw_keep_loss
        + keep_prototype_loss
        + deployment_allraw_rescue_loss + dual_crop_semantic_loss
        + dual_content_loss + dual_content_image_risk
        + dual_shared_semantic_loss + dual_role_image_risk
        + rescue_group_loss
        + 0.50 * object_quality_loss + 0.25 * iou_quality_loss + 0.5 * score_loss
        + 0.25 * ranking_loss + 0.25 * rescue_group_ranking
        + pair_classification_loss + pair_ranking_loss + pair_contrastive_loss
        + background_image_risk
        + prototype_classification_loss
        + float(model.candidate_roi_config["box_aux_weight"]) * (box_loss + 0.25 * negative_box_loss)
        + 0.001 * regularizer
        + anchor_strength * anchor_loss
    )


def _ground_truth_training_candidates(model, images: torch.Tensor) -> List[Candidate]:
    """Training-only positive screenshots for explicit fusion supervision."""
    candidates = []
    for batch_index, boxes in enumerate(_target_boxes(model, images)):
        for box in boxes:
            candidates.append(Candidate(
                batch_index=batch_index,
                anchor_index=-1,
                class_index=0,
                score=images.new_tensor(0.10),
                box_xyxy=box.detach(),
                source="gt",
                view_index=-1,
            ))
    return candidates


def _apply_box_delta(box_xyxy: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    center = (box_xyxy[:2] + box_xyxy[2:]) / 2
    size = (box_xyxy[2:] - box_xyxy[:2]).clamp_min(1)
    new_center = center + delta[:2] * size
    new_size = size * torch.exp(delta[2:])
    return torch.cat((new_center, new_size))


def _accumulate_candidate_diagnostics(
    model,
    candidates: Sequence[Candidate],
    images: torch.Tensor,
    keep_role: torch.Tensor,
    rescue_role: torch.Tensor,
    keep_raw: torch.Tensor,
    rescue_raw: torch.Tensor,
    corrected_logits: torch.Tensor,
    individual_raw: torch.Tensor,
    relation_raw: torch.Tensor,
    image_relation_raw: torch.Tensor,
    quality_raw: torch.Tensor,
    semantic_detail: torch.Tensor | None,
    cam_semantic_detail: torch.Tensor | None,
    box_center_semantic_detail: torch.Tensor | None,
    cross_view: torch.Tensor,
    box_deltas: torch.Tensor,
) -> None:
    """Accumulate GT-aware inference diagnostics without changing predictions."""
    targets = _target_boxes(model, images)
    if not isinstance(getattr(model, "batch", None), dict):
        return
    accumulator = getattr(model, "candidate_roi_diagnostics", {})
    corrected_scores = corrected_logits.detach().float().sigmoid()
    verbose_records = os.environ.get("BGD_VERBOSE_CANDIDATE_DIAGNOSTICS") == "1"
    for index, candidate in enumerate(candidates):
        if candidate.source == "gt" or not (bool(keep_role[index]) or bool(rescue_role[index])):
            continue
        role = "keep" if bool(keep_role[index]) else "rescue"
        gt = targets[candidate.batch_index]
        geometry_diagnostics = {}
        if len(gt):
            canonical_box = candidate.box_xyxy.detach().float().to(
                device=images.device
            )
            box = _candidate_deployment_source_box(
                candidate, index, cross_view, semantic_detail,
                model.candidate_roi_config,
            ).to(device=images.device, dtype=images.dtype)
            source_iou = box_iou(box[None], gt).flatten()
            quality = float(source_iou.max())
            matched_index = int(source_iou.argmax())
            corrected_xywh = _apply_box_delta(
                box, box_deltas[index].detach().to(box.dtype)
            )
            corrected_box = torch.cat((
                corrected_xywh[:2] - corrected_xywh[2:] / 2,
                corrected_xywh[:2] + corrected_xywh[2:] / 2,
            ))
            corrected_iou = float(box_iou(
                corrected_box[None], gt[matched_index:matched_index + 1]
            ).item())

            # Observational-only multiview geometry audit. Reconstruct the
            # transformed-view boxes from the same relative features consumed
            # by the fusion head, then compare parameter-free consensus rules
            # against the matched GT. This never changes model predictions.
            supported_boxes = [canonical_box]
            supported_scores = [candidate.score.detach().float()]
            view_qualities = []
            cluster_iou = float(model.candidate_roi_config.get(
                "consensus_cluster_iou", 0.50
            ))
            canonical_center = (canonical_box[:2] + canonical_box[2:]) / 2
            canonical_size = (canonical_box[2:] - canonical_box[:2]).clamp_min(1)
            for offset in (0, 6):
                if float(cross_view[index, offset + 1]) < cluster_iou:
                    view_qualities.append(-1.0)
                    continue
                relative = cross_view[index, offset + 2:offset + 6].detach().float()
                view_center = canonical_center + relative[:2] * canonical_size
                view_size = canonical_size * relative[2:].exp()
                view_box = torch.cat((
                    view_center - view_size / 2,
                    view_center + view_size / 2,
                ))
                supported_boxes.append(view_box)
                supported_scores.append(cross_view[index, offset].detach().float())
                view_qualities.append(float(box_iou(
                    view_box[None], gt[matched_index:matched_index + 1]
                ).item()))
            consensus_boxes = torch.stack(supported_boxes)
            equal_box = consensus_boxes.mean(0)
            median_box = consensus_boxes.median(0).values
            pairwise = box_iou(consensus_boxes, consensus_boxes)
            medoid_box = consensus_boxes[pairwise.sum(1).argmax()]
            score_weights = torch.stack(supported_scores).clamp_min(1e-6)
            score_weighted_box = (
                consensus_boxes * score_weights[:, None]
            ).sum(0) / score_weights.sum()
            detail_probability = (
                semantic_detail[index, -1].detach().float().clamp(0, 1)
                if semantic_detail is not None else canonical_box.new_tensor(0.0)
            )
            independent_view_scores = score_weights[1:]
            weakest_view_probability = (
                independent_view_scores.amin().clamp(0, 1)
                if len(independent_view_scores) else canonical_box.new_tensor(0.0)
            )
            semantic_or_view_probability = 1.0 - (
                1.0 - detail_probability
            ) * (1.0 - weakest_view_probability)
            detail_blended_box = canonical_box + detail_probability * (
                score_weighted_box - canonical_box
            )
            view_blended_box = canonical_box + weakest_view_probability * (
                score_weighted_box - canonical_box
            )
            semantic_or_view_blended_box = canonical_box + semantic_or_view_probability * (
                score_weighted_box - canonical_box
            )
            balanced_reliability = semantic_or_view_probability.clamp_min(0).sqrt()
            balanced_reliability_blended_box = canonical_box + balanced_reliability * (
                score_weighted_box - canonical_box
            )

            def matched_quality(candidate_box: torch.Tensor) -> float:
                return float(box_iou(
                    candidate_box[None], gt[matched_index:matched_index + 1]
                ).item())

            geometry_diagnostics = {
                "canonical_iou": matched_quality(canonical_box),
                "view1_gt_iou": view_qualities[0],
                "view2_gt_iou": view_qualities[1],
                "equal_fusion_iou": matched_quality(equal_box),
                "median_fusion_iou": matched_quality(median_box),
                "medoid_iou": matched_quality(medoid_box),
                "score_weighted_reconstructed_iou": matched_quality(score_weighted_box),
                "detail_blended_iou": matched_quality(detail_blended_box),
                "weak_view_blended_iou": matched_quality(view_blended_box),
                "semantic_or_weak_view_blended_iou": matched_quality(
                    semantic_or_view_blended_box
                ),
                "balanced_reliability_blended_iou": matched_quality(
                    balanced_reliability_blended_box
                ),
                "geometry_detail_probability": float(detail_probability),
                "geometry_weakest_view_probability": float(weakest_view_probability),
            }
        else:
            quality = 0.0
            corrected_iou = 0.0
            geometry_diagnostics = {
                "canonical_iou": 0.0,
                "view1_gt_iou": 0.0,
                "view2_gt_iou": 0.0,
                "equal_fusion_iou": 0.0,
                "median_fusion_iou": 0.0,
                "medoid_iou": 0.0,
                "score_weighted_reconstructed_iou": 0.0,
                "detail_blended_iou": 0.0,
                "weak_view_blended_iou": 0.0,
                "semantic_or_weak_view_blended_iou": 0.0,
                "balanced_reliability_blended_iou": 0.0,
                "geometry_detail_probability": 0.0,
                "geometry_weakest_view_probability": 0.0,
            }
        label = "positive" if quality >= 0.5 else ("negative" if quality < 0.3 else "ambiguous")
        source = "secondary" if candidate.source == "secondary" else "primary"
        key = f"{role}_{source}_{label}"
        raw = float((keep_raw if role == "keep" else rescue_raw)[index].detach().float())
        base = float(candidate.score.detach().float())
        corrected = float(corrected_scores[index])
        entry = accumulator.setdefault(key, {
            "count": 0, "raw_sum": 0.0, "raw_min": float("inf"),
            "raw_max": float("-inf"), "positive_evidence": 0,
            "crossed_boundary": 0, "base_sum": 0.0, "corrected_sum": 0.0,
            "iou_sum": 0.0, "corrected_iou_sum": 0.0,
            "iou_boundary_crossed": 0, "view_support_sum": 0,
            "signal_values": {
                "individual": [], "relation": [], "detail_probability": [],
                "cam_detail_probability": [], "box_detail_probability": [],
                "cross_score": [], "cross_iou": [], "quality_object": [],
                "quality_iou": [], "scene_background_raw": [],
            },
        })
        entry["count"] += 1
        entry["raw_sum"] += raw
        entry["raw_min"] = min(entry["raw_min"], raw)
        entry["raw_max"] = max(entry["raw_max"], raw)
        entry["positive_evidence"] += int(raw > 0)
        entry["crossed_boundary"] += int((base < 0.5) != (corrected < 0.5))
        entry["base_sum"] += base
        entry["corrected_sum"] += corrected
        entry["iou_sum"] += quality
        # Older checkpoints may carry an accumulated diagnostic dictionary
        # without the v46 geometry fields. Keep diagnostics checkpoint-safe;
        # they are observational only and must never stop validation.
        entry["corrected_iou_sum"] = entry.get("corrected_iou_sum", 0.0) + corrected_iou
        entry["iou_boundary_crossed"] = entry.get("iou_boundary_crossed", 0) + int(
            quality < 0.5 <= corrected_iou
        )
        entry["view_support_sum"] += int(candidate.view_support)
        role_column = 0 if role == "keep" else 1
        signals = entry.setdefault("signal_values", {
            "individual": [], "relation": [], "detail_probability": [],
            "cam_detail_probability": [], "box_detail_probability": [],
            "cross_score": [], "cross_iou": [], "quality_object": [],
            "quality_iou": [], "scene_background_raw": [],
        })
        # Checkpoint-safe diagnostic upgrade: old accumulated dictionaries do
        # not contain the two individual screenshot probability lists.
        signals.setdefault("cam_detail_probability", [])
        signals.setdefault("box_detail_probability", [])
        signals.setdefault("scene_background_raw", [])
        signals["individual"].append(float(individual_raw[index, role_column].detach().float()))
        signals["relation"].append(float(relation_raw[index, role_column].detach().float()))
        signals["detail_probability"].append(
            float(semantic_detail[index, -1].detach().float()) if semantic_detail is not None else float("nan")
        )
        signals["cam_detail_probability"].append(
            float(cam_semantic_detail[index, -1].detach().float())
            if cam_semantic_detail is not None else float("nan")
        )
        signals["box_detail_probability"].append(
            float(box_center_semantic_detail[index, -1].detach().float())
            if box_center_semantic_detail is not None else float("nan")
        )
        signals["cross_score"].append(float(cross_view[index, 0].detach().float()))
        signals["cross_iou"].append(float(cross_view[index, 1].detach().float()))
        signals["quality_object"].append(float(quality_raw[index, 0].detach().float().sigmoid()))
        signals["quality_iou"].append(float(quality_raw[index, 1].detach().float().sigmoid()))
        signals["scene_background_raw"].append(
            float(image_relation_raw[index].detach().float())
        )
        crossed_boundary = (base < 0.5) != (corrected < 0.5)
        if crossed_boundary or verbose_records:
            batch = getattr(model, "batch", {})
            image_files = batch.get("im_file", []) if isinstance(batch, dict) else []
            image_file = (
                str(image_files[candidate.batch_index])
                if candidate.batch_index < len(image_files) else ""
            )
            records = getattr(model, "candidate_roi_diagnostic_records", [])
            records.append({
                "image": image_file,
                "source": source,
                "label": label,
                "iou": quality,
                "corrected_iou": corrected_iou,
                "base": base,
                "corrected": corrected,
                "role_raw": raw,
                "individual": signals["individual"][-1],
                "relation": signals["relation"][-1],
                "detail_probability": signals["detail_probability"][-1],
                "cam_detail_probability": signals["cam_detail_probability"][-1],
                "box_detail_probability": signals["box_detail_probability"][-1],
                "cross_score": signals["cross_score"][-1],
                "cross_iou": signals["cross_iou"][-1],
                "cross_score_view2": float(cross_view[index, 6].detach().float()),
                "cross_iou_view2": float(cross_view[index, 7].detach().float()),
                "quality_object": signals["quality_object"][-1],
                "quality_iou": signals["quality_iou"][-1],
                "scene_background_raw": signals["scene_background_raw"][-1],
                "consensus": bool(candidate.consensus),
                "view_support": int(candidate.view_support),
                "pair_id": int(candidate.pair_id),
                "crossed_boundary": bool(crossed_boundary),
                **geometry_diagnostics,
            })
            model.candidate_roi_diagnostic_records = records
    model.candidate_roi_diagnostics = accumulator


def summarize_candidate_diagnostics(model) -> Dict[str, Dict[str, float]]:
    """Convert accumulated candidate sums into compact per-group means."""
    result = {}
    for key, entry in getattr(model, "candidate_roi_diagnostics", {}).items():
        count = max(1, int(entry["count"]))
        result[key] = {
            "count": int(entry["count"]),
            "raw_mean": entry["raw_sum"] / count,
            "raw_min": entry["raw_min"],
            "raw_max": entry["raw_max"],
            "positive_evidence_fraction": entry["positive_evidence"] / count,
            "crossed_boundary": int(entry["crossed_boundary"]),
            "base_mean": entry["base_sum"] / count,
            "corrected_mean": entry["corrected_sum"] / count,
            "iou_mean": entry["iou_sum"] / count,
            "corrected_iou_mean": entry.get("corrected_iou_sum", 0.0) / count,
            "iou_boundary_crossed": int(entry.get("iou_boundary_crossed", 0)),
            "view_support_mean": entry["view_support_sum"] / count,
        }
        for signal_name, values in entry.get("signal_values", {}).items():
            finite = torch.tensor(values, dtype=torch.float32)
            finite = finite[torch.isfinite(finite)]
            if not len(finite):
                continue
            quantiles = torch.quantile(finite, torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0)))
            result[key][f"{signal_name}_mean"] = float(finite.mean())
            result[key][f"{signal_name}_quantiles"] = [float(value) for value in quantiles]
    if os.environ.get("BGD_VERBOSE_CANDIDATE_DIAGNOSTICS") == "1":
        result["_rescue_records"] = getattr(model, "candidate_roi_diagnostic_records", [])
    return result


def _candidate_roi_impl(model, images, profile=False, visualize=False):
    config = model.candidate_roi_config
    if model.training and int(config["stage"]) < 2:
        for module in model.model:
            module.eval()
        model.detail_model.eval() if int(config["stage"]) == 0 else model.detail_model.train()
        model.candidate_roi_fusion.train()
    elif model.training:
        # Full fine-tuning while keeping small-batch BN running statistics stable.
        for child in model.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()

    features, detect_head = _features_before_detect(model, images, profile, visualize)
    for level, feature in enumerate(features):
        if not feature.requires_grad:
            features[level] = feature.detach().requires_grad_(True)
    activations: List[torch.Tensor | None] = [None] * len(features)
    handles = [
        detect_head.cv3[level][1].register_forward_hook(
            lambda _module, _inputs, output, level=level: activations.__setitem__(level, output)
        )
        for level in range(len(features))
    ]
    head_was_training = detect_head.training
    try:
        detect_head.eval()
        first_output = detect_head([feature.clone() for feature in features])
    finally:
        for handle in handles:
            handle.remove()
        detect_head.train(head_was_training)
    decoded, raw = first_output if isinstance(first_output, tuple) else (first_output, None)
    if raw is None or any(activation is None for activation in activations):
        raise RuntimeError("Candidate ROI fusion requires standard Detect decoded/raw outputs and all class activations")
    if model.training:
        # Expose the untouched student YOLO outputs for teacher distillation.
        # Fused residuals are deliberately excluded so Detail rescue remains
        # free to improve over the original detector.
        model.candidate_roi_base_raw = raw
    predicted_candidates = select_all_raw_candidates(decoded, float(config["candidate_conf"]))
    candidates = list(predicted_candidates)
    if model.training and bool(config.get("use_gt_train", True)):
        candidates.extend(_ground_truth_training_candidates(model, images))

    # Compute image-level presence from every FPN level before candidate
    # filtering.  Thus background images with no surviving raw candidate still
    # train the global expert, while candidate fusion reuses the same projected
    # feature maps below.
    projected_scene_levels = [
        model.candidate_roi_fusion.global_projection[level](feature)
        for level, feature in enumerate(features)
    ]
    scene_context_levels = [
        torch.cat((projected.mean((2, 3)), projected.amax((2, 3))), dim=1)
        for projected in projected_scene_levels
    ]
    image_presence_raw = model.candidate_roi_fusion.image_presence_head(
        torch.cat(scene_context_levels, dim=1)
    ).squeeze(1)
    image_presence_aux_loss = (
        _image_presence_loss(model, image_presence_raw, images)
        if model.training else images.new_zeros(())
    )

    cross_view, secondary_proposals, secondary_evidence = _cross_view_evidence(
        model, images, detect_head, candidates
    )
    if secondary_proposals:
        candidates.extend(secondary_proposals)
        cross_view = torch.cat((cross_view, secondary_evidence), 0)
    if not candidates:
        model.candidate_roi_aux_loss = (
            image_presence_aux_loss
            + sum(parameter.sum() for parameter in model.candidate_roi_fusion.parameters()) * 0
            if model.training else images.new_zeros(())
        )
        output = raw if model.training else (decoded, raw)
        return output if model.training else _detach_output_tree(output)
    expected_cross_features = int(model.candidate_roi_fusion.cross_view_features)
    if expected_cross_features == 15 and cross_view.shape[1] == 12:
        source_code = images.new_zeros((len(candidates), 3))
        for index, candidate in enumerate(candidates):
            source_code[index, {"prediction": 0, "secondary": 1, "gt": 2}[candidate.source]] = 1
        cross_view = torch.cat((cross_view, source_code), dim=1)
    elif expected_cross_features != cross_view.shape[1]:
        raise RuntimeError(
            f"Unsupported fusion cross-view width: head={expected_cross_features}, evidence={cross_view.shape[1]}"
        )
    cams = [_classic_gradcam(decoded, activation) for activation in activations]
    levels_and_indices = [
        _candidate_level(candidate, raw)
        if candidate.anchor_index >= 0
        else (candidate_fpn_level(candidate, tuple(images.shape[-2:])), -1)
        for candidate in candidates
    ]
    centers, center_successes, cam_success = [], [], 0
    for candidate, (level, _local_index) in zip(candidates, levels_and_indices):
        center, success = (
            ((candidate.box_xyxy[:2] + candidate.box_xyxy[2:]) / 2, False)
            if candidate.source == "gt"
            else _center_from_cam(
                candidate, cams[level], tuple(images.shape[-2:]),
                _crop_window(model, candidate.batch_index, int(config["crop_size"]))
            )
        )
        centers.append(center)
        center_successes.append(bool(success))
        cam_success += int(success)
    local_detail, semantic_detail, inverse, screenshot_boxes = _unique_crops(
        model, images, candidates, centers
    )
    # Retain the two unfused classifier outputs for explicit v51 supervision
    # and diagnostics. ``semantic_detail`` below remains the deployed
    # probability-weighted representation consumed by the fusion heads.
    cam_semantic_detail = semantic_detail
    candidate_boxes = torch.stack([candidate.box_xyxy for candidate in candidates]).to(
        device=images.device, dtype=features[0].dtype
    )
    candidate_centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) / 2
    # A Grad-CAM peak can occasionally land on contextual texture inside an
    # otherwise accurately localized YOLO box. Encode a second screenshot at
    # the candidate centre, preserving a separate aligned YOLO ROI for it.
    # Every primary proposal receives two independently centred screenshots:
    # one from classic Grad-CAM and one from the candidate box.  Earlier
    # versions skipped the second screenshot for three-view candidates and
    # then protected those candidates unconditionally, which made the Detail
    # branch unable to reject a geometrically stable false positive.
    box_center_list = [center.detach() for center in candidate_centers]
    (
        box_center_detail, box_center_semantic, _box_inverse,
        box_center_screenshot_boxes,
    ) = _unique_crops(model, images, candidates, box_center_list)
    dual_crop_role = torch.tensor([
        candidate.source == "prediction" for candidate in candidates
    ], device=images.device, dtype=torch.bool)
    if os.environ.get("BGD_DISABLE_DUAL_CROP") == "1":
        dual_crop_role.zero_()
    dual_crop_weights = images.new_zeros((len(candidates), 2))
    dual_crop_weights[:, 0] = 1.0
    if semantic_detail is not None and box_center_semantic is not None:
        crop_probabilities = torch.stack((
            semantic_detail[:, -1], box_center_semantic[:, -1]
        ), 1).to(dtype=images.dtype).clamp_min(1e-6)
        crop_policy = os.environ.get("BGD_DUAL_CROP_POLICY", "mean").lower()
        if crop_policy not in {"mean", "cam", "box", "legacy"}:
            raise ValueError(
                "BGD_DUAL_CROP_POLICY must be mean, cam, box, or legacy"
            )
        if crop_policy == "legacy":
            normalized_crop_weights = (
                crop_probabilities / crop_probabilities.sum(1, keepdim=True)
            )
        elif crop_policy == "cam":
            normalized_crop_weights = torch.zeros_like(crop_probabilities)
            normalized_crop_weights[:, 0] = 1.0
        elif crop_policy == "box":
            normalized_crop_weights = torch.zeros_like(crop_probabilities)
            normalized_crop_weights[:, 1] = 1.0
        else:
            # The old probability-weighted pool selected the screenshot that
            # was already most foreground-like.  This creates a positive
            # feedback loop on background candidates and makes the two routes
            # cease to be independent observations.  Equal pooling leaves
            # route selection to the supervised fused-content head, which
            # receives both probabilities separately below.
            normalized_crop_weights = torch.full_like(crop_probabilities, 0.5)
        dual_crop_weights[dual_crop_role] = normalized_crop_weights[dual_crop_role]
        local_detail = (
            dual_crop_weights[:, 0, None, None, None] * local_detail
            + dual_crop_weights[:, 1, None, None, None] * box_center_detail
        )
        semantic_hidden = (
            dual_crop_weights[:, :1] * semantic_detail[:, :-1]
            + dual_crop_weights[:, 1:] * box_center_semantic[:, :-1]
        )
        if crop_policy == "legacy":
            semantic_probability = torch.maximum(
                semantic_detail[:, -1:], box_center_semantic[:, -1:]
            )
        else:
            semantic_probability = (
                dual_crop_weights[:, :1] * semantic_detail[:, -1:]
                + dual_crop_weights[:, 1:] * box_center_semantic[:, -1:]
            )
        semantic_detail = torch.cat((semantic_hidden, semantic_probability), 1)
    candidate_sizes = (candidate_boxes[:, 2:] - candidate_boxes[:, :2]).clamp_min(1)
    cam_centers = torch.stack(centers).to(
        device=images.device, dtype=features[0].dtype
    )
    cam_offset = ((cam_centers - candidate_centers) / candidate_sizes).clamp(-1, 1)
    screenshot_sizes = (
        dual_crop_weights[:, :1]
        * (screenshot_boxes[:, 2:] - screenshot_boxes[:, :2]).to(
            device=images.device, dtype=features[0].dtype
        )
        + dual_crop_weights[:, 1:]
        * (box_center_screenshot_boxes[:, 2:] - box_center_screenshot_boxes[:, :2]).to(
            device=images.device, dtype=features[0].dtype
        )
    ).clamp_min(1)
    crop_ratio = torch.log(screenshot_sizes / candidate_sizes).clamp(-4, 4)
    cam_valid = torch.tensor(
        center_successes, device=images.device, dtype=features[0].dtype
    )[:, None]
    localization_geometry = torch.cat((cam_offset, crop_ratio, cam_valid), 1)
    geometry = _geometry(candidates, tuple(images.shape[-2:]), features[0].dtype, features[0].device)
    keep_raw = images.new_zeros(len(candidates))
    rescue_raw = images.new_zeros(len(candidates))
    quality_raw = images.new_zeros((len(candidates), 2))
    box_raw = images.new_zeros((len(candidates), 4))
    hidden_raw = images.new_zeros((len(candidates), 16))
    context_hidden_raw = images.new_zeros((len(candidates), 16))
    context_keep_raw = images.new_zeros(len(candidates))
    gate_means = []
    for level in range(len(features)):
        indices = [index for index, item in enumerate(levels_and_indices) if item[0] == level]
        if not indices:
            continue
        index_tensor = torch.tensor(indices, device=images.device, dtype=torch.long)
        boxes = torch.stack([candidates[index].box_xyxy for index in indices]).to(images.device)
        aligned_boxes = screenshot_boxes[index_tensor].to(
            device=images.device, dtype=features[level].dtype
        )
        batch_indices = torch.tensor([candidates[index].batch_index for index in indices], device=images.device)
        cam_global_roi = roi_align_tensor(
            features[level], aligned_boxes, batch_indices,
            tuple(images.shape[-2:]), (4, 4),
        )
        box_aligned_boxes = box_center_screenshot_boxes[index_tensor].to(
            device=images.device, dtype=features[level].dtype
        )
        box_global_roi = roi_align_tensor(
            features[level], box_aligned_boxes, batch_indices,
            tuple(images.shape[-2:]), (4, 4),
        )
        crop_weights = dual_crop_weights[index_tensor].to(dtype=cam_global_roi.dtype)
        global_roi = (
            crop_weights[:, 0, None, None, None] * cam_global_roi
            + crop_weights[:, 1, None, None, None] * box_global_roi
        )
        # A candidate-sized ROI alone cannot distinguish shattered glass from
        # visually similar window texture, reflection or rain.  Reuse the same
        # FPN projection for a 2x surrounding ROI and for whole-image moments;
        # this adds context without a second backbone.
        box_center = (boxes[:, :2] + boxes[:, 2:]) / 2
        box_size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(2)
        context_boxes = torch.cat((box_center - box_size, box_center + box_size), dim=1)
        context_boxes[:, (0, 2)] = context_boxes[:, (0, 2)].clamp(0, images.shape[-1])
        context_boxes[:, (1, 3)] = context_boxes[:, (1, 3)].clamp(0, images.shape[-2])
        global_context_roi = roi_align_tensor(
            features[level], context_boxes, batch_indices,
            tuple(images.shape[-2:]), (4, 4),
        )
        scene_context_all = scene_context_levels[level]
        scene_context = scene_context_all[batch_indices]
        base_logits = torch.stack([
            torch.logit(candidates[index].score.clamp(1e-5, 1 - 1e-5)) for index in indices
        ]).to(dtype=global_roi.dtype)
        fusion_base_logits = (
            base_logits
            if bool(config.get("use_base_logit_feature", True))
            else torch.zeros_like(base_logits)
        )
        (
            level_keep, level_rescue, level_box, level_quality, gate,
            level_hidden, level_context_hidden,
        ) = model.candidate_roi_fusion(
            level, global_roi, local_detail[index_tensor].to(dtype=global_roi.dtype),
            fusion_base_logits, geometry[index_tensor], localization_geometry[index_tensor],
            semantic_context=(
                semantic_detail[index_tensor].to(dtype=global_roi.dtype)
                if semantic_detail is not None else None
            ),
            cross_view=cross_view[index_tensor].to(dtype=global_roi.dtype),
            global_context_roi=global_context_roi,
            scene_context=scene_context,
        )
        keep_raw = keep_raw.index_copy(0, index_tensor, level_keep.to(keep_raw.dtype))
        rescue_raw = rescue_raw.index_copy(0, index_tensor, level_rescue.to(rescue_raw.dtype))
        quality_raw = quality_raw.index_copy(
            0, index_tensor, level_quality.to(quality_raw.dtype)
        )
        box_raw = box_raw.index_copy(0, index_tensor, level_box.to(box_raw.dtype))
        hidden_raw = hidden_raw.index_copy(0, index_tensor, level_hidden.to(hidden_raw.dtype))
        context_hidden_raw = context_hidden_raw.index_copy(
            0, index_tensor, level_context_hidden.to(context_hidden_raw.dtype)
        )
        level_context_keep = model.candidate_roi_fusion.keep_context_head(
            level_context_hidden
        ).squeeze(1)
        context_keep_raw = context_keep_raw.index_copy(
            0, index_tensor, level_context_keep.to(context_keep_raw.dtype)
        )
        gate_means.append(gate.detach().mean())

    # The individual branch sees each Grad-CAM/Detail/global ROI in isolation;
    # the relation branch sees the same embedding in the context of every
    # overlapping hypothesis from the two independent views.  Their geometric
    # conjunction implements a conservative two-expert decision without a
    # hand-selected score threshold.
    set_mean, set_max = _multiview_set_context(candidates, hidden_raw)
    relation_raw, relation_embedding = model.candidate_roi_fusion.relation_forward(
        hidden_raw, set_mean, set_max, cross_view.to(dtype=hidden_raw.dtype)
    )
    relation_raw = relation_raw.to(dtype=keep_raw.dtype)
    relation_embedding = relation_embedding.to(dtype=keep_raw.dtype)
    base_scores = torch.stack([candidate.score for candidate in candidates])
    candidate_batch_indices = torch.tensor(
        [candidate.batch_index for candidate in candidates],
        device=images.device, dtype=torch.long,
    )
    candidate_image_presence = image_presence_raw[candidate_batch_indices].to(
        dtype=keep_raw.dtype
    )
    component_raw, component_embedding, component_mask = (
        model.candidate_roi_fusion.component_forward(
            candidates,
            hidden_raw,
            context_hidden_raw,
            semantic_detail.to(dtype=hidden_raw.dtype) if semantic_detail is not None else None,
            cross_view.to(dtype=hidden_raw.dtype),
            base_scores,
            quality_raw.to(dtype=hidden_raw.dtype),
        )
    )
    relation_raw = torch.stack((
        relation_raw[:, 0],
        torch.where(component_mask, component_raw.to(relation_raw.dtype), relation_raw[:, 1]),
    ), dim=1)
    relation_embedding = torch.where(
        component_mask[:, None],
        component_embedding.to(relation_embedding.dtype),
        relation_embedding,
    )
    individual_raw = torch.stack((keep_raw, rescue_raw, context_keep_raw), dim=1)
    classifier_keep = -0.5 * (individual_raw[:, 0] + relation_raw[:, 0])
    classifier_rescue = 0.5 * (individual_raw[:, 1] + relation_raw[:, 1])
    deployment_raw = model.candidate_roi_fusion.deployment_forward(
        individual_raw,
        relation_raw,
        quality_raw,
        semantic_detail.to(dtype=hidden_raw.dtype) if semantic_detail is not None else None,
        cross_view.to(dtype=hidden_raw.dtype),
        base_scores,
    ).to(dtype=keep_raw.dtype)
    image_relation_raw, image_relation_mask = (
        model.candidate_roi_fusion.image_relation_forward(
            candidates,
            hidden_raw,
            context_hidden_raw,
            base_scores,
            cam_semantic_detail.to(dtype=hidden_raw.dtype)
            if cam_semantic_detail is not None else None,
            box_center_semantic.to(dtype=hidden_raw.dtype)
            if box_center_semantic is not None else None,
            quality_raw.to(dtype=hidden_raw.dtype),
            individual_raw[:, 0].to(dtype=hidden_raw.dtype),
            relation_raw[:, 0].to(dtype=hidden_raw.dtype),
            cross_view.to(dtype=hidden_raw.dtype),
            float(config.get("deployment_conf", 0.50)),
        )
    )
    image_relation_raw = image_relation_raw.to(dtype=keep_raw.dtype)
    if (
        os.environ.get("BGD_DISABLE_SCENE_BACKGROUND_EXPERT") == "1"
        or os.environ.get("BGD_DISABLE_IMAGE_RELATION_EXPERT") == "1"
    ):
        image_relation_raw = torch.zeros_like(image_relation_raw)
        image_relation_mask = torch.zeros_like(image_relation_mask)
    dual_content_raw, dual_role_raw = model.candidate_roi_fusion.dual_content_forward(
        hidden_raw,
        context_hidden_raw,
        cam_semantic_detail.to(dtype=hidden_raw.dtype)
        if cam_semantic_detail is not None else None,
        box_center_semantic.to(dtype=hidden_raw.dtype)
        if box_center_semantic is not None else None,
        quality_raw.to(dtype=hidden_raw.dtype),
        cross_view.to(dtype=hidden_raw.dtype),
    )
    dual_content_raw = dual_content_raw.to(dtype=keep_raw.dtype)
    dual_role_raw = dual_role_raw.to(dtype=keep_raw.dtype)
    # Learn the evidence combination end to end, while retaining the safe
    # one-way residual contract.  A fresh zero-initialized deployment head is
    # exactly the original YOLO prediction.  Training can only turn positive
    # background evidence into suppression and positive foreground evidence
    # into rescue; neither role can accidentally move a score the wrong way.
    # v71 uses one newly calibrated content margin for both one-way roles.
    # Negative content evidence may only suppress an existing detection;
    # positive content evidence may only rescue a supported missed candidate.
    # The role masks below keep those directions disjoint.  Older deployment,
    # scene and prototype heads remain representation auxiliaries but no
    # longer decide scores, preventing inherited calibration drift from being
    # mixed with the new signed head.
    keep_raw = dual_role_raw[:, 0].to(dtype=keep_raw.dtype)
    primary_three_view_consensus = torch.tensor(
        [
            candidate.source == "prediction" and candidate.consensus
            for candidate in candidates
        ],
        device=images.device,
        dtype=torch.bool,
    )
    rescue_raw = dual_role_raw[:, 1].to(dtype=keep_raw.dtype)
    rescue_quality_support = _rescue_quality_support(quality_raw)
    keep_prototype_similarity = (
        model.candidate_roi_fusion.keep_prototype_forward(hidden_raw)
        .to(dtype=keep_raw.dtype)
    )

    # The pretrained Detail classifier and the learned fused-ROI object
    # quality are independent object experts. For a primary hypothesis that
    # lacks three-view consensus, combine their odds geometrically (mean
    # log-odds) and expose only negative evidence. This is a probability-level
    # product-of-experts fusion: it has no selected confidence threshold and
    # can never boost or reject a three-view-protected YOLO detection.
    semantic_suppress_delta = torch.zeros_like(keep_raw)
    unprotected_semantic_suppress_delta = torch.zeros_like(keep_raw)
    fused_object_logit = torch.zeros_like(keep_raw)
    if semantic_detail is not None:
        detail_logit = torch.logit(
            semantic_detail[:, -1].float().clamp(1e-4, 1 - 1e-4)
        ).to(dtype=keep_raw.dtype)
        fused_object_logit = 0.5 * (
            detail_logit + quality_raw[:, 0].to(dtype=keep_raw.dtype)
        )
        unprotected_semantic_suppress_delta = fused_object_logit.clamp(
            min=-float(config["max_negative_delta"]), max=0.0
        )
        semantic_suppress_delta = unprotected_semantic_suppress_delta

    base_logits = torch.stack([torch.logit(candidate.score.clamp(1e-5, 1 - 1e-5)) for candidate in candidates])
    # Build the two inference roles before applying either residual. A primary
    # deployment survivor is handled only by the keep head; a low primary
    # consensus candidate or emitted secondary proposal is handled only by the
    # rescue head. Dense anchors and non-emitting secondary candidates remain
    # inspected by Grad-CAM/Detail, but cannot alter detector outputs.
    deployment_conf = float(config.get("deployment_conf", 0.50))
    deployment_nms_iou = float(config.get("deployment_nms_iou", 0.70))
    keep_role = torch.zeros(len(candidates), device=images.device, dtype=torch.bool)
    keep_group_role = torch.zeros_like(keep_role)
    keep_group_leader = torch.arange(
        len(candidates), device=images.device, dtype=torch.long
    )
    keep_group_protected = torch.zeros_like(keep_role)
    keep_group_relation_protected = torch.zeros_like(keep_role)
    rescue_role = torch.tensor(
        [
            candidate.source == "gt"
            or (candidate.source == "secondary" and candidate.emit)
            for candidate in candidates
        ],
        device=images.device,
        dtype=torch.bool,
    )
    for batch_index in range(images.shape[0]):
        high_primary = [
            index for index, candidate in enumerate(candidates)
            if candidate.batch_index == batch_index
            and candidate.source == "prediction"
            and float(base_scores[index].detach()) >= deployment_conf
        ]
        survivor_indices = []
        if high_primary:
            boxes = torch.stack([candidates[index].box_xyxy for index in high_primary]).float()
            scores = base_scores[torch.tensor(high_primary, device=images.device)].detach().float()
            classes = torch.tensor(
                [candidates[index].class_index for index in high_primary],
                device=images.device, dtype=torch.long,
            )
            keep = class_aware_nms(boxes, scores, classes, deployment_nms_iou)
            survivor_indices = [high_primary[index] for index in keep.tolist()]
            keep_role[torch.tensor(survivor_indices, device=images.device)] = True
            # Score correction must be consistent with the NMS component.
            # Otherwise suppressing its representative simply exposes an
            # untouched high-score alias at final NMS. Assign every member to
            # the same-class survivor that suppressed it, using the exact
            # deployment NMS IoU rather than introducing another threshold.
            survivor_tensor = torch.tensor(
                survivor_indices, device=images.device, dtype=torch.long
            )
            high_tensor = torch.tensor(
                high_primary, device=images.device, dtype=torch.long
            )
            survivor_boxes = torch.stack([
                candidates[index].box_xyxy for index in survivor_indices
            ]).float()
            component_iou = box_iou(boxes, survivor_boxes)
            high_classes = torch.tensor(
                [candidates[index].class_index for index in high_primary],
                device=images.device, dtype=torch.long,
            )
            survivor_classes = torch.tensor(
                [candidates[index].class_index for index in survivor_indices],
                device=images.device, dtype=torch.long,
            )
            component_iou = component_iou.masked_fill(
                high_classes[:, None] != survivor_classes[None, :], -1.0
            )
            best_iou, best_survivor = component_iou.max(1)
            grouped = best_iou >= deployment_nms_iou
            grouped_members = high_tensor[grouped]
            grouped_leaders = survivor_tensor[best_survivor[grouped]]
            keep_group_role[grouped_members] = True
            keep_group_leader[grouped_members] = grouped_leaders
            for leader in survivor_tensor.tolist():
                members = grouped_members[grouped_leaders == leader]
                protected = any(
                    candidates[index].consensus for index in members.tolist()
                )
                if protected:
                    keep_group_protected[members] = True
                # A score edit is shared by the full deployment-NMS
                # component.  Therefore the independent relation expert must
                # agree on background for every alias, not just its leader.
                # Any foreground-direction member vetoes only the scene
                # suppression; the learned local fusion path remains intact.
                relation_foreground = bool(
                    (relation_raw[members, 0].detach() <= 0).any()
                )
                if relation_foreground:
                    keep_group_relation_protected[members] = True
        # A low-score independently supported anchor that overlaps an already deployed box
        # is only a duplicate, not a recall opportunity.  Never let it outrank
        # or geometrically replace the pretrained survivor.
        low_supported = [
            index for index, candidate in enumerate(candidates)
            if candidate.batch_index == batch_index
            and candidate.source == "prediction"
            and float(base_scores[index].detach()) < deployment_conf
            and _primary_recall_supported(candidate)
        ]
        if low_supported and survivor_indices:
            low_boxes = torch.stack([candidates[index].box_xyxy for index in low_supported]).float()
            survivor_boxes = torch.stack([
                candidates[index].box_xyxy for index in survivor_indices
            ]).float()
            duplicate = box_iou(low_boxes, survivor_boxes).amax(1) >= float(
                config.get("consensus_cluster_iou", 0.50)
            )
            rescue_role[
                torch.tensor(low_supported, device=images.device)[duplicate]
            ] = False

        # Activate every non-duplicate independently supported candidate. The
        # learned rescue evidence remains positive-only and component NMS
        # below still emits at most one representative, retaining the v53
        # precision safeguards while extending coverage to two-view support.
        if (
            low_supported
            and os.environ.get("BGD_DISABLE_PRIMARY_CONSENSUS_RESCUE") != "1"
        ):
            low_supported_tensor = torch.tensor(
                low_supported, device=images.device, dtype=torch.long
            )
            rescue_role[low_supported_tensor] = True
            if survivor_indices:
                rescue_role[low_supported_tensor[duplicate]] = False

        # Select one learned representative from each overlapping primary
        # recall component.  All raw candidates have already passed through
        # Grad-CAM, Detail and global ROI fusion; this set-level NMS only stops
        # several aliases of the same missed object from crossing the final
        # confidence boundary independently.
        remaining_low = [index for index in low_supported if bool(rescue_role[index])]
        if len(remaining_low) > 1:
            low_tensor = torch.tensor(remaining_low, device=images.device, dtype=torch.long)
            learned_rank = (
                (
                    torch.sigmoid(rescue_raw[low_tensor].detach())
                    * rescue_quality_support[low_tensor].detach()
                ).float()
                + 1e-3 * base_scores[low_tensor].detach().float()
            )
            low_boxes = torch.stack([candidates[index].box_xyxy for index in remaining_low]).float()
            low_classes = torch.tensor(
                [candidates[index].class_index for index in remaining_low],
                device=images.device,
                dtype=torch.long,
            )
            selected = class_aware_nms(
                low_boxes, learned_rank, low_classes,
                float(config.get("consensus_cluster_iou", 0.50)),
            )
            selected_mask = torch.zeros(len(remaining_low), device=images.device, dtype=torch.bool)
            selected_mask[selected] = True
            rescue_role[low_tensor[~selected_mask]] = False
    correction_eligible = keep_role | rescue_role
    # Both heads are zero-initialized, so the fresh method is exactly the
    # pretrained detector. Training can only move a keep logit below zero to
    # suppress a false deployment survivor, or a rescue logit above zero to
    # promote a supported recall proposal.
    evidence_margin = float(config.get("evidence_margin", 0.0))
    # v75 score likelihood fusion.  The shared metric-memory sign determines
    # direction; the independently parameterized dense fused-content margin
    # and Detail/quality product continuously attenuate its magnitude.  This
    # is a soft AND over the Grad-CAM Detail screenshots and YOLO/global 4x4
    # token, with the native zero likelihood-ratio boundary and no detector
    # confidence tuning.
    dense_foreground_support = torch.sigmoid(dual_content_raw)
    semantic_foreground_support = torch.sigmoid(fused_object_logit)
    rescue_evidence = (
        torch.tanh((rescue_raw - evidence_margin).clamp_min(0))
        * dense_foreground_support
        * semantic_foreground_support
    )
    rejection_evidence = -torch.tanh((-keep_raw - evidence_margin).clamp_min(0))
    rescue_delta = (
        float(config["max_score_delta"])
        * rescue_evidence
        * rescue_quality_support
    )
    suppress_delta = float(config["max_negative_delta"]) * rejection_evidence
    score_delta = torch.where(rescue_role, rescue_delta, torch.zeros_like(rescue_delta))
    scene_background = torch.zeros_like(suppress_delta)
    image_absence_agreement_background = torch.zeros_like(suppress_delta)
    prototype_agreement_background = torch.zeros_like(suppress_delta)
    local_agreement_background = torch.zeros_like(suppress_delta)
    if os.environ.get("BGD_V51_ADDITIVE_SEMANTIC_SUPPRESSION") == "1":
        # Exact v51 ablation: either expert can independently suppress and the
        # two negative residuals are additive.
        joint_suppress_delta = (
            suppress_delta + semantic_suppress_delta
        ).clamp(min=-float(config["max_negative_delta"]), max=0.0)
    else:
        # v52 conservative expert agreement. The Detail/global semantic
        # product and the supervised deployment head are independent
        # background experts. Preserve a pretrained detection unless both
        # provide negative evidence, and deploy only the magnitude they share.
        # This continuous minimum introduces neither a confidence threshold
        # nor a trainable parameter and prevents an overconfident Detail crop
        # from vetoing YOLO on its own.
        learned_background = (-suppress_delta).clamp_min(0)
        semantic_background = (-semantic_suppress_delta).clamp_min(0)
        # v75 fixes the v74 deployment mismatch: v74's comment promised two
        # independent background experts, but only the role prototype was
        # actually deployed.  Require the shared prototype, dense fused-
        # content margin, and Detail/quality semantic product all to have a
        # native negative sign.  The smallest continuous magnitude is used,
        # so any foreground-looking expert vetoes suppression without an
        # added threshold.
        dense_content_background = float(config["max_negative_delta"]) * torch.tanh(
            (-dual_content_raw).clamp_min(0)
        )
        strict_quality_background = torch.minimum(
            learned_background, dense_content_background
        )
        if os.environ.get("BGD_V76_HARD_SEMANTIC_VETO") == "1":
            local_agreement_background = torch.minimum(
                strict_quality_background, semantic_background
            )
        else:
            # Detection validity and screenshot semantics answer different
            # questions.  A near-target duplicate can contain real broken-
            # glass texture while remaining a poor detection, so semantic
            # foreground must attenuate rather than completely veto the
            # learned IoU-validity judgement.  The square root is a symmetric
            # geometric balance: uncertain semantics retain useful quality
            # evidence, while confidently empty crops strengthen it.
            semantic_absence_probability = torch.sigmoid(
                -fused_object_logit.float()
            ).to(dtype=strict_quality_background.dtype)
            local_agreement_background = (
                strict_quality_background
                * semantic_absence_probability.clamp(0, 1).sqrt()
            )
        # The absolute scene expert can recover hard background candidates
        # missed by the local deployment head.  It is trained and deployed
        # only for candidates without three-view geometric consensus, and can
        # act only when the independent Detail/global semantic expert also
        # calls the candidate background.
        scene_background = float(config["max_negative_delta"]) * torch.tanh(
            image_relation_raw.clamp_min(0)
        )
        # v63: an absolute scene judgement may suppress a deployed candidate
        # only when the independently learned pair-relation classifier points
        # in the same background direction.  Zero is the classifier's native
        # decision boundary, not a confidence hyperparameter.  This preserves
        # the hard true positives for which the scene context is ambiguous but
        # the candidate-to-context relation remains foreground-like.
        relation_background_direction = relation_raw[:, 0] > 0
        scene_background = torch.where(
            image_relation_mask & relation_background_direction,
            scene_background,
            torch.zeros_like(scene_background),
        )
        scene_semantic_background = (
            -unprotected_semantic_suppress_delta
        ).clamp_min(0)
        scene_agreement_background = torch.minimum(
            scene_background, scene_semantic_background
        )
        # The image-level head is positive for object-present scenes and
        # negative for background-only scenes.  It can edit only unsupported
        # single-view primary detections, and still requires the independent
        # dual-screenshot Detail/global semantic expert to agree on
        # background.  The zero sign is the supervised classifier boundary;
        # no detection-confidence threshold is introduced here.
        single_view_primary = torch.tensor([
            candidate.source == "prediction"
            and candidate.view_support == 1
            and not candidate.consensus
            for candidate in candidates
        ], device=images.device, dtype=torch.bool)
        image_absence_background = float(config["max_negative_delta"]) * torch.tanh(
            (-candidate_image_presence).clamp_min(0)
        )
        if os.environ.get("BGD_DISABLE_IMAGE_PRESENCE_EXPERT") == "1":
            image_absence_background = torch.zeros_like(image_absence_background)
        image_absence_background = torch.where(
            single_view_primary,
            image_absence_background,
            torch.zeros_like(image_absence_background),
        )
        image_absence_agreement_background = torch.minimum(
            image_absence_background, scene_semantic_background
        )
        # v68: compare the complete Grad-CAM/Detail/global fused ROI token to
        # compact foreground and background memories.  This branch has a
        # native zero decision boundary (equal class similarity), operates
        # only on non-consensus candidates, and still requires the independent
        # dense semantic expert to agree.  It therefore targets hard FPs
        # without a selected detector-confidence threshold.
        prototype_background = float(config["max_negative_delta"]) * torch.tanh(
            (
                keep_prototype_similarity[:, 1]
                - keep_prototype_similarity[:, 0]
            ).clamp_min(0)
        )
        if os.environ.get("BGD_DISABLE_KEEP_PROTOTYPE_MEMORY") == "1":
            prototype_background = torch.zeros_like(prototype_background)
        prototype_background = torch.where(
            ~primary_three_view_consensus,
            prototype_background,
            torch.zeros_like(prototype_background),
        )
        prototype_agreement_background = torch.minimum(
            prototype_background, scene_semantic_background
        )
        joint_suppress_delta = -torch.maximum(
            torch.maximum(
                local_agreement_background, scene_agreement_background
            ),
            torch.maximum(
                image_absence_agreement_background,
                prototype_agreement_background,
            ),
        )
    def component_all_member_agreement(
        member_background: torch.Tensor,
    ) -> torch.Tensor:
        """Require continuous background agreement from every NMS alias."""
        component_background = torch.zeros_like(member_background)
        leaders = torch.unique(keep_group_leader[keep_group_role])
        for leader in leaders:
            members = keep_group_role & (keep_group_leader == leader)
            agreed = member_background[members].amin()
            component_background = torch.where(
                members, agreed, component_background
            )
        return component_background

    component_local_background = component_all_member_agreement(
        local_agreement_background
    )

    if os.environ.get("BGD_V51_ADDITIVE_SEMANTIC_SUPPRESSION") == "1":
        group_suppress_delta = joint_suppress_delta[keep_group_leader]
        group_suppress_delta = torch.where(
            keep_group_protected,
            torch.zeros_like(group_suppress_delta),
            group_suppress_delta,
        )
    else:
        # Replace the old binary "any consensus => protect everything" rule
        # with permutation-invariant component evidence.  A background edit
        # is shared by the NMS component only when every raw alias supplies
        # non-zero evidence in that expert; one foreground-looking member
        # continuously vetoes the edit by reducing the minimum to zero.
        group_local_suppress = -component_local_background
        group_scene_suppress = -component_all_member_agreement(
            scene_agreement_background
        )
        group_presence_suppress = -component_all_member_agreement(
            image_absence_agreement_background
        )
        group_prototype_suppress = -component_all_member_agreement(
            prototype_agreement_background
        )
        # Deploy only the newly calibrated paired-content expert.  The other
        # branches remain trained diagnostics, but combining their inherited
        # signs here would make the v71 experiment uninterpretable and could
        # reintroduce v69/v70's absolute-probability errors.
        group_suppress_delta = group_local_suppress
    score_delta = score_delta + torch.where(
        keep_group_role, group_suppress_delta, torch.zeros_like(group_suppress_delta)
    )
    if not bool(config.get("suppress_high_candidates", True)):
        # Recall-oriented mode: fusion may rescue a missed low-confidence
        # candidate, but can never erase an accepted pretrained detection.
        score_delta = torch.where(rescue_role, rescue_delta, torch.zeros_like(score_delta))
    consensus_scores = None
    if bool(config.get("consensus_rescue", False)):
        if semantic_detail is None:
            raise RuntimeError("consensus rescue requires the pretrained Detail classifier semantics")
        detail_probability = semantic_detail[:, -1].to(dtype=rescue_raw.dtype).clamp(1e-5, 1 - 1e-5)
        cross_probability = cross_view[:, 0].to(dtype=rescue_raw.dtype).clamp(1e-5, 1 - 1e-5)
        learned_probability = torch.sigmoid(rescue_raw).clamp(1e-5, 1 - 1e-5)
        # Product-of-experts in odds space. A candidate is rescued only when
        # the pretrained Detail classifier, the independent flip-scale view,
        # and the learned local/global ROI fusion jointly favor an object.
        consensus_scores = torch.sigmoid(
            torch.logit(detail_probability)
            + torch.logit(cross_probability)
            + torch.logit(learned_probability)
        )
        overlaps_accepted = torch.zeros_like(base_scores, dtype=torch.bool)
        for batch_index in range(images.shape[0]):
            low_indices = [
                index for index, item in enumerate(candidates)
                if item.batch_index == batch_index and float(base_scores[index]) < 0.5
            ]
            accepted_indices = [
                index for index, item in enumerate(candidates)
                if item.batch_index == batch_index and float(base_scores[index]) >= 0.5
            ]
            if low_indices and accepted_indices:
                low_boxes = torch.stack([candidates[index].box_xyxy for index in low_indices]).float()
                accepted_boxes = torch.stack([candidates[index].box_xyxy for index in accepted_indices]).float()
                overlap = box_iou(low_boxes, accepted_boxes).amax(1) >= 0.5
                overlaps_accepted[torch.tensor(low_indices, device=images.device)] = overlap.to(images.device)
        eligible_rescue = (
            (base_scores < 0.5)
            & (consensus_scores >= 0.5)
            & (
                ~overlaps_accepted
                | torch.tensor(
                    [candidate.source == "secondary" for candidate in candidates],
                    device=images.device,
                    dtype=torch.bool,
                )
            )
        )
        # Insert rescues at the lowest accepted priority. Existing YOLO boxes
        # (>0.5) therefore always win NMS, preventing a new proposal from
        # suppressing a baseline true positive.
        # A single nextafter step is lost by the subsequent logit -> sigmoid
        # round trip (and always lost in FP16), producing exactly 0.5 which
        # Ultralytics correctly rejects with its strict `score > conf` test.
        # 0.501 is dtype-stable while remaining below every normal accepted
        # baseline detection, so it changes eligibility, not ranking policy.
        insertion_scores = torch.full_like(consensus_scores, 0.501)
        rescued_scores = torch.where(
            eligible_rescue,
            insertion_scores,
            base_scores,
        )
        corrected_logits = torch.logit(rescued_scores.clamp(1e-5, 1 - 1e-5))
    else:
        corrected_logits = (
            base_logits + score_delta
            if bool(config.get("score_correction", True))
            else base_logits
        )
    max_shift, max_scale = float(config["max_box_shift"]), float(config["max_box_scale"])
    box_delta = torch.cat((max_shift * torch.tanh(box_raw[:, :2]), max_scale * torch.tanh(box_raw[:, 2:])), 1)
    # Stable boxes agree geometrically across transformed views.  v74 applied
    # every learned residual at full strength even to those already-correct
    # high-score boxes, which reduced mAP75.  Preserve baseline survivors in
    # proportion to their strongest independent-view IoU; low-score recall
    # proposals retain the full correction range.  This is continuous and
    # parameter-free, not a selected IoU cutoff.
    geometry_iou_columns = [1] + ([7] if cross_view.shape[1] > 7 else [])
    geometry_agreement = cross_view[:, geometry_iou_columns].float().amax(1).to(
        dtype=box_delta.dtype
    )
    baseline_survivor = base_scores >= deployment_conf
    box_quality_gate = torch.where(
        baseline_survivor,
        (1.0 - geometry_agreement).clamp(0, 1),
        torch.ones_like(geometry_agreement),
    )
    box_delta = box_delta * box_quality_gate[:, None]
    if not model.training:
        _accumulate_candidate_diagnostics(
            model, candidates, images, keep_role, rescue_role,
            keep_raw, rescue_raw, corrected_logits, individual_raw,
            relation_raw, image_relation_raw, quality_raw, semantic_detail,
            cam_semantic_detail, box_center_semantic,
            cross_view, box_delta,
        )

    fused_raw = [value.clone() for value in raw]
    for index, (candidate, (level, local_index)) in enumerate(zip(candidates, levels_and_indices)):
        if candidate.anchor_index < 0:
            continue
        height, width = raw[level].shape[-2:]
        y, x = divmod(local_index, width)
        fused_raw[level][candidate.batch_index, detect_head.reg_max * 4 + candidate.class_index, y, x] = corrected_logits[index]
    final_decoded = _decode_raw(detect_head, fused_raw)
    if bool(config.get("box_correction", True)):
        for index, candidate in enumerate(candidates):
            if candidate.anchor_index < 0 or not bool(correction_eligible[index]):
                continue
            source_box = _candidate_deployment_source_box(
                candidate, index, cross_view, semantic_detail, config
            )
            final_decoded[candidate.batch_index, :4, candidate.anchor_index] = _apply_box_delta(
                source_box.to(final_decoded.dtype), box_delta[index]
            )

    # Secondary boxes have no primary-view anchor to edit. Append them as
    # decoded proposals after they have passed through exactly the same
    # Grad-CAM screenshot, Detail encoder and local/global fusion head.  Zero
    # padding keeps a rectangular batch and is discarded by NMS.
    secondary_by_batch = [
        [index for index, candidate in enumerate(candidates)
         if candidate.batch_index == batch_index and candidate.source == "secondary" and candidate.emit]
        for batch_index in range(images.shape[0])
    ]
    max_secondary = max((len(indices) for indices in secondary_by_batch), default=0)
    if (
        not model.training
        and max_secondary
        and (bool(config.get("score_correction", True)) or bool(config.get("consensus_rescue", False)))
    ):
        extra_decoded = final_decoded.new_zeros(
            (images.shape[0], final_decoded.shape[1], max_secondary)
        )
        for batch_index, indices in enumerate(secondary_by_batch):
            for column, index in enumerate(indices):
                candidate = candidates[index]
                delta = box_delta[index] if bool(config.get("box_correction", True)) else box_delta[index] * 0
                source_box = _candidate_deployment_source_box(
                    candidate, index, cross_view, semantic_detail, config
                )
                extra_decoded[batch_index, :4, column] = _apply_box_delta(
                    source_box.to(final_decoded.dtype), delta.to(final_decoded.dtype)
                )
                extra_decoded[
                    batch_index, 4 + candidate.class_index, column
                ] = torch.sigmoid(corrected_logits[index]).to(final_decoded.dtype)
        final_decoded = torch.cat((final_decoded, extra_decoded), dim=2)

    candidate_aux_loss = (
        _candidate_auxiliary_loss(
            model, candidates, classifier_keep, classifier_rescue,
            keep_raw, rescue_raw, individual_raw, relation_raw, deployment_raw,
            dual_content_raw, dual_role_raw,
            image_relation_raw, image_relation_mask,
            relation_embedding, keep_prototype_similarity,
            quality_raw, semantic_detail, cross_view,
            cam_semantic_detail, box_center_semantic, dual_crop_role,
            screenshot_boxes, box_center_screenshot_boxes,
            corrected_logits, box_delta, images,
        )
        if model.training else images.new_zeros(())
    )
    model.candidate_roi_aux_loss = candidate_aux_loss + image_presence_aux_loss
    model.candidate_roi_runtime = {
        "candidate_count": len(candidates),
        "predicted_candidate_count": len(predicted_candidates),
        "gt_candidate_count": sum(candidate.source == "gt" for candidate in candidates),
        "secondary_candidate_count": sum(candidate.source == "secondary" for candidate in candidates),
        "secondary_emitted_count": sum(
            candidate.source == "secondary" and candidate.emit for candidate in candidates
        ),
        "multiview_supported_count": sum(candidate.view_support >= 2 for candidate in candidates),
        "emitted_component_size_mean": (
            sum(
                candidate.component_size for candidate in candidates
                if candidate.source == "secondary" and candidate.emit
            ) / max(1, sum(
                candidate.source == "secondary" and candidate.emit for candidate in candidates
            ))
        ),
        "keep_role_count": int(keep_role.sum()),
        "keep_group_member_count": int(keep_group_role.sum()),
        "keep_group_protected_count": int(keep_group_protected.sum()),
        "keep_group_relation_protected_count": int(
            keep_group_relation_protected.sum()
        ),
        "rescue_role_count": int(rescue_role.sum()),
        "correction_eligible_count": int(correction_eligible.sum()),
        "deployment_background_fraction": float(
            (deployment_raw.detach()[:, 0] > 0).float().mean()
        ),
        "deployment_background_margin_mean": float(
            deployment_raw.detach()[:, 0].float().clamp_min(0).mean()
        ),
        "dual_content_foreground_fraction": float(
            (dual_content_raw.detach() > 0).float().mean()
        ),
        "dual_content_margin_mean": float(
            dual_content_raw.detach().float().mean()
        ),
        "dual_role_keep_foreground_fraction": float(
            (dual_role_raw.detach()[:, 0] > 0).float().mean()
        ),
        "dual_role_keep_margin_mean": float(
            dual_role_raw.detach()[:, 0].float().mean()
        ),
        "dual_role_rescue_foreground_fraction": float(
            (dual_role_raw.detach()[:, 1] > 0).float().mean()
        ),
        "dual_role_rescue_margin_mean": float(
            dual_role_raw.detach()[:, 1].float().mean()
        ),
        "unique_crop_count": int(inverse.unique().numel()),
        "cam_success_count": cam_success,
        "dual_crop_candidate_count": int(dual_crop_role.sum()),
        "dual_crop_box_preferred_fraction": float(
            (dual_crop_weights[dual_crop_role, 1] > dual_crop_weights[dual_crop_role, 0])
            .float().mean()
        ) if bool(dual_crop_role.any()) else 0.0,
        "score_delta_abs_mean": float(score_delta.detach().abs().mean()),
        "semantic_suppress_delta_abs_mean": float(
            semantic_suppress_delta.detach().abs().mean()
        ),
        "component_local_background_mean": float(
            component_local_background.detach().float().mean()
        ),
        "component_local_background_fraction": float(
            (component_local_background.detach() > 0).float().mean()
        ),
        "expert_agreement_delta_abs_mean": float(
            joint_suppress_delta.detach().abs().mean()
        ),
        "scene_background_candidate_count": int(image_relation_mask.sum()),
        "scene_background_evidence_mean": (
            float(scene_background.detach()[image_relation_mask].float().mean())
            if bool(image_relation_mask.any()) else 0.0
        ),
        "image_presence_logit_mean": float(
            image_presence_raw.detach().float().mean()
        ),
        "image_presence_background_fraction": float(
            (image_presence_raw.detach() < 0).float().mean()
        ),
        "image_absence_evidence_mean": float(
            image_absence_background.detach().float().mean()
        ),
        "keep_prototype_background_fraction": float((
            keep_prototype_similarity.detach()[:, 1]
            > keep_prototype_similarity.detach()[:, 0]
        ).float().mean()),
        "keep_prototype_evidence_mean": float(
            prototype_agreement_background.detach().float().mean()
        ),
        "keep_evidence_mean": float(keep_raw.detach().mean()),
        "keep_evidence_positive_fraction": (
            float((keep_raw.detach()[keep_role] > evidence_margin).float().mean())
            if bool(keep_role.any()) else 0.0
        ),
        "rescue_evidence_mean": float(rescue_raw.detach().mean()),
        "rescue_evidence_positive_fraction": (
            float((rescue_raw.detach()[rescue_role] > evidence_margin).float().mean())
            if bool(rescue_role.any()) else 0.0
        ),
        "rescue_quality_support_mean": (
            float(rescue_quality_support.detach()[rescue_role].float().mean())
            if bool(rescue_role.any()) else 0.0
        ),
        "quality_object_probability_mean": float(
            quality_raw.detach().float()[:, 0].sigmoid().mean()
        ),
        "quality_iou_mean": float(
            quality_raw.detach().float()[:, 1].sigmoid().mean()
        ),
        "box_delta_abs_mean": float(box_delta.detach().abs().mean()),
        "box_quality_gate_mean": float(box_quality_gate.detach().mean()),
        "gate_mean": float(torch.stack(gate_means).mean()) if gate_means else 0.0,
        "cross_view_score_mean": float(cross_view[:, 0].detach().mean()),
        "cross_view_iou_mean": float(cross_view[:, 1].detach().mean()),
        "base_score_quantiles": [
            float(value) for value in torch.quantile(
                base_scores.detach().float(),
                base_scores.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0), dtype=torch.float32),
            )
        ],
        "keep_evidence_quantiles": [
            float(value) for value in torch.quantile(
                keep_raw.detach().float(),
                keep_raw.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0), dtype=torch.float32),
            )
        ],
        "rescue_evidence_quantiles": [
            float(value) for value in torch.quantile(
                rescue_raw.detach().float(),
                rescue_raw.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0), dtype=torch.float32),
            )
        ],
        "score_delta_quantiles": [
            float(value) for value in torch.quantile(
                score_delta.detach().float(),
                score_delta.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0), dtype=torch.float32),
            )
        ],
        "consensus_score_quantiles": (
            [
                float(value) for value in torch.quantile(
                    consensus_scores.detach().float(),
                    consensus_scores.new_tensor((0.0, 0.25, 0.5, 0.75, 1.0), dtype=torch.float32),
                )
            ]
            if consensus_scores is not None else []
        ),
    }
    output = fused_raw if model.training else (final_decoded, fused_raw)
    return output if model.training else _detach_output_tree(output)


def candidate_roi_fusion_predict_once(model, images, profile=False, visualize=False):
    """Checkpoint-safe forward replacement with Grad-CAM enabled in validators."""
    needs_grad = not torch.is_grad_enabled()
    inference_context = torch.inference_mode(False) if torch.is_inference_mode_enabled() else contextlib.nullcontext()
    grad_context = torch.enable_grad() if needs_grad else contextlib.nullcontext()
    with inference_context, grad_context:
        if torch.is_inference_mode_enabled() or needs_grad:
            images = images.detach().clone()
        return _candidate_roi_impl(model, images, profile, visualize)


class CandidateROIFYOLO(YOLO):
    """Ultralytics wrapper for candidate-level Grad-CAM Detail/ROI fusion."""

    def __init__(
        self,
        model,
        detail_weight=None,
        candidate_conf: float = 0.10,
        crop_size: int = 64,
        crop_source: str = "original",
        crop_source_scale: float = 0.0,
        hidden_channels: int = 16,
        max_score_delta: float = 4.0,
        max_negative_delta: float = 1.0,
        evidence_margin: float = 0.0,
        detail_encoder_mode: str = "block",
        max_box_shift: float = 0.15,
        max_box_scale: float = 0.30,
        positive_iou: float = 0.50,
        assignment_iou: float = 0.30,
        aux_weight: float = 0.50,
        box_aux_weight: float = 1.0,
        box_correction: bool = True,
        score_correction: bool = True,
        suppress_high_candidates: bool = True,
        use_base_logit_feature: bool = True,
        consensus_rescue: bool = False,
        secondary_proposals: bool = False,
        use_gt_train: bool = True,
        deployment_conf: float = 0.50,
        deployment_nms_iou: float = 0.70,
        anchor_yolo_weight=None,
        anchor_strength: float = 0.0,
        teacher_yolo_weight=None,
        distill_strength: float = 0.0,
        boundary_distill_strength: float = 0.0,
        scene_lr_multiplier: float = 1.0,
        content_lr_multiplier: float | None = None,
        transformer_fusion: bool = False,
        transformer_dim: int = 32,
        transformer_heads: int = 4,
        stage: int = 0,
        task: str = "detect",
    ):
        super().__init__(model=model, task=task)
        detect_head = self.model.model[-1]
        # Detect input channels are stable and directly observable from its regression branches.
        channels = [int(detect_head.cv2[level][0].conv.in_channels) for level in range(detect_head.nl)]
        device, dtype = next(self.model.parameters()).device, next(self.model.parameters()).dtype
        loaded = getattr(self.model, "candidate_roi_config", {})
        if detail_encoder_mode not in ("block", "classifier"):
            raise ValueError("detail_encoder_mode must be block or classifier")
        checkpoint_compatible_methods = {
            CANDIDATE_ROI_METHOD,
            "gradcam_detail_candidate_roi_v76_metric_aligned_safe_box_fusion",
            "gradcam_detail_candidate_roi_v75_shared_semantic_likelihood_fusion",
            "gradcam_detail_candidate_roi_v74_strict_role_prototype_memory",
            "gradcam_detail_candidate_roi_v73_role_decoupled_content_fusion",
            "gradcam_detail_candidate_roi_v72_decoupled_content_optimization",
            "gradcam_detail_candidate_roi_v70_dual_detail_component_agreement",
            "gradcam_detail_candidate_roi_v71_signed_dual_content_fusion",
            "gradcam_detail_candidate_roi_v69_signed_deployment_margin",
            "gradcam_detail_candidate_roi_v68_keep_prototype_memory",
            "gradcam_detail_candidate_roi_v67_hard_candidate_curriculum",
            "gradcam_detail_candidate_roi_v66_global_image_presence",
            "gradcam_detail_candidate_roi_v65_group_relation_consensus_veto",
            "gradcam_detail_candidate_roi_v64_balanced_background_curriculum",
            "gradcam_detail_candidate_roi_v63_relation_consensus_scene_veto",
            "gradcam_detail_candidate_roi_v62_relation_conditioned_scene_margin",
            "gradcam_detail_candidate_roi_v61_nonconsensus_scene_margin",
            "gradcam_detail_candidate_roi_v60_globally_balanced_scene_supervision",
            "gradcam_detail_candidate_roi_v59_differential_scene_optimization",
            "gradcam_detail_candidate_roi_v58_scene_background_expert",
            "gradcam_detail_candidate_roi_v57_image_relation_precision",
            "gradcam_detail_candidate_roi_v56_reliability_blended_geometry",
            "gradcam_detail_candidate_roi_v55_quality_consistent_rescue",
            "gradcam_detail_candidate_roi_v54_two_view_supported_rescue",
            "gradcam_detail_candidate_roi_v53_primary_consensus_rescue",
            "gradcam_detail_candidate_roi_v52_conservative_expert_agreement",
            "gradcam_detail_candidate_roi_v28_decoupled_context_prototype",
            "gradcam_detail_candidate_roi_v29_detached_joint_finetune",
            "gradcam_detail_candidate_roi_v30_l2sp_joint_finetune",
            "gradcam_detail_candidate_roi_v31_teacher_distilled_joint",
            "gradcam_detail_candidate_roi_v32_primary_multiview_box_fusion",
            "gradcam_detail_candidate_roi_v33_three_view_identity_rescue",
            "gradcam_detail_candidate_roi_v34_semantic_calibrated_three_view",
            "gradcam_detail_candidate_roi_v35_local_global_quality_gate",
            "gradcam_detail_candidate_roi_v36_context_veto_quality_gate",
            "gradcam_detail_candidate_roi_v37_learned_local_global_ranking",
            "gradcam_detail_candidate_roi_v38_interleaved_hardnegative_ranking",
            "gradcam_detail_candidate_roi_v39_allraw_deployment_supervision",
            "gradcam_detail_candidate_roi_v43_boundary_preserved_joint",
            "gradcam_detail_candidate_roi_v44_nearmiss_box_refinement",
            "gradcam_detail_candidate_roi_v45_balanced_nearmiss_refinement",
            "gradcam_detail_candidate_roi_v46_direct_ciou_refinement",
            "gradcam_detail_candidate_roi_v47_spatial_localization",
            "gradcam_detail_candidate_roi_v48_semantic_product_fusion",
            "gradcam_detail_candidate_roi_v49_component_consistent_suppression",
            "gradcam_detail_candidate_roi_v50_dual_screenshot_fusion",
            "gradcam_detail_candidate_roi_v51_dual_screenshot_supervision",
        }
        owns_method = (
            loaded.get("method") in checkpoint_compatible_methods
            and loaded.get("detail_encoder_mode", "block") == detail_encoder_mode
            and hasattr(self.model, "detail_model")
            and hasattr(self.model, "candidate_roi_fusion")
        )
        if owns_method:
            self.detail_load_report = {"source": "checkpoint", "preserved": True}
            self.model.detail_model.to(device=device, dtype=dtype)
            self.model.candidate_roi_fusion.to(device=device, dtype=dtype)
        else:
            if detail_weight is None:
                raise ValueError("A Detail checkpoint is required for a fresh candidate ROI fusion model")
            detail = DetailSemanticEncoder() if detail_encoder_mode == "classifier" else Detail_Net_attn_block()
            self.detail_load_report = (
                _load_full_detail_classifier(detail, Path(detail_weight))
                if detail_encoder_mode == "classifier"
                else load_detail_encoder_pretrained(detail, Path(detail_weight))
            )
            self.model.detail_model = detail.to(device=device, dtype=dtype)
            self.model.candidate_roi_fusion = CandidateROIFusion(
                channels,
                hidden_channels,
                local_channels=80 if detail_encoder_mode == "classifier" else 3,
                semantic_features=513 if detail_encoder_mode == "classifier" else 0,
                cross_view_features=15,
                transformer_fusion=transformer_fusion,
                transformer_dim=transformer_dim,
                transformer_heads=transformer_heads,
            ).to(device=device, dtype=dtype)
        if transformer_fusion and getattr(
            self.model.candidate_roi_fusion, "cross_attention_fusion", None
        ) is None:
            self.model.candidate_roi_fusion.cross_attention_fusion = LightweightCrossAttention(
                hidden_channels,
                attention_dim=transformer_dim,
                heads=transformer_heads,
            ).to(device=device, dtype=dtype)
        if not hasattr(self.model.candidate_roi_fusion, "keep_context_head"):
            # Architecture-safe upgrade for v28-v35 checkpoints.  The added
            # expert is inert until trained and contributes only 17 parameters.
            self.model.candidate_roi_fusion.keep_context_head = nn.Linear(16, 1).to(
                device=device, dtype=dtype
            )
            nn.init.zeros_(self.model.candidate_roi_fusion.keep_context_head.weight)
            nn.init.zeros_(self.model.candidate_roi_fusion.keep_context_head.bias)
        if not hasattr(self.model.candidate_roi_fusion, "deployment_head"):
            # Safe upgrade for v28-v36 checkpoints.  Zero initialization keeps
            # the inherited detector output exact until the learned ranking
            # objective has trained the new evidence combiner.
            cross_features = int(getattr(
                self.model.candidate_roi_fusion, "cross_view_features", 15
            ))
            deployment_features = 3 + 2 + 2 + 1 + cross_features + 1
            self.model.candidate_roi_fusion.deployment_projection = nn.Sequential(
                nn.Linear(deployment_features, 16),
                nn.SiLU(inplace=True),
            ).to(device=device, dtype=dtype)
            self.model.candidate_roi_fusion.deployment_head = nn.Linear(16, 2).to(
                device=device, dtype=dtype
            )
            nn.init.zeros_(self.model.candidate_roi_fusion.deployment_head.weight)
            nn.init.zeros_(self.model.candidate_roi_fusion.deployment_head.bias)
        if not hasattr(self.model.candidate_roi_fusion, "spatial_box_head"):
            # Checkpoint-safe v47 upgrade. Its zero output preserves the
            # inherited detector until the positional path is trained.
            self.model.candidate_roi_fusion.spatial_box_head = nn.Sequential(
                nn.Linear(16 + 5 + 4, 16),
                nn.SiLU(inplace=True),
                nn.Linear(16, 4),
            ).to(device=device, dtype=dtype)
            nn.init.zeros_(self.model.candidate_roi_fusion.spatial_box_head[-1].weight)
            nn.init.zeros_(self.model.candidate_roi_fusion.spatial_box_head[-1].bias)
        if (
            not hasattr(self.model.candidate_roi_fusion, "image_relation_head")
            or int(self.model.candidate_roi_fusion.image_relation_projection[0].in_features) != 16 * 3 + 8
            or loaded.get("method") == "gradcam_detail_candidate_roi_v57_image_relation_precision"
        ):
            # Checkpoint-safe v58 upgrade. v57's head had different relational
            # semantics, so reset it rather than silently reusing incompatible
            # weights. The zero output preserves the inherited detector until
            # the scene-background expert has been trained.
            self.model.candidate_roi_fusion.image_relation_projection = nn.Sequential(
                nn.Linear(16 * 3 + 8, 16),
                nn.SiLU(inplace=True),
            ).to(device=device, dtype=dtype)
            self.model.candidate_roi_fusion.image_relation_head = nn.Linear(
                16, 1
            ).to(device=device, dtype=dtype)
            nn.init.zeros_(self.model.candidate_roi_fusion.image_relation_head.weight)
            nn.init.zeros_(self.model.candidate_roi_fusion.image_relation_head.bias)
        if not hasattr(self.model.candidate_roi_fusion, "image_presence_head"):
            # Checkpoint-safe v66 upgrade.  The global presence classifier is
            # inert until trained and adds only 32 values per FPN level plus
            # one bias parameter.
            self.model.candidate_roi_fusion.image_presence_head = nn.Linear(
                32 * len(channels), 1
            ).to(device=device, dtype=dtype)
            nn.init.zeros_(self.model.candidate_roi_fusion.image_presence_head.weight)
            nn.init.zeros_(self.model.candidate_roi_fusion.image_presence_head.bias)
        if not hasattr(self.model.candidate_roi_fusion, "keep_prototypes"):
            # Checkpoint-safe v68 upgrade.  Equal foreground/background
            # memories produce an exact zero similarity margin before
            # supervised training separates their modes.
            shared = torch.empty(1, 4, 16, device=device, dtype=dtype)
            nn.init.normal_(shared, std=0.02)
            self.model.candidate_roi_fusion.keep_prototypes = nn.Parameter(
                shared.repeat(2, 1, 1)
            )
        if not hasattr(self.model.candidate_roi_fusion, "dual_content_head"):
            # Checkpoint-safe v71 upgrade.  This head deliberately starts from
            # zero even when older fusion heads are loaded: the inherited
            # detector and box path are therefore unchanged before training.
            self.model.candidate_roi_fusion.dual_content_projection = nn.Sequential(
                nn.Linear(16 * 3 + 13, 24),
                nn.LayerNorm(24),
                nn.SiLU(inplace=True),
                nn.Linear(24, 12),
                nn.SiLU(inplace=True),
            ).to(device=device, dtype=dtype)
            self.model.candidate_roi_fusion.dual_content_head = nn.Linear(
                12, 1
            ).to(device=device, dtype=dtype)
            nn.init.zeros_(self.model.candidate_roi_fusion.dual_content_head.weight)
            nn.init.zeros_(self.model.candidate_roi_fusion.dual_content_head.bias)
        elif loaded.get("method") == "gradcam_detail_candidate_roi_v71_signed_dual_content_fusion":
            # v71's new branch moved only about 0.007 logit and placed every
            # candidate on the foreground side.  Reinitialize that tiny branch
            # before applying v72's dedicated optimizer scale; reusing the
            # collapsed sign would make the optimization repair ambiguous.
            for module in self.model.candidate_roi_fusion.dual_content_projection.modules():
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    reset()
            self.model.candidate_roi_fusion.dual_content_head.reset_parameters()
            nn.init.zeros_(self.model.candidate_roi_fusion.dual_content_head.weight)
            nn.init.zeros_(self.model.candidate_roi_fusion.dual_content_head.bias)
        if not hasattr(self.model.candidate_roi_fusion, "dual_shared_prototypes"):
            # Checkpoint-safe v75 upgrade.  A v74 checkpoint contributes the
            # mean of its keep/rescue memories; older checkpoints receive
            # equal foreground/background modes for an exact zero margin.
            old_role_prototypes = getattr(
                self.model.candidate_roi_fusion, "dual_role_prototypes", None
            )
            if (
                isinstance(old_role_prototypes, torch.Tensor)
                and tuple(old_role_prototypes.shape) == (2, 2, 4, 12)
            ):
                shared = old_role_prototypes.detach().mean(0).to(
                    device=device, dtype=dtype
                )
            else:
                initial = torch.empty(1, 4, 12, device=device, dtype=dtype)
                nn.init.normal_(initial, std=0.02)
                shared = initial.repeat(2, 1, 1)
            self.model.candidate_roi_fusion.dual_shared_prototypes = nn.Parameter(
                shared.clone()
            )
        # Role-specific v74 memories and v73 linear heads are not part of the
        # effective v75 network. Remove them instead of reporting dead params.
        if hasattr(self.model.candidate_roi_fusion, "dual_role_prototypes"):
            delattr(self.model.candidate_roi_fusion, "dual_role_prototypes")
        # v73's linear role heads are not part of the effective v74 network.
        # Remove them on checkpoint upgrade rather than reporting dead
        # parameters or silently optimizing an unused branch.
        for legacy_name in ("dual_role_keep_head", "dual_role_rescue_head"):
            if hasattr(self.model.candidate_roi_fusion, legacy_name):
                delattr(self.model.candidate_roi_fusion, legacy_name)
        self.model.candidate_roi_config = {
            "method": CANDIDATE_ROI_METHOD,
            "detail_architecture": DETAIL_ARCH_VERSION,
            "detail_encoder_mode": detail_encoder_mode,
            "cam_method": "classic_gradcam_per_fpn_aggregate_all_scores",
            "candidate_selection": "all_raw_no_nms_no_topk",
            "candidate_conf": float(candidate_conf),
            "crop_size": int(crop_size),
            "crop_source": str(crop_source),
            "crop_source_scale": float(crop_source_scale),
            "crop_min_source_size": 64,
            "crop_max_source_size": 256,
            "fusion": (
                "paired_gradcam_and_candidate_center_detail_4x4_with_"
                "same_support_yolo_roi_4x4_equal_route_factorized_semantic_quality_fusion"
            ),
            "cross_view": "three_view_greedy_one_to_one_detail_global_component_and_weighted_box_fusion",
            "cross_view_specs": ((0.83, 3), (0.67, None)),
            "cross_view_features": 15,
            "consensus_cluster_iou": 0.50,
            "correction_policy": "learned_directional_score_and_group_balanced_direct_ciou_bounded_nearmiss_refinement_with_three_view_identity_protection",
            "candidate_discrimination": "learned_local_global_detail_context_multiview_role_ranking",
            "secondary_proposals": bool(secondary_proposals),
            "hidden_channels": int(hidden_channels),
            "transformer_fusion": bool(
                getattr(self.model.candidate_roi_fusion, "cross_attention_fusion", None)
                is not None
            ),
            "transformer_dim": int(transformer_dim),
            "transformer_heads": int(transformer_heads),
            "max_score_delta": float(max_score_delta),
            "max_negative_delta": float(max_negative_delta),
            "evidence_margin": float(evidence_margin),
            "max_box_shift": float(max_box_shift),
            "max_box_scale": float(max_box_scale),
            "positive_iou": float(positive_iou),
            "assignment_iou": float(assignment_iou),
            "aux_weight": float(aux_weight),
            "box_aux_weight": float(box_aux_weight),
            "box_correction": bool(box_correction),
            "score_correction": bool(score_correction),
            "suppress_high_candidates": bool(suppress_high_candidates),
            "use_base_logit_feature": bool(use_base_logit_feature),
            "consensus_rescue": bool(consensus_rescue),
            "use_gt_train": bool(use_gt_train),
            "supervision": "allraw_deployment_calibration_plus_group_balanced_direct_ciou_nearmiss_box_refinement_pair_prototype_ranking_set_mil",
            "deployment_conf": float(deployment_conf),
            "deployment_nms_iou": float(deployment_nms_iou),
            "anchor_yolo_weight": str(anchor_yolo_weight) if anchor_yolo_weight else "",
            "anchor_strength": float(anchor_strength),
            "teacher_yolo_weight": str(teacher_yolo_weight) if teacher_yolo_weight else "",
            "distill_strength": float(distill_strength),
            "boundary_distill_strength": float(boundary_distill_strength),
            "scene_lr_multiplier": float(scene_lr_multiplier),
            "content_lr_multiplier": float(
                scene_lr_multiplier
                if content_lr_multiplier is None else content_lr_multiplier
            ),
            "stage": int(stage),
        }
        self.model.candidate_roi_config.update({
            "correction_policy": (
                "conservative_deployment_and_dual_screenshot_semantic_expert_agreement_"
                "two_of_three_view_supported_primary_recall_rescue_"
                "continuous_fused_object_and_iou_quality_product_rescue_"
                "detail_and_weakest_view_reliability_blended_keep_geometry_"
                "nonconsensus_group_relation_consensus_scene_background_semantic_agreement_suppression_"
                "global_image_presence_single_view_veto_"
                "fused_roi_foreground_background_prototype_memory_"
                "signed_zero_boundary_deployment_calibration_"
                "all_primary_dual_screenshot_detail_global_product_of_experts_"
                "continuous_all_member_nms_component_background_agreement_"
                "signed_dual_content_relative_calibration_"
                "decoupled_new_head_optimization_"
                "strict_iou_shared_keep_rescue_multimode_semantic_likelihood_memory_"
                "continuous_semantic_reliability_weighted_detection_quality_"
                "and_crossview_geometry_preservation_"
                "metric_aligned_iou75_refinement_and_stable_box_identity_"
                "directional_score_and_"
                "gradcam_attention_spatial_direct_ciou_"
                "bounded_nearmiss_box_refinement"
            ),
            "supervision": (
                "allraw_deployment_calibration_plus_gradcam_attention_position_aware_"
                "dual_screenshot_crop_overlap_labeled_class_balanced_detail_semantics_"
                "two_view_supported_primary_and_secondary_group_balanced_direct_ciou_nearmiss_"
                "box_refinement_pair_prototype_ranking_quality_consistent_set_mil_"
                "nonconsensus_scene_background_global_balance_signed_margin_supervision_"
                "balanced_background_image_curriculum_global_presence_signed_margin_"
                "train_only_candidate_hardness_rank_curriculum"
                "_balanced_keep_prototype_classification"
                "_signed_unit_margin_deployment_calibration"
                "_all_primary_dual_screenshot_supervision"
                "_all_primary_signed_dual_content_and_image_hardest_risk"
                "_decoupled_new_head_learning_rate"
                "_globally_balanced_strict_iou_shared_keep_rescue_multimode_likelihood_risk"
                "_metric_aligned_below_iou75_refinement_stable_box_identity"
            ),
        })
        self.model.candidate_roi_anchor_parameters = {}
        if anchor_yolo_weight is not None and float(anchor_strength) > 0:
            anchor_model, _ = attempt_load_one_weight(
                str(anchor_yolo_weight), device=device
            )
            anchor_named = dict(anchor_model.named_parameters())
            self.model.candidate_roi_anchor_parameters = {
                name: anchor_named[name].detach().to(device=device, dtype=parameter.dtype).clone()
                for name, parameter in self.model.named_parameters()
                if name in anchor_named and anchor_named[name].shape == parameter.shape
            }
            LOGGER.info(
                "Candidate ROI L2-SP anchor: "
                f"{sum(value.numel() for value in self.model.candidate_roi_anchor_parameters.values()):,} "
                f"parameters from {anchor_yolo_weight}"
            )
        register_candidate_roi_teacher(self.model, None)
        if teacher_yolo_weight is not None and (
            float(distill_strength) > 0 or float(boundary_distill_strength) > 0
        ):
            teacher, _ = attempt_load_one_weight(str(teacher_yolo_weight), device=device)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            register_candidate_roi_teacher(self.model, teacher)
            LOGGER.info(
                "Candidate ROI teacher distillation: "
                f"{sum(parameter.numel() for parameter in teacher.parameters()):,} frozen "
                f"parameters from {teacher_yolo_weight}"
            )
        self.model._predict_once = self.model.candidate_roi_fusion_predict_once
        self.model.is_candidate_roi_fusion = True
        self.configure_stage(stage)

    def configure_stage(self, stage: int):
        stage = int(stage)
        for parameter in self.model.parameters():
            parameter.requires_grad_(stage == 2)
        if stage in (0, 1):
            for parameter in self.model.candidate_roi_fusion.parameters():
                parameter.requires_grad_(True)
        if stage == 0 and self.model.candidate_roi_config.get("detail_encoder_mode", "block") == "block":
            # The classifier checkpoint pretrains all three route backbones;
            # only these tiny 80->3 feature adapters are new.
            for name in ("conv_l", "conv_m", "conv_s", "bn_cat"):
                for parameter in getattr(self.model.detail_model, name).parameters():
                    parameter.requires_grad_(True)
        if stage == 1:
            for parameter in self.model.detail_model.parameters():
                parameter.requires_grad_(True)
        if stage not in (0, 1, 2):
            raise ValueError("stage must be 0 (fusion), 1 (fusion+Detail), or 2 (whole model)")
        dfl = getattr(getattr(self.model.model[-1], "dfl", None), "conv", None)
        if dfl is not None:
            for parameter in dfl.parameters():
                parameter.requires_grad_(False)
        self.model.candidate_roi_config["stage"] = stage
        count = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        LOGGER.info(f"Candidate ROI fusion stage {stage}: trainable parameters={count:,}")

    def train(self, **kwargs):
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        if kwargs.get("cfg"):
            overrides = yaml_load(check_yaml(kwargs["cfg"]))
        overrides.update(kwargs)
        overrides.update(mode="train", pretrained=False)
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
        # Checkpoints may contain diagnostics accumulated during their final
        # training validation. A new validation call must describe only the
        # requested split, otherwise train/val statistics leak into test
        # analysis and can dwarf the small number of deployed rescue boxes.
        self.model.candidate_roi_diagnostics = {}
        self.model.candidate_roi_diagnostic_records = []
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
