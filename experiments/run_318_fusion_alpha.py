"""Train the best318-era BGD framework from its two independent pretrained weights.

This intentionally does not initialize anything from ``best318.pt``. It uses
the YOLO and Detail checkpoints that preceded the best318 joint run and
restores the ZIP-era 4x4 Detail-to-Detect feature path. It supports fixed or
learnable single-head logit mixing::

    output = base_yolo + alpha * (zip_fused - base_yolo)

Learnable alpha is represented as ``sigmoid(fusion_alpha_logit_bias)`` so it
remains in (0, 1). It also supports the original direct single-head rule and
direct multi-head replacement; direct modes do not attach or log alpha.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import textwrap
import types
from contextlib import nullcontext
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO = ROOT / "yolo-runs" / "train" / "train10" / "weights" / "best.pt"
DEFAULT_DETAIL = ROOT / "run" / "detail_net_attn.pt"
DEFAULT_COMPRESSED_DETAIL = ROOT / "run" / "detail_net_atten" / "zip_compressed_d12_v1.pt"
DEFAULT_DATA = ROOT / "experiments" / "bg_local.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "val"))
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="joint checkpoint to evaluate or resume; preserves the trained YOLO, Detail, and fusion weights",
    )
    parser.add_argument(
        "--resume-training", action="store_true",
        help="resume train mode from --checkpoint, including optimizer, EMA, epoch, and scheduler state",
    )
    parser.add_argument("--yolo-weight", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--detail-weight", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument(
        "--detail-architecture",
        choices=("historical_wide",),
        default="historical_wide",
        help="historical wide Detail encoder used by the corrected release",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument(
        "--learnable-alpha", action="store_true",
        help=(
            "optimize a sigmoid-constrained scalar alpha initialized by "
            "--fusion-alpha; fixed-alpha behavior remains the default"
        ),
    )
    parser.add_argument(
        "--direct-fusion", action="store_true",
        help="reproduce the original best318 direct replacement, including its legacy scatter behavior",
    )
    parser.add_argument(
        "--corrected-direct-fusion", action="store_true",
        help=(
            "use single-head direct replacement with a corrected Detail scatter: "
            "write features in-place and map center coordinates from [x, y] to "
            "feature-map indexing [y, x]"
        ),
    )
    parser.add_argument(
        "--scatter-reduce", choices=("sum", "mean"), default="sum",
        help=(
            "aggregation for overlapping Detail patches in corrected single-head "
            "Direct mode; mean divides each feature-map location by its coverage count"
        ),
    )
    parser.add_argument(
        "--corrected-geometry", action="store_true",
        help=("opt into geometry v5: correct CAM H/W, track augmented native-image crops, "
              "scatter their actual projected footprints and isolate CAM gradients; "
              "requires --corrected-direct-fusion or input64 --multi-head-direct"),
    )
    parser.add_argument(
        "--geometry-fixed-grid", action="store_true",
        help=("keep all geometry v5 coordinate/gradient fixes, but inject the original "
              "4x4 Detail output at its actual crop center without footprint resizing; "
              "requires --corrected-geometry and --head-select mid/low (P4/P5)"),
    )
    parser.add_argument(
        "--geometry-input-support", action="store_true",
        help=("keep geometry fixes and the original CAM target, inject a fixed 64-input-pixel "
              "context (P3=8x8, P4=4x4, P5=2x2); requires --corrected-geometry"),
    )
    parser.add_argument(
        "--multi-head-direct", action="store_true",
        help="directly fuse every head in --multi-heads without introducing fusion_alpha",
    )
    parser.add_argument(
        "--multi-head-scale-aware", action="store_true",
        help=(
            "resize the 4x4 Detail output before multi-head scatter so a 64x64 "
            "crop occupies 64/stride cells at each level (P3=8, P4=4, P5=2 "
            "for imgsz=640); requires --multi-head-direct"
        ),
    )
    parser.add_argument(
        "--pre-detect-p4-direct", action="store_true",
        help=(
            "hook the P4 neck C2f immediately before Detect, concatenate its "
            "128-channel feature with the 3-channel spatial Detail feature, "
            "and directly recompute the complete P4 classification branch"
        ),
    )
    parser.add_argument(
        "--multi-heads", nargs="+", choices=("high", "mid", "low"),
        default=("high", "mid", "low"),
        help="Detect heads used by --multi-head-direct (default: all three)",
    )
    parser.add_argument("--candidate-conf", type=float, default=0.1)
    parser.add_argument("--head-select", choices=("high", "mid", "low"), default="mid")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--seed", type=int, default=0,
        help=(
            "random seed used both before constructing fresh fusion layers and by "
            "the Ultralytics trainer"
        ),
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="use automatic mixed precision, matching the best318 training run",
    )
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--rect-train", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--name", default="train10_detailpretrain_318frame_alpha05_e50")
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-conf", type=float, default=0.5)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--eval-batch", type=int, default=1)
    parser.add_argument("--wandb-project", default="BGD-YOLO")
    parser.add_argument("--wandb-entity", default="zing_c-ningbo-university")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-id", default=None, help="existing W&B run ID to resume")
    return parser.parse_args()


def install_legacy_modules(
    detail_architecture: str = "historical_wide",
    use_fusion_alpha: bool = True,
    use_corrected_direct_fusion: bool = False,
    use_multi_head_direct: bool = False,
    use_pre_detect_p4_direct: bool = False,
):
    # Reuse the import-only dependency shims and corrected residual forward
    # already audited by the exact best318 reproduction runner.
    sys.path.insert(0, str(ROOT / "experiments"))
    from eval_best318_legacy import install_compatibility_shims, patch_legacy_detail_forward

    install_compatibility_shims()
    legacy_root = ROOT / "experiments" / "legacy_bgd"
    sys.path.insert(0, str(legacy_root))
    sys.path.insert(1, str(ROOT))
    importlib.import_module("Resnet6")
    importlib.import_module("gradcam")
    legacy_val = importlib.import_module("val")
    if detail_architecture == "historical_wide":
        # Golden used the historical wide 16/32/64/128 constructor that remains
        # before the public-implementation marker in Resnet6.py.
        detail_source = (ROOT / "Resnet6.py").read_text(encoding="utf-8")
        marker = "# Active public Detail implementation."
        if marker not in detail_source:
            raise RuntimeError("Could not locate the historical wide Detail definition")
        detail_module = types.ModuleType("Resnet6")
        detail_module.__file__ = str(ROOT / "Resnet6.py")
        sys.modules["Resnet6"] = detail_module
        exec(
            compile(detail_source.split(marker, 1)[0], detail_module.__file__, "exec"),
            detail_module.__dict__,
        )
        legacy_val.Detail_Net_attn_block = detail_module.Detail_Net_attn_block
        patch_legacy_detail_forward(detail_module)
    elif detail_architecture == "zip_compressed_d12_v1":
        from detail_model import Detail_Net_attn_block

        legacy_val.Detail_Net_attn_block = Detail_Net_attn_block
    else:
        raise ValueError(f"Unsupported Detail architecture: {detail_architecture}")
    if use_pre_detect_p4_direct:
        patch_pre_detect_p4_direct(legacy_val)
    elif use_multi_head_direct:
        patch_multi_head_direct(legacy_val)
    elif use_corrected_direct_fusion:
        patch_corrected_single_head_direct(legacy_val)
    elif use_fusion_alpha:
        patch_fusion_alpha(legacy_val)
    return legacy_val


def patch_corrected_single_head_direct(legacy_val) -> None:
    """Correct and scale-align the single-head Detail scatter without alpha.

    ``find_max_heatmap_center_torch`` returns centers in ``[x, y]`` order,
    whereas a BCHW tensor must be indexed spatially as ``[y, x]``.  The ZIP
    implementation also discarded the result of the out-of-place ``add``.
    This patch fixes both issues and resizes the nominal 64x64 crop feature to
    cover ``64 / stride`` grid cells (P3=8, P4=4, P5=2), while preserving the
    original Detect-internal hook and direct classification-logit replacement.
    """
    original = textwrap.dedent(inspect.getsource(legacy_val.BGD_YOLO._predict_once))
    start_marker = "        detail_feature = torch.zeros([B, 3, self.activations[0].size()[2]"
    end_marker = "\n\n\n        concat_feature"
    if original.count(start_marker) != 1:
        raise RuntimeError("Could not uniquely locate the legacy single-head scatter block")
    start = original.index(start_marker)
    end = original.index(end_marker, start)
    new = textwrap.indent(textwrap.dedent("""
        detail_feature = torch.zeros(
            [B, 3, self.activations[0].size(2), self.activations[0].size(3)]
        ).to(self.device)
        feature_h = self.activations[0].size(2)
        feature_w = self.activations[0].size(3)
        resize_y = input_shape[2] / feature_h
        resize_x = input_shape[3] / feature_w
        patch_h = max(1, int(round(64 / resize_y)))
        patch_w = max(1, int(round(64 / resize_x)))

        runtime_config = getattr(self.model, "bgd_318_alpha_config", {})
        scatter_reduce = (
            runtime_config.get("scatter_reduce", "sum")
            if isinstance(runtime_config, dict) else "sum"
        )
        if scatter_reduce not in {"sum", "mean"}:
            raise RuntimeError(f"Unsupported corrected scatter reduction: {scatter_reduce}")
        detail_count = (
            torch.zeros([B, 1, feature_h, feature_w]).to(self.device)
            if scatter_reduce == "mean" else None
        )

        current_box_idx = 0
        for b in range(B):
            if idx2b[b] is None:
                continue
            detail_slice = range(current_box_idx, idx2b[b])
            detail4batch = detail_results[detail_slice]
            current_box_idx = idx2b[b]
            for ib in range(len(detail4batch)):
                current_box_output = detail4batch[ib]
                center_x = int(torch.round(centers[b][ib, 0] / resize_x).item())
                center_y = int(torch.round(centers[b][ib, 1] / resize_y).item())
                if current_box_output.shape[-2:] != (patch_h, patch_w):
                    detail_patch = current_box_output.unsqueeze(0)
                    if patch_h <= current_box_output.size(-2) and patch_w <= current_box_output.size(-1):
                        input_h, input_w = current_box_output.shape[-2:]
                        if input_h % patch_h == 0 and input_w % patch_w == 0:
                            kernel_h = input_h // patch_h
                            kernel_w = input_w // patch_w
                            detail_patch = torch.nn.functional.avg_pool2d(
                                detail_patch,
                                kernel_size=(kernel_h, kernel_w),
                                stride=(kernel_h, kernel_w),
                            )
                        else:
                            detail_patch = torch.nn.functional.interpolate(
                                detail_patch, size=(patch_h, patch_w), mode="area"
                            )
                    else:
                        detail_patch = torch.nn.functional.interpolate(
                            detail_patch, size=(patch_h, patch_w), mode="bilinear",
                            align_corners=False
                        )
                    current_box_output = detail_patch.squeeze(0)
                y0 = max(0, min(center_y - patch_h // 2, feature_h - patch_h))
                x0 = max(0, min(center_x - patch_w // 2, feature_w - patch_w))
                detail_feature[
                    b, :, y0:y0 + patch_h, x0:x0 + patch_w
                ].add_(current_box_output)
                if detail_count is not None:
                    detail_count[
                        b, :, y0:y0 + patch_h, x0:x0 + patch_w
                    ].add_(1.0)

        if detail_count is not None:
            detail_feature = detail_feature / detail_count.clamp_min(1.0)
    """).strip("\n"), "        ")
    namespace = dict(legacy_val.__dict__)
    exec(compile(original[:start] + new + original[end:],
                 "<best318-corrected-scaleaware-single-head-direct-predict-once>",
                 "exec"), namespace)
    patched = namespace["_predict_once"]
    patched.__module__ = legacy_val.__name__
    patched.__qualname__ = "BGD_YOLO._predict_once"
    legacy_val.BGD_YOLO._predict_once = patched


def patch_pre_detect_p4_direct(legacy_val) -> None:
    """Fuse Detail into the 128-channel P4 neck tensor before Detect.

    The original ZIP implementation hooks ``Detect.cv3[1][1]`` and therefore
    sees a 64-channel tensor.  This variant hooks model layer 18, the P4 C2f
    that feeds Detect, projects ``128 + 3`` channels back to 128, and runs the
    complete P4 classification branch.  Box regression remains untouched.
    """
    original = textwrap.dedent(inspect.getsource(legacy_val.BGD_YOLO._predict_once))
    needle = (
        "        out = m.cv3[self.head_dict[self.head_select]][2](concat_feature.to("
        "m.cv3[self.head_dict[self.head_select]][2].weight.dtype)).squeeze(1)\n"
        "        x[1][self.head_dict[self.head_select]][:, -1, ...] = out"
    )
    replacement = (
        "        selected_level = self.head_dict[self.head_select]\n"
        "        classifier = m.cv3[selected_level]\n"
        "        classifier_dtype = next(classifier.parameters()).dtype\n"
        "        out = classifier(concat_feature.to(classifier_dtype)).squeeze(1)\n"
        "        x[1][selected_level][:, -1, ...] = out"
    )
    if original.count(needle) != 1:
        raise RuntimeError("Could not uniquely patch the pre-Detect P4 fusion assignment")
    namespace = dict(legacy_val.__dict__)
    exec(compile(original.replace(needle, replacement),
                 "<best318-pre-detect-p4-direct-predict-once>", "exec"), namespace)
    patched = namespace["_predict_once"]
    patched.__module__ = legacy_val.__name__
    patched.__qualname__ = "BGD_YOLO._predict_once"
    legacy_val.BGD_YOLO._predict_once = patched


def patch_fusion_alpha(legacy_val) -> None:
    """Inject a baseline-preserving scalar at the ZIP fused class logits."""
    original = inspect.getsource(legacy_val.BGD_YOLO._predict_once)
    source = textwrap.dedent(original)
    needle = (
        "        out = m.cv3[self.head_dict[self.head_select]][2](concat_feature.to("
        "m.cv3[self.head_dict[self.head_select]][2].weight.dtype)).squeeze(1)\n"
        "        x[1][self.head_dict[self.head_select]][:, -1, ...] = out"
    )
    replacement = (
        "        fused_out = m.cv3[self.head_dict[self.head_select]][2](concat_feature.to("
        "m.cv3[self.head_dict[self.head_select]][2].weight.dtype)).squeeze(1)\n"
        "        selected_level = self.head_dict[self.head_select]\n"
        "        base_out = x[1][selected_level][:, -1, ...].clone()\n"
        "        if hasattr(self.model, 'fusion_alpha_logit_bias'):\n"
        "            fusion_alpha = torch.sigmoid(self.model.fusion_alpha_logit_bias)\n"
        "        else:\n"
        "            fusion_alpha = self.fusion_alpha\n"
        "        out = base_out + fusion_alpha * (fused_out - base_out)\n"
        "        x[1][selected_level][:, -1, ...] = out"
    )
    if source.count(needle) != 1:
        raise RuntimeError("Could not uniquely patch the ZIP fusion assignment")
    namespace = dict(legacy_val.__dict__)
    exec(compile(source.replace(needle, replacement), "<best318-alpha-predict-once>", "exec"), namespace)
    patched = namespace["_predict_once"]
    patched.__module__ = legacy_val.__name__
    patched.__qualname__ = "BGD_YOLO._predict_once"
    legacy_val.BGD_YOLO._predict_once = patched


def patch_multi_head_direct(legacy_val) -> None:
    """Replace classification logits at several Detect levels directly.

    One backward pass captures ordered P3/P4/P5 activations and gradients. The
    resulting CAMs are aggregated once for crop localization; Detail is run
    once, then its features are scattered and projected independently at each
    requested feature-map scale. New scale-aware models resize the nominal
    4x4 Detail feature to cover ``64 / stride`` cells (P3=8, P4=4, P5=2).
    Centers returned as ``[x, y]`` are mapped to BCHW spatial indices as
    ``[y, x]``. Checkpoints explicitly tagged with legacy/fixed-4x4 scatter
    retain their historical placement so previously reported results remain
    reproducible.
    """
    original = textwrap.dedent(inspect.getsource(legacy_val.BGD_YOLO._predict_once))
    start_marker = "        detail_feature = torch.zeros([B, 3, self.activations[0].size()[2]"
    end_marker = "        x[1][self.head_dict[self.head_select]][:, -1, ...] = out"
    if original.count(start_marker) != 1 or original.count(end_marker) != 1:
        raise RuntimeError("Could not uniquely locate the ZIP single-head fusion block")
    start = original.index(start_marker)
    end = original.index(end_marker, start) + len(end_marker)
    replacement = textwrap.indent(textwrap.dedent("""
            if len(self.activations) != len(self.multi_head_indices) or any(
                activation is None for activation in self.activations
            ):
                raise RuntimeError("Not all requested Detect head hooks produced activations")

            runtime_config = getattr(self.model, "bgd_318_alpha_config", {})
            legacy_xy_transpose = (
                isinstance(runtime_config, dict)
                and runtime_config.get("scatter") == "legacy"
            )
            scale_aware_scatter = (
                isinstance(runtime_config, dict)
                and runtime_config.get("multi_head_scale_aware", False)
            )

            for slot, (level_index, activation) in enumerate(zip(
                self.multi_head_indices, self.activations
            )):
                feature_h, feature_w = activation.shape[-2:]
                detail_feature = torch.zeros(
                    [B, 3, feature_h, feature_w],
                    device=self.device,
                    dtype=detail_results.dtype,
                )
                resize_y = input_shape[2] / feature_h
                resize_x = input_shape[3] / feature_w
                patch_h = max(1, int(round(64 / resize_y))) if scale_aware_scatter else 4
                patch_w = max(1, int(round(64 / resize_x))) if scale_aware_scatter else 4
                current_box_idx = 0
                for b in range(B):
                    if idx2b[b] is None:
                        continue
                    detail_slice = range(current_box_idx, idx2b[b])
                    detail4batch = detail_results[detail_slice]
                    current_box_idx = idx2b[b]
                    for ib, current_box_output in enumerate(detail4batch):
                        center_x = centers[b][ib, 0] / resize_x
                        center_y = centers[b][ib, 1] / resize_y
                        if legacy_xy_transpose:
                            center_h, center_w = center_x, center_y
                        else:
                            center_h, center_w = center_y, center_x
                        if scale_aware_scatter:
                            if current_box_output.shape[-2:] != (patch_h, patch_w):
                                detail_patch = current_box_output.unsqueeze(0)
                                input_h, input_w = current_box_output.shape[-2:]
                                if patch_h <= input_h and patch_w <= input_w:
                                    if input_h % patch_h == 0 and input_w % patch_w == 0:
                                        detail_patch = torch.nn.functional.avg_pool2d(
                                            detail_patch,
                                            kernel_size=(input_h // patch_h, input_w // patch_w),
                                            stride=(input_h // patch_h, input_w // patch_w),
                                        )
                                    else:
                                        detail_patch = torch.nn.functional.interpolate(
                                            detail_patch, size=(patch_h, patch_w), mode="area"
                                        )
                                else:
                                    detail_patch = torch.nn.functional.interpolate(
                                        detail_patch, size=(patch_h, patch_w), mode="bilinear",
                                        align_corners=False,
                                    )
                                current_box_output = detail_patch.squeeze(0)
                            fixed_y = int(torch.round(center_h).item())
                            fixed_x = int(torch.round(center_w).item())
                            y0 = max(0, min(fixed_y - patch_h // 2, feature_h - patch_h))
                            x0 = max(0, min(fixed_x - patch_w // 2, feature_w - patch_w))
                            detail_feature[
                                b, :, y0:y0 + patch_h, x0:x0 + patch_w
                            ].add_(current_box_output)
                        else:
                            # Preserve the fixed-4x4 geometry used by archived
                            # multi-head checkpoints.
                            fixed_y = int(max(2, min(center_h, feature_h - 3)))
                            fixed_x = int(max(2, min(center_w, feature_w - 3)))
                            detail_feature[
                                b, :,
                                max(fixed_y - 2, 0):min(fixed_y + 2, feature_h - 1),
                                max(fixed_x - 2, 0):min(fixed_x + 2, feature_w - 1),
                            ].add_(current_box_output)

                concat_feature = torch.cat(
                    (detail_feature, activation.to(self.device)), dim=1
                )
                conv = self.detail_model.conv_for_yolo_multi[slot]
                bn = self.detail_model.bn_for_yolo_multi[slot]
                relu = self.detail_model.relu_for_yolo_multi[slot]
                concat_feature = conv(concat_feature.to(conv.weight.dtype))
                concat_feature = bn(concat_feature.to(bn.weight.dtype))
                concat_feature = relu(concat_feature)
                classifier = m.cv3[level_index][2]
                out = classifier(concat_feature.to(classifier.weight.dtype)).squeeze(1)
                x[1][level_index][:, -1, ...] = out
    """).strip("\n"), "        ")
    namespace = dict(legacy_val.__dict__)
    exec(compile(original[:start] + replacement + original[end:],
                 "<best318-multi-head-direct-predict-once>", "exec"), namespace)
    patched = namespace["_predict_once"]
    patched.__module__ = legacy_val.__name__
    patched.__qualname__ = "BGD_YOLO._predict_once"
    legacy_val.BGD_YOLO._predict_once = patched

    def set_multi_head_hook(self):
        count = len(self.multi_head_indices)

        def capture(slot):
            def hook(_module, _inputs, output):
                if len(self.activations) != count:
                    self.activations = [None] * count
                if len(self.gradients) != count:
                    self.gradients = [None] * count
                self.activations[slot] = output
                if output.requires_grad:
                    def store(gradient):
                        if len(self.gradients) != count:
                            self.gradients = [None] * count
                        self.gradients[slot] = gradient
                    output.register_hook(store)
            return hook

        detect = self.model.model[-1]
        for slot, level_index in enumerate(self.multi_head_indices):
            layer = detect.cv3[level_index][1]
            self.handles.append(layer.register_forward_hook(capture(slot)))

    set_multi_head_hook.__module__ = legacy_val.__name__
    set_multi_head_hook.__qualname__ = "BGD_YOLO.set_hook"
    legacy_val.BGD_YOLO.set_hook = set_multi_head_hook


def grad_enabled_318_predict_once(detection_model, images, profile=False, visualize=False):
    """Keep Grad-CAM usable inside the trainer's inference-mode validation."""
    original = detection_model.bgd_318_original_predict_once
    config = getattr(detection_model, "bgd_318_alpha_config", {})
    if config.get("geometry_version") == 5:
        from experiments.direct_geometry_fusion import geometry_predict_once
        def original(value, profile=False, visualize=False):
            return geometry_predict_once(detection_model, value, profile, visualize)
    if not detection_model.training or torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        with torch.inference_mode(False), torch.enable_grad():
            images = images.detach().clone().requires_grad_(True)
            return original(images, profile, visualize)
    return original(images, profile, visualize)


def build_model(args: argparse.Namespace, legacy_val):
    target_layers = [18] if args.pre_detect_p4_direct else [-1]
    compressed_detail = args.detail_architecture == "zip_compressed_d12_v1"
    model = legacy_val.BGD_YOLO(
        str(args.yolo_weight),
        model_weight_only_yolo=True,
        # The legacy loader expects a raw historical state_dict.  The compact
        # checkpoint is metadata-wrapped, so load it below with strict coverage
        # and shape reporting instead of silently accepting zero matched keys.
        detail_weight=None if compressed_detail else str(args.detail_weight),
        target_layers=target_layers,
        detail_mode=True,
        conf_threshold=args.candidate_conf,
        iou_threshold=0.5,
        head_select=args.head_select,
    )
    head_dict = {"high": 0, "mid": 1, "low": 2}
    level = head_dict[args.head_select]
    detect = model.model.model[-1]
    detail = model.detail_model
    detail_preload = None
    if compressed_detail:
        from gradcam_fusion import load_detail_encoder_pretrained

        detail_preload = load_detail_encoder_pretrained(detail, args.detail_weight)
    # These layers did not belong to the independent Detail classifier
    # checkpoint; best318 also created them freshly for joint training.
    if args.pre_detect_p4_direct:
        feature_channels = int(detect.cv3[level][0].conv.in_channels)
        detail.conv_for_yolo = torch.nn.Conv2d(
            feature_channels + 3, feature_channels, 3, padding=1
        )
        detail.bn_for_yolo = torch.nn.BatchNorm2d(feature_channels)
        detail.relu_for_yolo = torch.nn.ReLU()
    elif args.multi_head_direct:
        head_names = list(dict.fromkeys(args.multi_heads))
        head_indices = [head_dict[name] for name in head_names]
        feature_channels = [int(detect.cv3[index][2].in_channels) for index in head_indices]
        detail.conv_for_yolo_multi = torch.nn.ModuleList([
            torch.nn.Conv2d(channels + 3, channels, 3, padding=1)
            for channels in feature_channels
        ])
        detail.bn_for_yolo_multi = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(channels) for channels in feature_channels
        ])
        detail.relu_for_yolo_multi = torch.nn.ModuleList([
            torch.nn.ReLU() for _ in feature_channels
        ])
        model.multi_head_names = head_names
        model.multi_head_indices = head_indices
        model.m_layers = [detect.cv3[index][1] for index in head_indices]
    else:
        final_classifier = detect.cv3[level][2]
        feature_channels = int(final_classifier.in_channels)
        detail.conv_for_yolo = torch.nn.Conv2d(feature_channels + 3, feature_channels, 3, padding=1)
        detail.bn_for_yolo = torch.nn.BatchNorm2d(feature_channels)
        detail.relu_for_yolo = torch.nn.ReLU()
    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    detail.to(device=device, dtype=dtype)
    model.model.detail_model = detail
    method = (
        (
            "best318_framework_multi_head_direct_lightdetail_d12_scaleaware_sum_v3"
            if compressed_detail
            else "best318_framework_multi_head_direct_scaleaware_sum_v3"
        )
        if args.multi_head_direct and args.multi_head_scale_aware
        else (
        "best318_framework_multi_head_direct_lightdetail_d12_xycorrected_v2"
        if args.multi_head_direct and compressed_detail
        else (
            "best318_framework_pre_detect_p4_direct_v1"
            if args.pre_detect_p4_direct else (
                "best318_framework_multi_head_direct_xycorrected_v2"
                if args.multi_head_direct else (
                    (
                        "best318_framework_corrected_single_head_direct_mean_scaleaware_v4"
                        if args.scatter_reduce == "mean"
                        else "best318_framework_corrected_single_head_direct_scaleaware_v3"
                    )
                    if args.corrected_direct_fusion else (
                        "best318_framework_direct_fusion_v1"
                        if args.direct_fusion else (
                            "best318_framework_learnable_logit_alpha_v1"
                            if args.learnable_alpha else "best318_framework_fixed_logit_alpha_v1"
                        )
                    )
                )
            )
        ))
    )
    fusion_config = {
        "method": method,
        "yolo_weight": str(args.yolo_weight.resolve()),
        "detail_weight": str(args.detail_weight.resolve()),
        "detail_architecture": args.detail_architecture,
        "detail_encoder_parameters": sum(
            parameter.numel() for name, parameter in detail.named_parameters()
            if not name.startswith("conv_for_yolo") and not name.startswith("bn_for_yolo")
        ),
        "initializes_from_best318": False,
        "candidate_conf": float(args.candidate_conf),
        "head_select": args.head_select,
        "head_selects": list(args.multi_heads) if args.multi_head_direct else [args.head_select],
        "multi_head_scale_aware": bool(
            args.multi_head_direct and args.multi_head_scale_aware
        ),
        "target_layers": target_layers,
        "scatter": (
            "in-place scale-aware Detail scatter (64/stride grid support) with center [x,y] mapped to BCHW [y,x]"
            if args.corrected_direct_fusion else (
                "in-place scale-aware multi-head Detail scatter (P3=8, P4=4, P5=2 at 640) with sum reduction and center [x,y] mapped to BCHW [y,x]"
                if args.multi_head_direct and args.multi_head_scale_aware else (
                    "in-place 4x4 Detail scatter with center [x,y] mapped to BCHW [y,x]"
                    if args.multi_head_direct else "legacy"
                )
            )
        ),
        "scatter_patch_hw": (
            {"high": [8, 8], "mid": [4, 4], "low": [2, 2]}[args.head_select]
            if args.corrected_direct_fusion and args.imgsz == 640 else (
                {"high": [8, 8], "mid": [4, 4], "low": [2, 2]}
                if args.multi_head_direct and args.multi_head_scale_aware and args.imgsz == 640
                else None
            )
        ),
        "scatter_reduce": (
            args.scatter_reduce if args.corrected_direct_fusion else (
                "sum" if args.multi_head_direct and args.multi_head_scale_aware else None
            )
        ),
        "fusion": (
            "P4 neck C2f (128ch) + spatial Detail (3ch), 131->128, direct P4 cls replacement"
            if args.pre_detect_p4_direct else (
                "zip_fused_logit at every selected Detect level (direct replacement)"
                if args.multi_head_direct else (
                "zip_fused_logit (corrected Detail scatter, direct replacement)"
                if args.corrected_direct_fusion else (
                "zip_fused_logit (original direct replacement)"
                if args.direct_fusion else "base_logit + alpha * (zip_fused_logit - base_logit)"
                )
                )
            )
        ),
    }
    if getattr(args, "corrected_geometry", False):
        from experiments.direct_geometry_fusion import METHOD, METHOD_FIXED_GRID, METHOD_INPUT_SUPPORT, METHOD_MULTI_INPUT_SUPPORT
        fixed_grid = bool(getattr(args, "geometry_fixed_grid", False))
        input_support = bool(getattr(args, "geometry_input_support", False))
        if fixed_grid and input_support:
            raise ValueError("Choose either fixed 4x4 or fixed input-64 support, not both")
        if fixed_grid and args.head_select not in {"mid", "low"}:
            raise ValueError("--geometry-fixed-grid requires the P4/P5 head")
        if args.multi_head_direct and (not input_support or not args.multi_head_scale_aware or args.scatter_reduce != 'sum'):
            raise ValueError('Corrected multi-head geometry requires input64, scale-aware and sum')
        fusion_config.update({
            "method": METHOD_MULTI_INPUT_SUPPORT if args.multi_head_direct else (
                METHOD_INPUT_SUPPORT if input_support else METHOD_FIXED_GRID if fixed_grid else METHOD),
            "geometry_version": 5,
            "geometry_scatter_mode": "input64" if input_support else "fixed_grid" if fixed_grid else "projected",
            "track_cam_values": input_support,
            "crop_size_source_hw": [64, 64],
            "cam_resize": "actual input (H,W), FP32 bilinear align_corners=False",
            "cam_target": "sum of all original sigmoid classification scores",
            "cam_gradients": "autograd.grad w.r.t. Detect activation only; preserve optimizer accumulation",
            "source_geometry": "actual resize + four-tile Mosaic + axis-aligned affine + flip",
            "window_size": "max(1, round(abs(source_to_input_axis_gain)*64)) independently for H/W",
            "scatter": (
                "fixed 64-input-pixel context at actual crop center; BCHW[y,x]; canvas clipping"
                if input_support else
                "fixed 4x4 Detail injection at actual projected crop center; in-place BCHW[y,x]; canvas clipping"
                if fixed_grid else
                "actual clamped source crop projected to input; in-place BCHW[y,x]; visibility clipping"
            ),
            "scatter_patch_hw": (
                {"high": [8, 8], "mid": [4, 4], "low": [2, 2]}[args.head_select]
                if input_support else [4, 4] if fixed_grid else "dynamic floor/ceil projected crop bounds per original image"
            ),
        })
        if args.multi_head_direct:
            fusion_config.update({
                'geometry_multi_head': True,
                'head_selects': head_names,
                'cam_aggregation': 'normalize each head grid, resize, mean across heads, normalize aggregate',
                'scatter_patch_hw': {name: {'high': [8, 8], 'mid': [4, 4], 'low': [2, 2]}[name] for name in head_names},
                'scatter_reduce': 'sum',
                'fusion': 'shared YOLO and Detail, independent 67->64 Conv/BN/ReLU per selected Detect classification head',
            })
    if detail_preload is not None:
        fusion_config["detail_pretrained_coverage"] = float(
            detail_preload["pretrained_coverage"]
        )
        fusion_config["detail_pretrained_tensors_loaded"] = len(detail_preload["loaded"])
        fusion_config["detail_random_output_tensors"] = list(detail_preload["missing"])
    if not (
        args.direct_fusion or args.corrected_direct_fusion
        or args.multi_head_direct or args.pre_detect_p4_direct
    ):
        if args.learnable_alpha:
            initial_alpha = torch.tensor(float(args.fusion_alpha), device=device, dtype=dtype)
            model.model.register_parameter(
                "fusion_alpha_logit_bias",
                torch.nn.Parameter(torch.logit(initial_alpha)),
            )
            fusion_config.update({
                "fusion_alpha_mode": "learnable_sigmoid",
                "fusion_alpha_initial": float(args.fusion_alpha),
                "fusion_alpha_parameter": "sigmoid(fusion_alpha_logit_bias)",
            })
        else:
            model.fusion_alpha = float(args.fusion_alpha)
            fusion_config["fusion_alpha"] = float(args.fusion_alpha)
    model.model.bgd_318_alpha_config = fusion_config
    # Trainer validation runs under torch inference mode.  The original ZIP
    # implementation assumed gradients were globally available and otherwise
    # fails while constructing Grad-CAM after the first training epoch.
    model.model.bgd_318_original_predict_once = model.model._predict_once
    model.model._predict_once = model.model.bgd_318_grad_predict_once
    # The original wrapper attaches Detail to the DetectionModel, so this is a
    # genuine whole-network fine-tune rather than a frozen auxiliary stage.
    for parameter in model.model.parameters():
        parameter.requires_grad_(True)
    return model


def load_joint_checkpoint(args: argparse.Namespace):
    """Load a trained joint checkpoint without rebuilding it from pretraining weights."""
    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    detection_model = model.model
    config = getattr(detection_model, "bgd_318_alpha_config", None)
    supported_methods = {
        "best318_framework_fixed_logit_alpha_v1",
        "best318_framework_learnable_logit_alpha_v1",
        "best318_framework_direct_fusion_v1",
        "best318_framework_corrected_single_head_direct_v2",
        "best318_framework_corrected_single_head_direct_scaleaware_v3",
        "best318_framework_corrected_single_head_direct_mean_scaleaware_v4",
        "best318_framework_multi_head_direct_v1",
        "best318_framework_multi_head_direct_lightdetail_d12_v1",
        "best318_framework_multi_head_direct_xycorrected_v2",
        "best318_framework_multi_head_direct_lightdetail_d12_xycorrected_v2",
        "best318_framework_multi_head_direct_scaleaware_sum_v3",
        "best318_framework_multi_head_direct_lightdetail_d12_scaleaware_sum_v3",
        "best318_framework_pre_detect_p4_direct_v1",
        "best318_framework_corrected_single_head_direct_geometry_v5",
        "best318_framework_corrected_single_head_direct_geometry_v5_fixed4",
        "best318_framework_corrected_single_head_direct_geometry_v5_input64",
        "best318_framework_multi_head_direct_geometry_v5_input64",
    }
    if not isinstance(config, dict) or config.get("method") not in supported_methods:
        raise RuntimeError(f"Not a supported best318 joint checkpoint: {args.checkpoint}")

    original = getattr(detection_model, "bgd_318_original_predict_once", None)
    wrapper = getattr(original, "__self__", None)
    if wrapper is None or not hasattr(detection_model, "detail_model"):
        raise RuntimeError(f"Joint BGD runtime is missing from checkpoint: {args.checkpoint}")

    # Keep the deserialized wrapper pointed at the exact trained model tensors.
    wrapper.model = detection_model
    wrapper.detail_model = detection_model.detail_model
    if "fusion_alpha" in config:
        wrapper.fusion_alpha = float(config["fusion_alpha"])
    if config["method"] == "best318_framework_learnable_logit_alpha_v1":
        parameter = getattr(detection_model, "fusion_alpha_logit_bias", None)
        if not isinstance(parameter, torch.nn.Parameter):
            raise RuntimeError("Learnable-alpha checkpoint is missing fusion_alpha_logit_bias")
    wrapper.conf_threshold = float(config["candidate_conf"])
    wrapper.head_select = config["head_select"]
    wrapper.target_layers = list(config["target_layers"])
    if config.get('geometry_multi_head', False) or config["method"] in {
        "best318_framework_multi_head_direct_v1",
        "best318_framework_multi_head_direct_lightdetail_d12_v1",
        "best318_framework_multi_head_direct_xycorrected_v2",
        "best318_framework_multi_head_direct_lightdetail_d12_xycorrected_v2",
        "best318_framework_multi_head_direct_scaleaware_sum_v3",
        "best318_framework_multi_head_direct_lightdetail_d12_scaleaware_sum_v3",
    }:
        head_dict = {"high": 0, "mid": 1, "low": 2}
        wrapper.multi_head_names = list(config["head_selects"])
        wrapper.multi_head_indices = [head_dict[name] for name in wrapper.multi_head_names]
        detect = detection_model.model[-1]
        wrapper.m_layers = [detect.cv3[index][1] for index in wrapper.multi_head_indices]
    elif config["method"] == "best318_framework_pre_detect_p4_direct_v1":
        wrapper.m_layers = [detection_model.model[18]]
    wrapper.validator = None
    wrapper.batch = None
    if config.get("geometry_version") == 5:
        os.environ["BGD_CORRECTED_GEOMETRY"] = "1"
        os.environ["BGD_INCLUDE_ORI_IMG"] = "0"
    return model


def learnable_alpha_value(detection_model) -> float | None:
    parameter = getattr(detection_model, "fusion_alpha_logit_bias", None)
    if parameter is None:
        return None
    return float(torch.sigmoid(parameter.detach().float()).cpu())


def print_configuration(args: argparse.Namespace, model) -> None:
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    runtime_config = dict(model.model.bgd_318_alpha_config)
    alpha = learnable_alpha_value(model.model)
    if alpha is not None:
        runtime_config["fusion_alpha_current"] = alpha
    print("BGD_318_FUSION_CONFIG " + json.dumps({
        **runtime_config,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "amp": args.amp,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "resume_training": args.resume_training,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }, sort_keys=True))


def main() -> None:
    args = parse_args()
    direct_modes = sum(bool(value) for value in (
        args.direct_fusion, args.corrected_direct_fusion,
        args.multi_head_direct, args.pre_detect_p4_direct
    ))
    if direct_modes > 1:
        raise ValueError(
            "--direct-fusion, --corrected-direct-fusion, --multi-head-direct, "
            "and --pre-detect-p4-direct "
            "are mutually exclusive"
        )
    if args.pre_detect_p4_direct and args.head_select != "mid":
        raise ValueError("--pre-detect-p4-direct requires --head-select mid (P4)")
    if args.multi_head_direct and len(set(args.multi_heads)) < 2:
        raise ValueError("--multi-head-direct requires at least two distinct heads")
    if args.multi_head_scale_aware and not args.multi_head_direct:
        raise ValueError("--multi-head-scale-aware requires --multi-head-direct")
    if args.learnable_alpha and direct_modes:
        raise ValueError("--learnable-alpha cannot be combined with a direct-fusion mode")
    if args.scatter_reduce != "sum" and not args.corrected_direct_fusion:
        raise ValueError("--scatter-reduce mean requires --corrected-direct-fusion")
    if args.corrected_geometry and not (args.corrected_direct_fusion or args.multi_head_direct):
        raise ValueError("--corrected-geometry requires --corrected-direct-fusion or --multi-head-direct")
    if args.corrected_geometry and args.multi_head_direct and not (
        args.geometry_input_support and args.multi_head_scale_aware and args.scatter_reduce == 'sum'
    ):
        raise ValueError('Corrected multi-head geometry requires input64, scale-aware and sum')
    if args.geometry_fixed_grid and (not args.corrected_geometry or args.head_select not in {"mid", "low"}):
        raise ValueError("--geometry-fixed-grid requires --corrected-geometry and --head-select mid/low")
    if args.geometry_input_support and (not args.corrected_geometry or args.geometry_fixed_grid):
        raise ValueError("--geometry-input-support requires --corrected-geometry and cannot combine with fixed-grid")
    if args.corrected_geometry:
        os.environ["BGD_CORRECTED_GEOMETRY"] = "1"
        # Native images are loaded lazily only for selected source tiles. Do
        # not transfer every full-resolution image from dataloader workers.
        os.environ["BGD_INCLUDE_ORI_IMG"] = "0"
        # Fresh Detail spatial projections / fusion layers are created before
        # trainer.train() normally seeds RNG. Seed *before* construction so
        # independent P4/P5 ablations really share the same initialization seed.
        from ultralytics.yolo.utils.torch_utils import init_seeds
        init_seeds(args.seed, deterministic=True)
    if args.resume_training and (args.mode != "train" or args.checkpoint is None):
        raise ValueError("--resume-training requires train mode and --checkpoint")
    if not direct_modes:
        valid_alpha = (
            0.0 < args.fusion_alpha < 1.0
            if args.learnable_alpha else 0.0 <= args.fusion_alpha <= 1.0
        )
        if not valid_alpha:
            interval = "(0, 1)" if args.learnable_alpha else "[0, 1]"
            raise ValueError(f"fusion-alpha must be in {interval}")
    required_paths = [args.data]
    if args.checkpoint is not None:
        required_paths.append(args.checkpoint)
    else:
        required_paths.extend((args.yolo_weight, args.detail_weight))
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["WANDB_MODE"] = args.wandb_mode

    with nullcontext(install_legacy_modules(
            detail_architecture=args.detail_architecture,
            use_fusion_alpha=not direct_modes,
            use_corrected_direct_fusion=args.corrected_direct_fusion,
            use_multi_head_direct=args.multi_head_direct,
            use_pre_detect_p4_direct=args.pre_detect_p4_direct,
        )) as legacy_val:
        model = load_joint_checkpoint(args) if args.checkpoint is not None else build_model(args, legacy_val)
        if model.model.bgd_318_alpha_config.get("track_cam_values", False) and args.mode == "train":
            from experiments.direct_geometry_monitor import reset_cam_epoch, log_cam_epoch
            model.add_callback("on_train_epoch_start", reset_cam_epoch)
            model.add_callback("on_fit_epoch_end", log_cam_epoch)
        print_configuration(args, model)

        if args.mode == "val":
            metrics = model.val(
                data=str(args.data), split=args.split, imgsz=args.imgsz,
                conf=args.eval_conf, iou=args.eval_iou, batch=args.eval_batch,
                workers=args.workers, device=args.device,
                project=str(ROOT / "experiments" / "runs"), name=args.name,
            )
            exact_metrics = {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map75": float(metrics.box.map75),
                "map50_95": float(metrics.box.map),
                "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
            }
            alpha = learnable_alpha_value(model.model)
            if alpha is not None:
                exact_metrics["fusion_alpha"] = alpha
            print("EXACT_METRICS " + json.dumps(exact_metrics, sort_keys=True))
            return

        wandb_run = None
        if args.wandb_mode != "disabled":
            import wandb
            wandb_config = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if not (
                    key == "fusion_alpha"
                    and direct_modes
                )
            }
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.name,
                mode=args.wandb_mode,
                id=args.wandb_id,
                resume="allow" if args.wandb_id else None,
                config=wandb_config,
                tags=(
                    "best318-framework", "independent-pretraining",
                    (
                        "pre-detect-p4-direct" if args.pre_detect_p4_direct
                        else (
                            "multi-head-direct" if args.multi_head_direct
                            else (
                                "corrected-single-head-direct" if args.corrected_direct_fusion
                                else (
                                "direct-fusion" if args.direct_fusion else (
                                    "learnable-alpha" if args.learnable_alpha else "fixed-alpha"
                                )
                                )
                            )
                        )
                    ),
                    "whole-finetune",
                ),
            )
            print(f"WANDB_URL {wandb_run.url}")
        try:
            geometry = {}
            if args.no_augment:
                geometry = {
                    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
                    "degrees": 0.0, "translate": 0.0, "scale": 0.0,
                    "shear": 0.0, "perspective": 0.0,
                    "flipud": 0.0, "fliplr": 0.0,
                }
            train_kwargs = {}
            if args.resume_training:
                train_kwargs["resume"] = str(args.checkpoint.resolve())
            model.train(
                data=str(args.data), epochs=args.epochs,
                optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
                momentum=args.momentum, weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                warmup_momentum=args.warmup_momentum,
                warmup_bias_lr=args.warmup_bias_lr, amp=args.amp,
                batch=args.batch, imgsz=args.imgsz, workers=args.workers, device=args.device,
                seed=args.seed, deterministic=True,
                patience=args.patience, save_period=args.save_period,
                project=str(ROOT / "experiments" / "runs"), name=args.name,
                rect=args.rect_train, **geometry, **train_kwargs,
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()
