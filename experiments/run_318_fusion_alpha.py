"""Train the best318-era BGD framework from its two independent pretrained weights.

This intentionally does not initialize anything from ``best318.pt``.  It uses
the YOLO and Detail checkpoints that preceded the best318 joint run, restores
the ZIP-era 4x4 Detail-to-Detect feature path, and adds one fixed logit mixing
hyperparameter::

    output = base_yolo + alpha * (zip_fused - base_yolo)

Thus alpha=0 is the original YOLO classification output and alpha=1 is the
original ZIP/best318 direct fusion rule.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import tempfile
import textwrap
import types
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO = ROOT / "yolo-runs" / "train" / "train10" / "weights" / "best.pt"
DEFAULT_DETAIL = ROOT / "run" / "detail_net_attn.pt"
DEFAULT_ARCHIVE = ROOT / "what is code.zip"
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
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument(
        "--direct-fusion", action="store_true",
        help="use the original best318 direct logit replacement without introducing fusion_alpha",
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
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


def install_legacy_modules(archive_path: Path, directory: str, use_fusion_alpha: bool = True):
    # Reuse the import-only dependency shims and corrected residual forward
    # already audited by the exact best318 reproduction runner.
    sys.path.insert(0, str(ROOT / "experiments"))
    from eval_best318_legacy import install_compatibility_shims, patch_legacy_detail_forward

    install_compatibility_shims()
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(directory)
    legacy_root = Path(directory) / "what is code"
    sys.path.insert(0, str(legacy_root))
    sys.path.insert(1, str(ROOT))
    importlib.import_module("Resnet6")
    importlib.import_module("gradcam")
    legacy_val = importlib.import_module("val")
    # The archived ZIP contains the later compressed Detail constructor, while
    # best318 used the historical wide 16/32/64/128 constructor that remains in
    # the project root.  Load that stable class under its original module name.
    detail_source = (ROOT / "Resnet6.py").read_text(encoding="utf-8")
    marker = "# Active public Detail implementation."
    if marker not in detail_source:
        raise RuntimeError("Could not locate the historical wide Detail definition")
    detail_module = types.ModuleType("Resnet6")
    detail_module.__file__ = str(ROOT / "Resnet6.py")
    sys.modules["Resnet6"] = detail_module
    exec(compile(detail_source.split(marker, 1)[0], detail_module.__file__, "exec"), detail_module.__dict__)
    legacy_val.Detail_Net_attn_block = detail_module.Detail_Net_attn_block
    patch_legacy_detail_forward(detail_module)
    if use_fusion_alpha:
        patch_fusion_alpha(legacy_val)
    return legacy_val


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
        "        out = base_out + float(self.fusion_alpha) * (fused_out - base_out)\n"
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


def grad_enabled_318_predict_once(detection_model, images, profile=False, visualize=False):
    """Keep Grad-CAM usable inside the trainer's inference-mode validation."""
    original = detection_model.bgd_318_original_predict_once
    if not detection_model.training or torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        with torch.inference_mode(False), torch.enable_grad():
            images = images.detach().clone().requires_grad_(True)
            return original(images, profile, visualize)
    return original(images, profile, visualize)


def build_model(args: argparse.Namespace, legacy_val):
    model = legacy_val.BGD_YOLO(
        str(args.yolo_weight),
        model_weight_only_yolo=True,
        detail_weight=str(args.detail_weight),
        target_layers=[-1],
        detail_mode=True,
        conf_threshold=args.candidate_conf,
        iou_threshold=0.5,
        head_select=args.head_select,
    )
    level = {"high": 0, "mid": 1, "low": 2}[args.head_select]
    final_classifier = model.model.model[-1].cv3[level][2]
    feature_channels = int(final_classifier.in_channels)
    detail = model.detail_model
    # These layers did not belong to the independent Detail classifier
    # checkpoint; best318 also created them freshly for joint training.
    detail.conv_for_yolo = torch.nn.Conv2d(feature_channels + 3, feature_channels, 3, padding=1)
    detail.bn_for_yolo = torch.nn.BatchNorm2d(feature_channels)
    detail.relu_for_yolo = torch.nn.ReLU()
    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    detail.to(device=device, dtype=dtype)
    model.model.detail_model = detail
    fusion_config = {
        "method": (
            "best318_framework_direct_fusion_v1"
            if args.direct_fusion else "best318_framework_fixed_logit_alpha_v1"
        ),
        "yolo_weight": str(args.yolo_weight.resolve()),
        "detail_weight": str(args.detail_weight.resolve()),
        "initializes_from_best318": False,
        "candidate_conf": float(args.candidate_conf),
        "head_select": args.head_select,
        "target_layers": [-1],
        "fusion": (
            "zip_fused_logit (original direct replacement)"
            if args.direct_fusion else "base_logit + alpha * (zip_fused_logit - base_logit)"
        ),
    }
    if not args.direct_fusion:
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
        "best318_framework_direct_fusion_v1",
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
    wrapper.conf_threshold = float(config["candidate_conf"])
    wrapper.head_select = config["head_select"]
    wrapper.target_layers = list(config["target_layers"])
    wrapper.validator = None
    wrapper.batch = None
    return model


def print_configuration(args: argparse.Namespace, model) -> None:
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print("BGD_318_ALPHA_CONFIG " + json.dumps({
        **model.model.bgd_318_alpha_config,
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
        "resume_training": args.resume_training,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.resume_training and (args.mode != "train" or args.checkpoint is None):
        raise ValueError("--resume-training requires train mode and --checkpoint")
    if not 0.0 <= args.fusion_alpha <= 1.0:
        raise ValueError("fusion-alpha must be in [0, 1]")
    required_paths = [args.archive, args.data]
    if args.checkpoint is not None:
        required_paths.append(args.checkpoint)
    else:
        required_paths.extend((args.yolo_weight, args.detail_weight))
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    os.environ["WANDB_MODE"] = args.wandb_mode

    with tempfile.TemporaryDirectory(prefix="bgd318_alpha_") as directory:
        legacy_val = install_legacy_modules(
            args.archive, directory, use_fusion_alpha=not args.direct_fusion
        )
        model = load_joint_checkpoint(args) if args.checkpoint is not None else build_model(args, legacy_val)
        print_configuration(args, model)

        if args.mode == "val":
            metrics = model.val(
                data=str(args.data), split=args.split, imgsz=640,
                conf=args.eval_conf, iou=args.eval_iou, batch=args.eval_batch,
                workers=args.workers, device=args.device,
                project=str(ROOT / "experiments" / "runs"), name=args.name,
            )
            print("EXACT_METRICS " + json.dumps({
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map75": float(metrics.box.map75),
                "map50_95": float(metrics.box.map),
                "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
            }, sort_keys=True))
            return

        wandb_run = None
        if args.wandb_mode != "disabled":
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.name,
                mode=args.wandb_mode,
                id=args.wandb_id,
                resume="allow" if args.wandb_id else None,
                config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                tags=(
                    "best318-framework", "independent-pretraining",
                    "direct-fusion" if args.direct_fusion else "fusion-alpha",
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
                batch=args.batch, imgsz=640, workers=args.workers, device=args.device,
                patience=args.patience, save_period=args.save_period,
                project=str(ROOT / "experiments" / "runs"), name=args.name,
                rect=args.rect_train, **geometry, **train_kwargs,
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()


if __name__ == "__main__":
    main()
