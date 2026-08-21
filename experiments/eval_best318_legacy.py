"""Reproduce the serialized best318 ZIP-era Grad-CAM/Detail evaluation."""

import argparse
import builtins
import importlib
import importlib.machinery
import json
import pickle
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, default=ROOT / "best318.pt")
    parser.add_argument("--archive", type=Path, default=ROOT / "what is code.zip")
    parser.add_argument("--data", type=Path, default=ROOT / "experiments" / "bg_local.yaml")
    parser.add_argument("--detail-conf", type=float, default=0.2)
    parser.add_argument("--detail-iou", type=float, default=0.5)
    parser.add_argument("--eval-conf", type=float, default=0.5)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="test_best318_zip_legacy")
    return parser.parse_args()


def install_compatibility_shims():
    """Provide small import-only shims for obsolete optional dependencies."""
    dill = types.ModuleType("dill")
    dill.__spec__ = importlib.machinery.ModuleSpec("dill", loader=None)
    dill.__version__ = "0.3.8"
    # Ultralytics opportunistically uses dill when stripping optimizers.  The
    # compatibility module only needs stdlib pickle semantics for this model.
    for name in ("Pickler", "Unpickler", "dump", "dumps", "load", "loads"):
        setattr(dill, name, getattr(pickle, name))
    dill_impl = types.ModuleType("dill._dill")
    dill_impl.__spec__ = importlib.machinery.ModuleSpec("dill._dill", loader=None)
    dill_impl._load_type = lambda name: getattr(types, name, getattr(builtins, name, None))
    dill._dill = dill_impl
    sys.modules["dill"] = dill
    sys.modules["dill._dill"] = dill_impl

    triton_ops = types.ModuleType("triton.ops")
    block_sparse = types.ModuleType("triton.ops.blocksparse")
    block_sparse.softmax = None
    sys.modules["triton.ops"] = triton_ops
    sys.modules["triton.ops.blocksparse"] = block_sparse

    package = types.ModuleType("pytorch_grad_cam")
    package.__path__ = []
    base_cam = types.ModuleType("pytorch_grad_cam.base_cam")
    utils = types.ModuleType("pytorch_grad_cam.utils")
    utils.__path__ = []
    image_utils = types.ModuleType("pytorch_grad_cam.utils.image")
    svd_utils = types.ModuleType("pytorch_grad_cam.utils.svd_on_activations")
    activation_utils = types.ModuleType("pytorch_grad_cam.activations_and_gradients")

    class ImportOnlyCAM:
        pass

    for name in (
        "GradCAMPlusPlus", "GradCAM", "XGradCAM", "EigenCAM", "HiResCAM",
        "LayerCAM", "RandomCAM", "EigenGradCAM",
    ):
        setattr(package, name, ImportOnlyCAM)
    base_cam.BaseCAM = ImportOnlyCAM
    activation_utils.ActivationsAndGradients = ImportOnlyCAM

    def scale_cam_image(cam, target_size=None):
        scaled = []
        for image in cam:
            image = image - np.min(image)
            image = image / (1e-7 + np.max(image))
            # Trainer validation evaluates the EMA model in FP16 on CUDA.
            # OpenCV resize has no float16 implementation, so normalize the
            # CAM dtype before crossing that library boundary.
            image = np.float32(image)
            if target_size is not None:
                image = cv2.resize(image, target_size)
            scaled.append(image)
        return np.float32(scaled)

    image_utils.scale_cam_image = scale_cam_image
    image_utils.show_cam_on_image = lambda image, _mask, *args, **kwargs: image
    svd_utils.get_2d_projection = lambda activations: activations.sum(axis=1)
    sys.modules.update({
        "pytorch_grad_cam": package,
        "pytorch_grad_cam.base_cam": base_cam,
        "pytorch_grad_cam.utils": utils,
        "pytorch_grad_cam.utils.image": image_utils,
        "pytorch_grad_cam.utils.svd_on_activations": svd_utils,
        "pytorch_grad_cam.activations_and_gradients": activation_utils,
    })


def patch_legacy_detail_forward(detail_module):
    """Support both the ZIP residual fix and the embedded pre-fix checkpoint."""
    nn = torch.nn

    def forward(self, inputs):
        inputs = self.pool(inputs)
        activation = getattr(self, "relu", None)

        def residual(main, shortcut, value):
            output = main(value) + shortcut(value)
            return activation(output) if activation is not None else output

        large = residual(self.large_route_l1, self.shortcut_L_l1, inputs)
        large = self.large_route_pool_1(large)
        large = residual(self.large_route_l2, self.shortcut_L_l2, large)
        large = self.large_route_pool_2(large)
        large = residual(self.large_route_l3, self.shortcut_L_l3, large)
        large = self.large_route_pool_3(large)
        large = residual(self.large_route_l4, self.shortcut_L_l4, large)

        medium = residual(self.medium_route_l1, self.shortcut_M_l1, inputs)
        medium = residual(self.medium_route_l2, self.shortcut_M_l2, medium)
        medium = self.medium_route_pool_1(medium)
        medium = residual(self.medium_route_l3, self.shortcut_M_l3, medium)

        small = residual(self.small_route_l1, self.shortcut_S_l1, inputs)
        small = residual(self.small_route_l2, self.shortcut_S_l2, small)

        merged = torch.cat((
            self.conv_l(self.atten_large(large)),
            self.conv_m(self.atten_medium(medium)),
            self.conv_s(self.atten_small(small)),
        ), dim=1)
        output_activation = activation or getattr(self, "relu4", nn.ReLU())
        return output_activation(self.bn_cat(merged))

    detail_module.Detail_Net_attn_block.forward = forward


def main():
    args = parse_args()
    for path in (args.weight, args.archive, args.data):
        if not path.exists():
            raise FileNotFoundError(path)

    install_compatibility_shims()
    with tempfile.TemporaryDirectory(prefix="best318_zip_eval_") as directory:
        with zipfile.ZipFile(args.archive) as archive:
            archive.extractall(directory)
        legacy_root = Path(directory) / "what is code"
        sys.path.insert(0, str(legacy_root))
        sys.path.insert(1, str(ROOT))
        detail_module = importlib.import_module("Resnet6")
        importlib.import_module("gradcam")
        legacy_val = importlib.import_module("val")
        patch_legacy_detail_forward(detail_module)

        checkpoint = torch.load(args.weight, map_location="cpu", weights_only=False)
        detection_model = checkpoint["model"].float()
        if not isinstance(detection_model.args, dict):
            detection_model.args = vars(detection_model.args).copy()
        legacy_wrapper = detection_model._predict_once.__self__
        legacy_wrapper.model = detection_model
        legacy_wrapper.detail_model = detection_model.detail_model
        # The checkpoint also serializes a bound forward method. Rebind the
        # ZIP source explicitly; otherwise Python can retain the workspace's
        # newer method body while keeping the old object's attributes.
        for method_name in (
            "_predict_once", "preprocess", "val", "set_requires_grad", "set_hook",
            "save_activation", "save_gradient", "get_cam", "compute_cam_per_layer",
            "aggregate_multi_layers", "get_cam_image", "get_cam_weights",
        ):
            if hasattr(legacy_val.BGD_YOLO, method_name):
                setattr(
                    legacy_wrapper, method_name,
                    types.MethodType(getattr(legacy_val.BGD_YOLO, method_name), legacy_wrapper),
                )
        legacy_wrapper.conf_threshold = float(args.detail_conf)
        legacy_wrapper.iou_threshold = float(args.detail_iou)
        legacy_wrapper.validator = None
        legacy_wrapper.batch = None

        original_forward = legacy_wrapper._predict_once

        def grad_enabled_forward(_model, images, profile=False, visualize=False):
            with torch.inference_mode(False), torch.enable_grad():
                images = images.detach().clone().requires_grad_(True)
                return original_forward(images, profile, visualize)

        detection_model._predict_once = types.MethodType(grad_enabled_forward, detection_model)
        print("LEGACY_CONFIG " + json.dumps({
            "detail_conf": args.detail_conf,
            "detail_iou": args.detail_iou,
            "eval_conf": args.eval_conf,
            "eval_iou": args.eval_iou,
            "batch": args.batch,
            "head_select": legacy_wrapper.head_select,
            "target_layers": legacy_wrapper.target_layers,
        }, sort_keys=True))
        metrics = legacy_wrapper.val(
            data=str(args.data), split="test", conf=args.eval_conf, iou=args.eval_iou,
            batch=args.batch, workers=args.workers, device=args.device,
            project=str(ROOT / "experiments" / "runs"), name=args.name,
        )
        print("EXACT_METRICS " + json.dumps({
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50), "map75": float(metrics.box.map75),
            "map50_95": float(metrics.box.map),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
