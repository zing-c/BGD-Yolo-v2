"""Import compatibility helpers required by the YOLO-FineDet runner."""

import builtins
import importlib.machinery
import pickle
import sys
import types

import cv2
import numpy as np
import torch

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
