# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import builtins
from copy import deepcopy
from pathlib import Path
import sys
import types

import timm
import torch
import torch.nn as nn

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv,DSConv,Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment, Detect_DyHead, Fusion, C2f_Faster, C2f_ODConv)
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.yolo.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.yolo.utils.plotting import feature_visualization
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)

from ultralytics.nn.backbone.convnextv2 import *
from ultralytics.nn.backbone.fasternet import *
from ultralytics.nn.backbone.efficientViT import *
from ultralytics.nn.backbone.EfficientFormerV2 import *

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            self.batch = x
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if hasattr(m, 'backbone'):
                x = m(x)
                for _ in range(5 - len(x)):
                    x.insert(0, None)
                for i_idx, i in enumerate(x):
                    if i_idx in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                x = x[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.
        Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
                
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information

        Args:
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)


    def _apply(self, fn):
        """
        `_apply()` is a function that applies a function to all the tensors in the model that are not
        parameters or registered buffers

        Args:
            fn: the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Detect_DyHead, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load the weights into the model.

        Args:
            weights (dict) or (torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        # Fusion paths need the unmodified batch metadata to map Grad-CAM
        # screenshots back to original images and to build candidate targets.
        # Keep it only for the duration of the forward/loss call so checkpoints
        # never retain a transient accelerator batch.
        self.batch = batch
        try:
            total_loss, loss_items = self.criterion(self.predict(batch['img']) if preds is None else preds, batch)
        finally:
            if hasattr(self, 'batch'):
                delattr(self, 'batch')

        # Global-local fusion V2 learns whether a selected crop actually covers
        # a ground-truth object.  Keep this auxiliary objective outside the
        # detector criterion so the standard three detection loss items and
        # their logging remain checkpoint-compatible.
        gl_config = getattr(self, 'gl_fusion_config', {})
        gl_aux_loss = getattr(self, 'gl_aux_loss', None)
        if (
            self.training
            and gl_config.get('method') in (
                'v2', 'v3', 'gradcam_global_local_v2', 'gradcam_global_local_v3'
            )
            and gl_aux_loss is not None
            and torch.is_tensor(gl_aux_loss)
        ):
            total_loss = total_loss + (
                float(gl_config.get('aux_weight', 0.2))
                * gl_aux_loss
                * batch['img'].shape[0]
            )
        zip_config = getattr(self, 'zip_bgd_config', {})
        zip_aux_loss = getattr(self, 'zip_bgd_aux_loss', None)
        if (
            self.training
            and zip_config.get('method') in (
                'zip_bgd_pretrained_semantic_score_aux_d12_v7',
                'zip_bgd_one_sided_quality_rescue_d12_v8',
                'zip_bgd_dense_distilled_grouped_rescue_d12_v9',
                'zip_bgd_boundary_calibrated_grouped_rescue_d12_v10',
                'zip_bgd_deployment_hard_negative_rescue_d12_v11',
                'zip_bgd_global_bce_hard_pairwise_rescue_d12_v12',
            )
            and zip_aux_loss is not None
            and torch.is_tensor(zip_aux_loss)
        ):
            total_loss = total_loss + (
                float(zip_config.get('detail_aux_weight', 0.5))
                * zip_aux_loss
                * batch['img'].shape[0]
            )
        zip_quality_loss = getattr(self, 'zip_bgd_quality_aux_loss', None)
        if (
            self.training
            and zip_config.get('method') in (
                'zip_bgd_one_sided_quality_rescue_d12_v8',
                'zip_bgd_dense_distilled_grouped_rescue_d12_v9',
                'zip_bgd_boundary_calibrated_grouped_rescue_d12_v10',
                'zip_bgd_deployment_hard_negative_rescue_d12_v11',
                'zip_bgd_global_bce_hard_pairwise_rescue_d12_v12',
            )
            and zip_quality_loss is not None
            and torch.is_tensor(zip_quality_loss)
        ):
            total_loss = total_loss + (
                float(zip_config.get('quality_aux_weight', 1.0))
                * zip_quality_loss
                * batch['img'].shape[0]
            )
        zip_filter_loss = getattr(self, 'zip_bgd_filter_aux_loss', None)
        if (
            self.training
            and zip_filter_loss is not None
            and torch.is_tensor(zip_filter_loss)
            and float(zip_config.get('filter_aux_weight', 0.0)) > 0
        ):
            total_loss = total_loss + (
                float(zip_config.get('filter_aux_weight', 1.0))
                * zip_filter_loss
                * batch['img'].shape[0]
            )
        zip_preserve_strength = float(zip_config.get('preserve_teacher_strength', 0.0))
        zip_dense_strength = float(zip_config.get('dense_distill_strength', 0.0))
        zip_student_raw = getattr(self, 'zip_bgd_student_raw', None)
        zip_base_raw = getattr(self, 'zip_bgd_base_raw', None)
        if (
            self.training and (zip_preserve_strength > 0 or zip_dense_strength > 0)
            and isinstance(zip_student_raw, (list, tuple))
        ):
            from zip_bgd_fusion import get_zip_bgd_teacher
            zip_teacher = get_zip_bgd_teacher(self)
            if zip_teacher is None:
                raise RuntimeError('ZIP baseline-preservation teacher is not registered')
            zip_teacher_parameter = next(zip_teacher.parameters())
            if zip_teacher_parameter.device != batch['img'].device:
                zip_teacher.to(batch['img'].device)
            zip_teacher.eval()
            with torch.no_grad():
                teacher_output = zip_teacher.predict(batch['img'])
                zip_teacher_raw = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
            regression_channels = int(self.model[-1].reg_max * 4)
            preserve_terms = []
            dense_terms = []
            for level_index, (student_level, teacher_level) in enumerate(zip(zip_student_raw, zip_teacher_raw)):
                teacher_level = teacher_level.to(device=student_level.device, dtype=student_level.dtype)
                teacher_probability = teacher_level[:, regression_channels:].sigmoid()
                teacher_confidence = teacher_probability.amax(1, keepdim=True)
                preserve_mask = (
                    teacher_confidence >= float(zip_config.get('deployment_conf', 0.5))
                ).to(dtype=student_level.dtype)
                normalizer = preserve_mask.sum().clamp_min(1)
                classification_drop = torch.relu(
                    teacher_level[:, regression_channels:] - student_level[:, regression_channels:]
                ).square().mean(1, keepdim=True)
                regression_drift = torch.nn.functional.smooth_l1_loss(
                    student_level[:, :regression_channels],
                    teacher_level[:, :regression_channels],
                    reduction='none',
                ).mean(1, keepdim=True)
                preserve_terms.append(
                    ((classification_drop + 0.5 * regression_drift) * preserve_mask).sum()
                    / normalizer
                )
                if zip_dense_strength > 0 and isinstance(zip_base_raw, (list, tuple)):
                    base_level = zip_base_raw[level_index]
                    class_weight = (0.02 + teacher_confidence).detach()
                    dense_classification = torch.nn.functional.binary_cross_entropy_with_logits(
                        base_level[:, regression_channels:], teacher_probability.detach(), reduction='none'
                    )
                    dense_classification = (
                        dense_classification * class_weight
                    ).sum() / (
                        class_weight.sum() * dense_classification.shape[1]
                    ).clamp_min(1)
                    regression_weight = teacher_confidence.detach()
                    dense_regression = torch.nn.functional.smooth_l1_loss(
                        base_level[:, :regression_channels],
                        teacher_level[:, :regression_channels].detach(),
                        reduction='none',
                    ).mean(1, keepdim=True)
                    dense_regression = (
                        dense_regression * regression_weight
                    ).sum() / regression_weight.sum().clamp_min(1)
                    dense_terms.append(dense_classification + 0.5 * dense_regression)
            if zip_preserve_strength > 0 and preserve_terms:
                total_loss = total_loss + (
                    zip_preserve_strength * torch.stack(preserve_terms).mean() * batch['img'].shape[0]
                )
            if zip_dense_strength > 0 and dense_terms:
                zip_dense_loss = torch.stack(dense_terms).mean()
                total_loss = total_loss + (
                    zip_dense_strength * zip_dense_loss * batch['img'].shape[0]
                )
                self.zip_bgd_dense_distill_loss = zip_dense_loss.detach()
        candidate_config = getattr(self, 'candidate_roi_config', {})
        candidate_aux_loss = getattr(self, 'candidate_roi_aux_loss', None)
        if (
            self.training
            and str(candidate_config.get('method', '')).startswith('gradcam_detail_candidate_roi_')
            and candidate_aux_loss is not None
            and torch.is_tensor(candidate_aux_loss)
        ):
            total_loss = total_loss + (
                float(candidate_config.get('aux_weight', 0.5))
                * candidate_aux_loss
                * batch['img'].shape[0]
            )
        teacher = None
        distill_strength = float(candidate_config.get('distill_strength', 0.0))
        boundary_distill_strength = float(
            candidate_config.get('boundary_distill_strength', 0.0)
        )
        if distill_strength > 0 or boundary_distill_strength > 0:
            # Imported lazily to avoid a module-import cycle.  The teacher is
            # held outside the student module tree so it is never serialized
            # into or counted as part of the deployable checkpoint.
            from candidate_roi_fusion import get_candidate_roi_teacher
            teacher = get_candidate_roi_teacher(self)
        student_raw = getattr(self, 'candidate_roi_base_raw', None)
        if (
            self.training and teacher is not None
            and (distill_strength > 0 or boundary_distill_strength > 0)
            and isinstance(student_raw, (list, tuple))
        ):
            teacher_parameter = next(teacher.parameters())
            if teacher_parameter.device != batch['img'].device:
                teacher.to(batch['img'].device)
            teacher.eval()
            with torch.no_grad():
                teacher_output = teacher.predict(batch['img'])
                teacher_raw = (
                    teacher_output[1]
                    if isinstance(teacher_output, tuple) else teacher_output
                )
            if not isinstance(teacher_raw, (list, tuple)) or len(teacher_raw) != len(student_raw):
                raise RuntimeError('Teacher/student raw detection outputs are incompatible')
            regression_channels = int(self.model[-1].reg_max * 4)
            distill_terms = []
            boundary_terms = []
            for student_level, teacher_level in zip(student_raw, teacher_raw):
                teacher_level = teacher_level.to(
                    device=student_level.device, dtype=student_level.dtype
                )
                teacher_probability = teacher_level[:, regression_channels:].sigmoid()
                # Preserve deployment-relevant high-confidence anchors most
                # strongly while retaining a small dense background signal.
                anchor_weight = (0.05 + teacher_probability.amax(1, keepdim=True)).detach()
                classification = torch.nn.functional.binary_cross_entropy_with_logits(
                    student_level[:, regression_channels:], teacher_probability,
                    reduction='none',
                )
                classification = (
                    classification * anchor_weight
                ).sum() / anchor_weight.sum().clamp_min(1)
                regression = torch.nn.functional.smooth_l1_loss(
                    student_level[:, :regression_channels],
                    teacher_level[:, :regression_channels],
                    reduction='none',
                ).mean(1, keepdim=True)
                regression = (
                    regression * anchor_weight
                ).sum() / anchor_weight.sum().clamp_min(1)
                distill_terms.append(classification + regression)
                # Preserve only the teacher anchors that form the original
                # deployed detector.  This is deliberately one-sided: a
                # student may improve a teacher detection or promote a missed
                # low-confidence anchor, but it is penalized for lowering an
                # original high-confidence class logit or moving its DFL box.
                # Unlike dense distillation, background anchors exert no
                # constraint and therefore cannot suppress Detail-guided
                # recall learning.
                teacher_confidence, teacher_class = teacher_probability.max(
                    1, keepdim=True
                )
                deployment_conf = float(candidate_config.get('deployment_conf', 0.5))
                deployment_mask = (teacher_confidence >= deployment_conf).to(
                    student_level.dtype
                )
                student_teacher_logit = student_level[
                    :, regression_channels:
                ].gather(1, teacher_class)
                teacher_selected_logit = teacher_level[
                    :, regression_channels:
                ].gather(1, teacher_class)
                confidence_deficit = torch.nn.functional.relu(
                    teacher_selected_logit.detach() - student_teacher_logit
                )
                boundary_classification = (
                    confidence_deficit.square()
                    * deployment_mask
                    * teacher_confidence.detach()
                ).sum() / (
                    deployment_mask * teacher_confidence.detach()
                ).sum().clamp_min(1)
                boundary_regression = torch.nn.functional.smooth_l1_loss(
                    student_level[:, :regression_channels],
                    teacher_level[:, :regression_channels],
                    reduction='none',
                ).mean(1, keepdim=True)
                boundary_regression = (
                    boundary_regression
                    * deployment_mask
                    * teacher_confidence.detach()
                ).sum() / (
                    deployment_mask * teacher_confidence.detach()
                ).sum().clamp_min(1)
                boundary_terms.append(boundary_classification + boundary_regression)
            if distill_strength > 0:
                distill_loss = torch.stack(distill_terms).mean()
                total_loss = total_loss + (
                    distill_strength * distill_loss * batch['img'].shape[0]
                )
                self.candidate_roi_distill_loss = distill_loss.detach()
            if boundary_distill_strength > 0:
                boundary_loss = torch.stack(boundary_terms).mean()
                total_loss = total_loss + (
                    boundary_distill_strength
                    * boundary_loss
                    * batch['img'].shape[0]
                )
                self.candidate_roi_boundary_distill_loss = boundary_loss.detach()
        if hasattr(self, 'candidate_roi_base_raw'):
            delattr(self, 'candidate_roi_base_raw')
        return total_loss, loss_items

    def init_criterion(self):
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Detect_DyHead, Segment, Pose)):
            s = 640  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(2, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def gradcam_global_local_predict_once(self, x, profile=False, visualize=False):
        """Run the optional global-local forward path with a pickle-stable method name."""
        from gradcam_fusion import gradcam_global_local_predict_once

        return gradcam_global_local_predict_once(self, x, profile, visualize)

    def zip_bgd_predict_once(self, x, profile=False, visualize=False):
        """Run the complete ZIP BGD path with a pickle-stable method name."""
        from zip_bgd_fusion import zip_bgd_predict_once

        return zip_bgd_predict_once(self, x, profile, visualize)

    def bgd_318_grad_predict_once(self, x, profile=False, visualize=False):
        """Keep Grad-CAM enabled for the best318-era EMA validation path."""
        from experiments.run_318_fusion_alpha import grad_enabled_318_predict_once

        return grad_enabled_318_predict_once(self, x, profile, visualize)

    def candidate_roi_fusion_predict_once(self, x, profile=False, visualize=False):
        """Run candidate-level Grad-CAM Detail/ROI fusion."""
        from candidate_roi_fusion import candidate_roi_fusion_predict_once

        return candidate_roi_fusion_predict_once(self, x, profile, visualize)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLOv5 augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        return v8DetectionLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8SegmentationLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg='yolov8n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8PoseLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self,
                 cfg=None,
                 model=None,
                 ch=3,
                 nc=None,
                 cutoff=10,
                 verbose=True):  # yaml, model, channels, number of classes, cutoff index, verbose flag
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg, ch, nc, verbose)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Create a YOLOv5 classification model from a YOLOv5 detection model."""
        from ultralytics.nn.autobackend import AutoBackend
        if isinstance(model, AutoBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        elif not nc and not self.yaml.get('nc', None):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):

    def __init__(self, cfg='rtdetr-l.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        from ultralytics.vit.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(num_classes=self.nc, use_vfl=True)


    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        img = batch['img']
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch['batch_idx']
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch['bboxes'][batch_idx == i].to(img.device))
            gt_class.append(batch['cls'][batch_idx == i].to(device=img.device, dtype=torch.long))
        targets = {'cls': gt_class, 'bboxes': gt_bbox}

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_out_bboxes, dec_out_logits, enc_topk_bboxes, enc_topk_logits, dn_meta = preds
        # NOTE: `dn_meta` means it's eval mode, loss calculation for eval mode is not supported.
        if dn_meta is None:
            return 0, torch.zeros(3, device=dec_out_bboxes.device)
        dn_out_bboxes, dec_out_bboxes = torch.split(dec_out_bboxes, dn_meta['dn_num_split'], dim=2)
        dn_out_logits, dec_out_logits = torch.split(dec_out_logits, dn_meta['dn_num_split'], dim=2)

        out_bboxes = torch.cat([enc_topk_bboxes.unsqueeze(0), dec_out_bboxes])
        out_logits = torch.cat([enc_topk_logits.unsqueeze(0), dec_out_logits])

        loss = self.criterion((out_bboxes, out_logits),
                              targets,
                              dn_out_bboxes=dn_out_bboxes,
                              dn_out_logits=dn_out_logits,
                              dn_meta=dn_meta)
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']])

    def predict(self, x, profile=False, visualize=False, batch=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False
            batch (dict): A dict including gt boxes and labels from dataloader.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLOv5 network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.yolo.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        return torch.load(file, map_location='cpu', weights_only=False), file  # load
    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'dill':
            # Some locally trusted, stripped legacy checkpoints used dill only
            # to serialize built-in container types (notably ``set``). Avoid a
            # network install by providing the tiny unpickling shim they need.
            # Remove the shim immediately after loading: leaving a partial dill
            # module in sys.modules breaks later torch.save() version checks.
            dill_module = types.ModuleType('dill')
            dill_impl = types.ModuleType('dill._dill')
            # Legacy dill checkpoints may reference runtime types such as
            # MethodType/CodeType which live in ``types``, not ``builtins``.
            dill_impl._load_type = lambda name: getattr(builtins, name, getattr(types, name, None))
            dill_module._dill = dill_impl
            missing = object()
            previous_dill = sys.modules.get('dill', missing)
            previous_dill_impl = sys.modules.get('dill._dill', missing)
            try:
                sys.modules['dill'] = dill_module
                sys.modules['dill._dill'] = dill_impl
                return torch.load(file, map_location='cpu', weights_only=False), file
            finally:
                if previous_dill is missing:
                    sys.modules.pop('dill', None)
                else:
                    sys.modules['dill'] = previous_dill
                if previous_dill_impl is missing:
                    sys.modules.pop('dill._dill', None)
                else:
                    sys.modules['dill._dill'] = previous_dill_impl
        if e.name == 'models':
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        # Checkpoints in this repository contain serialized nn.Module objects and
        # predate PyTorch 2.6, whose default changed to weights_only=True.
        return torch.load(file, map_location='cpu', weights_only=False), file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None  # combined args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        safe_to_fuse = fuse and hasattr(model, 'fuse') and not getattr(model, 'is_gradcam_global_local', False)
        ensemble.append(model.fuse().eval() if safe_to_fuse else model.eval())  # model in eval mode

    # Module compatibility updates
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Detect_DyHead, Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    safe_to_fuse = fuse and hasattr(model, 'fuse') and not getattr(model, 'is_gradcam_global_local', False)
    model = model.fuse().eval() if safe_to_fuse else model.eval()  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Detect_DyHead, Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
    
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    is_backbone = False
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        try:
            if m == 'node_mode':
                m = d[m]
                if len(args) > 0:
                    if args[0] == 'head_channel':
                        args[0] = int(d[args[0]])
            t = m
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        except:
            pass
        
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    try:
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except:
                        args[j] = a

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3, C2f_Faster, C2f_ODConv):
            if args[0] == 'head_channel':
                args[0] = d[args[0]]
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, C2f_Faster, C2f_ODConv):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Detect_DyHead, Segment, Pose, RTDETRDecoder):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is Fusion:
            args[0] = d[args[0]]
            c1, c2 = [ch[x] for x in f], (sum([ch[x] for x in f]) if args[0] == 'concat' else ch[f[0]])
            args = [c1, args[0]]
        elif isinstance(m, str):
            t = m
            m = timm.create_model(m, pretrained=args[0], features_only=True)
            c2 = m.feature_info.channels()
        elif m in {convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge,
                   fasternet_t0, fasternet_t1, fasternet_t2, fasternet_s, fasternet_m, fasternet_l,
                   efficientvit_b0, efficientvit_b1, efficientvit_b2, efficientvit_b3,
                   efficientformerv2_s0, efficientformerv2_s1, efficientformerv2_s2, efficientformerv2_l
                   }:
            m = m(*args)
            c2 = m.channel
        else:
            c2 = ch[f]

        if isinstance(c2, list):
            is_backbone = True
            m_ = m
            m_.backbone = True
        else:
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
        
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i + 4 if is_backbone else i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print

        save.extend(x % (i + 4 if is_backbone else i) for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist

        layers.append(m_)
        if i == 0:
            ch = []
        if isinstance(c2, list):
            ch.extend(c2)
            for _ in range(5 - len(ch)):
                ch.insert(0, 0)
        else:
            ch.append(c2)
    
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_stem(new_stem)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str) or (Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module) or (dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if 'detect' in m:
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, (Detect, Detect_DyHead)):
                return 'detect'
            elif isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # assume detect
