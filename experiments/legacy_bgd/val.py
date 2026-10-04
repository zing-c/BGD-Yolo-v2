# Ultralytics YOLO 🚀, AGPL-3.0 license
# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from ultralytics import yolo  # noqa
from ultralytics import YOLO
from ultralytics.yolo.data import ClassificationDataset, build_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER
from ultralytics.yolo.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.yolo.utils.plotting import plot_images
from ultralytics.yolo.cfg import get_cfg
from pathlib import Path
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import numpy as np
from typing import Callable, List, Optional, Tuple
from gradcam import convert_to_original_coords, find_max_heatmap_center_torch, yolov8_target_batch
from ultralytics.yolo.utils.ops import xywh2xyxy, non_max_suppression
import cv2
from Resnet6 import Detail_Net_attn, Detail_Net_attn_block
from math import ceil
import matplotlib.pyplot as plt
import time




from ultralytics.yolo.utils.tal import make_anchors,dist2bbox


class BGD_YOLO(YOLO):
    def __init__(self, model=None, task=None, detail_mode=True, target_layers=None,
                 cam_mode='Grad', output_type='all', conf_threshold=0.6, detail_weight=None,
                 iou_threshold=0.5,united_train=True,model_weight_only_yolo=True,head_select='low'):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(model, task)
        self.validator = None
        self.detail_mode = detail_mode
        self.model._predict_once = self._predict_once
        # self.model.model._predict_once = self.train_predict_once
        self.batch = None
        self.target_layers = target_layers
        self.handles = []
        self.united_train = united_train
        self.gradients = []
        self.activations = []
        self.m_layers = []
        self.cam_mode = cam_mode
        self.target = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_weight_only_yolo = model_weight_only_yolo
        self.head_select = head_select
        self.head_dict = {'high': 0, 'mid': 1, 'low': 2}

        if self.detail_mode:
            if str(model).endswith(".pt") is True:
                if detail_weight is None and self.model_weight_only_yolo is True:
                    detail_model = Detail_Net_attn_block()
                elif detail_weight is None and self.model_weight_only_yolo is False:  # validation
                    state_dict = torch.load(model, map_location="cpu", weights_only=False)
                    detail_dict = state_dict['detail_model']
                    detail_model = Detail_Net_attn_block()
                    detail_model.load_state_dict(detail_dict)
                elif detail_weight is not None and self.model_weight_only_yolo is True:
                    detail_dict = torch.load(detail_weight, map_location="cpu", weights_only=False)
                    detail_model = Detail_Net_attn_block()
                    detail_model.load_state_dict(detail_dict, strict=False)
            else:
                detail_model = Detail_Net_attn_block()

            # self.set_hook()
            self.target = yolov8_target_batch()
            # if detail_weight:
            #     detail_model.load_state_dict(torch.load(detail_weight, weights_only=False))
            #
            if self.model.training:
                self.detail_model = detail_model.train().to(self.device)
            self.detail_model = detail_model.eval().to(self.device)
            self.detail_input_buf = torch.empty((32, 3, 64, 64), device=self.device)  # max_batch 根据需求设置
            # prepare layer for cam calculation
            for l in self.target_layers:
                self.m_layers.append(self.model.model[l])
                for p in self.model.model[l].parameters():
                    p.requires_grad_(True)
                # add hooks to get gradients and activations
                # self.handles.append(
                #     self.model.model[l].register_forward_hook(self.save_activation))
                # self.handles.append(
                #     self.model.model[l].register_forward_hook(self.save_gradient))
        self.model.detail_model = self.detail_model
    def set_hook(self):
        for l in self.target_layers:

            if l == -1:
                detect_layer = self.model.model[-1]
                hook_conv = detect_layer.cv3[self.head_dict[self.head_select]][1]
                self.handles.append(hook_conv.register_forward_hook(self.save_activation))
                self.handles.append(hook_conv.register_forward_hook(self.save_gradient))
            else:
                self.handles.append(
                    self.model.model[l].register_forward_hook(self.save_activation))
                self.handles.append(
                    self.model.model[l].register_forward_hook(self.save_gradient))
    def remove_hook(self):
        for h in getattr(self, 'handles', []):
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def remove_hook_for_this_model(self, model):
        for m in model.modules():
            for name in ('_forward_hooks', '_forward_pre_hooks',
                         '_backward_hooks', '_backward_pre_hooks',
                         '_jvp_hooks'):
                d = getattr(m, name, None)
                if d:
                    for k in list(d.keys()):
                        del d[k]
    def train(self, **kwargs):
        from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel, SegmentationModel,
                                          attempt_load_one_weight, guess_model_task, nn, yaml_model_load)
        from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
        from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT,
                                            callbacks,
                                            is_git_dir, yaml_load)
        # Map head to model, trainer, validator, and predictor classes
        TASK_MAP = {
            'classify': [
                ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
                yolo.v8.classify.ClassificationPredictor],
            'detect': [
                DetectionModel, yolo.v8.detect.DetectionTrainer, yolo.v8.detect.DetectionValidator,
                yolo.v8.detect.DetectionPredictor],
            'segment': [
                SegmentationModel, yolo.v8.segment.SegmentationTrainer, yolo.v8.segment.SegmentationValidator,
                yolo.v8.segment.SegmentationPredictor],
            'pose': [PoseModel, yolo.v8.pose.PoseTrainer, yolo.v8.pose.PoseValidator, yolo.v8.pose.PosePredictor]}
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        check_pip_update_available()
        overrides = self.overrides.copy()
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides.update(kwargs)
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        # if overrides.get('resume'):
        #     overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task
        self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
        # if not overrides.get('resume'):  # manually set model only if not resuming
        #     self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
        #     self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.model = self.model
        self.model.is_BGD = True
        self.set_requires_grad()
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)

    def _store_grad(self, grad):
        self.gradients.append(grad)
    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation)
        # self.activations.append(activation.cpu())

    def save_gradient(self, module, input, output):

        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        # Gradients are computed in reverse order

        output.register_hook(self._store_grad)

    def release(self):
        for handle in self.handles:
            handle.remove()

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.5)[0]
        return result

    def set_requires_grad(self):
        for i in self.m_layers:
            for j in i.parameters():
                j.requires_grad = True


    def _predict_once(self, x, profile=False, visualize=False, batch=None):
        self.remove_hook_for_this_model(self.model.model)
        if not self.model.training and self.batch is not None:
            import copy
            self.model.model = copy.deepcopy(self.model.model)
        self.set_hook()
        input_shape = x.shape
        self.gradients = []
        self.activations = []
        # self.set_requires_grad()
        if self.united_train:
            self.detail_model
        if hasattr(self.model,'batch'):
            self.batch = self.model.batch
        original_x = x
        y, dt = [], []  # outputs

        for m in self.model.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self.model._profile_one_layer(m, x, dt)
            if hasattr(m, 'backbone'):
                x = m(x)
                for _ in range(5 - len(x)):
                    x.insert(0, None)
                for i_idx, i in enumerate(x):
                    if i_idx in self.model.save:
                        y.append(i)
                    else:
                        y.append(None)
                x = x[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.model.save else None)  # save output
            if visualize:
                self.model.feature_visualization(x, m.type, m.i, save_dir=visualize)

        self.detail_model = self.detail_model.eval()
        if self.model.training:
            self.detail_model = self.detail_model.train()
            shape = x[0].shape
            m.anchors, m.strides = (x.transpose(0, 1) for x in make_anchors(x, m.stride, 0.5))
            m.shape = shape

            x_cat = torch.cat([xi.view(shape[0], m.no, -1) for xi in x], 2)
            if m.export and m.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
                box = x_cat[:, :m.reg_max * 4]
                cls = x_cat[:, m.reg_max * 4:]
            else:
                box, cls = x_cat.split((m.reg_max * 4, m.nc), 1)
            dbox = dist2bbox(m.dfl(box), m.anchors.unsqueeze(0), xywh=True, dim=1) * m.strides
            y_ = torch.cat((dbox, cls.sigmoid()), 1)
            x =y if m.export else (y_, x)

        if self.detail_mode and self.batch is not None:
            # yolo_results = self.post_process(x)
            condition = x[0][:,-1] > self.conf_threshold
            # if not self.model.training:
            # condition[0][0:10] = True
            # yolo_results = x[0][0].T[condition]
            x0 = x[0]  # (32, 5, 8400)

            with torch.no_grad():
                results = []
                boxes_all = []
                x_idxes_all = []
                total_box_num = 0
                for b in range(x0.shape[0]):  # 遍历 batch
                    x_b = x0[b]  # (5, 8400)
                    cond_b = condition[b]  # (8400,)
                    total_box_num += cond_b.sum()
                    selected = x_b[:, cond_b]  # (5, n_b)
                    results.append(selected.T)  # 转为 (n_b, 5)
                    boxes_all.append(xywh2xyxy(selected.T[:, :4]))
                    x_idxes_all.append(cond_b.nonzero().tolist())

                r = torch.tensor([sum(l[0]) / 2 for l in self.batch['ratio_pad']])
                window_size = (64 * r).ceil()
            # calculate cam image
            grayscale_cam = self.get_cam(x)

            # print(f"cam_ALL:{(t10 - t5) * 1000}ms")
            if grayscale_cam is None or total_box_num == 0:
                return x
            # if  yolo_results.shape[0] != 0:
            #     pass

            # with torch.no_grad():
            #
            #     boxes = xywh2xyxy(yolo_results[:, :4])  # x1 y1 x2 y2
            #     x_idxes = condition.nonzero().tolist()

            centers, c2b = find_max_heatmap_center_torch(grayscale_cam, boxes_all, window_size=window_size)

            # detail_batch = self.get_crop_images(centers, r)
            # detail_batch = self.detail_input_buf[:len(centers)]
            # detail_batch.copy_(self.get_crop_images(centers, r))  # 保证为 torch tensor
            detail_batch, idx2b = self.get_crop_images(centers, r)
            # center_tensor = torch.stack(centers).squeeze() // 160
            detail_batch = detail_batch.to(self.device)

            self.detail_model.to(self.device)
            detail_results = self.detail_model(detail_batch.to(self.detail_model.shortcut_L_l1[0].weight.dtype))

            current_box_idx = 0
            B = len(idx2b)

            detail_feature = torch.zeros([B, 3, self.activations[0].size()[2], self.activations[0].size()[3]]).to(self.device)
            resize=input_shape[3]/self.activations[0].size()[3]
            for b in range(B):
                if idx2b[b] == None: continue
                slice = range(current_box_idx, idx2b[b])
                detail4batch = detail_results[slice]
                current_box_idx = idx2b[b]
                center_tensor = centers[b] // resize

                for ib in range(len(detail4batch)):
                    current_box_output = detail4batch[ib]
                    current_center = center_tensor[ib]
                    # fixed_center = [max(current_center[0], 2), max(current_center[1], 2)]   # 需要调整 避免中心点为 0 0
                    fixed_center = [
                        int(max(2, min(current_center[0], self.activations[0].size()[2] - 3))),
                        int(max(2, min(current_center[1], self.activations[0].size()[3] - 3)))
                    ]
                    detail_feature[b, :, max(int(fixed_center[0]) - 2, 0):min(int(fixed_center[0] + 2), self.activations[0].size()[2]-1),
                    max(int(fixed_center[1] - 2), 0):min(int(fixed_center[1] + 2), self.activations[0].size()[3]-1)].add(current_box_output)


            concat_feature = torch.cat((detail_feature, self.activations[0].to(self.device)), dim=1)
            concat_feature = self.detail_model.conv_for_yolo(concat_feature.to(self.detail_model.conv_for_yolo.weight.dtype))
            concat_feature = self.detail_model.bn_for_yolo(concat_feature.to(self.detail_model.bn_for_yolo.weight.dtype))
            concat_feature = self.detail_model.relu_for_yolo(concat_feature)
            out = m.cv3[self.head_dict[self.head_select]][2](concat_feature.to(m.cv3[self.head_dict[self.head_select]][2].weight.dtype)).squeeze(1)
            x[1][self.head_dict[self.head_select]][:, -1, ...] = out
            if not self.model.training:
                x = x[1]
                shape = x[0].shape
                m.anchors, m.strides = (x.transpose(0, 1) for x in make_anchors(x, m.stride, 0.5))
                m.shape = shape

                x_cat = torch.cat([xi.view(shape[0], m.no, -1) for xi in x], 2)
                if m.export and m.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
                    box = x_cat[:, :m.reg_max * 4]
                    cls = x_cat[:, m.reg_max * 4:]
                else:
                    box, cls = x_cat.split((m.reg_max * 4, m.nc), 1)
                dbox = dist2bbox(m.dfl(box), m.anchors.unsqueeze(0), xywh=True, dim=1) * m.strides
                y_ = torch.cat((dbox, cls.sigmoid()), 1)
                x = y if m.export else (y_, x)

            # for c_i, box_idxes in enumerate(c2b):
            #     for box_idx in box_idxes:
            #         # confs[box_idx] = detail_results.detach()[c_i][0]
            #         # x[0][0][-1][x_idxes[box_idx][0]] = torch.sigmoid(torch.sqrt(detail_results.detach()[c_i][0]*x[0][0][-1][x_idxes[box_idx][0]]))
            #         t=x[0][0][-1][x_idxes[box_idx][0]]/((1-detail_results[c_i][0])+x[0][0][-1][x_idxes[box_idx][0]])
            #         x[0][0][-1][x_idxes[box_idx][0]] =detail_results[c_i][0] *(1-t) +t* x[0][0][-1][x_idxes[box_idx][0]]
            #         # x[0][0][-1][x_idxes[box_idx][0]] =torch.sigmoid(detail_results[c_i][0])*t +(1-t)* x[0][0][-1][x_idxes[box_idx][0]]
            #         # x[0][0][-1][x_idxes[box_idx][0]] = sigmoid_variant( torch.sqrt(detail_results.detach()[c_i][0] * x[0][0][-1][x_idxes[box_idx][0]]), k=5)
            #     # 显式释放变量
        self.remove_hook()
        if self.model.training:
            x = x[1]
        else:
            torch.cuda.empty_cache()

        self.activations.clear()
        self.gradients.clear()
        return x

    def get_crop_images(self, centers, r):
        crop_images_all = []
        idx2b = []
        current_idx = 0
        for b in range(len(centers)):
            if len(centers[b]) ==0:
                # crop_images_all.append(None)
                idx2b.append(None)
                continue
            crop_images = []
            img_pre = []
            scaled_h = self.batch['ori_shape'][b][0] * self.batch['ratio_pad'][b][0][0]
            scaled_w = self.batch['ori_shape'][b][1] * self.batch['ratio_pad'][b][0][1]
            padded_h = (self.batch['resized_shape'][b][0] - scaled_h) / 2
            padded_w = (self.batch['resized_shape'][b][1] - scaled_w) / 2

            original_image = cv2.imread(self.batch['im_file'][b])
            mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet 均值
            std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet 标准差

            for center in centers[b]:
                x, y = center
                center_original = convert_to_original_coords(max(padded_w,x), max(padded_h,y), r[b], padded_w, padded_h)

                new_image = self.crop_image(original_image, center_original, crop_size=(64, 64))
                # cv2.imwrite(self.batch['im_file'][0][-9:-4]+'Crop.jpg', new_image)
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
                new_image = np.float32(new_image) / 255.0  # 归一化到 [0, 1]
                new_image = torch.from_numpy(np.transpose(new_image, axes=[2, 0, 1])).float()  # RGB CHW

                # 对每个通道进行归一化
                crop_image = (new_image - mean[:, None, None]) / std[:, None, None]
                crop_images_all.append(crop_image)
                current_idx += 1

            idx2b.append(current_idx)
            # crop_images_all.append(crop_images)

        return torch.stack(crop_images_all, dim=0), idx2b

    # def crop_image(self, image, center, crop_size=(64, 64)):
    #
    #     """
    #     基于中心点截取图像的一部分
    #     :param image: 输入图像 (np.ndarray)
    #     :param center: 中心点坐标 (cx, cy)
    #     :param crop_size: 截取区域的大小 (width, height)
    #     :return: 截取的图像区域 (np.ndarray)
    #     """
    #     h, w = image.shape[:2]
    #     cx, cy = center
    #     crop_w, crop_h = crop_size
    #     x1 = max(0, int(round(cx - crop_w / 2)))
    #     y1 = max(0, int(round(cy - crop_h / 2)))
    #     x2 = x1 + crop_w
    #     y2 = y1 + crop_h
    #
    #     cropped_image = image[y1:y2, x1:x2]
    #
    #     # 填充到目标尺寸
    #     if cropped_image.shape[0] < crop_h or cropped_image.shape[1] < crop_w:
    #         pad_h = max(0, crop_h - cropped_image.shape[0])
    #         pad_w = max(0, crop_w - cropped_image.shape[1])
    #         cropped_image = cv2.copyMakeBorder(
    #             cropped_image,
    #             pad_h // 2, pad_h - pad_h // 2,
    #             pad_w // 2, pad_w - pad_w // 2,
    #             cv2.BORDER_CONSTANT, value=(0, 0, 0)
    #         )
    #
    #     return cropped_image


    def crop_image(self, image, center, crop_size=(64, 64), border_value=(0, 0, 0)):
        """
        基于中心点裁出固定尺寸（width, height）的图像块。
        - 固定右下边界：x2=x1+width, y2=y1+height，杜绝 2r+1/取整导致的 65。
        - 贴边/越界时会先对整图做黑边填充，保证能取到完整块。
        """
        H_img, W_img = image.shape[:2]
        crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
        cx, cy = float(center[0]), float(center[1])

        # 若原图比目标块还小，先把整图 pad 到至少 crop_size
        need_pad_w = max(0, crop_w - W_img)
        need_pad_h = max(0, crop_h - H_img)
        if need_pad_w > 0 or need_pad_h > 0:
            pad_left = need_pad_w // 2
            pad_right = need_pad_w - pad_left
            pad_top = need_pad_h // 2
            pad_bot = need_pad_h - pad_top
            image = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right,
                                       borderType=cv2.BORDER_CONSTANT, value=border_value)
            W_img += need_pad_w
            H_img += need_pad_h
            cx += pad_left
            cy += pad_top

        # 计算左上角，用 round 保持对称，然后 clamp 到合法范围
        x1 = int(round(cx - crop_w / 2.0))
        y1 = int(round(cy - crop_h / 2.0))
        x1 = max(0, min(W_img - crop_w, x1))
        y1 = max(0, min(H_img - crop_h, y1))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        patch = image[y1:y2, x1:x2]

        # 兜底：极端情况下保证尺寸一致
        if patch.shape[0] != crop_h or patch.shape[1] != crop_w:
            patch = cv2.resize(patch, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        return patch

    def get_cam(self,x):
        # calculate gradient
        # self.set_hook()
        # try:
        if True:
            self.model.model.zero_grad()
            self.detail_model.zero_grad()
            loss = self.target(x[0])
            if torch.all(loss == 0):
                return None

            torch.autograd.backward(loss, grad_tensors=torch.ones_like(loss).detach(), retain_graph=self.model.training)

             # calculate cam image
            with torch.no_grad():
                cam_per_target_layer = self.compute_cam_per_layer()
                grayscale_cam = self.aggregate_multi_layers(cam_per_target_layer)
            self.model.model.zero_grad()
            self.detail_model.zero_grad()
            return grayscale_cam

        # finally:
        #     # self.remove_hook()
        #     # self.activations.clear()
        #     # self.gradients.clear()
        #     self.model.model.zero_grad()

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        if self.cam_mode=='GradCAM':
            sum_activations = np.sum(activations, axis=(2, 3))
            eps = 1e-7
            weights = grads * activations / \
                (sum_activations[:, :, None, None] + eps)
            weights = weights.sum(axis=(2, 3))
            return weights

        if self.cam_mode == 'Grad':
            if len(grads.shape) == 4:

                return np.mean(grads, axis=(2, 3))

            else:
                raise ValueError("Invalid grads shape."
                                 "Shape of grads should be 4 (2D image) or 5 (3D image).")

    def compute_cam_per_layer(
        self, eigen_smooth=False):
        activations_list = [a.cpu().data.numpy() for a in self.activations]
        grads_list = [g.cpu().data.numpy() for g in self.gradients]
        target_size = tuple(self.batch['resized_shape'][0])

        cam_per_target_layer = []
        # Loop over the saliency image from every layer


        for i in range(len(self.m_layers)):
            target_layer = self.m_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(activations=layer_activations,grads=layer_grads)

            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)

        return scale_cam_image(result)

    def get_cam_image(
        self,
        input_tensor=None,
        target_layer=None,
        targets=None,
        activations=None,
        grads=None,
        eigen_smooth=False,

    ):
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # 2D conv
        if len(activations.shape) == 4:

            weighted_activations = weights[:, :, None, None] * activations

        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)


        return cam

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.validator.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        nb = len(batch['img'])
        self.validator.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                   for i in range(nb)] if self.validator.args.save_hybrid else []

        # save batch here
        self.batch = batch
        return batch

    def val(self, data=None, **kwargs):
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed

        self.validator = yolo.v8.detect.DetectionValidator(args=args, _callbacks=self.callbacks)
        if self.detail_mode:
            self.validator.preprocess = self.preprocess
            self.set_requires_grad()
        self.validator(model=self.model)
        self.metrics = self.validator.metrics

        return self.validator.metrics





class ClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'classify'
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        # return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')
        return ('%22s' + '%11s' * 7) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP75', 'mAP50-95)')

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, task='classify')
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        n5 = min(len(self.model.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5])
        self.targets.append(batch['cls'])

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        return ClassificationDataset(root=img_path, args=self.args, augment=False)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        pf = '%22s' + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(images=batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=batch['cls'].squeeze(-1),
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names,
                    on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=torch.argmax(preds, dim=1),
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)  # pred


def val(cfg=DEFAULT_CFG, use_python=True,test_only=False):
    """Validate YOLO model using custom data."""
    import torch.nn as nn
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    data = cfg.data or 'mnist160'

    args = dict(model=model, data=data)
    if use_python:
        yolo_model = BGD_YOLO(model,target_layers=[21],detail_weight='run/detail_net_attn.pt',
                              detail_mode=True)

        yolo_model.val(**args)
    else:
        validator = ClassificationValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':


    val(test_only=False)
