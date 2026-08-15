
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
from gradcam import letterbox, find_max_heatmap_center, find_max_heatmap_center_torch, find_max_heatmap_center_torch_optimized,attempt_load_weights, convert_to_original_coords, yolov8_target,yolov8_target_batch
from ultralytics.yolo.utils.ops import xywh2xyxy, non_max_suppression
import cv2
from Resnet6 import Detail_Net_attn, Detail_Net_attn_block
from math import ceil
import matplotlib.pyplot as plt
import time
from functools import partial
import os


from ultralytics.yolo.utils.tal import make_anchors,dist2bbox


class BGD_YOLO(YOLO):
    def __init__(self, model=None, task=None, detail_mode=True, target_layers=None,
                     cam_mode='Grad', output_type='all', conf_threshold=0.6, detail_weight=None,
                     iou_threshold=0.5,united_train=True,model_weight_only_yolo=True,head_select='low',detail_position='neck',detail_num=2,
                     verbose_time=False,
                     debug_vis=True,
                     debug_vis_dir="runs/vis_crop_boxes",
                     debug_vis_max=1598):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(model, task)
        self.validator = None
        self.detail_mode = detail_mode
        self.model._predict_once = self._predict_once
        # self.model.model._predict_once = self.train_predict_once
        self.batch = None
        self.target_layers = target_layers
        self.handles=[]
        self.united_train = united_train
        self.m_layers = []
        self.cam_mode = cam_mode
        self.target = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_weight_only_yolo = model_weight_only_yolo
        self.head_select = head_select
        # self.head_dict = {'high': 0, 'mid': 1, 'low': 2}
        self.head_dict = { 'mid': 0, 'low': 1}
        self.detail_position =detail_position
        self.detail_num = detail_num
        self.detail_max_boxes = None  # Set to an int to enable top-k limiting.


        self.debug_vis = bool(debug_vis)
        self.debug_vis_dir = str(debug_vis_dir)
        self.debug_vis_max = int(debug_vis_max) if debug_vis_max is not None else 0
        self.debug_vis_count = 0
        try:
            self.debug_vis_nms_iou = float(os.getenv("BGD_DEBUG_VIS_NMS_IOU", "0.6"))
        except Exception:
            self.debug_vis_nms_iou = 0.6
        try:
            self.debug_vis_nms_max_boxes = int(os.getenv("BGD_DEBUG_VIS_NMS_MAX_BOXES", "50"))
        except Exception:
            self.debug_vis_nms_max_boxes = 50



        if self.detail_mode:
            if self.target_layers is None:
                self.target_layers = [-1]

            def _unwrap_state_dict(obj):
                # Supports either a raw state_dict or a checkpoint dict.
                if isinstance(obj, dict):
                    for k in ("detail_model", "state_dict", "model_state_dict"):
                        if k in obj and isinstance(obj[k], dict):
                            return obj[k]
                return obj

            detail_model = None
            if str(model).endswith(".pt") is True:
                if detail_weight is not None:
                    # Explicit detail weights take precedence (covers model_weight_only_yolo True/False).
                    loaded = torch.load(detail_weight, map_location="cpu")
                    detail_dict = _unwrap_state_dict(loaded)
                    detail_model = Detail_Net_attn()
                    detail_model.load_state_dict(detail_dict, strict=False)
                elif self.model_weight_only_yolo is True:
                    # Only YOLO weights are available, initialize detail model randomly.
                    detail_model = Detail_Net_attn()
                else:  # validation: expect detail weights stored in the YOLO checkpoint.
                    loaded = torch.load(model, map_location="cpu")
                    loaded = loaded if isinstance(loaded, dict) else {}
                    if "detail_model" not in loaded:
                        raise ValueError(
                            "detail_mode=True and model_weight_only_yolo=False, but no 'detail_model' "
                            f"was found in checkpoint: {model}. Provide `detail_weight=...` or set "
                            "`model_weight_only_yolo=True`."
                        )
                    detail_dict = _unwrap_state_dict(loaded)
                    detail_model = Detail_Net_attn()
                    detail_model.load_state_dict(detail_dict, strict=False)
            else:
                detail_model = Detail_Net_attn()

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
            # for l in self.target_layers:
            #     self.m_layers.append(self.model.model[l])
            #     for p in self.model.model[l].parameters():
            #         p.requires_grad_(True)
                # add hooks to get gradients and activations
                # self.handles.append(
                #     self.model.model[l].register_forward_hook(self.save_activation))
                # self.handles.append(
                #     self.model.model[l].register_forward_hook(self.save_gradient))
        # Expose the detail model to the underlying YOLO model (used by custom forward code).
        self.model.detail_model = self.detail_model
        self.model.detail_model.float()


    @staticmethod
    def _nms_xyxy_cpu(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thres: float):
        """Pure PyTorch NMS on CPU. boxes_xyxy: (N,4), scores: (N,). Returns keep indices (LongTensor)."""
        if boxes_xyxy is None or scores is None:
            return torch.empty((0,), dtype=torch.long)
        if boxes_xyxy.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        boxes = boxes_xyxy.to(device="cpu", dtype=torch.float32)
        sc = scores.to(device="cpu", dtype=torch.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        order = torch.argsort(sc, descending=True)
        keep = []
        while order.numel() > 0:
            i = int(order[0])
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]

            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)

            order = rest[iou <= float(iou_thres)]

        return torch.tensor(keep, dtype=torch.long)



    def _draw_text_box(self, img, text, x, y, font_scale=0.7, thickness=2,
                       text_color=(255, 255, 255), bg_color=(0, 0, 255), pad=3):
        if not text: return
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x0, y0 = int(max(0, x)), int(max(0, y))
        cv2.rectangle(img, (x0 - pad, y0 - th - baseline - pad), (x0 + tw + pad, y0 + pad), bg_color, -1)
        cv2.putText(img, text, (x0, y0 - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def _debug_vis_save(self, b_idx, img_hwc, img_path, boxes_xyxy_ori, crop_top_lefts_cpu,
                        crop_size, yolo_scores=None, detail_fused_scores=None, detail_individual_scores=None):
        if not self.debug_vis or self.debug_vis_count >= self.debug_vis_max: return

        out_dir = Path(self.debug_vis_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 图像处理
        img_np = img_hwc.cpu().numpy() if isinstance(img_hwc, torch.Tensor) else img_hwc
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # 1. 绘制预测大框 (红色)
        if boxes_xyxy_ori is not None:
            boxes = boxes_xyxy_ori.tolist()
            for bi, (x1, y1, x2, y2) in enumerate(boxes):
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)

                # 绘制融合后的总分 (大幅上移)
                parts = []
                if yolo_scores is not None: parts.append(f"Y:{yolo_scores[bi]:.3f}")
                if detail_fused_scores is not None: parts.append(f"D_Fused:{detail_fused_scores[bi]:.3f}")

                if parts:
                    # y 坐标减去 45 像素，避开内部截图框
                    self._draw_text_box(img_bgr, " ".join(parts), x1i, max(45, y1i - 45),
                                        font_scale=0.7, bg_color=(150, 0, 0))

        # 2. 绘制 5 个截图框 (绿色)
        if crop_top_lefts_cpu is not None:
            pts = crop_top_lefts_cpu.tolist()
            for i, (cx, cy) in enumerate(pts):
                cx, cy = int(cx), int(cy)
                cv2.rectangle(img_bgr, (cx, cy), (cx + crop_size, cy + crop_size), (0, 255, 0), 1)

                # 绘制单个截图经过 Sigmoid 后的原始分数
                if detail_individual_scores is not None:
                    score = detail_individual_scores[i]
                    self._draw_text_box(img_bgr, f"d:{score:.3f}", cx, max(12, cy - 2),
                                        font_scale=0.45, bg_color=(0, 0, 0), pad=2)

        fname = out_dir / f"{self.debug_vis_count:06d}_b{b_idx}.jpg"
        cv2.imwrite(str(fname), img_bgr)
        self.debug_vis_count += 1



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


    def save_activation(self, name, module, input, output):
        self.activations[name] = output

    def save_gradient(self, name, module, grad_input, grad_output):
        # grad_output 是 tuple，取第一个元素
        self.gradients[name] = grad_output[0]

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

    def _time_now(self):
        if self.profile_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _timing_add(self, key, dt):
        self.timing_stats[key] = self.timing_stats.get(key, 0.0) + dt
        self.timing_counts[key] = self.timing_counts.get(key, 0) + 1
        if getattr(self, "verbose_time", False):
            LOGGER.info(f"Timing {key}: {dt * 1000:.2f}ms")

    def _timing_report(self):
        if not self.timing_stats:
            return
        parts = []
        for k in sorted(self.timing_stats.keys()):
            total = self.timing_stats[k]
            cnt = self.timing_counts.get(k, 1)
            parts.append(f"{k}={total / cnt * 1000:.2f}ms")
        # LOGGER.info("Timing avg: " + ", ".join(parts))
        self.timing_stats.clear()
        self.timing_counts.clear()


    def _predict_once(self, x, profile=False, visualize=True, batch=None):
        # for param in self.model.model.parameters():
        #     param.requires_grad = False
        #
        # # 2. 确保 DetailModel 是开启梯度更新的
        # for param in self.model.detail_model.parameters():
        #     param.requires_grad = True



        grad_enabled = self.model.training

        if not self.model.training and self.batch is not None:
            import copy
            self.model.model = copy.deepcopy(self.model.model)

        input_shape = x.shape

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
                if m.type == 'ultralytics.nn.modules.head.Detect':
                    x_tmp = x
                    x = m([xi.clone() for xi in x] )
                else:
                    x = m(x)  # run

                y.append(x if m.i in self.model.save else None)  # save output
            if visualize:
                self.model.feature_visualization(x, m.type, m.i, save_dir=visualize)
        x_raw = x


        self.model.detail_model = self.model.detail_model.eval()
        if self.model.training:
            self.model.detail_model = self.model.detail_model.train()
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

        # 4. Detail Model 介入逻辑 (训练 & 验证 通用) ===============================
        if self.detail_mode and self.batch is not None:
            # x[0] shape: (Batch, 4+cls, 8400)
            preds = x[0]
            if grad_enabled:
                preds = preds.clone()  # 训练时避免原地操作报错，虽然下面是索引操作

            # Collect crops as batched tensors (N, 3, 64, 64) for efficient concatenation.
            crop_images_list = []
            mapping_indices = []  # (batch_idx, anchor_idx)

            # 预处理参数
            mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

            # 遍历 Batch
            debug_vis_payload = {} if self.debug_vis else None
            for b_idx in range(preds.shape[0]):
                # 转置方便处理: (8400, 4+cls)
                pred_b = preds[b_idx].T

                # --- 筛选框 ---
                # 训练初期分数很低，如果阈值太高(0.6)可能选不到框，导致 detail model 不训练
                # 建议：训练时适当降低阈值，或者使用 Top-K 策略
                current_conf = self.conf_threshold
                # if self.model.training:
                #     current_conf = 0.1 # 训练时降低阈值示例

                cls_scores, _ = pred_b[:, 4:].max(dim=1)
                mask = cls_scores > current_conf

                if not mask.any():
                    continue

                valid_indices = torch.nonzero(mask).squeeze(1)
                # 限制处理数量，防止训练时显存爆炸 (例如最多处理 100 个框)
                if len(valid_indices) > 100 and self.model.training:
                    # 随机采样或者取分数最高的 TopK
                    _, topk_idx = torch.topk(cls_scores[mask], 100)
                    valid_indices = valid_indices[topk_idx]

                valid_boxes = pred_b[valid_indices, :4]  # xywh
                valid_boxes_xyxy = xywh2xyxy(valid_boxes)

                # --- 准备截图源 ---
                use_tensor_crop = False # 训练时建议用 Tensor 截图

                if use_tensor_crop:
                    # 使用 batch['img'] (GPU Tensor) 截图
                    # 优点：速度快，保留梯度(如果需要)，且与 Mosaic 增强对齐
                    src_img = self.batch['img'][b_idx]  # (3, H, W), 0-1 float
                    img_h, img_w = src_img.shape[1], src_img.shape[2]
                else:
                    # 使用 im_file (磁盘读取) 截图
                    # 优点：清晰度高 (验证模式推荐)
                    img_path = self.batch['im_file'][b_idx]
                    img_original = cv2.imread(img_path)
                    # 坐标映射参数
                    r = self.batch['ratio_pad'][b_idx][0]
                    ori_h, ori_w = self.batch['ori_shape'][b_idx]
                    res_h, res_w = self.batch['resized_shape'][b_idx]
                    pad_w = (res_w - ori_w * r[1]) / 2
                    pad_h = (res_h - ori_h * r[0]) / 2

                # --- 遍历框并截图 ---
                for i, box_idx in enumerate(valid_indices):
                    bx1, by1, bx2, by2 = valid_boxes_xyxy[i]

                    # === 参数设置 ===
                    CROP_SIZE = 64
                    # 向内偏移量：
                    # 设为 4 或 8 比较合适。
                    OFFSET_VAL = 8

                    # === 第一步：计算截图框的【左上角】坐标 ===
                    if use_tensor_crop:
                        # 训练模式：直接基于 Tensor 坐标计算
                        crop_top_lefts = self.get_aligned_crop_coords(
                            bx1, by1, bx2, by2, CROP_SIZE, OFFSET_VAL
                        )
                    else:
                        # 验证模式：先映射回原图，再计算
                        # 映射 Box 到原图坐标系
                        b_x1_ori = (bx1 - pad_w) / r[1]
                        b_y1_ori = (by1 - pad_h) / r[0]
                        b_x2_ori = (bx2 - pad_w) / r[1]
                        b_y2_ori = (by2 - pad_h) / r[0]

                        crop_top_lefts = self.get_aligned_crop_coords(
                            b_x1_ori, b_y1_ori, b_x2_ori, b_y2_ori, CROP_SIZE, OFFSET_VAL
                        )

                    # === 第二步：执行截图 ===
                    for (cx, cy) in crop_top_lefts:
                        if use_tensor_crop:
                            # --- 训练模式 (Tensor切片) ---
                            # cx, cy 是截图框左上角
                            x1_c, y1_c = int(cx), int(cy)
                            x2_c, y2_c = x1_c + CROP_SIZE, y1_c + CROP_SIZE

                            # 边界安全检查 (Clamp) - 防止超出整张图的范围
                            # 就算 get_aligned_crop_coords 算得对，转 int 也可能有一像素误差，必须 clamp
                            x1_c = max(0, min(img_w, x1_c))
                            y1_c = max(0, min(img_h, y1_c))
                            x2_c = max(0, min(img_w, x2_c))
                            y2_c = max(0, min(img_h, y2_c))

                            # 切片
                            patch = src_img[:, y1_c:y2_c, x1_c:x2_c]

                            # Padding (如果切出来的尺寸不足 64x64)
                            # 比如 Box 在图像最边缘，或者 Box 很小
                            cur_h, cur_w = patch.shape[1], patch.shape[2]
                            pad_h = CROP_SIZE - cur_h
                            pad_w = CROP_SIZE - cur_w

                            if pad_h > 0 or pad_w > 0:
                                # 默认往右下填充，保持左上角对齐
                                patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h))

                            # 归一化
                            patch = (patch - mean[:, None, None]) / std[:, None, None]
                            crop_images_list.append(patch)

                        else:
                            # --- 验证模式 (OpenCV截图) ---
                            # cx, cy 是浮点数原图坐标，需要取整
                            ix1, iy1 = int(cx+0.5), int(cy+0.5)
                            ix2, iy2 = ix1 + CROP_SIZE, iy1 + CROP_SIZE

                            # 边界检查
                            h_ori, w_ori = img_original.shape[:2]
                            ix1 = max(0, min(w_ori, ix1))
                            iy1 = max(0, min(h_ori, iy1))
                            ix2 = max(0, min(w_ori, ix2))
                            iy2 = max(0, min(h_ori, iy2))

                            # Numpy 切片
                            patch_cv = img_original[iy1:iy2, ix1:ix2]

                            # Padding
                            cur_h, cur_w = patch_cv.shape[:2]
                            if cur_h != CROP_SIZE or cur_w != CROP_SIZE:
                                pad_b = CROP_SIZE - cur_h
                                pad_r = CROP_SIZE - cur_w
                                patch_cv = cv2.copyMakeBorder(
                                    patch_cv, 0, pad_b, 0, pad_r,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                                )

                            # 转 Tensor
                            patch_cv = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2RGB)
                            patch_t = torch.from_numpy(patch_cv).float() / 255.0
                            patch_t = patch_t.permute(2, 0, 1).to(self.device)
                            patch_t = (patch_t - mean[:, None, None]) / std[:, None, None]
                            crop_images_list.append(patch_t)

                    # 记录映射关系
                    mapping_indices.append((b_idx, box_idx))

            if crop_images_list:
                batch_crops = torch.stack(crop_images_list)

                # Detail Model 推理
                # 训练模式下，这里会有梯度
                model_dtype = next(self.model.detail_model.parameters()).dtype

                # 将输入张量转换为与模型一致的类型
                detail_logits = self.model.detail_model(batch_crops.to(device=self.device, dtype=model_dtype))

                # # . 批量计算 LSE


                #  确保 crop_images_list 的长度是 5 的倍数
                n_boxes = len(mapping_indices)
                dc_raw = torch.sigmoid(detail_logits.view(n_boxes, 5))
                logits_grouped = detail_logits.view(n_boxes, 5)

                # LSE Pooling: shape (N_boxes, )
                # fused_logits = torch.logsumexp(logits_grouped, dim=1)

                fused_logits = torch.mean(logits_grouped, dim=1)

                # 转为分数 (0-1)
                vote_scores = torch.sigmoid(fused_logits)

                # weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device=self.device)
                #
                # # 3. 先 Sigmoid 转成概率 (0~1)
                # probs_grouped = torch.sigmoid(logits_grouped)
                #
                # # 4. 加权求和
                # # probs_grouped: (N, 5)
                # # weights: (5,) -> 自动广播
                # # 结果: (N,)
                # vote_scores = (probs_grouped * weights).sum(dim=1)




                # 3. 构造全局 Mask
                # (Batch, 8400) 全 1
                total_anchors = x[0].shape[2]
                # score_mask = torch.ones(x[0].shape[0], total_anchors, device=self.device)

                # 解压索引
                b_idxs, a_idxs = zip(*mapping_indices)
                b_idxs = torch.tensor(b_idxs, device=self.device)
                a_idxs = torch.tensor(a_idxs, device=self.device)

                # # 填入分数
                # score_mask[b_idxs, a_idxs] = vote_scores
                #
                # # 4. 更新 x[0] (概率) -> 用于 Validation / Inference
                # x[0][:, 4:, :] = x[0][:, 4:, :] * score_mask.unsqueeze(1)

                # =================================================================
                # 【替换逻辑】 Part A: 更新 x[0] (Inference / Validation)
                # =================================================================
                # 直接将 Detail Model 的分数赋值给 x[0]，实现"替换"
                # index 4 是单类别的置信度位

                # 备份旧分数，用于计算 ratio (给训练用)
                old_scores = x[0][b_idxs, 4, a_idxs].clone()
                alpha = 0.5

                # 线性加权公式
                new_scores = alpha * vote_scores + (1 - alpha) * old_scores
                # new_scores = torch.sqrt(vote_scores * old_scores)
                # =======================================================

                # 2. 更新 x[0] (用于推理/验证)
                x[0][b_idxs, 4, a_idxs] = new_scores.to(x[0].dtype)
                if debug_vis_payload:
                    # 1. 将必要数据转到 CPU 准备构建映射
                    b_cpu = b_idxs.detach().to(device="cpu", dtype=torch.int64)
                    a_cpu = a_idxs.detach().to(device="cpu", dtype=torch.int64)
                    y_cpu = old_scores.detach().to(device="cpu", dtype=torch.float32)
                    d_cpu = vote_scores.detach().to(device="cpu", dtype=torch.float32)

                    # 将 (n_boxes, 5) 的原始 Sigmoid 分数转到 CPU (假设你之前已定义 dc_raw = torch.sigmoid(logits_grouped))
                    # 如果还没定义，请在 if 前加上: dc_raw = torch.sigmoid(detail_logits.view(n_boxes, 5))
                    dc_raw_cpu = dc_raw.detach().to(device="cpu", dtype=torch.float32)

                    # 2. 构建 score_map，每个 anchor 存 (YOLO分, 融合分, [5个原始分])
                    score_map = {
                        (int(b), int(a)): (float(y), float(d), dc.tolist())
                        for b, a, y, d, dc in zip(b_cpu, a_cpu, y_cpu, d_cpu, dc_raw_cpu)
                    }

                    for b_idx, payload in debug_vis_payload.items():
                        anchors = payload.get("anchor_indices")
                        if anchors is None: continue

                        y_list = []
                        d_list = []
                        d_indiv_list = [] # 存储该 batch 中所有 crop 的分数

                        for a in anchors.tolist():
                            yd = score_map.get((int(b_idx), int(a)))
                            if yd is None:
                                y_list.append(float("nan"))
                                d_list.append(float("nan"))
                                d_indiv_list.extend([float("nan")] * 5)
                            else:
                                y_list.append(yd[0])
                                d_list.append(yd[1])
                                d_indiv_list.extend(yd[2]) # 放入 5 个 crop 的 Sigmoid 分数

                        boxes_cpu = payload["boxes_xyxy_ori"]
                        crops_cpu = payload["crop_top_lefts_cpu"]
                        y_scores = torch.tensor(y_list, dtype=torch.float32)
                        d_scores = torch.tensor(d_list, dtype=torch.float32)
                        # 将原始分数列表转为 Tensor，方便后面做 NMS 过滤
                        di_scores = torch.tensor(d_indiv_list, dtype=torch.float32)

                        # 3. 可视化专用的 NMS 过滤 (防止框太多太乱)
                        if (
                                isinstance(boxes_cpu, torch.Tensor)
                                and boxes_cpu.numel()
                                and getattr(self, "debug_vis_nms_iou", 0) is not None
                                and float(self.debug_vis_nms_iou) > 0
                        ):
                            base_scores = y_scores.clone()
                            base_scores[~torch.isfinite(base_scores)] = 0.0
                            keep = self._nms_xyxy_cpu(boxes_cpu, base_scores, float(self.debug_vis_nms_iou))

                            if getattr(self, "debug_vis_nms_max_boxes", None):
                                keep = keep[: int(self.debug_vis_nms_max_boxes)]

                            if keep.numel():
                                boxes_cpu = boxes_cpu[keep]
                                y_scores = y_scores[keep]
                                d_scores = d_scores[keep]

                                # --- 重要修改：同步过滤 5 个点的截图及其分数 ---
                                if isinstance(crops_cpu, torch.Tensor) and crops_cpu.numel():
                                    # 生成对应的 crop 索引 (每个 box 对应 5 个连续的 crop)
                                    crop_keep = (keep.view(-1, 1) * 5 + torch.arange(5)).reshape(-1)
                                    crops_cpu = crops_cpu[crop_keep]
                                    # 同时过滤原始分数列表
                                    di_scores = di_scores[crop_keep]

                        # 4. 调用修改后的可视化函数
                        self._debug_vis_save(
                            b_idx=b_idx,
                            img_hwc=payload["img_hwc"],
                            img_path=payload["img_path"],
                            boxes_xyxy_ori=boxes_cpu,
                            crop_top_lefts_cpu=crops_cpu,
                            crop_size=payload["crop_size"],
                            yolo_scores=y_scores,
                            detail_fused_scores=d_scores,        # 融合后的总分 (大框用)
                            detail_individual_scores=di_scores   # LSE前的原始分 (小框用)
                        )
                # if self.profile_timing:
                #     t_fuse1 = self._time_now()
                #     self._timing_add("detail_fuse", t_fuse1 - t_fuse0)


                # # 执行替换
                # x[0][b_idxs, 4, a_idxs] = vote_scores*1.5

                # =================================================================
                # 【替换逻辑】 Part B: 更新 x[1] (Training Logits)
                # =================================================================
                if self.model.training:
                    # 构造"比率 Mask"，欺骗下面的乘法逻辑，实现"等效替换"
                    # Ratio = Target / Old
                    ratio = vote_scores / (old_scores + 1e-6)

                    # 构造全 1 的掩码
                    total_anchors = x[0].shape[2]
                    score_mask = torch.ones(x[0].shape[0], total_anchors, device=self.device)

                    # 填入比率 (而不是分数!)
                    score_mask[b_idxs, a_idxs] = ratio

                    # --- 下面这段代码保持不变 (利用乘法逻辑实现 Logits 更新) ---
                    split_sizes = [feat.shape[2] * feat.shape[3] for feat in x_raw]
                    mask_splits = torch.split(score_mask, split_sizes, dim=1)

                    new_x_raw = []
                    for i, raw_feat in enumerate(x_raw):
                        B, C, H, W = raw_feat.shape
                        layer_mask = mask_splits[i].view(B, 1, H, W)

                        bbox_logits = raw_feat[:, :4, ...]
                        cls_logits = raw_feat[:, 4:, ...]

                        # 核心逻辑:
                        # New_Prob = Old_Prob * Ratio
                        #          = Old_Prob * (New_Score / Old_Prob)
                        #          = New_Score (实现替换)
                        current_prob = cls_logits.sigmoid()
                        new_prob = current_prob * layer_mask

                        new_prob = torch.clamp(new_prob, min=1e-6, max=1-1e-6)
                        new_cls_logits = torch.logit(new_prob)

                        new_feat = torch.cat((bbox_logits, new_cls_logits), dim=1)
                        new_x_raw.append(new_feat)


                    # 更新 raw logits
                    x_raw = new_x_raw
                    x = (x[0], x_raw)



        if self.model.training:
            x = x[1]
        else:
           pass

        return x


    def get_aligned_crop_coords(self,bx1, by1, bx2, by2, crop_size=64, offset=0):
        """
        计算 5 个截图框的左上角坐标 (x1, y1)。
        逻辑：以 Box 边界为基准，向内偏移 offset，然后截取 crop_size 大小。
        """
        bw = bx2 - bx1
        bh = by2 - by1

        # === 情况 A: Box 比截图尺寸小 ===
        # 这种情况下无法“向内偏移”，一缩就缩没了。
        # 策略：强制取 Box 的几何中心，截取 64x64 (包含完整 Box + 填充)
        if bw < crop_size or bh < crop_size:
            # 计算中心点的左上角坐标
            cx = (bx1 + bx2) / 2 - crop_size / 2
            cy = (by1 + by2) / 2 - crop_size / 2
            # 返回 5 个一样的中心截图
            return [(cx, cy)] * 5

        # === 情况 B: Box 足够大，执行“边界对齐 + 向内偏移” ===

        # 1. 左上 (Top-Left):
        # 基准: Box左上角 (bx1, by1)
        # 动作: 向右(x+), 向下(y+) 偏移 offset
        tl_x = bx1 + offset
        tl_y = by1 + offset

        # 2. 右上 (Top-Right):
        # 基准: Box右边界 (bx2) 和 上边界 (by1)
        # 动作: 截图框右边对齐 (bx2 - offset) -> 左边 = bx2 - offset - crop_size
        #      截图框上边对齐 (by1 + offset)
        tr_x = bx2 - offset - crop_size
        tr_y = by1 + offset

        # 3. 左下 (Bottom-Left):
        # 基准: Box左边界 (bx1) 和 下边界 (by2)
        # 动作: 截图框左边对齐 (bx1 + offset)
        #      截图框下边对齐 (by2 - offset) -> 上边 = by2 - offset - crop_size
        bl_x = bx1 + offset
        bl_y = by2 - offset - crop_size

        # 4. 右下 (Bottom-Right):
        # 基准: Box右边界 (bx2) 和 下边界 (by2)
        # 动作: 截图框右边对齐 (bx2 - offset) -> 左边 = bx2 - offset - crop_size
        #      截图框下边对齐 (by2 - offset) -> 上边 = by2 - offset - crop_size
        br_x = bx2 - offset - crop_size
        br_y = by2 - offset - crop_size

        # 5. 中心 (Center): 保持绝对几何中心 (也可以选择向内偏移，但几何中心最稳定)
        cc_x = (bx1 + bx2) / 2 - crop_size / 2
        cc_y = (by1 + by2) / 2 - crop_size / 2

        # 返回 5 个截图框的【左上角】坐标
        return [
            (tl_x, tl_y),
            (tr_x, tr_y),
            (bl_x, bl_y),
            (br_x, br_y),
            (cc_x, cc_y)
        ]

    def _build_5point_crop_topleft_on_orig(self, boxes_xyxy_ori, ratio_pad, ori_shape, crop_size=64, offset_val=0):
        """
        构建每个 bbox 的 5 点截图左上角坐标，坐标系为原图坐标。
        返回 shape = (N*5, 2) 的 int64 Tensor，且做边界裁剪避免负索引。
        """
        if boxes_xyxy_ori.numel() == 0:
            return torch.empty((0, 2), device=boxes_xyxy_ori.device, dtype=torch.int64)

        device = boxes_xyxy_ori.device
        h_ori, w_ori = int(ori_shape[0]), int(ori_shape[1])
        max_h = max(h_ori, crop_size)
        max_w = max(w_ori, crop_size)
        max_x = max_w - crop_size
        max_y = max_h - crop_size

        bx1 = boxes_xyxy_ori[:, 0]
        by1 = boxes_xyxy_ori[:, 1]
        bx2 = boxes_xyxy_ori[:, 2]
        by2 = boxes_xyxy_ori[:, 3]

        bw = bx2 - bx1
        bh = by2 - by1

        tl_x = bx1 + offset_val
        tl_y = by1 + offset_val
        tr_x = bx2 - offset_val - crop_size
        tr_y = by1 + offset_val
        bl_x = bx1 + offset_val
        bl_y = by2 - offset_val - crop_size
        br_x = bx2 - offset_val - crop_size
        br_y = by2 - offset_val - crop_size
        cc_x = (bx1 + bx2) / 2 - crop_size / 2
        cc_y = (by1 + by2) / 2 - crop_size / 2

        coords = torch.stack(
            [
                torch.stack([tl_x, tl_y], dim=1),
                torch.stack([tr_x, tr_y], dim=1),
                torch.stack([bl_x, bl_y], dim=1),
                torch.stack([br_x, br_y], dim=1),
                torch.stack([cc_x, cc_y], dim=1),
            ],
            dim=1,
        )  # (N, 5, 2)

        small = (bw < crop_size) | (bh < crop_size)
        if small.any():
            cc = torch.stack([cc_x, cc_y], dim=1)  # (N, 2)
            coords[small] = cc[small].unsqueeze(1).expand(-1, 5, 2)

        coords[..., 0] = coords[..., 0].clamp(0, max_x)
        coords[..., 1] = coords[..., 1].clamp(0, max_y)

        coords = coords.round().to(torch.int64)
        return coords.view(-1, 2).to(device)
    # def get_crop_images(self, centers, r):
    #     crop_images_all = []
    #     idx2b = []
    #     current_idx = 0
    #     for b in range(len(centers)):
    #         if len(centers[b]) ==0:
    #             # crop_images_all.append(None)
    #             idx2b.append(None)
    #             continue
    #         crop_images = []
    #         img_pre = []
    #         scaled_h = self.batch['ori_shape'][b][0] * self.batch['ratio_pad'][b][0][0]
    #         scaled_w = self.batch['ori_shape'][b][1] * self.batch['ratio_pad'][b][0][1]
    #         padded_h = (self.batch['resized_shape'][b][0] - scaled_h) / 2
    #         padded_w = (self.batch['resized_shape'][b][1] - scaled_w) / 2
    #
    #         original_image = cv2.imread(self.batch['im_file'][b])
    #         mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet 均值
    #         std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet 标准差
    #
    #         for center in centers[b]:
    #             x, y = center
    #             center_original = convert_to_original_coords(max(padded_w,x), max(padded_h,y), r[b], padded_w, padded_h)
    #
    #             new_image = self.crop_image(original_image, center_original, crop_size=(64, 64))
    #             # cv2.imwrite(self.batch['im_file'][0][-9:-4]+'Crop.jpg', new_image)
    #             new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
    #             new_image = np.float32(new_image) / 255.0  # 归一化到 [0, 1]
    #             new_image = torch.from_numpy(np.transpose(new_image, axes=[2, 0, 1])).float()  # RGB CHW
    #
    #             # 对每个通道进行归一化
    #             crop_image = (new_image - mean[:, None, None]) / std[:, None, None]
    #             crop_images_all.append(crop_image)
    #             current_idx += 1
    #
    #         idx2b.append(current_idx)
    #         # crop_images_all.append(crop_images)
    #
    #     return torch.stack(crop_images_all, dim=0), idx2b

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

    def crop_image(self, image, centers, crop_size=(64, 64)):
        """
        同时裁剪多个位置的图像区域
        :param image: 输入图像 (np.ndarray)
        :param centers: 含有多个中心坐标的列表，格式为 [(x1, y1), (x2, y2), ...]
        :param crop_size: 每个裁剪区域的大小 (width, height)
        :return: 裁剪后的图像列表
        """
        H_img, W_img = image.shape[:2]
        crop_w, crop_h = crop_size
        crop_images = []

        # 对于每个中心点，裁剪图像
        for (cx, cy) in centers:
            # 计算裁剪区域的左上角
            x1 = int(round(cx - crop_w / 2.0))
            y1 = int(round(cy - crop_h / 2.0))

            # 确保裁剪区域不会超出图像边界
            x1 = max(0, min(W_img - crop_w, x1))
            y1 = max(0, min(H_img - crop_h, y1))

            # 计算右下角
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            # 提取裁剪区域
            patch = image[y1:y2, x1:x2]

            # 如果裁剪区域的尺寸不符合目标尺寸，进行调整
            if patch.shape[0] != crop_h or patch.shape[1] != crop_w:
                patch = cv2.resize(patch, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

            crop_images.append(patch)

        return crop_images

    # def crop_image(self, image, center, crop_size=(64, 64), border_value=(0, 0, 0)):
    #     """
    #     基于中心点裁出固定尺寸（width, height）的图像块。
    #     - 固定右下边界：x2=x1+width, y2=y1+height，杜绝 2r+1/取整导致的 65。
    #     - 贴边/越界时会先对整图做黑边填充，保证能取到完整块。
    #     """
    #     H_img, W_img = image.shape[:2]
    #     crop_w, crop_h = int(crop_size[0]), int(crop_size[1])
    #     cx, cy = float(center[0]), float(center[1])
    #
    #     # 若原图比目标块还小，先把整图 pad 到至少 crop_size
    #     need_pad_w = max(0, crop_w - W_img)
    #     need_pad_h = max(0, crop_h - H_img)
    #     if need_pad_w > 0 or need_pad_h > 0:
    #         pad_left = need_pad_w // 2
    #         pad_right = need_pad_w - pad_left
    #         pad_top = need_pad_h // 2
    #         pad_bot = need_pad_h - pad_top
    #         image = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right,
    #                                    borderType=cv2.BORDER_CONSTANT, value=border_value)
    #         W_img += need_pad_w
    #         H_img += need_pad_h
    #         cx += pad_left
    #         cy += pad_top
    #
    #     # 计算左上角，用 round 保持对称，然后 clamp 到合法范围
    #     x1 = int(round(cx - crop_w / 2.0))
    #     y1 = int(round(cy - crop_h / 2.0))
    #     x1 = max(0, min(W_img - crop_w, x1))
    #     y1 = max(0, min(H_img - crop_h, y1))
    #     x2 = x1 + crop_w
    #     y2 = y1 + crop_h
    #
    #     patch = image[y1:y2, x1:x2]
    #
    #     # 兜底：极端情况下保证尺寸一致
    #     if patch.shape[0] != crop_h or patch.shape[1] != crop_w:
    #         patch = cv2.resize(patch, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    #
    #     return patch


    def get_cam(self,x):
        # calculate gradient
        # self.set_hook()
        grayscale_cam = None
        try:
            if True:
                self.model.model.zero_grad()
                if hasattr(self, 'detail_model'):
                    self.detail_model.zero_grad()
                loss = self.target(x[0])
                if torch.all(loss == 0):
                    return None


                torch.autograd.backward(loss, grad_tensors=torch.ones_like(loss).detach(), retain_graph=self.model.training)

                 # calculate cam image
                with torch.no_grad():
                    cam_per_target_layer = self.compute_cam_per_layer()
                    grayscale_cam = self.aggregate_multi_layers(cam_per_target_layer)

        except Exception as e:
            print(f"Error in get_cam: {e}")
            # 打印详细报错信息，方便调试
            import traceback
            traceback.print_exc()

        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()  # 清空列表

            # 再次清空梯度，保持整洁
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

        target_size = tuple(self.batch['resized_shape'][0])

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for name, layer_activations in self.activations.items():
            if name not in self.gradients:
                print(f"Warning: Gradient for {name} not found.")
                continue
            layer_grads = self.gradients[name]
            layer_activations = layer_activations.cpu().data.numpy()
            layer_grads = layer_grads.cpu().data.numpy()

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
            # self.set_requires_grad()
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
        debug_vis = os.getenv("BGD_DEBUG_VIS", "0").strip().lower() not in ("0", "", "false", "no")
        debug_vis_dir = os.getenv("BGD_DEBUG_VIS_DIR", "runs/vis_crop_boxes")
        try:
            debug_vis_max = int(os.getenv("BGD_DEBUG_VIS_MAX", "50"))
        except Exception:
            debug_vis_max = 50

        yolo_model = BGD_YOLO(model,target_layers=[21],detail_weight='run/detail_net_attn.pt',
                              detail_mode=True,
                              debug_vis=debug_vis,
                              debug_vis_dir=debug_vis_dir,
                              debug_vis_max=debug_vis_max)

        yolo_model.val(**args)
    else:
        validator = ClassificationValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':


    val(test_only=False)
