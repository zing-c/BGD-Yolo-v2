# Ultralytics YOLO 🚀, AGPL-3.0 license
# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from PIL import Image
from ultralytics import yolo  # noqa
from ultralytics import YOLO
from ultralytics.yolo.data import ClassificationDataset, build_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER
from ultralytics.yolo.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.yolo.utils.ops import non_max_suppression
from ultralytics.yolo.utils.plotting import plot_images
from ultralytics.yolo.cfg import get_cfg
from pathlib import Path
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import numpy as np
from typing import Callable, List, Optional, Tuple
from gradcam import letterbox, find_max_heatmap_center, find_max_heatmap_center_torch, convert_to_original_coords, \
    yolov8_target, find_max_heatmap_center_torch_optimized
from Resnet6 import Detail_Net_attn
from gradcam import yolov8_heatmap
import cv2
from math import ceil
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
import matplotlib.pyplot as plt
import time
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

from ultralytics_new.yolo.utils.ops import xywh2xyxy


class BGD_YOLO(YOLO):
    def __init__(self, model=None, task=None, detail_mode=True, target_layers=None,
                 cam_mode='Grad', output_type='class', conf_threshold=0.5, detail_weight=None,
                 iou_threshold=0.5, plot=False, source=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(model, task)
        self.validator = None
        self.detail_mode = detail_mode
        self.model._predict_once = self._predict_once
        self.batch = None
        self.target_layers = target_layers
        self.handles = []
        self.plot = plot
        self.source = source

        self.gradients = []
        self.activations = []
        self.m_layers = []
        self.cam_mode = cam_mode
        self.target = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if self.detail_mode:

            self.target = yolov8_target(output_type=output_type, conf=0.5, ratio=0.1)
            detail_model = Detail_Net_attn()
            detail_model.load_state_dict(torch.load(detail_weight, weights_only=False))
            self.detail_model = detail_model.eval().to(self.device)
            self.detail_input_buf = torch.empty((32, 3, 64, 64), device=self.device)  # max_batch 根据需求设置
            # prepare layer for cam calculation
            for l in self.target_layers:
                self.m_layers.append(self.model.model[l])
                for p in self.model.model[l].parameters():
                    p.requires_grad_(True)
                # add hooks to get gradients and activations
                self.handles.append(
                    self.model.model[l].register_forward_hook(self.save_activation))
                self.handles.append(
                    self.model.model[l].register_forward_hook(self.save_gradient))



    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients
        output.register_hook(_store_grad)

    def release(self):
        for handle in self.handles:
            handle.remove()

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)[0]
        return result

    def set_requires_grad(self):
        for i in self.m_layers:
            for j in i.parameters():
                j.requires_grad = True

    def _predict_once(self, x, profile=False, visualize=False):
        input_shape = x.shape
        self.gradients = []
        self.activations = []
        original_image = None

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

        if self.detail_mode and self.batch is not None:


            # x = (self.post_process(x[0]), x[1])  # 创建一个新的元组，修改第一个元素

            condition = x[0][0][-1] > self.conf_threshold
            yolo_results = x[0][0].T[condition]

    # if self.renormalize:
    #     cam_image = yolov8_heatmap.renormalize_cam_in_bounding_boxes(boxes, img, grayscale_cam)
    #
    # save_filename = ''
    # cam_image = Image.fromarray(cam_image)
    # cam_image.save(save_filename)





    # calculate cam image

            grayscale_cam = self.get_cam(x)

            # print(f"cam_ALL:{(t10 - t5) * 1000}ms")

            # cam_image = show_cam_on_image(self.batch['im_file'], grayscale_cam, use_rgb=True)


            if grayscale_cam is None or yolo_results.shape[0] == 0:

                return x

            with torch.no_grad():
                boxes = xywh2xyxy(yolo_results[:, :4])  # x1 y1 x2 y2
                x_idxes = condition.nonzero().tolist()


                r = sum(self.batch['ratio_pad'][0][0]) / 2
                window_size = ceil(64 * r)

                centers, c2b = find_max_heatmap_center_torch_optimized(grayscale_cam, boxes, window_size=window_size)

                # detail_batch = self.get_crop_images(centers, r)
                detail_batch = self.detail_input_buf[:len(centers)]
                original_image = cv2.imread(self.batch['im_file'][0])
                detail_batch.copy_(self.get_crop_images(centers, r,original_image),)  # 保证为 torch tensor

                # self.detail_model.to(self.device)
                detail_results = self.detail_model(detail_batch)
                t=1
                for c_i, box_idxes in enumerate(c2b):
                    for box_idx in box_idxes:
                        # confs[box_idx] = detail_results.detach()[c_i][0]
                        x[0][0][-1][x_idxes[box_idx][0]] =  torch.sqrt(detail_results.detach()[c_i][0]*x[0][0][-1][x_idxes[box_idx][0]])
                        # x[0][0][-1][x_idxes[box_idx][0]] = detail_results[c_i][0] *t +(1-t)* x[0][0][-1][x_idxes[box_idx][0]]
                        #x[0][0][-1][x_idxes[box_idx][0]] =torch.sigmoid(detail_results[c_i][0])*t +(1-t)* x[0][0][-1][x_idxes[box_idx][0]]

                # 显式释放变量
                del detail_results

        x = (x[0].detach().requires_grad_(False),x[1])
        # torch.cuda.empty_cache()

        # if self.plot and original_image is not None:
        #     self.plot_image(preds=x,orig_imgs=original_image)
        return x

    def plot_image(self, preds=None, orig_imgs=None):
        """Postprocess predictions and returns a list of Results objects."""
        bboxes, scores = preds[:2]  # (1, bs, 300, 4), (1, bs, 300, nc)
        bboxes, scores = bboxes.squeeze_(0), scores.squeeze_(0)
        results = []
        img = self.batch['img']

        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            oh, ow = orig_img.shape[:2]
            if not isinstance(orig_imgs, torch.Tensor):
                pred[..., [0, 2]] *= ow
                pred[..., [1, 3]] *= oh
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def get_crop_images(self, centers, r, original_image):
        crop_images = []
        img_pre = []
        scaled_h = self.batch['ori_shape'][0][0] * self.batch['ratio_pad'][0][0][0]
        scaled_w = self.batch['ori_shape'][0][1] * self.batch['ratio_pad'][0][0][1]
        padded_h = (self.batch['resized_shape'][0][0] - scaled_h) / 2
        padded_w = (self.batch['resized_shape'][0][1] - scaled_w) / 2

        for center in centers:
            x, y = center
            center_original = convert_to_original_coords(max(padded_w,x), max(padded_h,y), r, padded_w, padded_h)
            new_image = self.crop_image(original_image, center_original, crop_size=(64, 64))
            # cv2.imwrite(self.batch['im_file'][0][-9:-4]+'Crop.jpg', new_image)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
            new_image = np.float32(new_image) / 255.0  # 归一化到 [0, 1]
            new_image = torch.from_numpy(np.transpose(new_image, axes=[2, 0, 1])).float()  # RGB CHW

            mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet 均值
            std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet 标准差

            # 对每个通道进行归一化
            crop_image = (new_image - mean[:, None, None]) / std[:, None, None]
            crop_images.append(crop_image)
        return torch.stack(crop_images, dim=0)

    def crop_image(self, image, center, crop_size=(64, 64)):

        """
        基于中心点截取图像的一部分
        :param image: 输入图像 (np.ndarray)
        :param center: 中心点坐标 (cx, cy)
        :param crop_size: 截取区域的大小 (width, height)
        :return: 截取的图像区域 (np.ndarray)
        """
        h, w = image.shape[:2]
        cx, cy = center
        crop_w, crop_h = crop_size
        x1 = max(0, int(cx - crop_w // 2))
        y1 = max(0, int(cy - crop_h // 2))
        x2 = min(w, int(cx + crop_w // 2))
        y2 = min(h, int(cy + crop_h // 2))


        cropped_image = image[y1:y2, x1:x2]

        # 填充到目标尺寸
        if cropped_image.shape[0] < crop_h or cropped_image.shape[1] < crop_w:
            pad_h = max(0, crop_h - cropped_image.shape[0])
            pad_w = max(0, crop_w - cropped_image.shape[1])
            cropped_image = cv2.copyMakeBorder(
                cropped_image,
                pad_h // 2, pad_h - pad_h // 2,
                pad_w // 2, pad_w - pad_w // 2,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        return cropped_image

    def get_cam(self,x):
        # calculate gradient
        self.model.model.zero_grad()
        loss = self.target(x[0])
        if loss==0:
            return None

        loss.backward(retain_graph=True )
        # calculate cam image
        with torch.no_grad():

            cam_per_target_layer = self.compute_cam_per_layer()
            grayscale_cam = self.aggregate_multi_layers(cam_per_target_layer)


        return grayscale_cam[0,:]

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
        return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')

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
        yolo_model = BGD_YOLO(model,target_layers=[18],detail_weight='run/detail_net_attn.pt',
                              detail_mode=False,plot=True)

        yolo_model.val(**args)
    else:
        validator = ClassificationValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val(test_only=False)


