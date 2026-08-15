from ultralytics import YOLO
import torch
from Resnet6 import Detail_Net
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics.yolo.v8.detect.val import DetectionValidator
from torchvision.ops import box_iou
from tqdm import tqdm
from gradcam import letterbox, find_max_heatmap_center, find_max_heatmap_center_torch, attempt_load_weights, convert_to_original_coords, yolov8_target, \
    yolov8_heatmap
from pytorch_grad_cam import GradCAM,XGradCAM,EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics.yolo.utils.ops import xywh2xyxy, non_max_suppression
from torchvision import transforms
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import dataloader
import yaml
from Resnet6 import ResNet6
import cv2
import numpy as np
import time
from torch.nn import functional as F


class Get_dataloder(Dataset):
    def __init__(self, data_yaml, img_size=640):
        super(Get_dataloder, self).__init__()
        """
        初始化数据集
        :param data_yaml: 数据集配置文件的路径
        :param img_size: 图片的尺寸
        :param transform: 数据预处理操作
        """
        with open(data_yaml, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        self.img_dir = os.path.join(self.data['path'], self.data['val'])  # 验证集图片路径
        self.label_dir = os.path.join(self.data['path'], 'labels', 'val')  # 标注文件

        self.img_size = img_size
        self.imgs = [os.path.join(self.img_dir, img) for img in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img_pre = img
        # 获取图像的原始尺寸
        img_height, img_width = img.shape[:2]

        # 使用 letterbox 方法调整图片大小
        img, ratio, (dw, dh) = letterbox(img, new_shape=(self.img_size, self.img_size))
        info = (dw, dh, ratio)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float() / 255.0

        # 转换为 RGB 格式并归一化
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
        # img = np.float32(img) / 255.0  # 归一化到 [0, 1]

        # 转换为 PyTorch 张量
        # img_tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).float()

        # 加载标注文件
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        true_boxes, true_labels = self.load_labels(label_path, img_width, img_height)

        return img_tensor, img_path, info, true_boxes, true_labels, img_pre,ratio[0]

    def load_labels(self, label_path, img_width, img_height):
        """
        加载标注文件并计算绝对坐标的 box
        :param label_path: 标注文件路径
        :param img_width: 图像的原始宽度
        :param img_height: 图像的原始高度
        :return: true_boxes (List[List[float]]), true_labels (List[int])
        """
        true_boxes = []
        true_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:  # YOLO 格式: class_id, x_center, y_center, width, height
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 转换为绝对坐标
                        x_min = (x_center - width / 2) * img_width
                        y_min = (y_center - height / 2) * img_height
                        x_max = (x_center + width / 2) * img_width
                        y_max = (y_center + height / 2) * img_height

                        true_boxes.append([x_min, y_min, x_max, y_max])
                        true_labels.append(class_id)
        return true_boxes, true_labels


class Validator:
    def __init__(self, yolo_weight, detail_weight, layers, device, conf_threshold=0.5, iou_threshold=0.7,
                 backward_type='box', ratio=0.1, grad_cam_method='GradCAM', target_layers=None):
        """
        初始化验证器
        :param yolo_model: YOLO 检测模型
        :param detail_model: 细节检测网络
        :param device: 设备 (cuda/cpu)
        :param conf_threshold: YOLO 置信度阈值
        :param grad_cam_method: Grad-CAM 方法名称
        :param target_layers: Grad-CAM 的目标层
        """
        self.yolo_weight = yolo_weight
        self.detail_weight = detail_weight
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.ratio = ratio
        self.backward_type = backward_type
        self.layers = layers

        ckpt = torch.load(self.yolo_weight)
        # yolo = YOLO(self.yolo_weight)
        # yolo.val(batch=1)
        self.model_names = ckpt['model'].names
        self.yolo = attempt_load_weights(self.yolo_weight, self.device)

        detail_model = Detail_Net()
        detail_model.load_state_dict(torch.load(self.detail_weight))
        detail_model = detail_model.to(self.device)
        detail_model.eval()
        # 将模型赋值给 self.ResNet6
        self.detail_model = detail_model

        for i,p in enumerate(self.yolo.parameters()):
            if i > layers[0]:
                p.requires_grad_(True)

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)[0]
        return result

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

    def validate(self, val_loader):
        """
        执行验证
        :param val_loader: 验证集 DataLoader
        :return: 整体 mAP, precision, recall
        """
        recorded_time = []
        # # 初始化 Grad-CAM
        # method = eval(self.method)(model=yolo_model, target_layers=target_layers)
        # method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        time_start = time.time()
        target = yolov8_target(self.backward_type, self.conf_threshold, ratio=self.ratio)
        target_layers = [self.yolo.model[l] for l in self.layers]
        cam = GradCAM(model=self.yolo, target_layers=target_layers)

        metric = MeanAveragePrecision(iou_type="bbox",class_metrics=True)

        # initial step
        time_end = time.time()
        recorded_time.append('Initial Time:'+str(time_end-time_start)+', ')
        # Forward yolo + heat map; Process outputs; Find center; Forward D-Model; Post Process
        time_seq = [0,0,0,0,0]
        for image, image_path, info, true_boxes, true_labels, img_pre,ratio in tqdm(val_loader, desc="Evaluating"):

            # 将单张图片移动到设备
            # image = image[0].unsqueeze(0).to(self.device)  # 添加批次维度

            # target = target.to(self.device)


            # 处理 YOLO 检测结果
            pred_boxes = []
            pred_confs = []
            pred_labels = []


            time_start = time.time()
            try:
                # CHW, RGB
                # 处理单张图片的检测结果
                grayscale_cam = cam(image, targets=[target])
            except AttributeError as e:
                continue
            time_end = time.time()
            time_seq[0] += time_end - time_start

            # Process outputs
            time_start = time.time()
            grayscale_cam = grayscale_cam[0, :]
            yolo_results = cam.outputs
            yolo_results = self.post_process(yolo_results)

            # 提取预测框、分数和类别
            boxes = yolo_results[:, :4]  # x1 y1 x2 y2
            confs = yolo_results[:, 4]  # 置信度

            # 获取当前图像的 info
            dw, dh, r = info[0].item(), info[1].item(), info[2][0].item()

            time_end = time.time()
            time_seq[1] += time_end - time_start

            time_start = time.time()
            # 在每个检测框内找到最大热力值窗口的中心点
            centers, c2b = find_max_heatmap_center_torch(grayscale_cam, boxes, window_size=64)
            time_end = time.time()
            time_seq[2] += time_end - time_start

            time_start = time.time()
            crop_images = []
            # 将中心点坐标转换回原始图像坐标系
            for center in centers:
                x, y = center
                center_original = convert_to_original_coords(x, y, r, dw, dh)

                # image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 数组 (H, W, C)
                # original_image = cv2.imread(image_path[0])
                original_image = img_pre[0].squeeze(0).numpy()
                new_image = self.crop_image(original_image, center_original, crop_size=(64, 64))
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
                new_image = np.float32(new_image) / 255.0  # 归一化到 [0, 1]
                new_image = torch.from_numpy(np.transpose(new_image, axes=[2, 0, 1])).float()  # RGB CHW

                mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet 均值
                std = torch.tensor([0.229, 0.224, 0.225])  # ImageNet 标准差

                # 对每个通道进行归一化
                crop_image = (new_image - mean[:, None, None]) / std[:, None, None]

                crop_images.append(crop_image)
            if len(crop_images) == 0:
                continue

            time_end = time.time()
            time_seq[3] += time_end - time_start
            time_start = time.time()

            batch = torch.stack(crop_images, dim=0)
            detail_results = self.detail_model(batch.to(self.device))
            predicted = (detail_results > 0.5).float()

            valid_indices = [i for i, pred in enumerate(predicted) if pred.item() == 1]
            for i in valid_indices:
                center = centers[i]  # 当前中心点
                box_indexes = c2b[i]
                for box_idx in box_indexes:
                    box = boxes[box_idx]
                    box_ = []
                    x_min = (box[0] - dw) / r
                    y_min = (box[1] - dh) / r
                    x_max = (box[2] - dw) / r
                    y_max = (box[3] - dh) / r
                    box_.append(x_min)
                    box_.append(y_min)
                    box_.append(x_max)
                    box_.append(y_max)

                    pred_boxes.append(box_)
                    pred_confs.append(confs[box_idx])
                    pred_labels.append(0)

                # # 如果需要重新归一化热力图
                # if self.renormalize:
                #     cam_image = self.renormalize_cam_in_bounding_boxes(boxes, img, grayscale_cam)

            time_end = time.time()
            time_seq[4] += time_end - time_start

                # 保存结果
            preds = [
                {
                    "boxes": torch.tensor(pred_boxes),
                    "scores": torch.tensor(pred_confs),
                    "labels": torch.tensor(pred_labels),
                }
            ]
            targets = [
                {
                    "boxes": torch.tensor(true_boxes),
                    "labels": torch.tensor(true_labels),
                }
            ]

            # 更新指标计算器
            metric.update(preds, targets)
        total_image = len(val_loader)
        print(recorded_time)
        # Forward yolo + heat map; Process outputs; Find center; Forward D-Model; Post Process
        print(f"Forward H-Model: {time_seq[0]/total_image}, Process Outputs: {time_seq[1]/total_image},"
              f"  Find Center: {time_seq[2]/total_image},"
              f" Crop Image: {time_seq[3]/total_image}, Forward D-Model: {time_seq[4]/total_image}")
        results = metric.compute()
        return {
            "mAP": results["map"].item(),  # mAP
            "mAP_50": results["map_50"].item(),  # mAP@0.5
            "mAP_75": results["map_75"].item(),  # mAP@0.75
            "precision": results["map_per_class"].mean().item(),  # 平均 Precision
            "recall": results["mar_100"].item(),  # Recall
        }


if __name__ == '__main__':
    init_time_l = [0, 0]
    init_time_l[0] = time.time()
    val_dataset = Get_dataloder(data_yaml='ultralytics/datasets/BG.yaml', img_size=640)
    val_dataset_loader = dataloader.DataLoader(val_dataset, batch_size=1, shuffle=True)

    val = Validator("runs/train9/weights/best.pt", detail_weight='run/best_model1.pt', device='cuda',
                    layers=[0,15,18,21], conf_threshold=0.2, iou_threshold=0.7)
    init_time_l[1] = time.time()
    init_time = init_time_l[1] - init_time_l[0]
    print('init_time:' + str(init_time))

    results = val.validate(val_dataset_loader)
    print(results)
    # print(torch.load('run/detail_net_attn.pt').keys())

