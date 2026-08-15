from ultralytics import YOLO
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm
from gradcam import letterbox, find_max_heatmap_center, attempt_load_weights, convert_to_original_coords,yolov8_target,yolov8_heatmap
from pytorch_grad_cam import GradCAM
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

class Get_dataloder(Dataset):
    def __init__(self, data_yaml, img_size=640):
        super(Get_dataloder, self).__init__()
        """
        初始化数据集
        :param data_yaml: 数据集配置文件的路径
        :param img_size: 图片的尺寸
        :param transform: 数据预处理操作
        """
        with open(data_yaml, 'r',encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
        self.img_dir = os.path.join(self.data['path'], self.data['val'])  # 验证集图片路径
        self.label_dir = os.path.join(self.data['path'], 'labels', 'val')  # 标注文件

        self.img_size = img_size
        self.imgs = [os.path.join(self.img_dir, img) for img in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img =cv2.imread(img_path)
        img_pre=img
        # 获取图像的原始尺寸
        img_height, img_width = img.shape[:2]


        # 使用 letterbox 方法调整图片大小
        img, ratio, (dw, dh) = letterbox(img, new_shape=(self.img_size, self.img_size))
        info=(dw, dh, ratio)



        # 转换为 RGB 格式并归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
        img = np.float32(img) / 255.0  # 归一化到 [0, 1]

        # 转换为 PyTorch 张量
        img_tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).float()

        # 加载标注文件
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        true_boxes, true_labels = self.load_labels(label_path, img_width, img_height)

        return img_tensor, img_path, info, true_boxes, true_labels,img_pre

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
    def __init__(self, yolo_weight,detail_weight,layers, device, conf_threshold=0.5, backward_type='box', ratio=0.1,grad_cam_method='GradCAM', target_layers=None):
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
        self.ratio=ratio
        self.backward_type = backward_type
        self.layers=layers

        ckpt = torch.load(self.yolo_weight)
        self.model_names = ckpt['model'].names
        self.yolo = attempt_load_weights(self.yolo_weight, self.device)

        detail_model=ResNet6()
        detail_model.load_state_dict(torch.load(self.detail_weight))
        detail_model = detail_model.to(self.device)
        detail_model.eval()
        # 将模型赋值给 self.ResNet6
        self.detail_model = detail_model



        for p in self.yolo.parameters():
            p.requires_grad_(True)




    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.5)[0]
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
        # # 初始化 Grad-CAM
        # method = eval(self.method)(model=yolo_model, target_layers=target_layers)
        # method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        target = yolov8_target(self.backward_type,  self.conf_threshold, ratio=self.ratio)
        target_layers = [self.yolo.model[l] for l in self.layers]
        cam = GradCAM(model=self.yolo, target_layers=target_layers)


        metric = MeanAveragePrecision(iou_type="bbox")




        #一个batch
        #with torch.no_grad():
        for image,image_path,info,true_boxes, true_labels,img_pre in tqdm(val_loader, desc="Evaluating"):
            start=time.time()

            # 将单张图片移动到设备
            # image = image[0].unsqueeze(0).to(self.device)  # 添加批次维度

            # target = target.to(self.device)

            # 处理 YOLO 检测结果
            pred_boxes = []
            pred_confs = []
            pred_labels=[]
            yolo_boxes=[]
            yolo_confs=[]
            yolo_labels=[]


            try:
                # 处理单张图片的检测结果
                grayscale_cam = cam(image, targets=[target])
            except AttributeError as e:
                continue

            grayscale_cam = grayscale_cam[0, :]
            yolo_results = cam.outputs
            yolo_results = self.post_process(yolo_results)   #参数问题?? iou


            # 提取预测框、分数和类别
            boxes = yolo_results[:, :4]  #x1 y1 x2 y2
            confs = yolo_results[:, 4]  # 置信度
            # labels = torch.zeros(len(boxes), dtype=torch.int)  # 类别标签全为 0
            yolo_boxes=boxes
            yolo_confs=confs
            yolo_lable=[0]*len(yolo_confs)


            # 获取当前图像的 info
            dw, dh, r = info[0].item(), info[1].item(), info[2][0].item()

            end = time.time()

            # 在每个检测框内找到最大热力值窗口的中心点
            centers, c2b = find_max_heatmap_center(grayscale_cam, boxes, window_size=64)



            crop_images = []
            # 将中心点坐标转换回原始图像坐标系
            for center in centers:
                x, y = center
                center_original = convert_to_original_coords(x, y, r, dw, dh)



                # image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 数组 (H, W, C)
                # original_image = cv2.imread(image_path[0])
                original_image=img_pre[0].squeeze(0).numpy()
                new_image = self.crop_image(original_image, center_original, crop_size=(64, 64))
                new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
                new_image = np.float32(new_image) / 255.0  # 归一化到 [0, 1]
                show_image=new_image

                mean = np.array([0.485, 0.456, 0.406])  # ImageNet 均值
                std = np.array([0.229, 0.224, 0.225])  # ImageNet 标准差
                new_image = (new_image - mean) / std

                crop_image = torch.from_numpy(np.transpose(new_image, axes=[2, 0, 1])).float()
                crop_images.append(crop_image)
            if len(crop_images) ==0:
                continue

            end2 = time.time()
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

                # 如果需要显示检测框
                # original_image = original_image[:, :, [2, 1, 0]]
                copy=original_image
                for i, data in enumerate(pred_boxes):

                    cam_image = self.draw_detections(
                        data[:4],
                        (255, 0, 0),
                        'a',
                        copy
                    )
            # if type(image_path)==str:
            #     image_path=tuple[image_path]
            #
            # image_path=image_path[0]  #??imagepath 为什么有时为tuple 有时为元组
            # file_name_with_extension = image_path.split('\\')[-1]  # 适用于 Windows 路径
            #
            # filename = os.path.join(r'E:\cv\cc\BGD\val_picture', file_name_with_extension)
            # if type(cam_image)==np.ndarray:
            #     cam_image=Image.fromarray(cam_image)
            #     cam_image.save(filename)
            # else:
            #
            #     cam_image.save(filename)
            #
            #
            # # show_image.squeeze(0).numpy().astype(np.uint8)
            # # c2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # c2_image = show_cam_on_image(image.squeeze(0).numpy().transpose(1,2,0), grayscale_cam, use_rgb=True)
            # filename2 = os.path.join(r'E:\cv\cc\BGD\demo_val_picture', file_name_with_extension)
            #
            # c2_image = Image.fromarray(c2_image)
            # c2_image.save(filename2)
            #
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

        yolo_preds=[
            {
                "boxes": torch.tensor(yolo_boxes),
                "scores": torch.tensor(yolo_confs),
                "labels": torch.tensor(yolo_labels),
            }
        ]

        metric.update(yolo_preds, targets)
        # 计算最终指标
        results = metric.compute()
        return {
            "mAP": results["map"].item(),  # mAP
            "mAP_50": results["map_50"].item(),  # mAP@0.5
            "mAP_75": results["map_75"].item(),  # mAP@0.75
            "precision": results["map_per_class"].mean().item(),  # 平均 Precision
            "recall": results["mar_100"].item(),  # Recall
        }




if __name__ == '__main__':

    val_dataset = Get_dataloder(data_yaml='ultralytics/datasets/BG.yaml', img_size=640)
    val_dataset_loader = dataloader.DataLoader(val_dataset, batch_size=1, shuffle=True)




    val=Validator("best.pt", detail_weight='run/detail_net_attn.pt', device='cuda',layers=[18],conf=0.5)
    results=val.validate(val_dataset_loader)
    print(results)
