import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ultralytics import YOLO
import yaml
import torch.optim as optim
import sys

# 自定义数据集类
class JointTrainingDataset(Dataset):
    def __init__(self, image_dir, yaml_file, yolo_model, confidence_threshold=0.5):
        self.image_dir = image_dir
        self.yaml_file = yaml_file
        self.confidence_threshold = confidence_threshold  # YOLOv8 置信度阈值
        self.yolo_model = yolo_model  # YOLOv8 模型

        # 加载 YAML 文件，解析破损位置标注
        with open(yaml_file, 'r', encoding='utf-8') as file:
            self.annotations = yaml.load(file, Loader=yaml.FullLoader)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 获取当前图像的路径
        image_name = list(self.annotations.keys())[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 读取图像
        image = cv2.imread(image_path)

        # 使用 YOLOv8 检测目标
        results = self.yolo_model(image)
        detections = results.pandas().xywh[0]  # 获取 YOLOv8 检测结果

        # 提取高置信度的检测框
        high_conf_detections = detections[detections['confidence'] > self.confidence_threshold]
        cropped_images = []

        for _, row in high_conf_detections.iterrows():
            # 获取边界框坐标
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cropped_image = image[y1:y2, x1:x2]  # 裁剪图像

            # 预处理裁剪后的图像（调整大小、归一化等）
            cropped_image = self.preprocess_detail_image(cropped_image)
            cropped_images.append(cropped_image)

        # 返回图像和细节标签（细节检测的输入）
        return image, cropped_images

    def preprocess_detail_image(self, cropped_image):
        resized_image = cv2.resize(cropped_image, (320, 320))  # 假设细节检测网络需要320x320大小
        normalized_image = resized_image / 255.0  # 归一化到 [0, 1]
        tensor_image = torch.tensor(normalized_image).permute(2, 0, 1).unsqueeze(0).float()  # 转为Tensor
        return tensor_image


# 细节检测网络模型
class DetailDetectionModel(torch.nn.Module):
    def __init__(self):
        super(DetailDetectionModel, self).__init__()
        # 定义网络结构
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def image_process(image):
    confidence_thres = 0.5
    # 使用 YOLOv8 检测目标
    results = self.yolo_model(image['img'].float().cuda())
    for i in range(len(results[0])):
        img_path = image['im_file'][i]
        ori_shape = image['ori_shape'][i]
        boxes = []
        scores = []
        class_ids = []

        # 计算边界框坐标的缩放因子
        x_factor = ori_shape[0] / self.input_width
        y_factor = ori_shape[0] / self.input_height
        outputs = np.transpose(np.squeeze(results[0][0].cpu().detach().numpy()))

        for row in outputs:
            # 从当前行中提取类别分数
            classes_scores = row[4:]
            # 如果最大值大于置信度阈值
            if classes_scores >= confidence_thres:
                # 获取具有最高分数的类别ID
                class_id = 1

                # 从当前行中提取边界框坐标
                x, y, w, h = row[0], row[1], row[2], row[3]




    # 用于存储检测到的边界框、置信度分数和类别ID的列表


    # 遍历输出数组中的每一行


    # 提取高置信度的检测框
    high_conf_detections = detections[detections['confidence'] > self.confidence_threshold]
    cropped_images = []

    for _, row in high_conf_detections.iterrows():
        # 获取边界框坐标
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cropped_image = image[y1:y2, x1:x2]  # 裁剪图像

        # 预处理裁剪后的图像（调整大小、归一化等）
        cropped_image = self.preprocess_detail_image(cropped_image)
        cropped_images.append(cropped_image)

    # 返回图像和细节标签（细节检测的输入）
    return image, cropped_images
# 联合训练过程
def joint_training(image_dir, yaml_file, yolo_model, detail_model, epochs=10, batch_size=16, lr=1e-4):
    # 准备数据集和数据加载器
    dataset = JointTrainingDataset(image_dir, yaml_file, yolo_model)

    data_loader = yolo_model.get_dataloader(data=yaml_file)
    # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化器
    optimizer = optim.Adam(list(yolo_model.model.parameters()) + list(detail_model.parameters()), lr=lr)

    # 损失函数
    yolo_loss_fn = torch.nn.CrossEntropyLoss()  # 假设使用交叉熵损失
    detail_loss_fn = torch.nn.MSELoss()  # 细节检测可能是回归问题，这里假设使用MSE损失

    # 训练循环
    for epoch in range(epochs):

        detail_model.train()

        running_loss = 0.0
        for img in data_loader:
            images, cropped_images_list = image_process(img)
            optimizer.zero_grad()

            # YOLOv8 检测
            yolo_outputs = yolo_model(images)

            # 细节检测（基于 YOLOv8 的高置信度检测框）
            detail_labels = []
            for cropped_images in cropped_images_list:
                detail_labels.append(detail_model(cropped_images))  # 细节检测网络输出

            # 计算损失
            yolo_loss = yolo_loss_fn(yolo_outputs, images)  # 假设yolo模型有对应的标签
            detail_loss = sum([detail_loss_fn(detail_output, cropped_images) for detail_output, cropped_images in zip(detail_labels, cropped_images_list)])



            # 反向传播和优化
            detail_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")


