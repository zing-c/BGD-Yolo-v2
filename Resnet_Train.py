from Resnet6 import ResNet6
from Resnet6 import Detail_Net
from Resnet6 import Detail_Net_attn_1
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import torchvision.models as models
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR
from detailModel.inceptionBlock import Inception_atten,Inception_atten_conv,InceptionResidual_atten
from Resnet6 import Detail_Net_attn,Detail_Net_attn_dialation
from detail_model import DETAIL_ARCH_VERSION

import cv2
import numpy as np
from PIL import Image


# class FocalLoss(nn.Module):
#     def __init__(self, neg_weight=0.5):
#         super(FocalLoss, self).__init__()
#         self.neg_weight = neg_weight
#
#     def forward(self, inputs, targets):
#         # 计算二元交叉熵损失
#         bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
#         # 计算概率
#         weighted_loss = torch.where(targets == 0, bce_loss * self.neg_weight, bce_loss)
#         # 返回平均损失
#         return weighted_loss.mean()
#
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, inputs, targets):
        eps = 1e-6  # 防止 log(0)
        inputs = inputs.clamp(min=eps, max=1.0 - eps)

        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
class GlassDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None,use_clahe=False):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir  # 数据集路径
        self.transform = transform
        self.use_clahe = use_clahe


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.iloc[idx, 0])  # 拼接图像路径
        label = int(self.data.iloc[idx, 1])  # 获取标签
        image = Image.open(img_name).convert('RGB')  # 读取图像

        if self.use_clahe:
            image = self.clahe_transform(image)

        if self.transform:
            image = self.transform(image)

        return image, label

def validator(val_ataloader,model,device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for images, labels in val_ataloader:
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images.to(device))
            predicted = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    labels_tensor = torch.as_tensor(np.asarray(all_labels)).reshape(-1).bool()
    predictions_tensor = torch.as_tensor(np.asarray(all_predicted)).reshape(-1).bool()
    true_positive = (labels_tensor & predictions_tensor).sum().item()
    false_positive = ((~labels_tensor) & predictions_tensor).sum().item()
    false_negative = (labels_tensor & (~predictions_tensor)).sum().item()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    model.train()
    return acc, precision, recall, f1

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 子图1：Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-o', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 子图2：准确率与指标
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], 'b-x', label='Val Acc (%)')
    plt.plot(epochs, [p*100 for p in history['val_precision']], 'g--', label='Val Precision (%)')
    plt.plot(epochs, [r*100 for r in history['val_recall']], 'm--', label='Val Recall (%)')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plot.png') # 保存结果图
    plt.close()

def mode(task='train'):
    device = 'cuda'

    # 模型实例化、损失函数、优化器
    #model = ResNet6()

    #model = Detail_Net()
    model = Detail_Net_attn()
    # model = Inception_atten(v1=True)
    # model = InceptionResidual_atten()

    #.resnet18(pretrained=False)  # pretrained=False 表示不加载预训练权重
    #model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #model.maxpool = nn.Identity()  # 用 Identity 层替换，相当于跳过这一层
    #model.fc = nn.Sequential(
    #    nn.Linear(model.fc.in_features, 1),  # Linear 层
    #    nn.Sigmoid()  # Sigmoid 激活函数
    #)
    model.to(device)
    criterion = FocalLoss() #nn.BCELoss()
    # criterion=nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    # ----------------------------------------------------------------------------------------------------------------
    history = {
        'epochs': [],
        'train_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    # 数据预处理
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if task == 'train':
        val_dataset = GlassDataset(csv_file='BGD_lable/val/lable.csv',
                                   data_dir='BGD_image/val',
                                   transform=transform,
                                   use_clahe=False)
        val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # 加载数据集
        dataset = GlassDataset(csv_file='BGD_lable/train/mixed_lable.csv',
                               data_dir='BGD_image/mixed_data',
                               transform=transform,
                               use_clahe=False)



        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


        # 训练过程
        num_epochs = 40
        max_acc = 0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in dataloader:
                labels = labels.float().unsqueeze(1).to(device)  # 将标签转换为浮点型并添加维度

                # 前向传播
                outputs = model(images.to(device))
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                running_loss += loss.item()

            # 每个 epoch 结束后更新学习率
            scheduler.step()
            avg_loss = running_loss / len(dataloader)
            # --- 修改点：让 validator 返回多个指标 ---
            # 假设你已经按我之前的建议修改了 validator 函数，使其返回 (acc, p, r, f1)
            acc, prec, rec, f1 = validator(val_dataloader, model, device)

            # --- 记录数据到 history ---
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(avg_loss)
            history['val_acc'].append(acc)
            history['val_precision'].append(prec)
            history['val_recall'].append(rec)
            history['val_f1'].append(f1)

            # --- 打印更详细的报告 ---
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Loss:   {avg_loss:.4f}')
            print(f'Acc:    {acc:.2f}% | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}')
            print('-' * 45)




            if max_acc <= acc and epoch > 4:
                max_acc = acc
                checkpoint_path = Path('run/detail_net_atten') / f'{DETAIL_ARCH_VERSION}.pt'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        'detail_arch': DETAIL_ARCH_VERSION,
                        'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'val_accuracy': acc,
                        'val_precision': prec,
                        'val_recall': rec,
                        'val_f1': f1,
                    },
                    checkpoint_path,
                )

        # === 关键：在此处添加调用语句 ===
        print("训练结束，正在生成可视化图表...")
        plot_training_history(history)


if __name__ == '__main__':
    mode('train')
