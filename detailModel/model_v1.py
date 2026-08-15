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
import torchvision.models as models

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from detailModel.inceptionBlock import Inception_atten,Inception_atten_conv,InceptionResidual_atten
from Resnet6 import Detail_Net_attn


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
    def __init__(self, csv_file, data_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir  # 数据集路径
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.iloc[idx, 0])  # 拼接图像路径
        label = int(self.data.iloc[idx, 1])  # 获取标签
        image = Image.open(img_name).convert('RGB')  # 读取图像

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

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in val_ataloader:
            labels = labels.float().unsqueeze(1).to(device)
            all_labels.extend(labels.cpu().numpy())  # 收集所有真实标签

            # 前向传播
            outputs = model(images.to(device))
            # 计算准确率
            predicted = (outputs > 0.5).float()  # 将概率转换为0或1
            all_predicted.extend(predicted.cpu().numpy())  # 收集所有预测标签
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # 计算精确率、召回率和F1分数
    precision = precision_score(all_labels, all_predicted)
    recall = recall_score(all_labels, all_predicted)
    f1 = f1_score(all_labels, all_predicted)
    model.train()

    print(
        f'Validation Accuracy: {100 * correct / total:.2f}%, '
        f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    return 100 * correct / total

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

    # 数据预处理
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if task == 'train':
        val_dataset = GlassDataset(csv_file='BGD_lable/val/lable.csv',
                                   data_dir='BGD_image/val',
                                   transform=transform)
        val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # 加载数据集
        dataset = GlassDataset(csv_file='BGD_lable/train/mixed_lable.csv',
                               data_dir='BGD_image/mixed_data',
                               transform=transform)



        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


        # 训练过程
        num_epochs = 80
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
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')
            acc = validator(val_dataloader, model, device)
            if max_acc <= acc and epoch > 4:
                max_acc = acc
                torch.save(model.state_dict(), 'run/detail_net_atten/exp3_4().pt')




if __name__ == '__main__':
    mode('train')
