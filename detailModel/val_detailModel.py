from inceptionBlock import Inception_atten,Inception_atten_conv
from inceptionBlock import InceptionResidual_atten

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from Resnet_Train import GlassDataset
from Resnet6 import Detail_Net_attn,Detail_Net_attn_1

def val_model(weight_path, model=Inception_atten):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = model().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载验证集
    val_dataset = GlassDataset(csv_file='../BGD_lable/val/lable.csv',
                               data_dir='../BGD_image/val',
                               transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 开始评估
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images.to(device))
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 指标计算
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f'\n[Evaluation]')
    print(f'Accuracy : {acc * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1 Score : {f1:.4f}')

if __name__ == '__main__':
    val_model(weight_path='../run/detail_net_atten/exp2.pt',model=Detail_Net_attn)
