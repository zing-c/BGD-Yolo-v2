import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

from Resnet_Train import GlassDataset
# 导入你自己的模型类
from Resnet6 import Detail_Net_attn,Detail_Net_attn_dialation
from Resnet6 import Detail_Net  # 假设你想尝试加载不同的模型结构
# 导入你定义的 Dataset 类
# 如果在同一个文件，请忽略这里的 import
# from your_train_script import GlassDataset
import numpy as np
import shutil
import torch.nn.functional as F
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def plot_probability_histogram(all_probs, all_labels):
    """
    all_probs: 模型输出的概率 (0~1)
    all_labels: 真实标签 (0 或 1)
    """
    probs = all_probs.detach().cpu().numpy() if hasattr(all_probs, 'detach') else all_probs
    labels = all_labels.detach().cpu().numpy() if hasattr(all_labels, 'detach') else all_labels

    probs = probs.ravel()
    labels = labels.ravel()
    plt.figure(figsize=(10, 6))

    # 绘制正样本（有缺陷）的预测概率分布
    # plt.hist(probs[labels == 1], bins=50, alpha=0.5, label='Positive (Defect)', color='red')
    # 绘制负样本（合格玻璃）的预测概率分布
    plt.hist(probs[labels == 0], bins=50, alpha=0.5, label='Negative (Normal)', color='blue')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
    plt.title('Probability Distribution per Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def extract_error_samples(model, device, val_dataloader, save_dir='run/detail_model/error_analysis'):
    """
    提取 FP (误报) 和 FN (漏检) 的图片并分类保存
    """
    model.eval()

    # 创建保存目录
    fp_dir = os.path.join(save_dir, 'False_Positives_误报') # 正常判为破损
    fn_dir = os.path.join(save_dir, 'False_Negatives_漏检') # 破损判为正常
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    print(f"开始提取错误样本到: {save_dir} ...")

    # 注意：为了获取文件名，我们需要 val_dataset 的原始信息
    # 假设你的 GlassDataset 的图像路径存储在 self.data_info 中
    dataset = val_dataloader.dataset

    error_count = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.cpu().numpy()

            outputs = model(images)
            # 如果你的模型内部带 sigmoid，直接用 outputs
            # 如果不带，请加上 outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).cpu().numpy().astype(int).flatten()

            # 找到当前 batch 在 dataset 中的索引
            batch_start = i * val_dataloader.batch_size

            for j in range(len(preds)):
                true_label = int(labels[j])
                pred_label = int(preds[j])

                # 获取原始文件路径 (根据你的 GlassDataset 结构调整)

                img_name = dataset.data.iloc[batch_start + j, 0]
                src_path = os.path.join(dataset.data_dir, img_name)

                # 情况1: FP (误报) - 标签0，预测1
                if true_label == 0 and pred_label == 1:
                    shutil.copy(src_path, os.path.join(fp_dir, f"FP_{img_name}"))
                    error_count += 1

                # 情况2: FN (漏检) - 标签1，预测0
                elif true_label == 1 and pred_label == 0:
                    shutil.copy(src_path, os.path.join(fn_dir, f"FN_{img_name}"))
                    error_count += 1

    print(f"完成！共提取 {error_count} 张错误图片。")
    print(f"请查看目录: {save_dir}")

def find_optimal_temperature_logit_free(model, device, val_dataloader):
    """
    针对自带 Sigmoid 的模型寻找最优 T
    """
    model.eval()
    all_probs = []
    all_labels = []

    print("正在收集验证集概率输出以计算最优 T...")
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            # 此时 outputs 是经过模型内部 sigmoid 后的结果 (0-1)
            outputs = model(images)
            all_probs.append(outputs.cpu())
            all_labels.append(labels.float().unsqueeze(1).cpu())

    all_probs = torch.cat(all_probs).clamp(1e-7, 1 - 1e-7)
    all_labels = torch.cat(all_labels)

    # 反推 Logits: z = log(p / (1-p))
    logits = torch.log(all_probs / (1 - all_probs))

    def objective(t):
        # 计算校准后的损失
        loss = F.binary_cross_entropy_with_logits(logits / t[0], all_labels)
        return loss.item()

    res = minimize(objective, x0=[1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
    optimal_t = res.x[0]

    print(f"优化完成！最优温度 T: {optimal_t:.4f}")
    return optimal_t

def load_checkpoint(model, model_type, checkpoint_path, device):
    """
    支持局部加载权重的函数
    """
    if os.path.exists(checkpoint_path):
        print(f"正在加载权重文件: {checkpoint_path}")
        if model_type=='Detail_model':
            state_dict = torch.load(checkpoint_path, map_location=device)
        elif model_type=='BGD_model':
            state_dict = torch.load(checkpoint_path, map_location=device)['detail_model']

        # 这种方式可以过滤掉不匹配的 key，实现“局部加载”
        model_dict = model.state_dict()
        # 1. 过滤掉不存在于当前模型中的 key
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

        print(f"成功匹配到的参数层数: {len(pretrained_dict)} / 总层数: {len(model_dict)}")

        # 2. 更新现有的 state_dict
        model_dict.update(pretrained_dict)
        # 3. 加载
        model.load_state_dict(model_dict)
    else:
        print("未找到权重文件，请检查路径。")
    return model

def run_evaluation(model_type,checkpoint_path,sigmoid_need,use_T=False, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 实例化当前要验证的模型
    model = Detail_Net_attn()


    model = load_checkpoint(model, model_type, checkpoint_path, device)
    model.to(device)
    model.eval()

    # 3. 数据预处理 (必须与训练时一致)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 加载验证集 (指向你的测试集/验证集路径)
    val_dataset = GlassDataset(
        csv_file='BGD_lable/val/lable.csv',
        data_dir='BGD_image/val',
        transform=transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    opt_t = 1
    if use_T:
        print(">>> 模式：使用 Temperature Scaling 校准")
        opt_t = find_optimal_temperature_logit_free(model, device, val_dataloader)
    else:
        print(">>> 模式：标准推理（不使用 T）")

    # 5. 开始推理
    all_labels = []
    all_predicted_probs = []

    print("开始验证...")


    # 5. 开始推理 (修复版)
    all_labels = []
    all_predicted_probs = []

    print("开始验证...")



    with torch.no_grad():
        for images, labels in val_dataloader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1) # 保持 (B, 1)

            probs = model(images)
            if  sigmoid_need is True:
                probs = torch.sigmoid(probs)
            else:
                probs = probs

            if use_T:
                # 1. 还原 Logits
                logits = torch.log(probs.clamp(1e-7, 1-1e-7) / (1 - probs.clamp(1e-7, 1-1e-7)))
                # 2. 应用 T 缩放并重新计算 Sigmoid
                probs = torch.sigmoid(logits / opt_t)

            all_labels.extend(labels.cpu().numpy())
            all_predicted_probs.extend(probs.cpu().numpy())


    all_predicted_probs = np.array(all_predicted_probs)
    all_predicted = (all_predicted_probs > threshold).astype(float)
    all_labels = np.array(all_labels)

    extract_error_samples(model, device, val_dataloader)
    plot_probability_histogram(all_predicted_probs, all_labels)

    # 6. 计算详细指标
    accuracy = accuracy_score(all_labels, all_predicted)  # 新增
    precision = precision_score(all_labels, all_predicted)
    recall = recall_score(all_labels, all_predicted)
    f1 = f1_score(all_labels, all_predicted)
    conf_matrix = confusion_matrix(all_labels, all_predicted)

    print("\n" + "="*30)
    print("验证结果报告:")
    print("="*30)
    print(f"Accuracy  (准确率): {accuracy:.4f}")  # 新增打印
    print(f"Precision (精确率): {precision:.4f}")
    print(f"Recall    (召回率): {recall:.4f}")
    print(f"F1 Score  (F1分数): {f1:.4f}")
    print("-" * 30)

    print("混淆矩阵 (Confusion Matrix):")
    print(conf_matrix)
    print("-" * 30)
    print("分类报告:")
    print(classification_report(all_labels, all_predicted, target_names=['Class 0', 'Class 1']))

if __name__ == '__main__':
    # 确保 GlassDataset 在当前作用域可用
    run_evaluation(model_type='Detail_model',# Detail_model or BGD_model
                   checkpoint_path ='run/detail_net_atten/exp3_4_1.pt',
                   use_T=False,
                   sigmoid_need=False,
                   threshold=0.5
                   )




