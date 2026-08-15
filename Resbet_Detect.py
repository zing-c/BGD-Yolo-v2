import torch
from torchvision import transforms
from PIL import Image
import os
from Resnet6 import ResNet6
from PIL import ImageDraw

# 加载模型
def load_model(model_path):
    """
    加载训练好的模型
    :param model_path: 模型权重文件路径
    :return: 加载的模型
    """
    model = ResNet6()  # 实例化模型
    model.load_state_dict(torch.load(model_path))  # 加载权重
    model.eval()  # 设置为评估模式
    return model

# 数据预处理
def preprocess_image(image_path):
    """
    对输入图片进行预处理
    :param image_path: 图片路径
    :return: 预处理后的图片张量
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图片尺寸
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    image = Image.open(image_path).convert('RGB')  # 读取图片并转换为RGB格式
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image

# 预测单张图片
def predict_image(model, image_path):
    """
    对单张图片进行预测
    :param model: 加载的模型
    :param image_path: 图片路径
    :return: 预测结果（0 或 1）和概率值
    """
    image = preprocess_image(image_path)  # 预处理图片
    with torch.no_grad():  # 禁用梯度计算
        output = model(image)  # 前向传播
        probability = output.item()  # 获取概率值
        prediction = 1 if probability > 0.5 else 0  # 根据阈值分类
    return prediction, probability

# 批量预测
def predict_folder(model, image_folder):
    """
    对文件夹中的所有图片进行预测
    :param model: 加载的模型
    :param image_folder: 图片文件夹路径
    """
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):  # 过滤图片文件
            image_path = os.path.join(image_folder, filename)
            prediction, probability = predict_image(model, image_path)
            print(f'Image: {filename}, Prediction: {prediction}, Probability: {probability:.4f}')

def visualize_prediction(image_path, prediction, probability):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    text = f'Prediction: {prediction}, Probability: {probability:.4f}'
    draw.text((10, 10), text, fill="red")  # 在图片上绘制文本
    image.show()  # 显示图片


if __name__ == '__main__':
    # 加载模型
    model_path = 'run/detail_net_attn.pt'  # 替换为你的模型路径
    model = load_model(model_path)

    # 预测单张图片
    image_path = 'BGD_image/train/000345.jpg'  # 替换为你的图片路径
    prediction, probability = predict_image(model, image_path)

    print(f'Image: {image_path}, Prediction: {prediction}, Probability: {probability:.4f}')
    visualize_prediction(image_path,prediction, probability)
    # # 预测文件夹中的所有图片
    # image_folder = 'test_images'  # 替换为你的图片文件夹路径
    # predict_folder(model, image_folder)