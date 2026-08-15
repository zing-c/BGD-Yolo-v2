import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
import csv
import tqdm
np.random.seed(0)
import matplotlib.pyplot as plt
from ultralytics import nn
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.yolo.utils.torch_utils import intersect_dicts
from ultralytics.yolo.utils.ops import xywh2xyxy, non_max_suppression
try:
    from pytorch_grad_cam import (
        EigenCAM,
        EigenGradCAM,
        GradCAM,
        GradCAMPlusPlus,
        HiResCAM,
        LayerCAM,
        RandomCAM,
        XGradCAM,
    )
    from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
    from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
except ModuleNotFoundError:  # optional dependency (only needed for CAM visualization)
    GradCAMPlusPlus = GradCAM = XGradCAM = EigenCAM = HiResCAM = LayerCAM = RandomCAM = EigenGradCAM = None
    show_cam_on_image = scale_cam_image = None
    ActivationsAndGradients = None
from torch.utils.data import DataLoader, Dataset
from Resnet6 import ResNet6

class ImageDataset(Dataset):
    """自定义数据集类，用于加载图像"""
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.img_files[idx])
        img = cv2.imread(img_path)

        img, r, (dw, dh) = letterbox(img)
        info = (dw, dh, r)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).float() / 255.0
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.float32(img)

        return img_tensor, self.img_files[idx],info


def lable_annotationp(image_name, image_shape, txt_folder, new_image_name, center, output_csv):
    """
    检查截取区域与标注框的交集

    :param image_name: 原始图像文件名（如 "000001.jpg"）
    :param image_shape: 原始图像形状 (height, width, channels)
    :param txt_folder: 存放标注txt文件的文件夹路径
    :param new_image_name: 新生成的裁剪图片文件名
    :param center: 截取区域中心坐标
    :param output_csv: 输出结果CSV文件路径
    """
    # 获取图像宽高
    height, width = image_shape[:2]

    # 构建对应txt文件路径
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    txt_path = os.path.join(txt_folder, f"{base_name}.txt")

    has_overlap_flag = 0

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析标注行（格式：class x_center y_center width height）
                parts = line.split()
                if len(parts) != 5:
                    continue

                try:
                    # 转换为归一化浮点数
                    _, x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts)
                except:
                    continue

                # 转换为绝对坐标
                x_center = x_center_norm * width
                y_center = y_center_norm * height
                w = w_norm * width
                h = h_norm * height

                # 计算实际框坐标
                x1_box = x_center - w/2
                y1_box = y_center - h/2
                x2_box = x_center + w/2
                y2_box = y_center + h/2

                # 检查与截取区域是否相交
                if is_overlap(center, (x1_box, y1_box, x2_box, y2_box)):
                    has_overlap_flag = 1
                    break  # 发现相交立即终止循环
                else:
                    has_overlap_flag=0

    # 写入结果到CSV
    write_to_csv(output_csv, new_image_name, has_overlap_flag)

def is_overlap(center, rect_b):
    cx, cy = center
    """
    检查两个矩形是否有交集
    :param rect_a: (x1, y1, x2, y2)
    :param rect_b: (x1, y1, x2, y2)
    :return: bool
    """
    return rect_b[0] < cx < rect_b[2] and rect_b[1] < cy < rect_b[3]

def write_to_csv(output_csv, new_image_name, has_overlap):
    """
    将结果写入CSV文件
    """
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['image_name', 'has_overlap'])
        writer.writerow([new_image_name, has_overlap])
def new_datasets(image_path, center, crop_size=(64, 64),save_path=None, output_csv=None,txt_folder=None):
    """
    基于中心点截取图像的一部分。

    参数:
        image_path (str): 图像路径。
        center (tuple): 中心点坐标 (cx, cy)。
        crop_size (tuple): 截取区域的大小 (width, height)。

    返回:
        cropped_image (np.ndarray): 截取的图像区域。
    """
    # 加载图像
    image = cv2.imread(image_path)
    image_name=os.path.basename(image_path)
    image_shape = image.shape


    if image is None:
        raise ValueError("图像加载失败，请检查路径是否正确。")

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算截取区域的左上角和右下角坐标
    cx, cy = center
    crop_width, crop_height = crop_size
    x1 = max(0, int(cx - crop_width // 2))
    y1 = max(0, int(cy - crop_height // 2))
    x2 = min(width, int(cx + crop_width // 2))
    y2 = min(height, int(cy + crop_height // 2))

    crop_rect = (x1, y1, x2, y2)  # 保存截取区域的坐标

    # 截取图像
    cropped_image = image[y1:y2, x1:x2]

    # 如果截取区域小于目标尺寸，填充到目标尺寸
    if cropped_image.shape[0] < crop_height or cropped_image.shape[1] < crop_width:
        pad_height = max(0, crop_height - cropped_image.shape[0])
        pad_width = max(0, crop_width - cropped_image.shape[1])
        cropped_image = cv2.copyMakeBorder(
            cropped_image,
            pad_height // 2, pad_height - pad_height // 2,
            pad_width // 2, pad_width - pad_width // 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)  # 用黑色填充
        )


    # 如果指定了保存路径，则保存截取的图像
    if save_path is not None:
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)

        # 获取当前文件夹中的文件数量，用于生成新的文件名
        existing_files = os.listdir(save_path)
        existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.endswith('.jpg')]
        next_number = max(existing_numbers) + 1 if existing_numbers else 0

        # 生成文件名并保存
        save_filename = f"{next_number:06d}.jpg"  # 使用 6 位数字格式

        lable_annotationp(image_name,image_shape,txt_folder,save_filename,center,output_csv)

        save_filepath = os.path.join(save_path, save_filename)
        cv2.imwrite(save_filepath, cropped_image)




def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
       调整图像尺寸并填充，同时计算填充信息。

       参数:
           im (np.ndarray): 输入图像，形状为 [H, W, C]。
           new_shape (tuple): 目标尺寸，默认为 (640, 640)。
           color (tuple): 填充颜色，默认为灰色 (114, 114, 114)。
           auto (bool): 是否自动计算填充，默认为 True。
           scaleFill (bool): 是否拉伸图像，默认为 False。
           scaleup (bool): 是否允许放大图像，默认为 True。
           stride (int): 填充的步长，默认为 32。

       返回:
           im (np.ndarray): 调整后的图像，形状为 [new_shape[0], new_shape[1], C]。
           ratio (tuple): 缩放比例 (width_ratio, height_ratio)。
           (dw, dh) (tuple): 填充的宽度和高度。
       """


    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # 强制调整图像尺寸为 new_shape
    im_resized = cv2.resize(im, new_shape, interpolation=cv2.INTER_LINEAR)

    # 重新计算 dw 和 dh
    # 由于强制调整尺寸为 new_shape，填充的宽度和高度需要重新计算
    # 新的 dw 和 dh 是强制调整后的填充信息
    new_dw = (new_shape[1] - new_unpad[0]) / 2
    new_dh = (new_shape[0] - new_unpad[1]) / 2

    return im, ratio, (dw, dh)

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


def convert_to_original_coords(x_padded, y_padded, r, dw, dh):
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    x_original = (x_padded - left) / r
    y_original = (y_padded - top) / r
    return (x_original, y_original)


def find_max_heatmap_center_intergralImage(heatmap, boxes, window_size=64):
    centers = []
    c2b = []

    # 预计算整个热力图的积分图
    integral = cv2.integral(heatmap)[1:, 1:]  # OpenCV的积分图会多一行一列

    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < window_size or box_height < window_size:
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if center not in centers:
                centers.append(center)
                c2b.append([box_idx])
            else:
                c2b[centers.index(center)].append(box_idx)
            continue

        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, integral.shape[1]), min(y2, integral.shape[0])

        max_value = -1
        max_center = (0, 0)

        # 计算box区域的积分图
        box_integral = integral[y1:y2, x1:x2]
        # 填充边界，方便计算
        box_integral = np.pad(box_integral, ((1, 0), (1, 0)), mode='constant')

        for i in range(1, box_integral.shape[0] - window_size + 1):
            for j in range(1, box_integral.shape[1] - window_size + 1):
                # 使用积分图快速计算窗口和
                total = (box_integral[i + window_size - 1, j + window_size - 1]
                         - box_integral[i - 1, j + window_size - 1]
                         - box_integral[i + window_size - 1, j - 1]
                         + box_integral[i - 1, j - 1])

                if total > max_value:
                    max_value = total
                    max_center = (x1 + j - 1 + window_size // 2,
                                  y1 + i - 1 + window_size // 2)

        if max_center not in centers:
            centers.append(max_center)
            c2b.append([box_idx])
        else:
            c2b[centers.index(max_center)].append(box_idx)

    return centers, c2b



def find_max_heatmap_center_torch_optimized(heatmap, boxes, window_size=64):
    device = boxes.device
    heatmap_tensor = torch.from_numpy(heatmap).float().to(device)
    H, W = heatmap_tensor.shape

    # 预计算卷积结果（使用积分图优化）
    if window_size > 1:
        # 创建积分图
        integral = torch.cumsum(torch.cumsum(heatmap_tensor, dim=0), dim=1)
        integral = torch.nn.functional.pad(integral, (1, 0, 1, 0))  # 左上padding

        # 计算每个窗口的和
        conv_result = (integral[window_size:, window_size:] -
                       integral[window_size:, :-window_size] -
                       integral[:-window_size, window_size:] +
                       integral[:-window_size, :-window_size])
    else:
        conv_result = heatmap_tensor

    boxes = boxes.clone().detach().to(torch.long)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H)

    # 计算有效区域（向量化）
    valid_width = (boxes[:, 2] - boxes[:, 0]) >= window_size
    valid_height = (boxes[:, 3] - boxes[:, 1]) >= window_size
    valid_mask = valid_width & valid_height

    # 初始化中心坐标
    centers = torch.zeros((len(boxes), 2), dtype=torch.long).to(device)

    # 处理无效框（向量化）
    invalid_boxes = boxes[~valid_mask]
    if len(invalid_boxes) > 0:
        centers[~valid_mask] = torch.stack([
            (invalid_boxes[:, 0] + invalid_boxes[:, 2]) // 2,
            (invalid_boxes[:, 1] + invalid_boxes[:, 3]) // 2
        ], dim=1)

    # 批量处理有效框（完全向量化）
    valid_boxes = boxes[valid_mask]
    if len(valid_boxes) > 0:
        conv_coords = valid_boxes.clone()
        conv_coords[:, 2] = (conv_coords[:, 2] - window_size + 1).clamp(min=0)
        conv_coords[:, 3] = (conv_coords[:, 3] - window_size + 1).clamp(min=0)

        # 改为逐个处理，但保持高效
        max_coords = torch.zeros((len(valid_boxes), 2), dtype=torch.long).to(device)
        for i, (x1, y1, x2, y2) in enumerate(conv_coords):
            if x2 > x1 and y2 > y1:
                region = conv_result[y1:y2, x1:x2]
                if region.numel() > 0:
                    idx = torch.argmax(region)
                    max_y, max_x = np.unravel_index(idx.item(), region.shape)
                    max_coords[i] = torch.tensor([max_x + x1, max_y + y1])
                else:
                    # 回退到中心
                    max_coords[i] = torch.tensor([
                        (valid_boxes[i, 0] + valid_boxes[i, 2]) // 2,
                        (valid_boxes[i, 1] + valid_boxes[i, 3]) // 2
                    ])
            else:
                # 区域太小，回退到中心
                max_coords[i] = torch.tensor([
                    (valid_boxes[i, 0] + valid_boxes[i, 2]) // 2,
                    (valid_boxes[i, 1] + valid_boxes[i, 3]) // 2
                ])

        centers[valid_mask] = max_coords + window_size // 2

        # 合并重复中心点
    unique_centers, inverse_indices = torch.unique(centers, dim=0, return_inverse=True)
    c2b = [[] for _ in range(len(unique_centers))]
    for idx, center_idx in enumerate(inverse_indices):
        c2b[center_idx].append(idx)

    return unique_centers.tolist(), c2b
def find_max_heatmap_center_torch(heatmap, boxes, window_size=64):
    B, H, W = heatmap.shape
    device = boxes[0].device
    heatmap = torch.from_numpy(heatmap).float().to(device)
    centers = []
    c2b_all = []

    for b in range(B):
        box_b = boxes[b].clone().detach().to(torch.long)  # (N_i, 4)

        heatmap_b = heatmap[b].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        ws = int(window_size[b].item())

        # Create convolution layer
        conv = torch.nn.Conv2d(1, 1, kernel_size=ws, stride=1, padding=0, bias=False)
        conv.weight.data.fill_(1.0)
        conv = conv.to(device)
        conv.requires_grad_(False)

        with torch.no_grad():
            conv_result = conv(heatmap_b).squeeze(0).squeeze(0)  # (H-w+1, W-w+1)

        box_b[:, [0, 2]] = box_b[:, [0, 2]].clamp(0, W)
        box_b[:, [1, 3]] = box_b[:, [1, 3]].clamp(0, H)
        # Problem

        valid_w = (box_b[:, 2] - box_b[:, 0]) >= ws
        valid_h = (box_b[:, 3] - box_b[:, 1]) >= ws
        valid = valid_w & valid_h

        center_b = torch.zeros((len(box_b), 2), dtype=torch.long, device=device)
        # if len(box_b)==0: continue
        # 无效框直接中心
        invalid_boxes = box_b[~valid]
        center_b[~valid] = torch.stack([
            (invalid_boxes[:, 0] + invalid_boxes[:, 2]) // 2,
            (invalid_boxes[:, 1] + invalid_boxes[:, 3]) // 2
        ], dim=1)

        # 有效框通过卷积最大响应计算
        valid_boxes = box_b[valid]
        if len(valid_boxes) > 0:
            conv_coords = valid_boxes.clone()
            conv_coords[:, 2] = (conv_coords[:, 2] - ws//2 + 1).clamp(min=0)
            conv_coords[:, 3] = (conv_coords[:, 3] - ws//2 + 1).clamp(min=0)

            max_coords = []
            for box in conv_coords:
                x1, y1, x2, y2 = box
                region = conv_result[y1:y2, x1:x2]
                if region.numel() == 0:
                    max_coords.append([-1, -1])
                    continue
                idx = torch.argmax(region)
                max_y, max_x = np.unravel_index(idx.item(), region.shape)
                max_coords.append([max_x + x1, max_y + y1])


            max_coords = torch.tensor(max_coords, device=device)

            fallback = (max_coords == -1).any(dim=1)
            # 有效框在 center_b 中的行号（避免链式布尔索引）
            idx_valid = valid.nonzero(as_tuple=False).squeeze(1)

            fallback_boxes = valid_boxes[fallback]

            centers_fallback = torch.stack([
                (fallback_boxes[:, 0] + fallback_boxes[:, 2]) // 2,
                (fallback_boxes[:, 1] + fallback_boxes[:, 3]) // 2
            ], dim=1)

            # 非回退的：卷积最大响应左上角 + ws//2 → 原图中心
            centers_from_conv = max_coords + (ws // 2)

            # 组装为与 valid_boxes 对齐的结果，然后一次性写回
            centers_new = torch.empty((valid_boxes.size(0), 2),
                                      dtype=center_b.dtype, device=device)
            centers_new[fallback] = centers_fallback.to(center_b.dtype)
            centers_new[~fallback] = centers_from_conv[~fallback].to(center_b.dtype)

            center_b[idx_valid] = centers_new

        centers.append(center_b)

        # 去重 + 反向索引
        uc, inv = torch.unique(center_b, dim=0, return_inverse=True)
        c2b = [[] for _ in range(len(uc))]
        for idx, cid in enumerate(inv):
            c2b[cid].append(idx)
        c2b_all.append(c2b)

    return centers, c2b_all
    # # 转换为PyTorch张量
    # device = heatmap.device
    # heatmap_tensor = torch.from_numpy(heatmap).float().to(device)
    # centers = []
    # c2b_all = []
    # B, H, W = heatmap_tensor.shape
    #
    # for b in range(B):
    # # 预计算卷积结果
    # # conv_layer = torch.nn.Conv2d(1, 1, kernel_size=window_size, stride=1, padding=0, bias=False)
    # # conv_layer.weight.data = torch.ones(1, 1, window_size, window_size).to(device)
    # # conv_layer.requires_grad_(False)
    # #
    # # with torch.no_grad():
    # #     conv_result = conv_layer(heatmap_tensor.view(1, 1, H, W)).squeeze()  # (H_conv, W_conv)
    # conv_result = []
    # for b, b_ws in enumerate(window_size.tolist()):
    #     b_ws = int(b_ws)
    #     conv_layer = torch.nn.Conv2d(1, 1, kernel_size=b_ws, stride=1, padding=0, bias=False)
    #     conv_layer.weight.data = torch.ones(1, 1, b_ws, b_ws).to(device)
    #     conv_layer.requires_grad_(False)
    #     with torch.no_grad():
    #         conv_result.append(conv_layer(heatmap_tensor[b].view(1, 1, H, W)).squeeze())  # (H_conv, W_conv)
    #
    # # 处理所有检测框
    # boxes = [box.clone().detach().to(torch.long) for box in boxes]
    # boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W)
    # boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H)
    #
    # # 计算有效区域
    # valid_width = (boxes[:, 2] - boxes[:, 0]) >= window_size
    # valid_height = (boxes[:, 3] - boxes[:, 1]) >= window_size
    # valid_mask = valid_width & valid_height
    #
    # # 初始化中心坐标
    # centers = torch.zeros((len(boxes), 2), dtype=torch.long).to(device)
    #
    # # 处理无效框（直接取中心）
    # invalid_boxes = boxes[~valid_mask]
    # centers[~valid_mask] = torch.stack([
    #     (invalid_boxes[:, 0] + invalid_boxes[:, 2]) // 2,
    #     (invalid_boxes[:, 1] + invalid_boxes[:, 3]) // 2
    # ], dim=1)
    #
    # # 批量处理有效框
    # valid_boxes = boxes[valid_mask]
    # if len(valid_boxes) > 0:
    #     # 计算卷积结果中的有效区域
    #     conv_coords = valid_boxes.clone()
    #     conv_coords[:, 2] = (conv_coords[:, 2] - window_size + 1).clamp(min=0)
    #     conv_coords[:, 3] = (conv_coords[:, 3] - window_size + 1).clamp(min=0)
    #
    #     # 批量提取区域并找到最大值
    #     max_coords = []
    #     for box in conv_coords:
    #         x1, y1, x2, y2 = box
    #         region = conv_result[y1:y2, x1:x2]
    #         if region.numel() == 0:
    #             max_coords.append((-1, -1))
    #             continue
    #         idx = torch.argmax(region)
    #         max_y, max_x = np.unravel_index(idx.item(), region.shape)
    #         max_coords.append((max_x + x1, max_y + y1))
    #
    #     # 转换到原图坐标
    #     max_coords = torch.tensor(max_coords).to(device)
    #     centers[valid_mask] = max_coords * 1 + window_size // 2  # 计算中心点
    #
    #     # 处理无效区域（当卷积结果为空时回退到中心）
    #     fallback = (max_coords == -1).any(dim=1)
    #     fallback_boxes = valid_boxes[fallback]
    #     centers[valid_mask][fallback] = torch.stack([
    #         (fallback_boxes[:, 0] + fallback_boxes[:, 2]) // 2,
    #         (fallback_boxes[:, 1] + fallback_boxes[:, 3]) // 2
    #     ], dim=1)
    #
    # # 合并重复中心点
    # unique_centers, inverse_indices = torch.unique(centers, dim=0, return_inverse=True)
    # c2b = [[] for _ in range(len(unique_centers))]
    # for idx, center_idx in enumerate(inverse_indices):
    #     c2b[center_idx].append(idx)

    # return unique_centers.tolist(), c2b


def find_max_heatmap_center(heatmap, boxes, window_size=64):
    """
    在每个检测框内找到一个 windousize的窗口，其热力值最大，并返回窗口的中心点。

    参数:
        heatmap (np.ndarray): 热力图，形状为 (H, W)。
        boxes (np.ndarray): 检测框坐标，形状为 (N, 4)，格式为 (x1, y1, x2, y2)。
        window_size (int): 滑动窗口的大小，默认为 64。

    返回:
        centers (list): 每个检测框内最大热力值窗口的中心点坐标，格式为 [(x, y), ...]。
    """
    centers = []
    c2b = []
    for box_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # 检测框坐标
        box_width = x2 - x1
        box_height = y2 - y1

        # 如果检测框小于窗口大小，直接取检测框的中心
        if box_width < window_size or box_height < window_size:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            max_center = (center_x, center_y)
            if max_center not in centers:
                centers.append(max_center)
                c2b.append([box_idx])
            else:
                c2b[centers.index(max_center)].append(box_idx)
            continue

        if x1<0 :
            x1=0
        if y1<0:
            y1=0

        # 提取检测框内的热力图区域
        box_heatmap = heatmap[y1:y2, x1:x2]

        max_value = -1
        max_center = (0, 0)

        # 滑动窗口，找到热力值最大的窗口
        for i in range(0, box_height - window_size + 1):
            for j in range(0, box_width - window_size + 1):
                window = box_heatmap[i:i + window_size, j:j + window_size]
                window_sum = np.sum(window)

                if window_sum > max_value:
                    max_value = window_sum
                    max_center = (x1 + j + window_size // 2, y1 + i + window_size // 2)
        if max_center not in centers:
            centers.append(max_center)
            c2b.append([box_idx])
        else:
            c2b[centers.index(max_center)].append(box_idx)
    return centers, c2b



class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    def post_process(self, result):

        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
    def __init__(self, output_type, conf, ratio) -> None:
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes, post_boxes = self.post_process(data)
        result = []
        for i in range(int(post_result.size(0) )):
            if float(post_result[i].max()) < self.conf:
                break
            if self.output_type == 'class' or self.output_type == 'all':
                result.append(post_result[i].max())
            elif self.output_type == 'box' or self.output_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class yolov8_target_batch(torch.nn.Module):
    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        return torch.transpose(logits_, 1,2)  ,torch.transpose(boxes_, 1, 2) ,xywh2xyxy(boxes_.cpu().detach().numpy())
    def __init__(self) -> None:
        super().__init__()
        # self.output_type = output_type
        # self.conf = conf
        # self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes, post_boxes = self.post_process(data)
        results = []
        B, N, _ = pre_post_boxes.shape
        for b in range(B):  # 遍历每张图
            val = sum(post_result[b])  # Tensor
            results.append(val)
        return torch.stack(results)



class yolov8_heatmap:
    def __init__(self, yolo_weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, task='train',renormalize=True,):
        if GradCAM is None:
            raise ModuleNotFoundError(
                "Missing optional dependency 'pytorch_grad_cam'. Install it to use yolov8_heatmap/GradCAM features."
            )
        device = torch.device(device)
        ckpt = torch.load(yolo_weight)
        yolo_model_names = ckpt['model'].names
        yolo_model = attempt_load_weights(yolo_weight, device)
        yolo_model.info()
        for p in yolo_model.parameters():
            p.requires_grad_(True)
        yolo_model.eval()
        self.task=task
        self.conf_threshold = conf_threshold

        # detail_model = ResNet6()
        # detail_model.load_state_dict(torch.load(self.detail_weight))
        # detail_model.eval()

        target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [yolo_model.model[l] for l in layer]
        method = eval(method)(yolo_model, target_layers)
        # method.activations_and_grads = ActivationsAndGradients(yolo_model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(yolo_model_names), 3)).astype(np.uint8)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img


    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, batch, save_path,img_path):
        """批量处理图像"""
        images, filenames, infos = batch  # 解包 batch，获取 images, filenames, infos
        images = images.to(self.device)

        try:
            grayscale_cams = self.method(images, [self.target])
            preds = self.method.outputs[0]
            preds = self.post_process(preds)
        except AttributeError as e:
            return
        # # 模型推理
        # with torch.no_grad():
        #     grayscale_cams = self.method(images, [self.target])
        #     preds = self.yolo_model(images)[0]

        # 处理每张图像
        pred_boxes = []
        pred_scores = []

        for i in range(len(images)):
            grayscale_cam = grayscale_cams[i, :]
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            pred = preds[i]
            boxes = pred[:,:4].cpu().detach().numpy().astype(np.int32)

            # 获取当前图像的 info
            dw, dh, r = infos[0][i].item(), infos[1][i].item(), infos[2][0][i].item()
            # 在每个检测框内找到最大热力值窗口的中心点
            centers,_ = find_max_heatmap_center(grayscale_cam, boxes)


            if self.task=='train':
                # 将中心点坐标转换回原始图像坐标系
                for center in centers:
                    x, y = center
                    center_original = convert_to_original_coords(x, y, r, dw, dh)
                    # 建立新的数据集
                    # new_datasets(
                    #     image_path=os.path.join(img_path, filenames[i]),
                    #     center=center_original,
                    #     save_path='BGD_image/demo',
                    #     output_csv='BGD_lable/val/demo.csv',
                    #     txt_folder='/home/bme-2020/czy/mydata/labels/val'
                    # )

                # 如果需要重新归一化热力图
                if self.renormalize:
                    cam_image = self.renormalize_cam_in_bounding_boxes(boxes, img, grayscale_cam)

                # 如果需要显示检测框
                if self.show_box:
                    for data in pred:
                        data = data.cpu().detach().numpy()
                        cam_image = self.draw_detections(
                            data[:4],
                            self.colors[int(data[4:].argmax())],
                            f'{self.yolo_model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
                            cam_image
                        )

                # 在图像上绘制中心点
                for center in centers:
                    x, y = center
                    cv2.circle(cam_image, (x, y), radius=5, color=(0, 0, 0), thickness=-1)  # 绘制绿色圆点

                # 保存结果
                save_filename = os.path.join(save_path,
                                             f"{os.path.splitext(filenames[i])[0]}.jpg")
                cam_image = Image.fromarray(cam_image)
                cam_image.save(save_filename)

            if self.task=='val':
                # 将中心点坐标转换回原始图像坐标系
                for center in centers:
                    x, y = center
                    center_original = convert_to_original_coords(x, y, r, dw, dh)
                    cropped_image=crop_image(img, center_original)
                    cropped_tensor = torch.from_numpy(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    detail_output = self.detail_model(cropped_tensor)
                    detail_label = detail_output.argmax(dim=1)
                    if detail_label.item()==1 :
                        for bbox in boxes:
                            x_min, y_min, x_max, y_max = bbox
                            if (x_min <= x <= x_max) and (y_min <= y <= y_max):
                                pred_boxes.append(bbox)
                                pred_scores.append()









    # def process(self, batch, save_path):
    #     # img process
    #     images, filenames,infos = batch
    #     images = images.to(self.device)
    #
    #
    #     try:
    #         grayscale_cam = self.method(tensor, [self.target])
    #     except AttributeError as e:
    #         return
    #
    #     grayscale_cam = grayscale_cam[0, :]
    #     cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    #
    #     pred = self.model(tensor)[0]
    #     pred = self.post_process(pred)
    #
    #
    #     # 获取检测框坐标
    #     boxes = pred[:, :4].cpu().detach().numpy().astype(np.int32)
    #
    #     # 在每个检测框内找到最大热力值窗口的中心点
    #     centers = find_max_heatmap_center(grayscale_cam, boxes)
    #
    #     # center_original=[]
    #     for center in centers:
    #         x, y = center
    #         center_original=convert_to_original_coords(x,y,ratio,dw,dh)
    #         new_datasets(image_path=img_path,center=center_original, save_path='BGD_image/train2', output_csv='BGD_lable/train/lable2.csv',txt_folder=r'E:\cv\czy\mydata\labels\test')
    #
    #
    #     if self.renormalize:
    #         cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
    #                                                            grayscale_cam)
    #     if self.show_box:
    #         for data in pred:
    #             data = data.cpu().detach().numpy()
    #             cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())],
    #                                              f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
    #                                              cam_image)
    #     # 在图像上绘制中心点
    #     for center in centers:
    #         x, y = center
    #         cv2.circle(cam_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # 绘制绿色圆点
    #
    #
    #
    #     cam_image = Image.fromarray(cam_image)
    #     cam_image.save(save_path)


    def __call__(self, img_path, save_path, grad_name,batch_size=8):
        #路径设置
        # output_csv = 'BGD_lable/val'
        # img_folder ='BGD_image/val'

        # remove dir if exist

        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        #     shutil.rmtree(img_folder)
           # shutil.rmtree(output_csv)




        # make dir if not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # if os.path.isdir(img_path):
        #     for img_path_ in os.listdir(img_path):
        #         name = img_path_.rsplit('.')[0]
        #         end_name = img_path_.rsplit('.')[-1]
        #         self.process(f'{img_path}/{img_path_}', f'{save_path}/{name}_{grad_name}.{end_name}')
        # else:
        #     self.process(img_path, f'{save_path}/result_{grad_name}.jpg')
        # 如果 img_path 是文件夹

        if os.path.isdir(img_path):
            # 创建数据集和数据加载器
            dataset = ImageDataset(img_path)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            # 批量处理图像
            for batch in tqdm.tqdm(dataloader, desc="Processing images"):
                # 调用 process 函数处理当前批次的图像
                self.process(batch, save_path,img_path)


        # 如果 img_path 是单张图片
        elif os.path.isfile(img_path):
            # 加载单张图像
            img = cv2.imread(img_path)

            # 获取 letterbox 处理后的信息
            img,ratio,(dw,dh)=letterbox(img)

            info = (dw, dh, ratio)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.float32(img) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).float()

            # 构建保存路径
            name, ext = os.path.splitext(os.path.basename(img_path))
            save_file_path = os.path.join(save_path, f"{name}{ext}")


            # 将单张图像包装成一个批次
            batch = (
                img_tensor.unsqueeze(0),  # 增加批次维度
                [os.path.basename(img_path)],  # 文件名列表
                [info]  # info 列表
            )

            img_path = os.path.split(img_path)[0]
            # 调用 process 函数处理单张图像
            self.process(batch, save_path,img_path)

        else:
            raise ValueError(f"无效的路径: {img_path}。请确保路径是图片文件或包含图片的文件夹。")



def get_params():
    # 绘制热力图方法列表
    grad_list = [
        'GradCAM',
         # 'GradCAMPlusPlus',
        # 'XGradCAM',
        # 'EigenCAM',
        # 'HiResCAM',
        # 'LayerCAM',
        # 'RandomCAM',
        # 'EigenGradCAM'
    ]
    # 自定义需要绘制热力图的层索引，可以用列表绘制不同层的热力图,如[10, 12, 14, 16, 18]，将多层的话会将结果进行汇总到一张图上
    layers = [21]
    for grad_name in grad_list:
        params = {

            'yolo_weight': "/home/bme-2020/czy/BGD-Yolo/runs/yolo/train10/weights/best.pt",  # 训练好的权重路径
            'device': 'cuda:0',       # cpu或者cuda:0
            'method': grad_name,
            # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
            'layer': layers,  # 计算梯度的层, 指定层的索引
            'backward_type': 'all',  # class, box, all
            'conf_threshold': 0.2,  # 置信度阈值默认0.2, 根据情况调节
            'ratio': 0.1,  # 建议0.02-0.1，取前多少数据，默认是0.02，只取置信度排序后的前百分之2的目标进行计算热力图。
            'show_box': True,  # 是否显示检测框
            'renormalize': False  # 是否优化热力图显示结果
        }
        yield params


def process_folder(model, image_folder, save_folder, method):
    """
    遍历文件夹中的所有图片并处理
    :param model: 模型对象
    :param image_folder: 图片文件夹路径
    :param save_folder: 保存结果的文件夹路径
    :param method: 方法名称
    """
    # 确保保存文件夹存在
    os.makedirs(save_folder, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(image_folder):
        # 检查文件是否为图片（支持常见图片格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            # 构建完整路径
            image_path = os.path.join(image_folder, filename)
            # 处理图片，并传递保存路径和文件名
            model(image_path, save_folder, filename, method)
# 示例使用
if __name__ == '__main__':
    for each in get_params():
        model = yolov8_heatmap(**each)
    model(img_path="/home/bme-2020/czy/mydata/images/test",
               save_path='/home/bme-2020/czy/runs/yolo-21',
               grad_name=each['method'],
               batch_size=1,
            )
