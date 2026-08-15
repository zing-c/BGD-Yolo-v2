import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def fourier_transform_and_save(input_dir, output_dir, save_magnitude=True, save_phase=True, visualize=False):
    """
    对输入文件夹中的所有图像进行傅里叶变换并保存结果

    参数:
        input_dir: 输入图像文件夹路径
        output_dir: 输出结果文件夹路径
        save_magnitude: 是否保存幅度谱
        save_phase: 是否保存相位谱
        visualize: 是否显示转换示例（显示第一张图的处理结果）
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    magnitude_dir = os.path.join(output_dir, 'magnitude')
    phase_dir = os.path.join(output_dir, 'phase')

    if save_magnitude:
        os.makedirs(magnitude_dir, exist_ok=True)
    if save_phase:
        os.makedirs(phase_dir, exist_ok=True)

    # 获取所有图像文件
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not img_files:
        print(f"警告: 输入文件夹 {input_dir} 中没有找到图像文件!")
        return

    # 处理第一张图的示例可视化
    example_shown = False

    for img_file in tqdm(img_files, desc="Processing images"):
        # 读取图像并转为灰度
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"警告: 无法读取图像 {img_file}, 跳过")
            continue

        # 傅里叶变换
        dft = np.fft.fft2(img)
        dft_shift = np.fft.fftshift(dft)  # 将低频移到中心

        # 计算幅度谱和相位谱
        magnitude = np.log1p(np.abs(dft_shift))  # log(1 + |F(u,v)|)
        phase = np.angle(dft_shift)

        # 归一化到0-255
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        phase_norm = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 保存结果
        base_name = os.path.splitext(img_file)[0]

        if save_magnitude:
            cv2.imwrite(os.path.join(magnitude_dir, f'{base_name}.jpg'), magnitude_norm)

        if save_phase:
            cv2.imwrite(os.path.join(phase_dir,  f'{base_name}.jpg'), phase_norm)

        # 示例可视化（仅显示第一张图）
        if visualize and not example_shown:
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.imshow(img, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(magnitude, cmap='jet')
            plt.title('Magnitude Spectrum (log scale)')
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(phase, cmap='hsv')
            plt.title('Phase Spectrum')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            example_shown = True

    print(f"\n处理完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    # 使用示例
    input_folder = "/home/bme-2020/czy/BGD-Yolo/BGD_image/mixed_data"  # 替换为你的输入文件夹路径
    output_folder ="/home/bme-2020/czy/fornier/mixed_data"  # 替换为输出文件夹路径

    # 调用函数（参数说明见函数文档）
    fourier_transform_and_save(
        input_dir=input_folder,
        output_dir=output_folder,
        save_magnitude=True,
        save_phase=True,
        visualize=True  # 显示第一张图的处理示例
    )