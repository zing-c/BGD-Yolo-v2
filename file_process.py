import os
import random
import re
from shutil import move


def create_numbered_txt_files(folder_path, start_num, end_num):
    """
    在指定文件夹中创建从start_num到end_num的编号txt文件

    参数:
        folder_path: 目标文件夹路径
        start_num: 起始编号(包含)
        end_num: 结束编号(包含)
    """
    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)

    # 验证数字范围
    if start_num > end_num:
        print("错误：起始编号不能大于结束编号")
        return

    if start_num < 0 or end_num < 0:
        print("错误：编号不能为负数")
        return

    # 创建文件
    created_count = 0
    for num in range(start_num, end_num + 1):
        filename = f"{num:05d}.txt"  # 05d格式
        filepath = os.path.join(folder_path, filename)

        # 创建空文件
        with open(filepath, 'w') as f:
            pass  # 创建空文件

        created_count += 1

    print(f"成功创建 {created_count} 个文件，从 {start_num:05d}.txt 到 {end_num:05d}.txt")

def shuffle_and_rename_images(target_folder, start_num=None):
    # 获取目标文件夹中所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    target_files = [f for f in os.listdir(target_folder)
                    if f.lower().endswith(image_extensions)]

    if not target_files:
        print("目标文件夹中没有图片文件")
        return

    # 如果没有提供起始编号，则询问用户
    if start_num is None:
        while True:
            try:
                start_num = int(input("请输入起始编号(例如输入23会生成00023.jpg): "))
                if start_num < 0:
                    print("编号不能为负数，请重新输入")
                    continue
                break
            except ValueError:
                print("请输入有效的数字")

    # 打乱目标文件夹中的文件顺序
    random.shuffle(target_files)

    # 创建临时文件夹用于安全重命名
    temp_dir = os.path.join(target_folder, "temp_rename")
    os.makedirs(temp_dir, exist_ok=True)

    # 先把所有文件移动到临时文件夹
    for file in target_files:
        src = os.path.join(target_folder, file)
        dst = os.path.join(temp_dir, file)
        move(src, dst)

    # 从临时文件夹移回并重命名
    target_files = os.listdir(temp_dir)  # 获取打乱后的文件列表
    for i, file in enumerate(target_files, start=1):
        ext = os.path.splitext(file)[1].lower()
        new_name = f"{start_num + i - 1:05d}{ext}"  # -1因为i从1开始
        src = os.path.join(temp_dir, file)
        dst = os.path.join(target_folder, new_name)
        move(src, dst)

    # 删除临时文件夹
    os.rmdir(temp_dir)
    print(f"已完成重命名，共处理了{len(target_files)}个文件")
    print(f"命名范围: {start_num:05d} 到 {start_num + len(target_files) - 1:05d}")


if __name__ == "__main__":
    #
    # #refactor the name of filr
    # target_path = input("请输入要重命名的目标文件夹路径: ")
    #
    # if os.path.isdir(target_path):
    #     # 询问用户是否要从参考文件夹获取起始编号
    #     use_reference = input("是否要从参考文件夹获取起始编号? (y/n): ").lower() == 'y'
    #
    #     if use_reference:
    #         reference_path = input("请输入参考文件夹路径: ")
    #         if os.path.isdir(reference_path):
    #             # 获取参考文件夹中所有图片文件
    #             image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    #             reference_files = [f for f in os.listdir(reference_path)
    #                                if f.lower().endswith(image_extensions)]
    #
    #             if reference_files:
    #                 # 找出参考文件夹中数字最大的文件
    #                 reference_files.sort(key=lambda x: [int(c) if c.isdigit() else c
    #                                                     for c in re.split('([0-9]+)', x)])
    #                 last_file = reference_files[-1]
    #                 match = re.search(r'(\d+)', last_file)
    #                 start_num = int(match.group(1)) + 1 if match else 0
    #                 print(f"自动获取的起始编号: {start_num:05d}")
    #             else:
    #                 print("参考文件夹中没有图片文件，将手动输入起始编号")
    #                 start_num = None
    #         else:
    #             print("参考文件夹路径无效，将手动输入起始编号")
    #             start_num = None
    #     else:
    #         start_num = None
    #
    #     shuffle_and_rename_images(target_path, start_num)
    # else:
    #     print("目标文件夹路径无效")

    # 用户输入
    folder = input("请输入要创建文件的文件夹路径: ")
    start = int(input("请输入起始编号: "))
    end = int(input("请输入结束编号: "))

    create_numbered_txt_files(folder, start,end)