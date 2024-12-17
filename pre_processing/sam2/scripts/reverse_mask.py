from PIL import Image
import numpy as np
import os

def invert_mask(mask_path, output_path):
    """
    反转单张 mask 图像的黑白颜色。
    :param mask_path: 输入 mask 图像路径
    :param output_path: 输出反转后的 mask 图像路径
    """
    # 打开 mask 图像
    mask = Image.open(mask_path)

    # 确保 mask 是灰度图像
    mask = mask.convert("L")  # "L"模式为灰度图

    # 将 mask 转换为 numpy 数组
    mask_np = np.array(mask)

    # 反转黑白
    inverted_mask = 255 - mask_np  # 255减去每个像素值，黑变白，白变黑

    # 将反转后的 mask 转回 PIL 图像
    inverted_mask_img = Image.fromarray(inverted_mask)

    # 保存反转后的 mask 图像
    inverted_mask_img.save(output_path)

def batch_invert_masks(input_folder, output_folder):
    """
    批量反转文件夹中的所有 mask 图像。
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 检查文件是否是图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            output_path = os.path.join(output_folder, filename)
            invert_mask(input_path, output_path)
            print(f"处理完成: {input_path} -> {output_path}")

# 使用示例
input_folder = "/data1/hlp/workspace/Tracking/sam2/data/mask/1149_1"  # 输入文件夹路径
output_folder = "/data1/hlp/workspace/Tracking/sam2/data/mask/1149_1_re"  # 输出文件夹路径

batch_invert_masks(input_folder, output_folder)
