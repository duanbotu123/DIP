from PIL import Image
import numpy as np

def extract_masked_area(image_path, mask_path, output_path):
    # 打开原始图像和 mask 图像
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # 确保原始图像和 mask 都是 RGB 格式
    image = image.convert("RGB")
    mask = mask.convert("L")  # 转换为灰度图像，L模式为单通道

    # 将图像和 mask 转换为 numpy 数组
    image_np = np.array(image)
    mask_np = np.array(mask)

    # 使用 mask 提取原图的区域，mask 中值大于0的部分保留，其他部分设为白色
    masked_image = np.copy(image_np)
    masked_image[mask_np == 0] = [0, 0, 0]  # 设置为白色背景

    # 将结果转换回 PIL 图像并保存
    result_img = Image.fromarray(masked_image)
    result_img.save(output_path)

# 使用示例
image_path = "/data1/hlp/workspace/Tracking/sam2/data/head.png"  # 输入图像路径
mask_path = "/data1/hlp/workspace/Tracking/sam2/data/mask_head.png"    # 输入 mask 图像路径
output_path = "/data1/hlp/workspace/Tracking/sam2/data/headwithmask.png"  # 输出图像路径

extract_masked_area(image_path, mask_path, output_path)

