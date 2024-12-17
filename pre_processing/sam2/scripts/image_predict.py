import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
import argparse

def main(args):
    # 加载模型
    checkpoint = args.checkpoint
    model_cfg = args.model_cfg
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # 创建输出文件夹
    os.makedirs(args.mask, exist_ok=True)

    # 遍历输入图像文件夹
    for image in os.listdir(args.image):
        image_path = os.path.join(args.image, image)
        image_name = os.path.splitext(image)[0]
        os.makedirs(args.mask, exist_ok=True)  
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # 加载图像
            image = Image.open(image_path)
            print(f"Processing {image_path}, Mode: {image.mode}")
            # 读取关键点
            import json
            json_file_path = os.path.join(args.annots, f'{image_name}.json')
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                annots_data = data["annots"][0]
                body_points = sorted(annots_data["keypoints"], key=lambda x: x[2], reverse=True)[:10]
                body_points = np.array(body_points)[:,:2]
            # 示例输入点
            num_points = len(body_points)
            label = np.ones(num_points)
            input_point = body_points
            input_label = label

            # 设置图像并预测
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
            )

        # 根据得分排序
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]

        # 获取最高得分的 mask
        mask = masks[0]
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # 归一化到 0-1
        mask = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask)

        # 保存 mask 到输出文件夹
        mask_path = os.path.join(args.mask, f'{image_name}.jpg')
        print(f"Saving mask to {mask_path}, Mode {mask_image.mode}")
        mask_image.save(mask_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--annots', type=str, required=True, help='输入关键点路径')
    parser.add_argument('--mask', type=str, required=True, help='输出 mask 文件夹路径')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam2.1_hiera_large.pt', help='模型权重路径')
    parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml', help='模型配置文件路径')
    args = parser.parse_args()

    main(args)
