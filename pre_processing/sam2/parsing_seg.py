from __future__ import annotations
from email.mime import image
import argparse, os, sys, pathlib, json
import numpy as np
from PIL import Image
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from pathlib import Path
from typing import List, Union, Literal
from alive_progress import alive_bar

# sapiens return npy
# classes=('Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
#                  'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
#                  'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
#                  'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
#                  'Right-shoe')

# 要分隔的class：Hat，UpperClothes，Dress，Coat，Socks，Pants，Scarf，Skirt，shoes，其他
def calculate_center(pixels):
    """计算像素区域的中心点
    
    Args:
        pixels: 输入数据:
            - 二值mask (numpy array)
    Returns:
        (center_x, center_y): 中心点坐标的元组
    """
    # 对于二值mask,使用图像矩
    y_indices, x_indices = np.nonzero(pixels)
    center_x = int(np.mean(x_indices))
    
    return center_x, center_y