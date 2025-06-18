from __future__ import annotations
from email.mime import image
from re import S
import argparse, os, sys, pathlib, json
from turtle import st, width
import numpy as np
from PIL import Image
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from pathlib import Path
from typing import List, Union, Literal
from alive_progress import alive_bar
import sam2

# sapiens return npy
# parsing classes
classes=('Background', 'Hat', 'Hair', 'Glove', 'Sunglasses',
                 'UpperClothes', 'Dress', 'Coat', 'Socks', 'Pants',
                 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
                 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe',
                 'Right-shoe')
seg_classes = ('head_mask', 'uppercloth_mask', 'dress_mask', 'coat_mask', 'socks_mask', 'pants_mask', 'skirt_mask', 'skin_mask', 'shoes_mask')
class_colors = {
    'head_mask': (255, 0, 0),       # 红色
    'uppercloth_mask': (0, 255, 0),   # 绿色
    'dress_mask': (0, 0, 255),      # 蓝色
    'coat_mask': (255, 255, 0),    # 黄色
    'socks_mask': (255, 0, 255),    # 品红色/洋红色
    'pants_mask': (0, 255, 255),    # 青色
    'skirt_mask': (255, 165, 0),    # 橙色
    'skin_mask': (128, 0, 128),    # 紫色
    'shoes_mask': (165, 42, 42)     # 棕色 (sienna)
}
head_zone = [1, 2, 4, 10, 13] # hat, hair, sunglasses, jumpsuits, face
uppercloth_zone = [5]
dress_zone = [6]
coat_zone = [7]
socks_zone = [8]
pants_zone = [9]
skirt_zone = [12]
skin_zone = [14,15,16,17] # left-arm, right-arm, left-leg, right-leg
shoes_zone = [18,19] # left-shoe, right-shoe
# this code shows the continuous object tracking plus reverse tracking

'''
Step 1: Environment settings and model initialization
'''
# init sam image predictor and video predictor model
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

root_dir = "/home/hlp/data/vton/zf/views/01"
video_dir = root_dir + "/ori_imgs"
output_dir = root_dir + "/sem_seg"
parsing_dir = root_dir + "/sapiens_2b"
os.makedirs(output_dir, exist_ok=True)

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = predictor.init_state(video_path=video_dir)
step = 30
# init the output masks
frame_masks = []
"""
Step 2: Propagate the video predictor to get the segmentation results for each frame
"""
print("Total frames:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
    # get different masks
    print(f"Processing frames {start_frame_idx} to {min(start_frame_idx + step, len(frame_names)) - 1}")
    mask_name = os.path.splitext(frame_names[start_frame_idx])[0] + "_seg.npy"
    mask_path = os.path.join(parsing_dir, mask_name) 
    mask = np.load(mask_path, allow_pickle=True)
    height, width = mask.shape
    sematic_masks = {
    'skin_mask': np.isin(mask, skin_zone),
    'uppercloth_mask': np.isin(mask, uppercloth_zone),
    'dress_mask': np.isin(mask, dress_zone),
    'coat_mask': np.isin(mask, coat_zone),
    'socks_mask': np.isin(mask, socks_zone),
    'pants_mask': np.isin(mask, pants_zone),
    'skirt_mask': np.isin(mask, skirt_zone),
    'head_mask': np.isin(mask, head_zone),
    'shoes_mask': np.isin(mask, shoes_zone)
    }
    # propagate the video predictor
    all_segments = {}
    for seg_idx, seg_label in enumerate(seg_classes):
        predictor.reset_state(inference_state)
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=start_frame_idx,
        obj_id=(seg_idx+1),
        mask=sematic_masks[seg_label],
        )
        video_segments = {}  # output the following {step} frames tracking masks
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, max_frame_num_to_track=step-1, start_frame_idx=start_frame_idx):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        all_segments[seg_label] = video_segments

    """
    Step 3: save the tracking masks
    """
    
    for frame_idx in range(min(step, len(frame_names) - start_frame_idx)):
        frame_masks.append(np.zeros((height, width, 3), dtype=np.uint8))

    for seg_label, video_segments in all_segments.items():
        for out_frame_idx, out_masks in video_segments.items():
            final_mask = frame_masks[out_frame_idx]
            for out_obj_id, out_mask in out_masks.items():
                color = class_colors[seg_label]
                final_mask[np.squeeze(out_mask)] = color

for i, frame_name in enumerate(frame_names):
    output_path = os.path.join(output_dir, frame_name)
    Image.fromarray(frame_masks[i]).save(output_path)