import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset
from .OSX import cfg
from .data_utils import process_bbox, generate_patch_image


class Folder_Dataset(Dataset):
    def __init__(self, imgs_folder):
        rects_folder = imgs_folder.replace('ori_imgs', 'ldmks')
        self.dets_names = natsorted([f for f in os.listdir(rects_folder) if f.endswith('.rect')])

        self.imgs_folder = imgs_folder
        self.dets_folder = rects_folder

    def __len__(self):
        return len(self.dets_names)

    def __getitem__(self, idx):
        
        img_ori = cv2.imread(os.path.join(self.imgs_folder, self.dets_names[idx][:-5] + '.jpg'))
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        det_rect = np.loadtxt(os.path.join(self.dets_folder, self.dets_names[idx]), dtype=np.float32)
        det_bbox = np.array(det_rect)
        det_bbox[2], det_bbox[3] = det_rect[2] - det_rect[0], det_rect[3] - det_rect[1] ### convert xyxy to xywh

        det_bbox = process_bbox(det_bbox, img_ori.shape[1], img_ori.shape[0]) ### process bbox

        img, img2bb_trans, bb2img_trans = generate_patch_image(img_ori, det_bbox, 1.0, 0.0, False, cfg.input_img_shape)
        name_pre = self.dets_names[idx][:-5]
        
        return torch.from_numpy(img) , torch.from_numpy(img_ori), torch.from_numpy(det_bbox), name_pre
