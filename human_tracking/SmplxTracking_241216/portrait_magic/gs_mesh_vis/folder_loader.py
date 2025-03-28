import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

def remod(x, mod=64):
    return ((x-1)//mod + 1)*mod

class Folder_Dataset(Dataset):
    def __init__(self, data_dir, start_rate = 0., end_rate = 1.):
        self.data_dir = data_dir
        rect_dir = os.path.join(data_dir, 'ldmks')
        self.rect_paths = natsorted([os.path.join(rect_dir, f) for f in os.listdir(rect_dir) if f.endswith('.rect')])
        img = cv2.imread(self.rect_paths[0].replace('ldmks', 'ori_imgs').replace('.rect', '.jpg'))
        self.img_size = img.shape[:2]


        start_id = int(len(self.rect_paths) * start_rate + .5)
        end_id = int(len(self.rect_paths) * end_rate + .5)
        self.load_ids = np.arange(start_id, end_id)

    def __len__(self):
        return self.load_ids.shape[0]

    def __getitem__(self, idx):
        load_idx = self.load_ids[idx]
        img_gt = cv2.imread(self.rect_paths[load_idx].replace('ldmks', 'train_imgs').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGRA2RGBA)

        img_parsing = cv2.imread(self.rect_paths[load_idx].replace('ldmks', 'parsing').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)

        return torch.from_numpy(img_gt), torch.as_tensor(img_parsing), load_idx
