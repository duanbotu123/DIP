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

        # self.rect_paths = self.rect_paths[start_id:end_id]

    def __len__(self):
        return self.load_ids.shape[0]

    def __getitem__(self, idx):
        load_idx = self.load_ids[idx]
        img_gt = cv2.imread(self.rect_paths[load_idx].replace('ldmks', 'train_imgs').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        wb_ldmks = np.loadtxt(self.rect_paths[load_idx].replace('.rect', '.wb'), dtype=np.float32)
        l, t, r, b = wb_ldmks[71, 0], wb_ldmks[74, 1], wb_ldmks[77, 0], wb_ldmks[80, 1]


        # rect_h = remod(int(b-t))
        # rect_w = remod(int(r-l))
        # rect_l = int(l*.5+r*.5) - rect_w//2
        # rect_t = int(t*.5+b*.5) - rect_h//2

        box_size = 400
        lmin = 0
        lmax = img_gt.shape[1] - box_size
        tmin = 0
        tmax = img_gt.shape[0] - box_size

        rect_l = np.random.randint(lmin, lmax)
        rect_t = np.random.randint(tmin, tmax)


        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGRA2RGBA)
        return torch.from_numpy(img_gt), load_idx, torch.tensor([rect_l, rect_t, box_size, box_size], dtype=torch.int32)
