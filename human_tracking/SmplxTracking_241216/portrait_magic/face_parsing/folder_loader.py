import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

def cocowb_to_5landmarks(wb_ldmlks):
    leye = (wb_ldmlks[59, :2] + wb_ldmlks[62, :2]) * .5
    reye = (wb_ldmlks[65, :2] + wb_ldmlks[68, :2]) * .5
    nose, lmouth, rmouth = wb_ldmlks[53, :2], wb_ldmlks[71, :2], wb_ldmlks[77, :2]
    return np.array([leye, reye, nose, lmouth, rmouth])


class Folder_Dataset(Dataset):
    def __init__(self, folder):
        self.wb_ldmks_paths = natsorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wb')])

    def __len__(self):
        return len(self.wb_ldmks_paths)

    def __getitem__(self, idx):
        img_ori = cv2.imread(self.wb_ldmks_paths[idx].replace('ldmks', 'ori_imgs').replace('.wb', '.jpg'))
        # img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        img_name = os.path.basename(self.wb_ldmks_paths[idx])
        wb_ldmks = np.loadtxt(self.wb_ldmks_paths[idx], dtype=np.float32)
        ldmks_5p = cocowb_to_5landmarks(wb_ldmks)
        
        return torch.from_numpy(img_ori), torch.from_numpy(ldmks_5p), img_name
