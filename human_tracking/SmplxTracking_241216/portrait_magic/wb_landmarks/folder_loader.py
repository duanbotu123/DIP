import os
import cv2
import torch
from natsort import natsorted
from torch.utils.data import Dataset


class Folder_Dataset(Dataset):
    def __init__(self, folder):

        self.img_paths = natsorted([os.path.join(folder, f)
                                   for f in os.listdir(folder) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        img_ori = cv2.imread(self.img_paths[idx])
        img_name = os.path.basename(self.img_paths[idx])
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        
        return torch.from_numpy(img_ori), img_name
