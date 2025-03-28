import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset


class Folder_Dataset(Dataset):
    def __init__(self, folder):
        self.rect_paths = natsorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.rect')])
        self.smplx_track_dict = torch.load(os.path.join(folder.replace('ldmks', 'body_track'), 'smplx_track.pth'))
        self.track_save_keys = ['body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr', 'global_orient', 'transl']
        # assert(len(self.rect_paths) == self.smplx_track_dict['expr'].shape[0])

    def __len__(self):
        return min(len(self.rect_paths), self.smplx_track_dict['expr'].shape[0])

    def __getitem__(self, idx):
        img_ori = cv2.imread(self.rect_paths[idx].replace('ldmks', 'ori_imgs').replace('.rect', '.jpg'))
        seg_img = cv2.imread(self.rect_paths[idx].replace('ldmks', 'seg_masks').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        parsing_img = cv2.imread(self.rect_paths[idx].replace('ldmks', 'parsing').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        img_name = self.rect_paths[idx].split('/')[-1].replace('.rect', '.jpg')
        ret_dict = {}
        ret_dict['img_ori'] = torch.from_numpy(img_ori)
        ret_dict['img_ori'] = torch.from_numpy(img_ori)
        ret_dict['seg_img'] = torch.from_numpy(seg_img)
        ret_dict['parsing_img'] = torch.from_numpy(parsing_img)
        ret_dict['img_name'] = img_name
        ret_dict['shape'] = self.smplx_track_dict['shape'][0]
        ret_dict['cam_para'] = self.smplx_track_dict['cam_para'][0]
        for key in self.track_save_keys:
            ret_dict[key] = self.smplx_track_dict[key][idx]
        return ret_dict
