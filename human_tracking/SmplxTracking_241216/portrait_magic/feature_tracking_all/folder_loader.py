import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset


class Folder_Dataset(Dataset):
    def __init__(self, folder):
        self.dst_size = (512, 384)
        self.rect_paths = natsorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.rect')])
        self.cal_warp_info()

    def __len__(self):
        return len(self.rect_paths)
    
    def cal_warp_info(self):
        sel_ids = np.arange(0, len(self.rect_paths), max(1, len(self.rect_paths) // 20))
        body_rects = []
        for i in range(sel_ids.shape[0]):
            body_rects.append(np.loadtxt(self.rect_paths[sel_ids[i]], dtype=np.float32))
        body_rects = np.array(body_rects)
        body_rect = np.median(body_rects, axis=0)

        len_per_pixel = max((body_rect[2] - body_rect[0]) / self.dst_size[1], (body_rect[3] - body_rect[1]) / self.dst_size[0]) * 1.1
        warp_scale = 1. / len_per_pixel
        cx = (body_rect[0] + body_rect[2]) / 2.
        cy = body_rect[1]*.55 + body_rect[3]*.45
        warp_tx, warp_ty = self.dst_size[1] * .5 - cx*warp_scale, self.dst_size[0] * .5 - cy*warp_scale
        self.warp_mat = np.array([[warp_scale, 0, warp_tx], [0, warp_scale, warp_ty]], dtype=np.float32)

    def get_batched_items(self, sel_ids):
        batch_img_croped, batch_parsing_croped, batch_warp_mat, batch_img_name = [], [], [], []
        for sel_id in sel_ids:
            img_croped, parsing_croped, warp_mat, img_name = self.__getitem__(sel_id)
            batch_img_croped.append(img_croped)
            batch_parsing_croped.append(parsing_croped)
            batch_warp_mat.append(warp_mat)
            batch_img_name.append(img_name)
        return torch.stack(batch_img_croped, dim=0), torch.stack(batch_parsing_croped, dim=0), torch.stack(batch_warp_mat, dim=0), batch_img_name
    
    def get_batched_tracks(self, sel_ids):
        batch_tracks, batch_warp_mat = [], []
        for sel_id in sel_ids:
            tracks = np.loadtxt(self.rect_paths[sel_id].replace('ldmks', 'track_feature').replace('.rect', '.track'), dtype=np.int64)
            batch_tracks.append(tracks)
            batch_warp_mat.append(self.warp_mat)
        return torch.from_numpy(np.array(batch_tracks)), torch.from_numpy(np.array(batch_warp_mat))


    def __getitem__(self, idx):
        img_ori = cv2.imread(self.rect_paths[idx].replace('ldmks', 'ori_imgs').replace('.rect', '.jpg'))
        parsing_img = cv2.imread(self.rect_paths[idx].replace('ldmks', 'parsing').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        seg_img = cv2.imread(self.rect_paths[idx].replace('ldmks', 'seg_masks').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        parsing_map = np.stack([parsing_img, seg_img], axis=-1)
        img_name = self.rect_paths[idx].split('/')[-1].replace('.rect', '.jpg')
        img_croped = cv2.warpAffine(img_ori, self.warp_mat, (self.dst_size[1], self.dst_size[0]))
        img_croped = cv2.cvtColor(img_croped, cv2.COLOR_BGR2RGB)
        parsing_croped = cv2.warpAffine(parsing_map, self.warp_mat, (self.dst_size[1], self.dst_size[0]), flags=cv2.INTER_NEAREST)
        return torch.from_numpy(img_croped), torch.from_numpy(parsing_croped), torch.from_numpy(self.warp_mat), img_name
