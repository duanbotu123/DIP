import os
import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

def remod(x, mod=64):
    return ((x-1)//mod + 1)*mod

class Folder_Dataset(Dataset):
    def __init__(self, data_dir, head_data_dir, start_rate = 0., end_rate = 1.):
        self.data_dir = data_dir
        rect_dir = os.path.join(data_dir, 'ldmks')
        self.rect_paths = natsorted([os.path.join(rect_dir, f) for f in os.listdir(rect_dir) if f.endswith('.rect')])
        img = cv2.imread(self.rect_paths[0].replace('ldmks', 'ori_imgs').replace('.rect', '.jpg'))
        self.img_size = img.shape[:2]


        self.head_data_dir = head_data_dir
        head_rect_dir = os.path.join(head_data_dir, 'ldmks')
        self.head_rect_paths = natsorted([os.path.join(head_rect_dir, f) for f in os.listdir(head_rect_dir) if f.endswith('.rect')])
        head_img = cv2.imread(self.head_rect_paths[0].replace('ldmks', 'ori_imgs').replace('.rect', '.jpg'))
        self.head_img_size = head_img.shape[:2]


        start_id = int(len(self.rect_paths) * start_rate + .5)
        end_id = int(len(self.rect_paths) * end_rate + .5)
        self.load_ids = np.arange(start_id, end_id)

    def __len__(self):
        return self.load_ids.shape[0]

    def __getitem__(self, idx):
        load_idx = self.load_ids[idx]
        img_gt = cv2.imread(self.rect_paths[load_idx].replace('ldmks', 'train_imgs').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        img_ldmks = np.loadtxt(self.rect_paths[load_idx].replace('.rect', '.wb'), dtype=np.float32)
        face_sel_ldmks = img_ldmks[[59, 60, 67, 68, 71, 77, 80], :]
        inner_l, inner_r, inner_t, inner_b = int(np.min(face_sel_ldmks[:, 0])), int(np.max(face_sel_ldmks[:, 0])), int(np.min(face_sel_ldmks[:, 1])), int(np.max(face_sel_ldmks[:, 1]))
        img_parsing = cv2.imread(self.rect_paths[load_idx].replace('ldmks', 'parsing').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        img_parsing[inner_t:inner_b, inner_l:inner_r] = 11
        wb_ldmks = np.loadtxt(self.rect_paths[load_idx].replace('.rect', '.wb'), dtype=np.float32)
        l, t, r, b = wb_ldmks[71, 0], wb_ldmks[74, 1], wb_ldmks[77, 0], wb_ldmks[80, 1]
        box_size = 400
        lmin = 0
        lmax = img_gt.shape[1] - box_size
        tmin = 0
        tmax = img_gt.shape[0] - box_size
        rect_l = np.random.randint(lmin, lmax)
        rect_t = np.random.randint(tmin, tmax)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGRA2RGBA)

        head_img_gt = cv2.imread(self.head_rect_paths[load_idx].replace('ldmks', 'train_imgs').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        head_ldmks = np.loadtxt(self.head_rect_paths[load_idx].replace('.rect', '.wb'), dtype=np.float32)
        head_face_top_y = int(np.min(head_ldmks[40:50, 1]))
        head_img_parsing = cv2.imread(self.head_rect_paths[load_idx].replace('ldmks', 'parsing').replace('.rect', '.png'), cv2.IMREAD_UNCHANGED)
        # head_img_parsing[:head_face_top_y] = 0
        head_wb_ldmks = np.loadtxt(self.head_rect_paths[load_idx].replace('.rect', '.wb'), dtype=np.float32)
        l, t, r, b = head_wb_ldmks[71, 0], head_wb_ldmks[74, 1], head_wb_ldmks[77, 0], head_wb_ldmks[80, 1]
        head_box_size = 400
        lmin = 0
        lmax = head_img_gt.shape[1] - head_box_size
        tmin = 0
        tmax = head_img_gt.shape[0] - head_box_size
        head_rect_l = np.random.randint(lmin, lmax)
        head_rect_t = np.random.randint(tmin, tmax)
        head_img_gt = cv2.cvtColor(head_img_gt, cv2.COLOR_BGRA2RGBA)
        return torch.from_numpy(img_gt), torch.from_numpy(img_parsing), torch.tensor([rect_l, rect_t, box_size, box_size], dtype=torch.int32), torch.from_numpy(head_img_gt), torch.from_numpy(head_img_parsing), torch.tensor([head_rect_l, head_rect_t, head_box_size, head_box_size], dtype=torch.int32), load_idx
