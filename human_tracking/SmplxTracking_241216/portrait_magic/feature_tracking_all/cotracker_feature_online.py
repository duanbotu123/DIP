import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.ops import sample_farthest_points
from .folder_loader import Folder_Dataset
from ..utils_funcs.video_writer import VideoWriter
from ..render_utils.render_nvdiff import MeshRenderer
from ..third_parties.cotracker3.utils.visualizer import Visualizer
from ..third_parties.cotracker3.predictor import CoTrackerOnlinePredictor


# 'label_names':  {'0: background', '1: neck', '2: face', '3: cloth', '4: rr', '5: lr', '6: rb', '7: lb', '8: re',
#                     '9: le', '10: nose', '11: imouth', '12: llip', '13: ulip', '14: hair',
#                     '15: eyeg', '16: hat', '17: earr', '18: neck_l'}
sample_nums = np.array([0, 40, 60, 200, 15, 15, 8, 8, 8, 8, 30, 0, 15, 15, 30, 10, 0, 5, 20], dtype=np.float32) ##### < 1 for probility; > 1 form number

sample_nums = (sample_nums * 2)

def correct_parsing_map(parsing_map):
    ##### parsing_map: (b, h, w, 2) -> (b, h, w)
    full_parsing = torch.zeros_like(parsing_map[..., 0]) + 3
    valid_parsing_mask = (parsing_map[..., 0] > 0).byte()
    full_parsing = full_parsing * (1 - valid_parsing_mask) + parsing_map[..., 0] * valid_parsing_mask
    full_parsing[parsing_map[..., 1] < 1] = 0
    return full_parsing

def get_query_points_from_parsing(parsing_map, canonical_id = 0):
    ##### parsing_map: (h, w)
    sample_pts = []
    for i in range(sample_nums.shape[0]):
        if sample_nums[i] < 1e-3:
            continue
        valid_indices = torch.nonzero(parsing_map == i).float()
        if valid_indices.shape[0] < 20:
            continue
        if sample_nums[i] <= 1:
            sample_num = int(valid_indices.shape[0] * sample_nums[i])
        else:
            sample_num = min(valid_indices.shape[0], int(sample_nums[i]))
        sample_points = sample_farthest_points(valid_indices.unsqueeze(0), K = sample_num, random_start_point = True)[0].squeeze(0)
        sample_points = torch.cat((torch.zeros_like(sample_points[:, :1]) + canonical_id, sample_points, torch.zeros_like(sample_points[:, :1]) + i), dim=1)
        sample_pts.append(sample_points)
    return torch.cat(sample_pts, dim=0)

class CotrackOnlineFeature():
    def __init__(self):
        self.device = 'cuda:0'
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = CoTrackerOnlinePredictor(checkpoint=os.path.join(file_dir_path, 'scaled_online.pth'))
        self.step = self.model.step
        self.model.to(self.device).eval()
        self.mesh_renderer = MeshRenderer()

        self.viser = Visualizer('./debug/')

    def set_folder(self, imgs_folder):
        self.folder_dataset = Folder_Dataset(imgs_folder.replace('ori_imgs', 'ldmks'))
        self.pre_num = 1
        self.save_id = 0
    

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        folder_dataset = Folder_Dataset(imgs_folder.replace('ori_imgs', 'ldmks'))
        folder_loader = torch.utils.data.DataLoader(
            folder_dataset, batch_size=self.step, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)

        video_out = None

        is_first_step = True

        tracker_viser = Visualizer('./debug/', show_first_frame=0, linewidth=1)
        for img_rgb, parsing_map, warp_mat, img_names in tqdm(folder_loader, desc='cotracker feature'):            
            parsing_map = parsing_map.to(self.device)
            parsing_map = correct_parsing_map(parsing_map)

            imgs = img_rgb.to(self.device).permute(0, 3, 2, 1).float()

            if is_first_step:
                query_pts_info = get_query_points_from_parsing(parsing_map[0])
                self.model(imgs[None], is_first_step, query_pts_info[:, :3][None])
                is_first_step = False
                imgs_pre = imgs.clone()
                continue
            video_chunk = torch.cat((imgs_pre, imgs), dim=0)
            tracks, visibilities = self.model(video_chunk[None], is_first_step, None)
            imgs_pre = imgs.clone()

        tracks, visibilities = tracks.squeeze(0), visibilities.squeeze(0)
        cur_idx = 0

        for img_rgb, parsing_map, _, img_names in tqdm(folder_loader, desc='cotracker feature'):   
            parsing_map = parsing_map.to(self.device)
            parsing_map = correct_parsing_map(parsing_map)        
            imgs = img_rgb.to(self.device).permute(0, 3, 2, 1).float()
            cur_size = imgs.shape[0]
            for i in range(cur_size):
                ys, xs = tracks[cur_idx + i, :, 0], tracks[cur_idx + i, :, 1]
                parsing_i = parsing_map[i][torch.clamp((ys+.5).long(), 0, img_rgb.shape[1]-1), torch.clamp((xs+.5).long(), 0, img_rgb.shape[2]-1)]
                track_i_parsing_not_same = (torch.abs(parsing_i - query_pts_info[:, 3]) > .5)
                visibilities[cur_idx + i][track_i_parsing_not_same] = False
                track_info = torch.stack(((xs+.5).int(), (ys+.5).int(), visibilities[cur_idx + i].int()), dim=-1)
                np.savetxt(os.path.join(save_folder, img_names[i][:-4] + '.track'), track_info.detach().cpu().numpy(), '%d')
                
            if debug_dir != 'none':
                if video_out is None:
                    video_out = VideoWriter(os.path.join(debug_dir, 'tracking_cofeature_online.mp4'))
                vis_video = tracker_viser.visualize(imgs.permute(0, 1, 3, 2).unsqueeze(0), tracks[cur_idx:cur_idx+cur_size, :, [1,0]].unsqueeze(0), visibilities[cur_idx:cur_idx+cur_size].unsqueeze(0).unsqueeze(-1), save_video=False)[0].permute(0, 2, 3, 1).byte().cpu().numpy()
                for i in range(cur_size):
                    video_out.write_frame(vis_video[i])
            cur_idx += cur_size

        if video_out is not None:
            video_out.close()
                