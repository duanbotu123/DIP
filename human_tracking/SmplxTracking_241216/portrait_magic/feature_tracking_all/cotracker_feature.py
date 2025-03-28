import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.ops import sample_farthest_points
from .folder_loader import Folder_Dataset
from ..utils_funcs.video_writer import VideoWriter
from ..third_parties.cotracker.predictor import CoTrackerPredictor
from ..render_utils.render_nvdiff import MeshRenderer
from ..dmm_models.smplx.smplx.lbs import vertices2landmarks
from ..render_utils.geometry_utils import proj_pts
from ..body_tracking.loss_utils import cal_euclidean_distance
from ..third_parties.cotracker.utils.visualizer import Visualizer


# 'label_names':  {'0: background', '1: neck', '2: face', '3: cloth', '4: rr', '5: lr', '6: rb', '7: lb', '8: re',
#                     '9: le', '10: nose', '11: imouth', '12: llip', '13: ulip', '14: hair',
#                     '15: eyeg', '16: hat', '17: earr', '18: neck_l'}
# sample_nums = np.array([0, 20, 150, 100, 10, 10, 10, 10, 8, 8, 20, 0, 25, 25, 50, 10, 0, 5, 10], dtype=np.float32) ##### < 1 for probility; > 1 form number

sample_nums = np.array([0, 40, 100, 150, 15, 15, 8, 8, 8, 8, 30, 0, 15, 15, 80, 10, 0, 5, 20], dtype=np.float32) ##### < 1 for probility; > 1 form number

def correct_parsing_map(parsing_map):
    ##### parsing_map: (b, h, w, 2) -> (b, h, w)
    full_parsing = torch.zeros_like(parsing_map[..., 0]) + 3
    valid_parsing_mask = (parsing_map[..., 0] > 0).byte()
    full_parsing = full_parsing * (1 - valid_parsing_mask) + parsing_map[..., 0] * valid_parsing_mask
    full_parsing[parsing_map[..., 1] < 200] = 0
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

class CotrackFeature():
    def __init__(self):
        self.device = 'cuda:0'
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = CoTrackerPredictor(checkpoint=os.path.join(file_dir_path, 'cotracker_stride_4_wind_8.pth'))
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
            folder_dataset, batch_size=25, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
        canonical_frame = None
        batch_id = 0

        video_out = None

        tracker_viser = Visualizer('./debug/', show_first_frame=0, linewidth=1)
        for img_rgb, parsing_map, warp_mat, img_names in tqdm(folder_loader, desc='cotracker feature'):            
            parsing_map = parsing_map.to(self.device)
            parsing_map = correct_parsing_map(parsing_map)

            ori_size = img_rgb.shape[0]
            min_size = 10

            imgs = img_rgb.to(self.device).permute(0, 3, 2, 1).float()

            if ori_size < min_size:
                parsing_map = torch.cat((parsing_map, parsing_map[-1:].repeat(min_size-ori_size, 1, 1)), dim=0)
                imgs = torch.cat((imgs, imgs[-1:].repeat(min_size-ori_size, 1, 1, 1)), dim=0)

            if canonical_frame is None:
                canonical_frame = imgs[:1].clone()
                query_pts_info = get_query_points_from_parsing(parsing_map[0])
            
            if batch_id > 0:
                tracking_frames = torch.cat((canonical_frame, imgs), dim=0)
            else:
                tracking_frames = imgs
            with torch.inference_mode():
                tracks, visibilities = self.model.compute_tracks_without_resize(tracking_frames.unsqueeze(0), query_pts_info[:, :3].unsqueeze(0))

                tracks, visibilities = tracks.squeeze(0), visibilities.squeeze(0) #### (T, N, 2), (T, N)
                start_id = int(batch_id>0) * canonical_frame.shape[0]

                # vis_sum = torch.sum(visibilities[start_id:].float(), dim=0)
                # vis_valid = (vis_sum > int((visibilities.shape[0]-start_id) * .7))
                # visibilities[:, ~vis_valid] = False

                for i in range(ori_size):
                    ys, xs = tracks[start_id + i, :, 0], tracks[start_id + i, :, 1]
                    parsing_i = parsing_map[i][torch.clamp((ys+.5).long(), 0, img_rgb.shape[1]-1), torch.clamp((xs+.5).long(), 0, img_rgb.shape[2]-1)] ### N
                    track_i_parsing_not_same = (torch.abs(parsing_i - query_pts_info[:, 3]) > .5)
                    visibilities[start_id + i][track_i_parsing_not_same] = False
                    # track_i_parsing_not_same = ((parsing_i == 0) | (parsing_i == 11))

                    track_info = torch.stack(((xs+.5).int(), (ys+.5).int(), visibilities[start_id + i].int()), dim=-1)
                    np.savetxt(os.path.join(save_folder, img_names[i][:-4] + '.track'), track_info.detach().cpu().numpy(), '%d')
                
            if debug_dir != 'none':
                if video_out is None:
                    video_out = VideoWriter(os.path.join(debug_dir, 'tracking_cofeature.mp4'))
                vis_video = tracker_viser.visualize(tracking_frames.permute(0, 1, 3, 2).unsqueeze(0), tracks[..., [1,0]].unsqueeze(0), visibilities.unsqueeze(0).unsqueeze(-1), save_video=False)[0].permute(0, 2, 3, 1).byte().cpu().numpy()
                for i in range(start_id, vis_video.shape[0]):
                    # print(features[i].shape)
                    video_out.write_frame(vis_video[i])
            batch_id += 1

        if video_out is not None:
            video_out.close()
                
