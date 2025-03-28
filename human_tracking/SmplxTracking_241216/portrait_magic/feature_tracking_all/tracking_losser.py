import torch
import numpy as np
from .folder_loader import Folder_Dataset
from ..render_utils.render_nvdiff import MeshRenderer_Cuda
from ..dmm_models.smplx.smplx.lbs import vertices2landmarks
from ..render_utils.geometry_utils import proj_pts
from ..body_tracking.loss_utils import cal_euclidean_distance


class TrackingLosser():
    def __init__(self):
        self.device = 'cuda:0'
        self.mesh_renderer = MeshRenderer_Cuda()
    
    def set_folder(self, imgs_folder):
        self.folder_dataset = Folder_Dataset(imgs_folder.replace('ori_imgs', 'ldmks'))
        self.pre_num = 1
        self.save_id = 0

    def cal_tracking_loss(self, vertices_cam, cam_para, sel_ids, tris, bary_info = None):
        tracks_info, warp_mat = self.folder_dataset.get_batched_tracks(sel_ids) ########### (x, y, is_visible)
        tracks_info = tracks_info.to(self.device)
        warp_mat = warp_mat.to(self.device)
        T, N = tracks_info.shape[:2]

        render_cam = cam_para.clone()
        render_cam[:, [0, 2]] *= warp_mat[:, 0, 0:1]
        render_cam[:, [1, 3]] *= warp_mat[:, 1, 1:2]
        render_cam[:, 2] += warp_mat[:, 0, 2]
        render_cam[:, 3] += warp_mat[:, 1, 2]

        if bary_info is None:
            render_h, render_w = self.folder_dataset.dst_size
            with torch.inference_mode():
                rast_out = self.mesh_renderer.forward_rasterization(vertices_cam, render_cam, tris, [render_h, render_w]) ########### (T, h, w, 4)
                ##### parsing_info corrected visibility; tri id info
                tris_info = []
                for i in range(T):
                    ys, xs = tracks_info[i, :, 1], tracks_info[i, :, 0]
                    tris_info.append(rast_out[i][torch.clamp(ys, 0, render_h-1), torch.clamp(xs, 0, render_w-1)][:, [0,1,3]])
                
                tris_info = torch.stack(tris_info, dim=0) ################ (T, N, 3) (u, v, tri_id)

                tracks_tid = (tris_info.reshape(-1, 3)[:, 2]-1).long()
                tracks_coords = torch.cat((tris_info.reshape(-1, 3)[:, :2], 1. - torch.sum(tris_info.reshape(-1, 3)[:, :2], dim=-1, keepdim=True)), dim=-1)

                tracks_3d_is_valid = (tris_info[:, :, 2]>0).unsqueeze(-1)
                tracks_tid[tracks_tid<0] = 0.
                
                tracks_pts3d = vertices2landmarks(vertices_cam[:1], tris.squeeze(0).long(), tracks_tid.unsqueeze(0), tracks_coords.unsqueeze(0)).reshape(T, N, 3)
                tracks_pts3d[~(tracks_3d_is_valid.squeeze(-1))] = 1e5

                mean_tracks_3d = torch.mean(tracks_pts3d*tracks_3d_is_valid.float(), dim=0, keepdim=True) / (torch.mean(tracks_3d_is_valid.float(), dim=0, keepdim=True) + 1e-5)
                center_dis = torch.mean((tracks_pts3d - mean_tracks_3d)**2, dim=-1) ########### (T, N)
                
                track_in_mesh = (torch.sum(tracks_3d_is_valid.float().squeeze(-1) * tracks_info[..., 2].float(), dim=0) > .5)
                argmin_index = torch.min(center_dis, dim=0)[1]
                bary_info = tris_info[argmin_index, torch.arange(0, N)] ############### (N, 3)
                bary_info[~track_in_mesh, 2] = 0

        track_valid_ids = (bary_info[:, 2] > .5)
        track_tid = (bary_info[track_valid_ids, 2]-1).long().unsqueeze(0)
        track_coords = torch.cat((bary_info[track_valid_ids, :2], 1. - torch.sum(bary_info[track_valid_ids, :2], dim=-1, keepdim=True)), dim=-1).unsqueeze(0)
        track_pts_cam = vertices2landmarks(vertices_cam, tris.squeeze(0).long(), track_tid.expand(T, -1), track_coords.expand(T, -1, -1))  ############# (B, N', 3)
        track_proj_pts = proj_pts(track_pts_cam, render_cam.clone().detach()) ########### (b, N', 2)

        return cal_euclidean_distance(track_proj_pts, tracks_info[:, track_valid_ids, :2].float(), tracks_info[:, track_valid_ids, 2:].float()), bary_info
    