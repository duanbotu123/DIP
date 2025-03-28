# from ...dmm_models import SMPLX, smplx_model_path
from exavatar.fitting.common.utils.smpl_x import smpl_x
from ...render_utils.geometry_utils import bary_inerpolation
from ...dmm_models.adnerf_rendering import rendering_part_ids_path, template_mesh_file, gaussian_info_file, smplx_smooth_wts_file
from pytorch3d.io import load_obj
import torch
import torch.nn as nn
import numpy as np
from ...render_utils.geometry_utils import pts2obj
import pickle
from icecream import ic
from ...render_utils.supp_utils import a_in_b_torch
from simple_knn._C import distCUDA2
import copy

class GaussianGeo_osot_ex(nn.Module):
    def __init__(self, device, smplx_share):
        super().__init__()
        self.device = device
        
        # shape_dim, expr_dim = 300, 100
        # self.smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=expr_dim, num_betas=shape_dim, use_pca=False).to(self.device).eval()
        self.smplx_layer = copy.deepcopy(smpl_x.layer).to(self.device)

        self.rendering_part_ids = np.loadtxt(rendering_part_ids_path, dtype=np.int64)

        gaussian_info = torch.load(gaussian_info_file)
        bary_info = gaussian_info['gs_tris'].to(self.device) #### （*， 3)
        self.lap_indices = gaussian_info['lap_indices'].to(self.device)
        self.upper_teeth_vid = gaussian_info['upper_vid']
        self.downer_teeth_vid = gaussian_info['downer_vid']
        self.upper_teeth_inds = gaussian_info['upper_teeth_inds'].reshape(-1)
        self.downer_teeth_inds = gaussian_info['downer_teeth_inds'].reshape(-1)
        self.gaussian_pnum = bary_info.shape[0]
        self.gs_tids = bary_info[:, 0].long()
        self.gs_coords = torch.cat((bary_info[:, 1:], 1. - torch.sum(bary_info[:, 1:], dim=-1, keepdim=True)), dim=-1)
        _, faces, _ = load_obj(template_mesh_file)
        self.tris = faces.verts_idx.to(self.device).long()

        self.smplx_share = smplx_share
        self.compute_canonical_points()

        smplx_smooth_wts = torch.from_numpy(np.load(smplx_smooth_wts_file)).reshape(-1, 1).to(self.device)
        self.verts_smooth_wts = smplx_smooth_wts[self.rendering_part_ids, :]

        # osot
        self.gs_vids = self.tris[self.gs_tids] # pnum, 3
        with open('exavatar/fitting/common/utils/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
            hand_vertex_idx = pickle.load(f, encoding='latin1')
        self.rhand_vid = hand_vertex_idx['right_hand']
        self.lhand_vid = hand_vertex_idx['left_hand']
        self.rhand_vid_tensor = torch.tensor(self.rhand_vid, dtype=torch.int64).to(self.device)
        self.lhand_vid_tensor = torch.tensor(self.lhand_vid, dtype=torch.int64).to(self.device)
        self.is_lhand = (
            a_in_b_torch(self.gs_vids[:,0], self.lhand_vid_tensor)
            & a_in_b_torch(self.gs_vids[:,1], self.lhand_vid_tensor)
            & a_in_b_torch(self.gs_vids[:,2], self.lhand_vid_tensor)
        ) # (pnum,)
        self.is_rhand = (
            a_in_b_torch(self.gs_vids[:,0], self.rhand_vid_tensor)
            & a_in_b_torch(self.gs_vids[:,1], self.rhand_vid_tensor)
            & a_in_b_torch(self.gs_vids[:,2], self.rhand_vid_tensor)
        ) # (pnum,)


    def compute_canonical_points(self):
        # smplx_out = self.smplx_model.forward(betas = self.shape_code)
        self.root_pose_default = torch.zeros(1, 3).to(self.device)
        self.body_pose_default = torch.zeros(1, 63).to(self.device)
        self.jaw_pose_default = torch.zeros(1, 3).to(self.device)
        self.leye_pose_default = torch.zeros(1, 3).to(self.device)
        self.reye_pose_default = torch.zeros(1, 3).to(self.device)
        self.lhand_pose_default = torch.zeros(1, 45).to(self.device)
        self.rhand_pose_default = torch.zeros(1, 45).to(self.device)
        self.expr_default = torch.zeros(1, 50).to(self.device)

        smplx_out = self.smplx_layer(global_orient=self.root_pose_default, body_pose=self.body_pose_default, jaw_pose=self.jaw_pose_default, leye_pose=self.leye_pose_default, reye_pose=self.reye_pose_default, left_hand_pose=self.lhand_pose_default, right_hand_pose=self.rhand_pose_default, expression=self.expr_default, betas=self.smplx_share['shape'], face_offset=self.smplx_share['face_offset'], joint_offset=self.smplx_share['joint_offset'], locator_offset=self.smplx_share['locator_offset'])
        
        part_verts = smplx_out.vertices[0:1, self.rendering_part_ids, :] ### (1, nv, 3)
        self.canonical_verts = part_verts.clone()
        self.canonical_points = bary_inerpolation(part_verts, self.tris, self.gs_tids.unsqueeze(0), self.gs_coords.unsqueeze(0)).detach() ### (1, pcs, 3)
        self.canonical_points[:, self.upper_teeth_inds, 2] -= 0.015 #### teeth 1.5cm back 
        self.canonical_points[:, self.downer_teeth_inds, 2] -= 0.015 #### teeth 1.5cm back 


    def compute_canonical_points_rott(self, frames_smplx_dict, canonical_gs_dis, canonical_verts_dis):
        b = frames_smplx_dict['expr'].shape[0]
        # smplx_out, verts_rott, verts_canonical = self.smplx_model.forward(betas=self.shape_code.expand(b, -1), body_pose=frames_smplx_dict['body_pose'], left_hand_pose=frames_smplx_dict['lhand_pose'], right_hand_pose=frames_smplx_dict['rhand_pose'], expression=frames_smplx_dict['expr'], jaw_pose=frames_smplx_dict['jaw_pose'], leye_pose=frames_smplx_dict['leye_pose'], reye_pose=frames_smplx_dict['reye_pose'], global_orient=frames_smplx_dict['global_orient'], with_rott_return=True)
        smplx_out, verts_rott, verts_canonical = self.smplx_layer.forward_rott(global_orient=frames_smplx_dict['root_pose'], body_pose=frames_smplx_dict['body_pose'], jaw_pose=frames_smplx_dict['jaw_pose'], leye_pose=frames_smplx_dict['leye_pose'], reye_pose=frames_smplx_dict['reye_pose'], left_hand_pose=frames_smplx_dict['lhand_pose'], right_hand_pose=frames_smplx_dict['rhand_pose'], expression=frames_smplx_dict['expr'], betas=frames_smplx_dict['shape'], face_offset=frames_smplx_dict['face_offset'], joint_offset=frames_smplx_dict['joint_offset'], locator_offset=frames_smplx_dict['locator_offset'], with_rott_return=True)
        
        canonical_part_verts = verts_canonical[:, self.rendering_part_ids, :]

        part_verts = canonical_part_verts + canonical_verts_dis
        part_verts_rott = verts_rott[:, self.rendering_part_ids]

        upper_teeth_mask = part_verts.new_zeros(1, self.gaussian_pnum, 1)
        upper_teeth_mask[:, self.upper_teeth_inds] = 1.
        downer_teeth_mask = part_verts.new_zeros(1, self.gaussian_pnum, 1)
        downer_teeth_mask[:, self.downer_teeth_inds] = 1.

        gs_points = bary_inerpolation(canonical_part_verts, self.tris, self.gs_tids.unsqueeze(0).expand(b, -1), self.gs_coords.unsqueeze(0).expand(b,-1,-1)) ### (b, pcs, 3)
        gs_points = gs_points*(1.-upper_teeth_mask) + self.canonical_points*upper_teeth_mask
        gs_points = gs_points*(1.-downer_teeth_mask) + self.canonical_points*downer_teeth_mask
        gs_points += canonical_gs_dis
        gs_points_rott = bary_inerpolation(part_verts_rott, self.tris, self.gs_tids.unsqueeze(0).expand(b, -1), self.gs_coords.unsqueeze(0).expand(b, -1, -1)) ### (b, pcs, 3, 4)
        gs_points_rott = gs_points_rott*(1.-upper_teeth_mask[..., None]) + part_verts_rott[:, self.upper_teeth_vid].unsqueeze(1)*upper_teeth_mask[..., None]
        gs_points_rott = gs_points_rott*(1.-downer_teeth_mask[..., None]) + part_verts_rott[:, self.downer_teeth_vid].unsqueeze(1)*downer_teeth_mask[..., None]

        return gs_points, gs_points_rott, part_verts, part_verts_rott
    
    @torch.no_grad()
    def cal_neighbor_dist(self):
        neighbor_diss = self.canonical_points[0, self.lap_indices[:, :1]] - self.canonical_points[0, self.lap_indices[:, 1:]]
        neighbor_dist = torch.max(torch.norm(neighbor_diss, dim=-1), dim=-1)[0]
        median_dist = torch.median(neighbor_dist).item()
        arg_min, arg_max = .5*median_dist, 1.5*median_dist
        neighbor_dist[neighbor_dist<arg_min] = arg_min
        neighbor_dist[neighbor_dist>arg_max] = arg_max
        return neighbor_dist
        
    @torch.no_grad()
    def cal_neighbor_dist_osot(self):
        dist2 = torch.clamp_min(distCUDA2(self.canonical_points[0]), 0.0000001)

        return torch.sqrt(dist2)
