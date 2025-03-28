import torch
import torch.nn as nn
import numpy as np
import os
from .geometry_osot import GaussianGeo_osot
from .camera import Camera
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, so3_exp_map
from pytorch3d.structures import Meshes
from ...render_utils.geometry_utils import strip_symmetric, build_scaling_rotation, focal2fov, proj_pts, pts2obj, compute_rotation_from_normals
from ...render_utils import MeshRenderer_Cuda
from pytorch3d.loss import mesh_normal_consistency
from ...render_utils.geometry_utils import bary_inerpolation
import ipdb
from icecream import ic
from PIL import Image

class GaussianScene_osot_iso(nn.Module):
    def __init__(self, img_size, shape, sh_degree = 3, device = 'cuda:0'):
        super().__init__()
        self.img_size = img_size
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.device = device

        self.geo_creator = GaussianGeo_osot(device, shape)
        self.is_lhand = self.geo_creator.is_lhand
        self.is_rhand = self.geo_creator.is_rhand
        self.lap_indices = self.geo_creator.lap_indices
        self.mesh_renderer = MeshRenderer_Cuda().to(self.device)
        self.init_attributes()
        self.setup_functions()

    def cal_attr_lap_loss(self, attr):
        attr_dis = attr[self.lap_indices[:, 0]] - torch.mean(attr[self.lap_indices[:, 1:]], dim=1)
        # osot
        weight = torch.ones((self.geo_creator.gaussian_pnum, 1)).float().to(self.device)
        weight[self.is_lhand,:] = 100
        weight[self.is_rhand,:] = 100
        return torch.mean(attr_dis**2*weight)
    
    def cal_lap_loss(self):
        return self.cal_attr_lap_loss(self._features_dc.squeeze(1)) * 1e-3 + self.cal_attr_lap_loss(self._xyz_canonical_dis[0]) + self.cal_attr_lap_loss(self.scaling) # + self.cal_attr_lap_loss(self._opacity)

    def cal_mesh_lap_loss(self):
        verts_dis_laplacian = torch.mm(self.laplacian_L, self._verts_canonical_dis[0])
        verts_dis_laplacian *= self.geo_creator.verts_smooth_wts
        return torch.mean(torch.abs(verts_dis_laplacian))

    def cal_dis_loss(self):
        ### make dis not large
        return torch.mean(self._xyz_canonical_dis**2) + torch.mean(self._verts_canonical_dis**2)
    
    def cal_scale_loss(self):
        ### make scale not far from initial
        canonical_sacle_ratio = self.scaling_activation(self._scaling) / self.canonical_scales
        return torch.sum(torch.relu(canonical_sacle_ratio - 2.0)) + torch.sum(torch.relu(0.2 - canonical_sacle_ratio)) 

    def cal_gs_mesh_consistent_loss(self):
        ### make gs dis close to underlying mesh driven dis
        mesh_driven_gs_dis = bary_inerpolation(self._verts_canonical_dis, self.geo_creator.tris, self.geo_creator.gs_tids.unsqueeze(0), self.geo_creator.gs_coords.unsqueeze(0)) ### (1, pcs, 3)
        return torch.mean(torch.abs(mesh_driven_gs_dis - self._xyz_canonical_dis))

    def init_attributes(self):
        pc_num = self.geo_creator.gaussian_pnum
        print('point number', pc_num)
        features = torch.zeros((pc_num, 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)

        ### init scale based on the dist to neighbor gs centers, and make the GS close to planer
        gs_neighbor_dist = self.geo_creator.cal_neighbor_dist_osot()
        self.canonical_scales = gs_neighbor_dist[..., None]#.repeat(1, 3) * 1.0
        self.canonical_scales /= 2.5

        scales = torch.log(self.canonical_scales)
        print('init scale', torch.min(torch.exp(scales)), torch.max(torch.exp(scales)), torch.mean(torch.exp(scales)))
        rots = torch.zeros((pc_num, 4), device=self.device)
        rots[:, 0] = 1
        # opacities = torch.logit(0.1 * torch.ones((pc_num, 1), dtype=torch.float, device=self.device))
        opacities = torch.logit(1.0 * torch.ones((pc_num, 1), dtype=torch.float, device=self.device))

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) #### (pcs, 1, 3)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._xyz_canonical_dis = nn.Parameter(torch.zeros_like(self.geo_creator.canonical_points).requires_grad_(True)) ### (1, pcs, 3)
        self._verts_canonical_dis = nn.Parameter(torch.zeros_like(self.geo_creator.canonical_verts).requires_grad_(True)) ### (1, nv, 3)
        # self._body_pose_dis = nn.Parameter(self.geo_creator.canonical_points.new_zeros((self.transl.shape[0], 63)).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self.scaling = self._scaling.repeat(1, 3)
        # self._rotation_base = nn.Parameter(rots.requires_grad_(True))
        self._rotation_base = rots
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._opacity = opacities
        self.max_radii2D = torch.zeros((pc_num), device=self.device)

        mesh_canonical = Meshes(verts=[self.geo_creator.canonical_verts[0]], faces=[self.geo_creator.tris])
        self.laplacian_L = mesh_canonical.laplacian_packed() ### (n_v, n_v) sparse matrix

    
    def save_updated_canonical(self, save_path):
        pts2obj(self.geo_creator.canonical_points + self._xyz_canonical_dis, save_path)

    def capture(self):
        # return (
        #     self.active_sh_degree, self._xyz_canonical_dis, self._verts_canonical_dis, self._features_dc, self._features_rest, self._scaling, self._rotation_base, self._opacity#, self._body_pose_dis
        # )
        return (
            self.active_sh_degree, self._xyz_canonical_dis, self._verts_canonical_dis, self._features_dc, self._features_rest, self._scaling, self._rotation_base, self._opacity, self.optimizer.state_dict()
        )
    
    def restore(self, model_dict):
        (self.active_sh_degree, _xyz_canonical_dis, _verts_canonical_dis, _features_dc, _features_rest, _scaling, _rotation_base, _opacity, opt_dict) = model_dict
        self._xyz_canonical_dis.data = _xyz_canonical_dis.clone()
        self._verts_canonical_dis.data = _verts_canonical_dis.clone()
        self._features_dc.data = _features_dc.clone()
        self._features_rest.data = _features_rest.clone()
        self._scaling.data = _scaling.clone()
        self._rotation_base.data = _rotation_base.clone()
        self._opacity.data = _opacity.clone()
        # self._body_pose_dis.data = _body_pose_dis.clone()
        self.optimizer.load_state_dict(opt_dict)

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit
        self.rotation_activation = torch.nn.functional.normalize 
    
    def cal_rott_loss(self):
        return torch.mean(torch.abs(self.delta_rott)) + self.cal_attr_lap_loss(self.delta_rott) * 1e2

    def update_xyz_rot(self, frame_ids, with_mesh_rendering = True, in_train = False, mode = 'body'):
        batch_size = frame_ids.shape[0]
        frame_smplx_dict = {}
        for key in self.smplx_track_keys[0:8]:
            if mode == 'head':
                frame_smplx_dict[key] = getattr(self, 'head_' + key)[frame_ids].clone()
            else:
                frame_smplx_dict[key] = getattr(self, key)[frame_ids].clone()
        if mode == 'mixture':
            frame_smplx_dict['expr'] = self.head_expr[frame_ids].clone()
            frame_smplx_dict['jaw_pose'] = self.head_jaw_pose[frame_ids].clone()
            frame_smplx_dict['leye_pose'] = self.head_leye_pose[frame_ids].clone()
            frame_smplx_dict['reye_pose'] = self.head_reye_pose[frame_ids].clone()
        if mode != 'head':
            frame_smplx_dict['body_pose'] += self._body_pose_dis[frame_ids]
        gs_points, gs_rott, verts, verts_rott = self.geo_creator.compute_canonical_points_rott(frame_smplx_dict, self._xyz_canonical_dis, self._verts_canonical_dis)

        ### compute gs rotation based on underlying mesh
        meshes_canonical = Meshes(verts=verts, faces=self.geo_creator.tris[None, ...].expand(batch_size, -1, -1))
        with torch.no_grad():
            face_normals = meshes_canonical.faces_normals_padded() ### (b, nf, 3)
            rotation_ = compute_rotation_from_normals(face_normals.reshape(-1,3)).reshape(batch_size, -1, 3, 3)
            gs_canonical_rotation = rotation_[:, self.geo_creator.gs_tids] ### (b, pcs, 3, 3)
        gs_points_homo = torch.cat((gs_points, torch.ones_like(gs_points[..., :1])), dim=-1)
        self._xyz = torch.bmm(gs_rott.reshape(-1,3,4), gs_points_homo.reshape(-1,4,1)).reshape(batch_size, -1, 3) ### (b, pcs, 3)
        gs_rotation = torch.bmm(gs_rott[:, :, :3, :3].reshape(-1, 3, 3), gs_canonical_rotation.reshape(-1,3,3)).reshape(batch_size, -1, 3, 3)
        self._rotation = matrix_to_quaternion(gs_rotation)

        verts_homo = torch.cat((verts, torch.ones_like(verts[..., :1])), dim=-1)
        verts_current = torch.bmm(verts_rott.reshape(-1,3,4), verts_homo.reshape(-1,4,1)).reshape(batch_size, -1, 3) ### (b, v_num, 3)
        verts_cam = torch.bmm(verts_current, self.rots[frame_ids].permute(0,2,1)) + self.transs[frame_ids].unsqueeze(1)

        if not with_mesh_rendering:
            return 

        if not in_train: ### only return mesh vis
            mesh_vis = self.mesh_renderer.forward_visualization_geo(verts_cam, self.geo_creator.tris[None, ...].int().expand(batch_size, -1, -1), self.cam_para.expand(batch_size, -1), self.img_size)
            return mesh_vis
        
        else: ### return differentiable mesh mask, and mesh normal consistent loss
            diff_mask = self.mesh_renderer.forward_differentiable_mask(verts_cam, self.geo_creator.tris[None, ...].int().expand(batch_size, -1, -1), self.cam_para.expand(batch_size, -1), self.img_size)
            meshes_consistent = Meshes(verts=verts[0:1], faces=self.geo_creator.tris[None, ...].expand(1, -1, -1))
            loss_normal_consistent = mesh_normal_consistency(meshes_consistent)
            return diff_mask, loss_normal_consistent
    
    def update_xyz_rot_osot(self, smplx_inputs, cam_inputs, with_mesh_rendering = True, in_train = False):
        batch_size = smplx_inputs['body_pose'].shape[0]
        gs_points, gs_rott, verts, verts_rott = self.geo_creator.compute_canonical_points_rott(smplx_inputs, self._xyz_canonical_dis, self._verts_canonical_dis)

        gs_points_homo = torch.cat((gs_points, torch.ones_like(gs_points[..., :1])), dim=-1)
        self._xyz = torch.bmm(gs_rott.reshape(-1,3,4), gs_points_homo.reshape(-1,4,1)).reshape(batch_size, -1, 3) ### (b, pcs, 3)
        self._rotation = self._rotation_base.unsqueeze(0).expand(batch_size, -1, -1)
        self.scaling = self._scaling.repeat(1, 3)

        verts_homo = torch.cat((verts, torch.ones_like(verts[..., :1])), dim=-1)
        verts_current = torch.bmm(verts_rott.reshape(-1,3,4), verts_homo.reshape(-1,4,1)).reshape(batch_size, -1, 3) ### (b, v_num, 3)
        verts_cam = torch.bmm(verts_current, cam_inputs['R'].permute(0,2,1)) + cam_inputs['t'].unsqueeze(1)

        if not with_mesh_rendering:
            return 
        
        cam_para = cam_inputs['cam_para']
        if not in_train: ### only return mesh vis
            mesh_vis = self.mesh_renderer.forward_visualization_geo(verts_cam, self.geo_creator.tris[None, ...].int().expand(batch_size, -1, -1), cam_para.expand(batch_size, -1), self.img_size)
            return mesh_vis
        
        else: ### return differentiable mesh mask, and mesh normal consistent loss
            diff_mask = self.mesh_renderer.forward_differentiable_mask(verts_cam, self.geo_creator.tris[None, ...].int().expand(batch_size, -1, -1), cam_para.expand(batch_size, -1), self.img_size)
            meshes_consistent = Meshes(verts=verts[0:1], faces=self.geo_creator.tris[None, ...].expand(1, -1, -1))
            loss_normal_consistent = mesh_normal_consistency(meshes_consistent)
            return diff_mask, loss_normal_consistent

    @property
    def get_scaling(self):
        # return self.scaling_activation(self._scaling)
        return self.scaling_activation(self.scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) ### batched
    
    @property
    def get_xyz(self):
        return self._xyz ### batched, (b, pcs, 3)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) #### (pcs, 1, 3)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.geo_creator.gaussian_pnum, 1), device=self.device)
        self.denom = torch.zeros((self.geo_creator.gaussian_pnum, 1), device=self.device)
        optable_params = [
            {'params': [self._xyz_canonical_dis], 'lr': training_args.position_lr_init, "name": "xyz"},
            {'params': [self._verts_canonical_dis], 'lr': training_args.position_lr_init, "name": "vert_xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._body_pose_dis], 'lr': training_args.bodypose_lr * 0., "name": "bodypose"},
            # {'params': [self._rotation_base], 'lr': training_args.rotation_lr * 0., "name": "rotation"}, ### make rotation parallel to underlying mesh, do not optimize
        ]
        # for param_group in optable_params:
        #     param_group['lr'] *= 0.1
        self.optimizer = torch.optim.Adam(optable_params, lr=0.0, eps=1e-15)

    def decay_lr(self, decay_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay_rate

    def forward(self, x):
        pass
    