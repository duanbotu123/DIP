import torch
import torch.nn as nn
import numpy as np
import os
from .geometry import ADGaussianGeo
from .camera import Camera
from simple_knn._C import distCUDA2
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, so3_exp_map
from ...render_utils.geometry_utils import strip_symmetric, build_scaling_rotation, focal2fov, proj_pts, pts2obj

class Pose2Rott(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pose2Rott, self).__init__()

        # self.fc1 = nn.Linear(in_dim, 256)
        # self.fc2 = nn.Linear(256, out_dim)

        # self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(in_dim, out_dim)

        self.activation = nn.LeakyReLU(0.01)

        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, pose):
        # return self.fc2(self.activation(self.fc1(pose)))
        return self.fc2(pose)


class GaussianScene(nn.Module):
    def __init__(self, data_dir, img_size, sh_degree = 3, device = 'cuda:0'):
        super(GaussianScene, self).__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.device = device

        self.smplx_track_keys = ['body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'expr', 'leye_pose', 'reye_pose', 'cam_trans', 'cam_angle', 'shape', 'cam_para']
        self.load_smplx_track()
        canonical_hair_centers = torch.load(os.path.join(data_dir, 'train_imgs/hair_canonical_pts.pt')).to(self.device)
        canonical_hair_centers = None
        self.geo_creator = ADGaussianGeo(device, self.shape, canonical_hair_centers)
        self.lap_indices = self.geo_creator.lap_indices
        self.init_attributes()
        self.setup_functions()

    def load_smplx_track(self):
        smplx_track_dict = torch.load(os.path.join(self.data_dir, 'body_track', 'smplx_track.pth'))
        for key in self.smplx_track_keys:
            self.__setattr__(key, torch.from_numpy(smplx_track_dict[key]).to(self.device))
        self.cameras = []
        rots = so3_exp_map(self.cam_angle)
        transs = self.cam_trans.clone()
        cam_para = self.cam_para[0]
        rots[:, :, 1:] *= -1
        transs[:, 1:] *= -1
        fx, fy = float(cam_para[0]), float(cam_para[1])
        print(cam_para, self.img_size)
        FovX = focal2fov(fx, self.img_size[1])
        FovY = focal2fov(fy, self.img_size[0])
        for i in range(self.cam_angle.shape[0]):
            camera_indiv = Camera(rots[i].cpu().numpy(), transs[i].cpu().numpy(), FovX, FovY, self.img_size, device=self.device)
            self.cameras.append(camera_indiv)

    def load_driven_track(self, smplx_track_file):
        smplx_track_dict = torch.load(smplx_track_file)
        for key in self.smplx_track_keys:
            self.__setattr__('driven_' + key, torch.from_numpy(smplx_track_dict[key]).to(self.device))
        self.driven_cameras = []
        rots = so3_exp_map(self.driven_cam_angle)
        transs = self.driven_cam_trans.clone()
        cam_para = self.driven_cam_para[0]
        rots[:, :, 1:] *= -1
        transs[:, 1:] *= -1
        fx, fy = float(cam_para[0]), float(cam_para[1])
        driven_img_size = (int(cam_para[3]*2+.5), int(cam_para[2]*2+.5))
        print(cam_para, driven_img_size)
        FovX = focal2fov(fx, driven_img_size[1])
        FovY = focal2fov(fy, driven_img_size[0])
        for i in range(self.driven_cam_angle.shape[0]):
            camera_indiv = Camera(rots[i].cpu().numpy(), transs[i].cpu().numpy(), FovX, FovY, driven_img_size, device=self.device)
            self.driven_cameras.append(camera_indiv)
        return self.driven_cam_angle.shape[0]

    def cal_attr_lap_loss(self, attr):
        attr_dis = attr[self.lap_indices[:, 0]] - torch.mean(attr[self.lap_indices[:, 1:]], dim=1)
        return torch.mean(attr_dis**2)

    def cal_dis_loss(self):
        return torch.mean(self._xyz_canonical_dis**2)
    
    def cal_scale_loss(self):
        arg_max = 6e-3
        return torch.mean(torch.relu(self.scaling_activation(self._scaling)[self.scaling_activation(self._scaling)>arg_max] - arg_max))


    def init_attributes(self):
        pc_num = self.geo_creator.gaussian_pnum
        print('point number', pc_num)
        features = torch.zeros((pc_num, 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        pts2obj(self.geo_creator.canonical_points, 'canonical.obj')
        dist2 = torch.clamp_min(distCUDA2(self.geo_creator.canonical_points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        print('init scale', torch.exp(torch.min(scales)), torch.exp(torch.max(scales)), torch.exp(torch.mean(scales)))
        rots = torch.zeros((pc_num, 4), device=self.device)
        rots[:, 0] = 1
        opacities = torch.logit(0.1 * torch.ones((pc_num, 1), dtype=torch.float, device=self.device))

        self._body_pose_delta = nn.Parameter(torch.zeros_like(self.body_pose).requires_grad_(True))
        self._lhand_pose_delta = nn.Parameter(torch.zeros_like(self.lhand_pose).requires_grad_(True))
        self._rhand_pose_delta = nn.Parameter(torch.zeros_like(self.rhand_pose).requires_grad_(True))

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._xyz_canonical_dis = nn.Parameter(torch.zeros_like(self.geo_creator.canonical_points).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation_base = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((pc_num), device=self.device)

        self.pose2rotter = Pose2Rott(self.body_pose.shape[1] + self.lhand_pose.shape[1] + self.rhand_pose.shape[1], self.geo_creator.skin_part_pnum*4*3).to(self.device)

    def cal_lap_loss(self):
        return self.cal_attr_lap_loss(self._features_dc) * 1e-3 + self.cal_attr_lap_loss(self._xyz_canonical_dis) + self.cal_attr_lap_loss(self._scaling) + self.cal_attr_lap_loss(self._opacity)

    def save_updated_canonical(self, save_path):
        pts2obj(self.geo_creator.canonical_points + self._xyz_canonical_dis, save_path)

    def capture(self):
        return (
            self.active_sh_degree, self._xyz_canonical_dis, self._features_dc, self._features_rest, self._scaling, self._rotation_base, self._opacity, self.optimizer.state_dict(), self._body_pose_delta, self._lhand_pose_delta, self._rhand_pose_delta, self.pose2rotter.state_dict(),
        )
    
    def restore(self, model_dict):
        (self.active_sh_degree, _xyz_canonical_dis, _features_dc, _features_rest, _scaling, _rotation_base, _opacity, opt_dict, _body_pose_delta, _lhand_pose_delta, _rhand_pose_delta, pose2rott_dict) = model_dict
        self._xyz_canonical_dis.data = _xyz_canonical_dis.clone()
        self._features_dc.data = _features_dc.clone()
        self._features_rest.data = _features_rest.clone()
        self._scaling.data = _scaling.clone()
        self._rotation_base.data = _rotation_base.clone()
        self._opacity.data = _opacity.clone()
        self._body_pose_delta.data = _body_pose_delta.clone()
        self._lhand_pose_delta.data = _lhand_pose_delta.clone()
        self._rhand_pose_delta.data = _rhand_pose_delta.clone()
        self.pose2rotter.load_state_dict(pose2rott_dict)
        # self.optimizer.load_state_dict(opt_dict)

        print('load scale', torch.exp(torch.min(_scaling)), torch.exp(torch.max(_scaling)), torch.exp(torch.mean(_scaling)))

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

    def update_xyz_rot_driven(self, idx):
        frame_smplx_dict = {}
        frame_idx = idx%(self.body_pose.shape[0])
        for key in self.smplx_track_keys[0:7]:
            frame_smplx_dict[key] = getattr(self, key)[frame_idx:frame_idx+1]
        frame_smplx_dict['body_pose'][:, 11*3:12*3] = self.driven_body_pose[idx:idx+1][:, 11*3:12*3]
        frame_smplx_dict['body_pose'][:, 14*3:15*3] = self.driven_body_pose[idx:idx+1][:, 14*3:15*3]
        pcs_rott, _ = self.geo_creator.compute_canonical_points_rott(frame_smplx_dict) #### (1, pc_num, 3, 4)
        pcs_canonical_homo = torch.cat((self.geo_creator.canonical_points.clone().detach() + self._xyz_canonical_dis, torch.ones_like(self._xyz_canonical_dis[..., :1])), dim=-1)
        self._xyz = torch.bmm(pcs_rott[0], pcs_canonical_homo.unsqueeze(-1)).squeeze(-1)
        self._rotation = quaternion_multiply(matrix_to_quaternion(pcs_rott[0, :, :3, :3]), self._rotation_base)

    def cal_pose_loss(self, frame_ids):
        return torch.mean(torch.abs(self._body_pose_delta[frame_ids].clone())) + torch.mean(torch.abs(self._lhand_pose_delta[frame_ids].clone())) + torch.mean(torch.abs(self._rhand_pose_delta[frame_ids].clone()))
    
    def cal_rott_loss(self):
        return torch.mean(torch.abs(self.delta_rott)) + self.cal_attr_lap_loss(self.delta_rott) * 1e2

    def update_xyz_rot(self, idx, with_debug_proj = False, inverse_frame_expr = False, with_pose_delta = False, with_pose_disturb = False):
        frame_smplx_dict = {}
        for key in self.smplx_track_keys[0:7]:
            if (key == 'expr' or key == 'jaw_pose') and inverse_frame_expr:
                frames_num = getattr(self, key).shape[0]
                frame_smplx_dict[key] = getattr(self, key)[frames_num-1-idx:frames_num-idx].clone()
            else:
                frame_smplx_dict[key] = getattr(self, key)[idx:idx+1].clone()

        if with_pose_delta:
            frame_smplx_dict['body_pose'] += self._body_pose_delta[idx:idx+1].clone()
            frame_smplx_dict['lhand_pose'] += self._lhand_pose_delta[idx:idx+1].clone()
            frame_smplx_dict['rhand_pose'] += self._rhand_pose_delta[idx:idx+1].clone()

        if with_pose_disturb:
            frame_smplx_dict['body_pose'] += torch.randn_like(frame_smplx_dict['body_pose']) * 1e-3
            frame_smplx_dict['lhand_pose'] += torch.randn_like(frame_smplx_dict['lhand_pose']) * 1e-3
            frame_smplx_dict['rhand_pose'] += torch.randn_like(frame_smplx_dict['rhand_pose']) * 1e-3

        pcs_rott, _ = self.geo_creator.compute_canonical_points_rott(frame_smplx_dict) #### (1, pc_num, 3, 4)

        skin_part_num = self.geo_creator.skin_part_pnum

        self.delta_rott = self.pose2rotter(torch.cat((frame_smplx_dict['body_pose'], frame_smplx_dict['lhand_pose'], frame_smplx_dict['rhand_pose']), dim=-1).detach()).reshape(-1, 3, 4)
        pcs_rott[0, :skin_part_num] = pcs_rott[0, :skin_part_num] + self.delta_rott

        pcs_canonical_homo = torch.cat((self.geo_creator.canonical_points.clone().detach() + self._xyz_canonical_dis, torch.ones_like(self._xyz_canonical_dis[..., :1])), dim=-1)
        self._xyz = torch.bmm(pcs_rott[0], pcs_canonical_homo.unsqueeze(-1)).squeeze(-1)
        self._rotation = quaternion_multiply(matrix_to_quaternion(pcs_rott[0, :, :3, :3]), self._rotation_base)
        if with_debug_proj:
            rot = so3_exp_map(self.cam_angle[idx:idx+1])
            trans = self.cam_trans[idx:idx+1].clone()
            pts_cam = torch.bmm(self._xyz.unsqueeze(0), rot.permute(0,2,1)) + trans.unsqueeze(1)
            proj_pts_ = proj_pts(pts_cam, self.cam_para)
            return (proj_pts_[0] + .5).long()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
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
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation_base], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._body_pose_delta, self._lhand_pose_delta, self._rhand_pose_delta], 'lr': training_args.pose_lr, "name": "pose"},
            {'params': list(self.pose2rotter.parameters()), 'lr': training_args.pose2rott_lr, "name": "pose2rott"},
        ]
        self.optimizer = torch.optim.Adam(optable_params, lr=0.0, eps=1e-15)

    def forward(self, x):
        pass
    