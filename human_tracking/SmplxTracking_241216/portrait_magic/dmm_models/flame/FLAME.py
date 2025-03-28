"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
"""
# Modified from smplx code for FLAME
import torch
import torch.nn as nn
import numpy as np
import pickle
from .lbs import lbs
from pytorch3d.io import load_obj, save_obj
from ..smplx import smplx_model_path
import os

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, config, optimize_basis=False):
        super(FLAME, self).__init__()
        with open(config.flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.optimize_basis = optimize_basis
        self.cfg = config
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        self.n_vertices = self.v_template.shape[0]
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.n_shape], shapedirs[:, :, 300:300+config.n_exp]], 2)

        if optimize_basis:
            self.register_parameter('shapedirs', torch.nn.Parameter(shapedirs))
        else:
            self.register_buffer('shapedirs', shapedirs)

        self.n_shape = config.n_shape
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        # 
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long();
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose, requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(config.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long().permute(1, 0))
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype).permute(1, 0, 2))

        neck_kin_chain = [];
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))


        _, faces, aux = load_obj(config.template_mesh_file)
        uv_coords = aux.verts_uvs[None, ...].float()
        uv_coords[..., 1] = 1. - uv_coords[..., 1]
        tris_uv = faces.textures_idx[None, ...].int()
        tris = faces.verts_idx[None, ...].int()

        self.register_buffer('uv_coords', uv_coords)
        self.register_buffer('tris_uv', tris_uv)
        self.register_buffer('tris', tris)

        pts96_fid = np.loadtxt(config.pts96_fid_path, dtype=np.int64)
        pts96_bary_coords = np.loadtxt(config.pts96_bary_path, dtype=np.float32)
        pts96_bary_coords = np.concatenate((pts96_bary_coords, 1.-np.sum(pts96_bary_coords,axis=1,keepdims=True)), axis=1)

        wflw_to_pts51 = np.loadtxt(os.path.join(smplx_model_path, 'wflw_to_pts51.txt'), dtype=np.int64)
        pts51_bary_coords = pts96_bary_coords[wflw_to_pts51]
        pts51_fid = pts96_fid[wflw_to_pts51]
        pts68_bary_coords = np.concatenate((pts51_bary_coords[:17], pts51_bary_coords), axis=0)
        pts68_fid = np.concatenate((pts51_fid[:17], pts51_fid), axis=0)

        self.register_buffer('pts68_fid', torch.as_tensor(pts68_fid))
        self.register_buffer('pts68_bary_coords', torch.as_tensor(pts68_bary_coords))

        ## for tex
        self.n_tex = config.n_tex
        tex_space = np.load(config.tex_space_path)
        texture_mean = tex_space['MU'].reshape(1, -1)
        texture_basis = tex_space['PC'].reshape(-1, 199)[:, :self.n_tex]
        texture_mean = torch.from_numpy(texture_mean).float()[None,...]
        texture_basis = torch.from_numpy(texture_basis).float()[None,...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward_geo(self, shape_params, expression_params, jaw_pose_params = None, pose_params=None, eye_pose_params=None, neck_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if neck_pose_params is None:
            neck_pose_params = self.neck_pose.expand(batch_size, -1)
        if jaw_pose_params is None:
            jaw_pose_params = self.neck_pose.expand(batch_size, -1)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([pose_params[:, :3], neck_pose_params, jaw_pose_params, eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)
        return vertices
    
    def forward_tex(self, texcode):
        texture = self.texture_mean + (self.texture_basis*texcode[:,None,:]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0,3,1,2)
        texture = torch.nn.functional.interpolate(texture, [256,256])
        texture = texture.permute(0,2,3,1)[:, :, :, [2,1,0]]
        return texture
    

    def get_3dlandmarks(self, geometry, faces_idx, bary_coords):
        ## geometry: (b, n, 3), faces_idx: (L), bary_coords: (L, 3)
        ## return: (b, L, 3)
        b, L = geometry.shape[0], faces_idx.shape[0]
        fv_idx = self.faces_tensor[faces_idx] # (l, 3)
        fv_lms = geometry[:, fv_idx.reshape(-1), :].reshape(b, L, 3, 3)
        fv_coords = bary_coords.reshape(1, L, 3, 1)
        return torch.sum(fv_lms*fv_coords, dim=2)
    
    def save_mesh(self, file, geometry):
        save_obj(file, geometry.squeeze(0), self.faces_tensor)