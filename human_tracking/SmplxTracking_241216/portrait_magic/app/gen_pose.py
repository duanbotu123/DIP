from portrait_magic.dmm_models import SMPLX, smplx_model_path
import torch
import numpy as np
import os

shape_dim, expr_dim = 10, 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=expr_dim, num_betas=shape_dim, use_pca = False).cuda().eval()
print(smplx_model.joint_names)
for param in smplx_model.parameters():
    param.requires_grad = False

# Initialize all SMPLX parameters to zeros
paras_dict = {
    'shape': torch.zeros(1, shape_dim, device=device),
    'body_pose': torch.zeros(1, 21*3, device=device),
    'lhand_pose': torch.zeros(1, 15*3, device=device),
    'rhand_pose': torch.zeros(1, 15*3, device=device),
    'jaw_pose': torch.zeros(1, 3, device=device),
    'expr': torch.zeros(1, expr_dim, device=device),
    'leye_pose': torch.zeros(1, 3, device=device),
    'reye_pose': torch.zeros(1, 3, device=device),
    'global_orient': torch.zeros(1, 3, device=device),
    'transl': torch.zeros(1, 3, device=device)
}

smplx_out = smplx_model.forward(betas = paras_dict['shape'], body_pose = paras_dict['body_pose'], left_hand_pose = paras_dict['lhand_pose'], right_hand_pose = paras_dict['rhand_pose'], jaw_pose = paras_dict['jaw_pose'], expression = paras_dict['expr'], with_iris_return=False, leye_pose=paras_dict['leye_pose'], reye_pose=paras_dict['reye_pose'], global_orient=paras_dict['global_orient'], transl=paras_dict['transl'])
joints = smplx_out.joints

print(f'joints shape: {joints.shape}')
print(f'joints: {joints}')