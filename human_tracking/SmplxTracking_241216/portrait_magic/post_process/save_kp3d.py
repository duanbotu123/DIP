import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import re
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm
from ..dmm_models import FLAME, flame_config, SMPLX, smplx_model_path
from ..dmm_models.utils.keypoints_mapping import convert_kps

device = 'cuda:0'
shape_dim, expr_dim = 10, 10
smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=expr_dim, num_betas = shape_dim, use_face_contour= True, use_pca = False).cuda().eval()

for param in smplx_model.parameters():
    param.requires_grad = False

# load npz
npz_path = ''
data = np.load()

# 每帧保存一个3d关键点文件
smplx_out = smplx_model.forward(betas = paras_dict['shape'][:1], body_pose = paras_dict['body_pose'], left_hand_pose = paras_dict['lhand_pose'], right_hand_pose = paras_dict['rhand_pose'], jaw_pose = paras_dict['jaw_pose'], expression = paras_dict['expr'], with_iris_return=False, leye_pose=paras_dict['leye_pose'], reye_pose=paras_dict['reye_pose'], global_orient=paras_dict['global_orient'], transl=paras_dict['transl'])
pts3d, confidence_ = convert_kps(smplx_out.joints)
# 每帧保存一个smplx


