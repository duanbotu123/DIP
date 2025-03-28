#--------------------------------------
#功能: 存储默认参数
#--------------------------------------

import os
from yacs.config import CfgNode as CN

flame_config = CN()
flame_config.n_shape = 300
flame_config.n_exp = 100
flame_config.n_tex = 100

file_dir_path = os.path.dirname(os.path.realpath(__file__))
flame_config.data_dir = os.path.join(file_dir_path, 'FLAME2020')
flame_config.flame_model_path = os.path.join(flame_config.data_dir, 'generic_model.pkl')
flame_config.flame_lmk_embedding_path = os.path.join(flame_config.data_dir, 'landmark_embedding.npy')
flame_config.template_mesh_file = os.path.join(flame_config.data_dir, 'head_template_mesh.obj')
flame_config.tex_space_path = os.path.join(flame_config.data_dir, 'FLAME_albedo_from_BFM.npz')
flame_config.pts68_fid_path = os.path.join(flame_config.data_dir, 'pts68_tri_ids.txt')
flame_config.pts68_bary_path = os.path.join(flame_config.data_dir, 'pts68_bary_coords.txt')

model_config = CN()
model_config.sh_degree = 0

pipeline_config = CN()
pipeline_config.convert_SHs_python = False
pipeline_config.compute_cov3D_python = False
pipeline_config.debug = False

optim_config = CN()
optim_config.iterations = 900_000
optim_config.position_lr_init = 0.00016
optim_config.position_lr_final = optim_config.position_lr_init
optim_config.position_lr_delay_mult = 0.01
optim_config.position_lr_max_steps = 30_000
optim_config.feature_lr = 0.0025
optim_config.pose_lr = 0.00001
optim_config.bodypose_lr = 0.00001
optim_config.pose2rott_lr = 0.000001
optim_config.opacity_lr = 0.05
optim_config.scaling_lr = 0.005
optim_config.rotation_lr = 0.001
optim_config.percent_dense = 0.01
optim_config.lambda_dssim = 0.2
optim_config.densification_interval = 100
optim_config.opacity_reset_interval = 3000
optim_config.densify_from_iter = 500
optim_config.densify_until_iter = 15_000
optim_config.densify_grad_threshold = 0.0002
