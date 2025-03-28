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
flame_config.pts96_fid_path = os.path.join(flame_config.data_dir, 'wflw_tri_ids.txt')
flame_config.pts96_bary_path = os.path.join(flame_config.data_dir, 'wflw_bary_coords.txt')
