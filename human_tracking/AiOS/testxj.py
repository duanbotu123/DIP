import pickle
import torch
import numpy as np
import ipdb

pkl_path = "/home/juyonggroup/xiangjun/github/AiOS/demo/actor2_out/actor2_out/predictions/000000_personId_0.pkl"
# with open(pkl_path, 'rb') as f:  # 使用 'rb' 模式打开文件
f = open(pkl_path, 'rb')
data = pickle.load(f)
params = data['params'] 
# ['transl', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'reye_pose', 'leye_pose', 'jaw_pose', 'expression', 'betas']
ipdb.set_trace()