#### simple pytorch implementation of bundle adjustment

import torch
import numpy as np
from ...render_utils.geometry_utils import forward_rott_proj
from ..loss_utils import cal_euclidean_distance

def set_tensors_grad(tensor_list, requires_grad):
    for tensor in tensor_list:
        tensor.requires_grad = requires_grad

def cal_dis_given_focal(ldmks, img_size, focal, pts3d):
    '''
    ldmks: (b, n, 2), img_size: [h, w] -> [fx, fy, cx, cy]
    '''

    b, n, _ = ldmks.shape
    h, w = img_size
    
    if pts3d is None:
        pts_3d = ldmks.new_zeros(1, n, 3)
    else:
        pts_3d = pts3d.clone()
        
    euler_angles = ldmks.new_zeros(b, 3)
    transs = ldmks.new_zeros(b, 3)
    focal_len = ldmks.new_zeros(1, 1)
    principle = ldmks.new_zeros(1, 2)
    focal_len[0, 0], principle[0, 0], principle[0, 1], transs[:, 2] = focal, w/2., h/2., -2.


    set_tensors_grad([euler_angles, transs], True)
    optimizer_extrinsic = None
    optimizer_extrinsic = torch.optim.Adam([transs, euler_angles], lr=.1)
    
    iter_num = 5000
    for iter in range(iter_num):
        cam_para = torch.cat((focal_len, focal_len, principle), dim=1)
        proj_pts = forward_rott_proj(pts_3d.expand(b, -1, -1), euler_angles, transs, cam_para.expand(b, -1))
        loss_ldmks = torch.mean(cal_euclidean_distance(proj_pts, ldmks))

        min_z, max_z = -.7, -3.0
        loss_z_dis = torch.mean(torch.relu(transs[:, 2] - min_z) + torch.relu(max_z - transs[:, 2]))

        optimizer_extrinsic.zero_grad()

        (loss_z_dis*1e1 + loss_ldmks).backward()
        optimizer_extrinsic.step()

        # if iter % 100 == 0:
        #     print('iter: {}, loss: {}, cam: {}'.format(iter, loss_ldmks.item(), cam_para.detach().cpu().numpy()))
        if iter % (iter_num//5) == 0 and iter>0:
            for param_group in optimizer_extrinsic.param_groups:
                param_group['lr'] *= .2

    
    return loss_ldmks.item(), torch.mean(transs[:, 2]).item()

def cal_best_focal_in_range(ldmks, img_size, start_focal, end_focal, step, pts3d = None):
    arg_dis = 1e10
    arg_focal = None
    arg_z = None
    for focal in range(start_focal, end_focal, step):
        dis, trans_z = cal_dis_given_focal(ldmks, img_size, focal, pts3d)
        
        if dis<arg_dis:
            arg_dis = dis
            arg_focal = focal
            arg_z = trans_z
    return arg_focal, arg_dis, arg_z

def cal_best_focal(ldmks, img_size, pts3d = None):
    '''
    ldmks: (b, n, 2), img_size: [h, w] -> [fx, fy, cx, cy]
    '''
    
    start_focal, end_focal = 1200, 2300
    steps = [400, 160, 64]
    
    arg_arg_dis = 1e10
    arg_arg_focal = None

    for i in range(len(steps)):
        arg_focal, arg_dis, arg_z = cal_best_focal_in_range(ldmks, img_size, start_focal, end_focal, steps[i], pts3d)
        print(arg_focal, arg_dis, arg_z)
        if arg_dis < arg_arg_dis:
            arg_arg_dis = arg_dis
            arg_arg_focal = float(arg_focal)
        if i < len(steps) - 1:
            start_focal = arg_focal - (steps[i]-steps[i+1])
            end_focal = arg_focal + (steps[i]-steps[i+1])
    return arg_arg_focal

