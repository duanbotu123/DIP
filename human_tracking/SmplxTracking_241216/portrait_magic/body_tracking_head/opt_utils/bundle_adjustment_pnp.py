#### simple pytorch implementation of bundle adjustment

import torch
import numpy as np
from ...render_utils.geometry_utils import forward_rott_proj, solve_pnp_mix, euler2rot, rot_trans_pts, proj_pts
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
    
    cam_para = torch.tensor((focal, focal, w/2., h/2.), dtype=torch.float32, device = ldmks.device).reshape(1, 4)

    rots_init, trans_init = solve_pnp_mix(pts3d.expand(b, -1, -1), ldmks, cam_para.expand(b, -1))
    
    euler_angles = ldmks.new_zeros(b, 3)
    transs = ldmks.new_zeros(b, 1, 3)

    set_tensors_grad([euler_angles, transs], True)
    optimizer_extrinsic = None
    optimizer_extrinsic = torch.optim.Adam([transs, euler_angles], lr=1e-1)

    iter_num = 5000
    for iter in range(iter_num):
        rots = torch.bmm(euler2rot(euler_angles), rots_init)
        trans = transs + trans_init
        cam_pts = torch.bmm(pts3d.expand(b, -1, -1), rots.permute(0,2,1)) + trans
        proj_pts_ = proj_pts(cam_pts, cam_para)
        loss_ldmks = torch.mean(cal_euclidean_distance(proj_pts_, ldmks))

        min_z, max_z = -.7, -3.0
        loss_z_dis = torch.mean(torch.relu(trans[..., 2] - min_z) + torch.relu(max_z - trans[..., 2]))

        optimizer_extrinsic.zero_grad()
        
        (loss_z_dis*1e1 * 0. + loss_ldmks).backward()
        optimizer_extrinsic.step()

        # if iter % 100 == 0:
        #     print('iter: {}, loss: {}, cam: {}'.format(iter, loss_ldmks.item(), cam_para.detach().cpu().numpy()))
        if iter % (iter_num//5) == 0 and iter>0:
            for param_group in optimizer_extrinsic.param_groups:
                param_group['lr'] *= .15
    
    return loss_ldmks.item(), torch.mean(trans[..., 2]).item()

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

