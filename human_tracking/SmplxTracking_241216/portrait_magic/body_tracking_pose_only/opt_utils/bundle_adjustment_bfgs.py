# simple pytorch implementation of bundle adjustment

import torch
import torch.nn as nn
import numpy as np
from ...render_utils.geometry_utils import solve_pnp_cv, proj_pts, rot_trans_pts, euler2rot
from ..loss_utils import cal_euclidean_distance
from torchmin import minimize


def set_tensors_grad(tensor_list, requires_grad):
    for tensor in tensor_list:
        tensor.requires_grad = requires_grad


class PoseEstimator():
    def __init__(self, pts3d, pts2d, cam_para, init_rots=None, init_transs=None, confidence = None):
        if pts3d.shape[0] != pts2d.shape[0]:
            pts3d = pts3d.expand(pts2d.shape[0], -1, -1)
        self.pts3d, self.pts2d, self.cam_para = pts3d, pts2d, cam_para
        self.b = pts2d.shape[0]
        self.init_rots = init_rots
        self.init_transs = init_transs
        self.confidence = confidence

    def cal_error(self, pose_params):
        euler_angles = pose_params[:, :3]
        transs = pose_params[:, 3:]
        rots = euler2rot(euler_angles)
        if self.init_rots is not None:
            rots = torch.bmm(rots, self.init_rots)
            transs = transs + self.init_transs

        proj_pts_ = proj_pts(rot_trans_pts(self.pts3d, rots, transs), self.cam_para)
        loss_ldmks = torch.mean(cal_euclidean_distance(proj_pts_, self.pts2d, self.confidence))
        min_z, max_z = -.5, -3.0
        loss_z_dis = torch.mean(torch.relu(
            transs[:, 2] - min_z) + torch.relu(max_z - transs[:, 2]))
        return loss_z_dis*1e1 + loss_ldmks

def refine_rt(pts3d, ldmks, cam_para, init_rots, init_trans, confidence = None):
    '''
    pts3d: (b, n, 3), ldmks: (b, n, 2), cam_para: (b, 4), init_rots: (b, 3, 3), init_trans: (b, 1, 3), confidence: (b, n, 1) -> (b, 3, 3), (b, 1, 3)
    '''
    b, n, _ = ldmks.shape
    proj_pts_ = proj_pts(rot_trans_pts(pts3d, init_rots, init_trans.squeeze(1)), cam_para)
    init_dis = torch.mean(cal_euclidean_distance(proj_pts_, ldmks, confidence=confidence)).item()
    pose_params = ldmks.new_zeros(b, 6)
    set_tensors_grad([pose_params], True)
    iter_nums = 50
    funcer = PoseEstimator(pts3d, ldmks, cam_para, init_rots, init_trans.squeeze(1), confidence)
    res = minimize(funcer.cal_error, pose_params, method='bfgs', max_iter=iter_nums, disp=0, tol=1e-10,
                   options=dict(line_search='strong-wolfe', lr=1.))
    pose_params = res.x.clone()
    euler_angles = pose_params[:, :3]
    trans = pose_params[:, 3:].unsqueeze(1)
    rots = euler2rot(euler_angles)
    rots = torch.bmm(rots, init_rots)
    trans = trans + init_trans
    proj_pts_ = proj_pts(rot_trans_pts(pts3d, rots, trans.squeeze(1)), cam_para)
    final_dis = torch.mean(cal_euclidean_distance(proj_pts_, ldmks, confidence=confidence)).item()
    print('refine rt', init_dis, '->', final_dis, 'with distance', torch.mean(trans[:, :, 2]).item())

    return rots, trans

def cal_dis_given_focal(ldmks, img_size, focal, pts3d):
    '''
    ldmks: (b, n, 2), img_size: [h, w] -> [fx, fy, cx, cy]
    '''

    with_pnp = True

    b, n, _ = ldmks.shape
    h, w = img_size

    cam_para = torch.tensor((focal, focal, w/2., h/2.),
                            dtype=torch.float32, device=ldmks.device).reshape(1, 4)

    pose_params = ldmks.new_zeros(b, 6)

    if not with_pnp:
        pose_params[:, 5] = -1.5

    if with_pnp:
        init_rots, init_transs = solve_pnp_cv(
            pts3d.expand(b, -1, -1), ldmks, cam_para.expand(b, -1))
        init_transs = init_transs.squeeze(1)
        iter_nums = 100
    else:
        init_rots, init_transs = None, None
        iter_nums = 300

    set_tensors_grad([pose_params], True)

    funcer = PoseEstimator(pts3d, ldmks, cam_para, init_rots, init_transs)
    res = minimize(funcer.cal_error, pose_params, method='bfgs', max_iter=iter_nums, disp=0, tol=1e-10,
                   options=dict(line_search='strong-wolfe', lr=1.))
    pose_params = res.x.clone()

    euler_angles = pose_params[:, :3]
    transs = pose_params[:, 3:]

    rots = euler2rot(euler_angles)
    if with_pnp:
        rots = torch.bmm(rots, init_rots)
        transs = transs + init_transs
    proj_pts_ = proj_pts(rot_trans_pts(
        pts3d.expand(b, -1, -1), rots, transs), cam_para)
    loss_ldmks = torch.mean(cal_euclidean_distance(proj_pts_, ldmks))

    return loss_ldmks.item(), torch.mean(transs[:, 2]).item()


def cal_best_focal_in_range(ldmks, img_size, start_focal, end_focal, step, pts3d=None):
    arg_dis = 1e10
    arg_focal = None
    arg_z = None
    for focal in range(start_focal, end_focal, step):
        dis, trans_z = cal_dis_given_focal(ldmks, img_size, focal, pts3d)

        if dis < arg_dis:
            arg_dis = dis
            arg_focal = focal
            arg_z = trans_z
    return arg_focal, arg_dis, arg_z


def cal_best_focal(ldmks, img_size, pts3d=None):
    '''
    ldmks: (b, n, 2), img_size: [h, w] -> [fx, fy, cx, cy]
    '''

    start_focal, end_focal = 1200, 2300
    steps = [400, 160, 64]

    arg_arg_dis = 1e10
    arg_arg_focal = None
    arg_arg_z = None

    for i in range(len(steps)):
        arg_focal, arg_dis, arg_z = cal_best_focal_in_range(
            ldmks, img_size, start_focal, end_focal, steps[i], pts3d)
        # print(arg_focal, arg_dis, arg_z)
        if arg_dis < arg_arg_dis:
            arg_arg_dis = arg_dis
            arg_arg_focal = float(arg_focal)
            arg_arg_z = arg_z
        if i < len(steps) - 1:
            start_focal = arg_focal - (steps[i]-steps[i+1])
            end_focal = arg_focal + (steps[i]-steps[i+1])
    return arg_arg_focal, arg_arg_dis, arg_arg_z
