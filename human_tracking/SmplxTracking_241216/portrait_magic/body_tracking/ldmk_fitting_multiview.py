from email.mime import image
from enum import Flag
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import re
import cv2
from onnx import save
from regex import F, P
import torch
import numpy as np
import pickle
from tqdm import tqdm
from copy import deepcopy
from natsort import natsorted
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import so3_exp_map, so3_log_map
from ..render_utils import MeshRenderer_Cuda, enlarge_human_masks
from ..third_parties.MICA import MICAModel
from ..render_utils.geometry_utils import solve_pnp_cv, proj_pts, proj_pts_mv
from .opt_utils.bundle_adjustment_bfgs import cal_best_focal
from .loss_utils import cal_euclidean_distance, cal_l2_loss, cal_smooth_loss, cal_euclidean_distance_mv
from ..dmm_models import FLAME, flame_config, SMPLX, smplx_model_path
from ..dmm_models.utils.keypoints_mapping import convert_kps
from ..utils_funcs import VideoWriter
from ..feature_tracking_all import TrackingLosser
from ..visualization.visualizer import Renderer, merge, load_image, images_to_video

class LDMK_Fitting_multiview():
    def __init__(self):
        self.device = 'cuda:0'
        self.shape_dim, self.expr_dim = 10, 10
        self.smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=self.expr_dim, num_betas = self.shape_dim, use_face_contour= True, use_pca = False).cuda().eval()

        for param in self.smplx_model.parameters():
            param.requires_grad = False

        _, faces, aux = load_obj(os.path.join(smplx_model_path, 'smplx_uv.obj'))
        self.uv_coords = aux.verts_uvs[None, ...].float().cuda()
        self.uv_coords[..., 1] = 1. - self.uv_coords[..., 1]
        self.tris_uv = faces.textures_idx[None, ...].cuda().int()
        self.tris = faces.verts_idx[None, ...].cuda().int()

        self.mesh_renderer = MeshRenderer_Cuda().cuda()
        self.confidence_min = .5

        self.upper_body_inds = torch.tensor([2, 5, 8, 11, 12, 13, 14], dtype=torch.int64)
        self.lower_body_inds = torch.tensor([0, 1, 3, 4, 6, 7, 9, 10, 15, 16, 17, 18, 19, 20], dtype=torch.int64)

        direct_concat_inds = torch.cat((self.upper_body_inds, self.lower_body_inds), dim=0)
        self.get_body_inds = torch.argsort(direct_concat_inds)

        self.paras_keys = ['upper_body_pose', 'lower_body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'shape', 'expr', 'global_orient', 'transl']
        self.is_opt_dict_init = {'upper_body_pose': False, 'lower_body_pose': False, 'lhand_pose': False, 'rhand_pose': False, 'jaw_pose': False, 'leye_pose': False, 'reye_pose': False, 'shape': False, 'expr': False, 'global_orient': False, 'transl': False}
        self.with_losses_dict_init = {'mask': False, 'normal': False, 'feature': False, 'smooth': False}
        self.shared_keys = ['shape'] ### shared across frames, for full body
        # self.shared_keys = ['shape', 'lower_body_pose', 'global_orient', 'transl'] ### for half body 

        # self.fixed_keys = ['lhand_pose', 'rhand_pose', 'lower_body_pose'] # paper body video input
        self.fixed_keys = [] # other input

        self.confidence_scale = torch.ones((1, 133, 1), dtype=torch.float32).to(self.device)
        # self.confidence_scale[:, [22+62, 22+63, 22+64, 22+66, 22+67, 22+68], :] = 3. ### make mouth accurate
        # self.confidence_scale[:, [22+38, 22+39, 22+41, 22+42, 22+44, 22+45, 22+47, 22+48], :] = 3. ### make eye accurate
        
        self.smoother_buffers, self.smoother_wts, self.smoother_pre_values = {}, {}, {}
        self.smoother_wts['verts'] = 1e4 * 1e-1 * 0.1
        self.smoother_wts['upper_body_poses'] = 1e2 * 0.1
        self.smoother_wts['lower_body_poses'] = 1e3 * 0.1
        self.smoother_wts['lhand_pose'] = 1e2 * 0.1
        self.smoother_wts['rhand_pose'] = 1e2 * 0.1
        self.smoother_wts['global_orient'] = 1e3 * 1. * 0.1
        self.smoother_wts['transl'] = 1e3 * 1. * 0.1
        self.smoother_wts['expr'] = 1. * 0.1
        self.smoother_wts['jaw'] = 2. * 0.1

        for smoother_key in self.smoother_wts.keys():
            self.smoother_pre_values[smoother_key] = None

        self.tracking_losser = TrackingLosser()
        self.smplx_sample_ids = None
        

    def cal_loss(self, print_info = None, init_body_pose = None, debug = False):
        paras_dict = {}
        confidence = None
        if self.tmp_fixed_dict['confidence'] is not None:
            confidence = self.tmp_fixed_dict['confidence'].clone()
        extri = self.tmp_fixed_dict['extri']
        intri = self.tmp_fixed_dict['intri']
        
        pts2d_gt = self.tmp_fixed_dict['pts2d_gt']
        pts2d_gt.requires_grid = False
        pts3d_gt = self.tmp_fixed_dict['pts3d_gt'][...,:3]
        pts3d_gt.requires_grid = False
        confidence3d = self.tmp_fixed_dict['pts3d_gt'][...,3]
        confidence3d.requires_grid = False

        weights = self.tmp_fixed_dict['weights']
        weights.requires_grid = False
        for para_key in self.paras_keys:
            if para_key in self.tmp_fixed_dict:
                paras_dict[para_key] = self.tmp_fixed_dict[para_key]
            else:
                assert(para_key in self.opted_paras_dict.keys())
                paras_dict[para_key] = self.opted_paras_dict[para_key]
        body_joint_num = 21
        batch_size = paras_dict['expr'].shape[0]
        
        # for key in paras_dict.keys():
        #     print(f'{key}: {paras_dict[key].shape}')
        #     print(paras_dict[key].device)
            
        for para_key in self.paras_keys:
            if para_key in self.shared_keys:
                paras_dict[para_key] = paras_dict[para_key].expand(batch_size, *(paras_dict[para_key]).shape[1:])


        paras_dict['body_pose'] = torch.cat((paras_dict['upper_body_pose'], paras_dict['lower_body_pose']), dim=1)[:, self.get_body_inds, :].reshape(-1, body_joint_num * 3)
        

        smplx_out = self.smplx_model.forward(betas = paras_dict['shape'][:1], body_pose = paras_dict['body_pose'], left_hand_pose = paras_dict['lhand_pose'], right_hand_pose = paras_dict['rhand_pose'], jaw_pose = paras_dict['jaw_pose'], expression = paras_dict['expr'], with_iris_return=False, leye_pose=paras_dict['leye_pose'], reye_pose=paras_dict['reye_pose'], global_orient=paras_dict['global_orient'], transl=paras_dict['transl'])

        pts3d, confidence_ = convert_kps(smplx_out.joints)

        # debug_pts3d = np.array(pts3d[0].detach().cpu().numpy())
        # debug_ = '/home/hlp/data/smpl_multi_recon_test/debug'
        # np.save(os.path.join(debug_, 'pts3d.npy'), debug_pts3d)
        # print(f'pts3d shape: {pts3d.shape}')
        # print(f'confidence_ shape: {confidence_.shape}')

        pts3d_cams = []
        cam_para_intris = []
        camera_id = self.cameras.keys()
        # for id in extri.keys():
        for id in camera_id:
            # print(f'id:{id}')
            R = torch.tensor(extri[id]['R'], dtype=torch.float32).to(self.device).reshape(1, 3, 3).detach().clone()
            T = torch.tensor(extri[id]['T'], dtype=torch.float32).to(self.device).reshape(1, 3).detach().clone()
            K = torch.tensor(intri[id]['K'], dtype=torch.float32).to(self.device).reshape(1, 3, 3).detach().clone()
            # print(f'view: {id}, K: {K}, R: {R}, T: {T}')
            pts3d_cam = torch.matmul(pts3d, R.transpose(1, 2)) + T
            # pts3d_cam = torch.matmul(pts3d,R) + T
            pts3d_cams.append(pts3d_cam)
            cam_para_intri = torch.tensor((K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]),dtype=torch.float32).to(self.device).reshape(1, 4).expand(batch_size, -1).detach()
            cam_para_intris.append(cam_para_intri)
        cam_para_intris = torch.stack(cam_para_intris, dim=0).to(self.device)
        cam_para_intris.requires_grad = False
        pts3d_cams = torch.stack(pts3d_cams, dim=0).to(self.device)

        if confidence is not None:
            confidence *= confidence_.unsqueeze(0).unsqueeze(-1)
        else:
            confidence = confidence_.unsqueeze(0).unsqueeze(-1).expand(len(Rs),-1,-1,-1)
        if confidence3d is not None:
            confidence3d *= confidence_
        else:
            confidence3d = confidence_
        
        confidence *= self.confidence_scale.unsqueeze(0)
        confidence3d *= self.confidence_scale.squeeze(-1)
        confidence.requires_grid = False
        confidence_.requires_grad = False
        confidence3d.requires_grad = False
        proj_pts_ = proj_pts_mv(pts3d_cams, cam_para_intris) #shape [v, b, n, 2]
        
        # if debug:
        #     proj_pts_debug = np.array(proj_pts_.detach().cpu().numpy())
        #     debug_ = '/home/hlp/data/smpl_multi_recon_test/debug'
        #     for i, id in enumerate(camera_id):
        #         os.makedirs(os.path.join(debug_,id),exist_ok=True)
        #         if print_info is not None:
        #             import re
        #             match_num = re.search(r'\d+', print_info)
        #             match_type = re.search(r'\S+(?= )', print_info)
        #             if match_num and match_type:
        #                 number = int(match_num.group())
        #                 opt_type = match_type.group()
        #                 os.makedirs(os.path.join(debug_,id,opt_type),exist_ok=True)
        #                 np.save(os.path.join(debug_, id, opt_type, f'proj_pts_{number}_2d.npy'), proj_pts_debug[i])
        #                 np.save(os.path.join(debug_, id, opt_type, f'proj_pts_gt.npy'), pts2d_gt[i].detach().cpu().numpy())

        loss = torch.tensor(0., device = self.device)
        loss_ldmks = torch.mean(cal_euclidean_distance_mv(x1=proj_pts_, x2=pts2d_gt, confidence=confidence, weights=weights))
        loss_3d = torch.mean(cal_euclidean_distance(x1=pts3d_gt, x2=pts3d, confidence=confidence3d.unsqueeze(-1)))
        # print(f'pts3d shape: {pts3d.shape}')
        # print(f'pts3d_gt shape: {pts3d_gt.shape}')
        # loss_ldmks = torch.mean(cal_euclidean_distance(pts3d, pts3d_gt, confidence_))
        loss_shape, loss_exp = cal_l2_loss(paras_dict['shape'][0]), cal_l2_loss(paras_dict['expr'])

        loss_mask = torch.tensor(0., device = loss_ldmks.device)
        loss_smooth = torch.tensor(0., device = loss_ldmks.device)
        loss_feature = torch.tensor(0., device = loss_ldmks.device)
        loss_normal = torch.tensor(0., device = loss_ldmks.device)

        loss_pose = torch.tensor(0., device = loss_ldmks.device)

        if self.with_losses_dict['smooth']:
            temporal_values, concat_values = {}, {}
            if self.smplx_sample_ids is None:
                _, self.smplx_sample_ids = sample_farthest_points(smplx_out.vertices[:1], K = 1000)
                self.smplx_sample_ids = self.smplx_sample_ids.squeeze(0)
                is_head_part = (smplx_out.vertices[0, self.smplx_sample_ids, 1]) > float(smplx_out.joints[0, 12, 1])
                self.sample_smooth_weight = torch.ones_like(smplx_out.vertices[0:1, self.smplx_sample_ids, 0:1])
                self.sample_smooth_weight[:, is_head_part] = .1
            temporal_values['verts'] = smplx_out.vertices[:, self.smplx_sample_ids]
            temporal_values['upper_body_poses'] = paras_dict['upper_body_pose']
            temporal_values['lower_body_poses'] = paras_dict['lower_body_pose']
            temporal_values['lhand_pose'] = paras_dict['lhand_pose']
            temporal_values['rhand_pose'] = paras_dict['rhand_pose']
            temporal_values['expr'] = paras_dict['expr']
            temporal_values['jaw'] = paras_dict['jaw_pose']
            temporal_values['global_orient'] = paras_dict['global_orient']
            temporal_values['transl'] =  paras_dict['transl']
            for smoother_key in self.smoother_wts.keys():
                concat_values[smoother_key] = temporal_values[smoother_key].clone()
                if self.smoother_pre_values[smoother_key] is not None:
                    concat_values[smoother_key] = torch.cat((self.smoother_pre_values[smoother_key], concat_values[smoother_key]), dim = 0)
                if smoother_key == 'verts':
                    loss_smooth += cal_smooth_loss(concat_values[smoother_key], attn_weight=self.sample_smooth_weight) * self.smoother_wts[smoother_key]
                elif smoother_key == 'lower_body_poses':
                    loss_smooth += cal_smooth_loss(concat_values[smoother_key], with_grad=True) * self.smoother_wts[smoother_key]
                else:
                    loss_smooth += cal_smooth_loss(concat_values[smoother_key]) * self.smoother_wts[smoother_key]
                self.smoother_buffers[smoother_key] = temporal_values[smoother_key].clone()
        
        if init_body_pose is not None:
            loss_pose = torch.mean(torch.abs(paras_dict['body_pose'] - init_body_pose))
        if print_info is not None:   
            print(print_info, 'ldmk2d: ', loss_ldmks.item(), 'ldmk3d: ', loss_3d.item(), 'smooth: ', loss_smooth.item(), 'shape: ', loss_shape.item(), 'exp: ', loss_exp.item(), 'pose: ', loss_pose.item())

        loss = loss_ldmks * self.ldmk_ratio * 1. + loss_3d * 50. + loss_shape * 1e-1 * 0.1 + loss_exp * .5 + loss_pose * 1e-1 + loss_smooth * 10.
        
        return loss

        # if (not self.with_losses_dict['mask']) and (not self.with_losses_dict['feature']):
        #     if print_info is not None:
        #         print(print_info, 'ldmk: ', loss_ldmks.item(), 'shape: ', loss_shape.item(), 'exp: ', loss_exp.item())
        #     return loss_ldmks * 1.  + loss_shape *1e-1 * 0.1 + loss_exp * .5

        # vertices_cam = smplx_out.vertices.clone()

        # # 'label_names':  {'0: background', '1: neck', '2: face', '3: cloth', '4: rr', '5: lr', '6: rb', '7: lb', '8: re', '9: le', '10: nose', 
        # #                  '11: imouth', '12: llip', '13: ulip', '14: hair', '15: eyeg', '16: hat', '17: earr', '18: neck_l'}       
        # if self.with_losses_dict['mask']:
        #     render_cam_para = cam_para.clone()
        #     render_cam_para[:, [0, 2]] *= self.resize_scale[1]
        #     render_cam_para[:, [1, 3]] *= self.resize_scale[0]
        #     rendered_masks, rendered_normals = self.mesh_renderer.forward_differentiable_mask_normal(vertices_cam, self.tris, render_cam_para, self.mask_size)
        #     valid_mask = enlarge_human_masks(self.masks)
        #     valid_mask[(self.parsing_maps == 14) | (self.parsing_maps == 16)] = 0. #### exclude hair & hat
        #     loss_mask = cal_l2_loss(rendered_masks.squeeze(-1) - self.masks, valid_mask)

        #     if self.with_losses_dict['normal']:
        #         normal_valid_mask = self.masks.clone()
        #         normal_valid_mask[(self.parsing_maps == 14) | (self.parsing_maps == 16) | (self.parsing_maps == 15) | (self.parsing_maps == 11)] = 0.
        #         loss_normal = cal_l2_loss(rendered_normals - self.normal_maps.detach(), normal_valid_mask.detach().unsqueeze(-1))

        # if self.with_losses_dict['feature']:
        #     if self.with_fixed_bary:
        #         loss_feature, _ = self.tracking_losser.cal_tracking_loss(vertices_cam, cam_para, self.sel_ids, self.tris, self.bary_info)
        #     else:
        #         loss_feature, self.bary_info = self.tracking_losser.cal_tracking_loss(vertices_cam, cam_para, self.sel_ids, self.tris)
        # if print_info is not None:   
        #     print(print_info, 'mask:', loss_mask.item(), 'ldmk: ', loss_ldmks.item(), 'normal: ', loss_normal.item(), 'feature: ', loss_feature.item(), 'smooth: ', loss_smooth.item(), 'shape: ', loss_shape.item(), 'exp: ', loss_exp.item())

        # ldmk_mask_ratio = loss_ldmks.item() / (loss_mask.item() + 1e-5) * self.ldmk_mask_ratio
        # ldmk_normal_ratio = loss_ldmks.item() / (loss_mask.item() + 1e-5) * .0
        # ldmk_feature_ratio = loss_ldmks.item() / (loss_feature.item() + 1e-5) * self.ldmk_feature_ratio

        # return (loss_ldmks*self.ldmk_ratio + loss_mask*ldmk_mask_ratio  + loss_feature*ldmk_feature_ratio + loss_normal*ldmk_normal_ratio) * 1. + loss_shape * 4e-3 * 0.1 + loss_exp *.25 + loss_smooth  + loss_pose * 1e-1

    def opt_smplx_paras(self, 
                        paras_dict_init, 
                        cam_para, 
                        pts2d_gt,
                        pts3d_gt, 
                        is_opt_dict, 
                        with_losses_dict, 
                        confidence = None, 
                        with_fixed_bary = False, 
                        adam_lr = 1e-2, 
                        iter_nums = 300, 
                        ldmk_ratio = 1., 
                        ldmk_mask_ratio = 1., 
                        ldmk_feature_ratio = .2, 
                        current_step = "", 
                        init_body_pose = None):
        self.is_opt_dict = is_opt_dict
        self.with_losses_dict = with_losses_dict
        self.with_fixed_bary = with_fixed_bary
        self.ldmk_ratio = ldmk_ratio
        self.ldmk_mask_ratio = ldmk_mask_ratio
        self.ldmk_feature_ratio = ldmk_feature_ratio
        self.tmp_fixed_dict = {}
        self.tmp_fixed_dict['extri'] = cam_para['extri']
        self.tmp_fixed_dict['intri'] = cam_para['intri']
        self.tmp_fixed_dict['pts2d_gt'] = pts2d_gt.detach()
        self.tmp_fixed_dict['pts3d_gt'] = pts3d_gt.detach()
        self.tmp_fixed_dict['confidence'] = confidence
        self.tmp_fixed_dict['weights'] = self.cal_weights()
        self.opted_paras_dict = {}
        opted_param_groups = []
        for para_key in self.paras_keys:
            if is_opt_dict[para_key] and (para_key not in self.fixed_keys):
                self.opted_paras_dict[para_key] = paras_dict_init[para_key].detach().clone()
                self.opted_paras_dict[para_key].requires_grad = True
                lr_scale = 10. if para_key == 'expr' else 1.
                if para_key == 'shape':
                    lr_scale = 1.
                opted_param_groups.append({'params': self.opted_paras_dict[para_key], 'lr': adam_lr*lr_scale})
            else:
                self.tmp_fixed_dict[para_key] = paras_dict_init[para_key].detach().clone()
        
        optimizer = torch.optim.Adam(opted_param_groups)
        for iter in range(iter_nums):
            optimizer.zero_grad()
            is_debug = False
            print_info = None
            if iter % 50 == 0:
                print_info = current_step + ' iter: ' + str(iter)
                is_debug = True
            loss = self.cal_loss(print_info, init_body_pose = init_body_pose, debug = is_debug)
            loss.backward()
            optimizer.step()
            if iter % (iter_nums//3) == 0 and iter>0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= .2

        for key in self.opted_paras_dict.keys():
            if key in self.shared_keys:
                paras_dict_init[key] = self.opted_paras_dict[key][0:1].detach().clone()
            else:
                paras_dict_init[key] = self.opted_paras_dict[key].detach().clone()
        return paras_dict_init

    def load_camera(self, extri_path, intri_path):
        extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
        names_node = extri.getNode("names")
        names = [names_node.at(i).string() for i in range(names_node.size())]
        extri_params = {name: {
            'R': extri.getNode(f"Rot_{name}").mat().flatten().tolist(),
            'T': extri.getNode(f"T_{name}").mat().flatten().tolist()
        } for name in names}
        extri.release()
        intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
        intri_params = {name: {
            'K': intri.getNode(f"K_{name}").mat().flatten().tolist(),
            'dist': intri.getNode(f"dist_{name}").mat().flatten().tolist()
        } for name in names}
        intri.release()
        return {'extri':extri_params, 'intri':intri_params}

    def load_masks(self, mask_folder, ldmks_names, dst_size):
        masks = []
        parsing_maps = []
        normal_maps = []
        for ldmk_name in ldmks_names:
            mask = cv2.imread(os.path.join(mask_folder, ldmk_name.replace('.wb', '.png')), cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (dst_size[1], dst_size[0]))
            # mask[mask<220] = 0
            masks.append(mask)
            parsing_map = cv2.imread(os.path.join(mask_folder.replace('seg_masks', 'parsing'), ldmk_name.replace('.wb', '.png')), cv2.IMREAD_UNCHANGED)
            parsing_map = cv2.resize(parsing_map, (dst_size[1], dst_size[0]), interpolation=cv2.INTER_NEAREST)
            parsing_maps.append(parsing_map)
            # normal_map = np.load(os.path.join(mask_folder.replace('seg_masks', 'normal'), ldmk_name.replace('.wb', '.npy')))
            # normal_map = cv2.resize(normal_map, (dst_size[1], dst_size[0]), interpolation=cv2.INTER_NEAREST)
            normal_map = np.zeros((dst_size[0], dst_size[1], 3), dtype=np.float32)
            normal_maps.append(normal_map)
        return np.array(masks), np.array(parsing_maps), np.array(normal_maps)

    def cal_weights(self):
        pts2d_gt = self.tmp_fixed_dict['pts2d_gt']
        min_xy = torch.amin(pts2d_gt, dim=2)  # shape: [view, batch, 2]
        max_xy = torch.amax(pts2d_gt, dim=2)  # shape: [view, batch, 2]
        wh = max_xy - min_xy  # shape: [view, batch, 2]
        area = wh[..., 0] * wh[..., 1]  # shape: [view, batch]
        max_area = area.max(dim=1, keepdim=True)[0]
        weights = area / (max_area + 1e-8)
        return weights
        
    def undistort(self, imgs_folder, cameras):
        for camera in cameras:
            img_folder = os.path.join(imgs_folder, camera)
            undis_folder = img_folder.replace('images', 'undis_images')
            print(undis_folder)
            os.makedirs(undis_folder, exist_ok=True)
            K = np.array(cameras[camera]['K'].reshape(3, 3))
            dist = np.array(cameras[camera]['dist'])
            for img_name in os.listdir(img_folder):
                img = cv2.imread(os.path.join(img_folder, img_name))
                h, w = img.shape[:2]
                undist_img = cv2.undistort(img, K, dist, None)
                cv2.imwrite(os.path.join(undis_folder, img_name), undist_img)
    
    # def undis_det(self, ldmks, cameras):

    def batch_triangulate(self, keypoints_, Pall, keypoints_pre=None, lamb=1e3):
        # keypoints: (nViews, nJoints, 3)
        # Pall: (nViews, 3, 4)
        # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
        v = (keypoints_[:, :, -1]>0).sum(axis=0)
        valid_joint = np.where(v > 1)[0]
        keypoints = keypoints_[:, valid_joint]
        conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
        # P2: P矩阵的最后一行：(1, nViews, 1, 4)
        P0 = Pall[None, :, 0, :]
        P1 = Pall[None, :, 1, :]
        P2 = Pall[None, :, 2, :]
        # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
        uP2 = keypoints[:, :, 0].T[:, :, None] * P2
        vP2 = keypoints[:, :, 1].T[:, :, None] * P2
        conf = keypoints[:, :, 2].T[:, :, None]
        Au = conf * (uP2 - P0)
        Av = conf * (vP2 - P1)
        A = np.hstack([Au, Av])
        if keypoints_pre is not None:
            # keypoints_pre: (nJoints, 4)
            B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
            B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
            confpre = lamb * keypoints_pre[valid_joint, 3]
            # 1, 0, 0, -x0
            # 0, 1, 0, -y0
            # 0, 0, 1, -z0
            # 0, 0, 0,   0
            B[:, 3, 3] = 0
            B = B * confpre[:, None, None]
            A = np.hstack((A, B))
        u, s, v = np.linalg.svd(A)
        X = v[:, -1, :]
        X = X / X[:, 3:]
        # out: (nJoints, 4)
        result = np.zeros((keypoints_.shape[1], 4))
        result[valid_joint, :3] = X[:, :3]
        result[valid_joint, 3] = conf3d
        return result

    def vis_smpl(self, vertices, faces, images, save_folder, nf, sub_vis, add_back=True):
        os.makedirs(os.path.join(save_folder, 'smplx'), exist_ok=True)
        outname = os.path.join(save_folder, 'smplx', '{:06d}.jpg'.format(nf))
        render_data = {}
        assert vertices.shape[1] == 3 and len(vertices.shape) == 2, 'shape {} != (N, 3)'.format(vertices.shape)
        pid = 0
        render_data[pid] = {'vertices': vertices, 'faces': faces, 
            'vid': pid, 'name': 'human_{}_{}'.format(nf, pid)}
        cameras = {'K': [], 'R':[], 'T':[]}
        for key in cameras.keys():
            cameras[key] = np.stack([self.cameras[cam][key] for cam in sub_vis])
        images = images
        self._vis_smpl(render_data, images, cameras, outname, add_back=add_back)
    
    def _vis_smpl(self, render_data, images, cameras, outname, add_back):
        render = Renderer(height=1024, width=1024, faces=None)
        render_results = render.render(render_data, cameras, images, add_back=add_back)
        image_vis = merge(render_results)
        cv2.imwrite(outname, image_vis)
        return image_vis

    def run_folder(self, imgs_folder, save_folder, debug_dir='none', sub_vis = None):
        # self.tracking_losser.set_folder(imgs_folder)
        # calculate the number of frames
        self.views = sorted(os.listdir(imgs_folder))
        frames = []
        for view in self.views:
            single_view_images_folder = os.path.join(imgs_folder, view, 'ori_imgs')
            image_count = len([f for f in os.listdir(single_view_images_folder) if f.lower().endswith(('.png', '.jpg'))])
            frames.append(image_count)
        print(f'frames: {frames}')
        frames_num = min(frames)

        recon_smplx_params = {'shape': [], 'body_pose': [], 'lhand_pose': [], 'rhand_pose': []}
        ldmks_all_view = []
        recon_folder = imgs_folder.replace('images', 'smplx_recon')
        smpl_names = np.array(natsorted([f for f in os.listdir(recon_folder) if f.endswith('_personId_0.pkl')]))[:frames_num]
        for smpl_name in smpl_names:
            aios_recon_file = os.path.join(recon_folder, smpl_name)
            with open(aios_recon_file, 'rb') as f:
                recon_params = pickle.load(f)['params']
            recon_smplx_params['shape'].append(np.array(recon_params['betas']).reshape(-1))
            recon_smplx_params['body_pose'].append(np.array(recon_params['body_pose']).reshape(-1))
            recon_smplx_params['lhand_pose'].append(np.array(recon_params['left_hand_pose']).reshape(-1))
            recon_smplx_params['rhand_pose'].append(np.array(recon_params['right_hand_pose']).reshape(-1))
        print(f'len(recon_smplx_params[shape]): {len(recon_smplx_params["shape"])}')
        print(f'len(recon_smplx_params[body_pose]): {len(recon_smplx_params["body_pose"])}')
        print(f'len(recon_smplx_params[lhand_pose]): {len(recon_smplx_params["lhand_pose"])}')
        print(f'len(recon_smplx_params[rhand_pose]): {len(recon_smplx_params["rhand_pose"])}')

        # for view in sorted(os.listdir(imgs_folder)):
        for view in self.views:
            single_view_folder = os.path.join(imgs_folder, view)

            ldmks_folder = os.path.join(single_view_folder, 'ldmks')
            ldmks_names = np.array(natsorted([f for f in os.listdir(ldmks_folder) if f.endswith('.wb')]))[:frames_num]

            ldmks_all = []
        
            for ldmk_name in ldmks_names:
                ldmks = np.loadtxt(os.path.join(ldmks_folder, ldmk_name), dtype=np.float32)
                ldmks_all.append(ldmks)
            ldmks_all_view.append(ldmks_all)
        
        # recon_load_params = torch.load(os.path.join(recon_folder, 'smplx_recon.pth'), map_location = 'cpu')
        # recon_smplx_params['shape'] = recon_load_params['shape']
        # recon_smplx_params['body_pose'] = recon_load_params['body_pose'].reshape(-1, 63)
        # recon_smplx_params['lhand_pose'] = recon_load_params['lhand_pose'].reshape(-1, 45)
        # recon_smplx_params['rhand_pose'] = recon_load_params['rhand_pose'].reshape(-1, 45)
        # body_pose_init = torch.as_tensor(recon_load_params['body_pose'].reshape(-1, 63)).to(self.device)

        body_pose_init = torch.as_tensor(recon_smplx_params['body_pose']).to(self.device)

        ldmks_all_view = np.array(ldmks_all_view) # shape [view, batch, 133, 3]

        recon_smplx_params = {k: np.array(v) for k, v in recon_smplx_params.items()}
        

        ### Step 1: load camera parameters(extri and intri) and other parameters

        # cam_para = torch.tensor((best_focal, best_focal, img.shape[1]/2., img.shape[0]/2.), dtype=torch.float32).cuda().reshape(1, 4)
        extri_path = imgs_folder.replace('images', 'extri.yml')
        intri_path = imgs_folder.replace('images', 'intri.yml')
        cam_para = self.load_camera(extri_path, intri_path)
        self.cameras = {}
        Ps = []
        for view in self.views:
            camera = {}
            R = np.array(cam_para['extri'][view]['R']).reshape(3, 3)
            T = np.array(cam_para['extri'][view]['T']).reshape(3, 1)
            K = np.array(cam_para['intri'][view]['K']).reshape(3, 3)
            camera['R'] = R.tolist()
            camera['T'] = T.tolist()
            camera['K'] = K.tolist()
            self.cameras[view] = camera
            P = np.dot(K, np.hstack((R, T)))
            Ps.append(P)
        Ps = np.array(Ps)

        ### initialize all para_keys: ['upper_body_pose', 'lower_body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'shape', 'expr', 'global_orient', 'transl']
        ldmks_2d_all_view = torch.as_tensor(ldmks_all_view[..., :2]).to(self.device)
        ldmks_confidence_all_view = torch.as_tensor(ldmks_all_view[..., 2:]).to(self.device)
        ldmks_confidence_all_view[ldmks_confidence_all_view <= self.confidence_min] = 0.
        print(f'ldmks_2d_all_view shape: {ldmks_2d_all_view.shape}')
        # undistort images and keypoints2d

        # reconstruct 3d keypoints
        ldmks_3d = []
        for frame in tqdm(range(ldmks_all_view.shape[1]),desc='triangulation'):
            keypoint_2d = ldmks_all_view[:,frame,...]
            ldmks_3d_frame = self.batch_triangulate(keypoint_2d, Ps)
            ldmks_3d.append(ldmks_3d_frame)
        ldmks_3d = np.array(ldmks_3d)
        ldmks_3d_gt = torch.tensor(ldmks_3d.tolist()).to(self.device)
        ldmks_3d_gt.requires_grad = False
        print(f'ldmks_3d_gt shape: {ldmks_3d_gt.shape}')

        # frames_num = ldmks_2d.shape[0]
        self.frames_params = {}
        self.frames_params['shape'] = torch.zeros((1, self.shape_dim), dtype=torch.float32)
        self.frames_params['expr'] = torch.zeros((frames_num, self.expr_dim), dtype=torch.float32)
        self.frames_params['global_orient'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        # self.frames_params['global_orient'][:,1] = torch.tensor([np.pi]*frames_num)
        self.frames_params['transl'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        self.frames_params['upper_body_pose'] = torch.zeros((frames_num, self.upper_body_inds.shape[0],3), dtype=torch.float32)
        self.frames_params['lower_body_pose'] = torch.zeros((frames_num, self.lower_body_inds.shape[0], 3), dtype=torch.float32)
        self.frames_params['lhand_pose'] = torch.zeros((frames_num, 45), dtype=torch.float32)
        self.frames_params['rhand_pose'] = torch.zeros((frames_num, 45), dtype=torch.float32)
        self.frames_params['jaw_pose'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        self.frames_params['leye_pose'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        self.frames_params['reye_pose'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        mean_betas = np.mean(recon_smplx_params['shape'], axis=0, keepdims=True)
        print(f"self.frames_params['shape'] shape: {self.frames_params['shape'].shape}")
        print(f"mean_betas shape: {mean_betas.shape}")
        self.frames_params['shape'][:, :mean_betas.shape[1]] = torch.as_tensor(mean_betas)
        self.frames_params['upper_body_pose'] = torch.as_tensor(recon_smplx_params['body_pose']).reshape(-1, 21, 3)[:, self.upper_body_inds, :]
        self.frames_params['lower_body_pose'] = torch.as_tensor(recon_smplx_params['body_pose']).reshape(-1, 21, 3)[:, self.lower_body_inds, :]
        self.frames_params['lhand_pose'] = torch.as_tensor(recon_smplx_params['lhand_pose'])
        self.frames_params['rhand_pose'] = torch.as_tensor(recon_smplx_params['rhand_pose'])
        for para_key in self.paras_keys:
            if para_key in self.shared_keys:
                self.frames_params[para_key] = self.frames_params[para_key][0:1]  ### make the shared param only have single value
            self.frames_params[para_key] = self.frames_params[para_key].to(self.device)
        for key in self.frames_params.keys():
            print(f'{key} : {self.frames_params[key].shape}')
            print(self.frames_params[key].device)
        # resize_scale = min(1., 400./max(img.shape[:2]))  ### resolution for differentiable rendering loss computation
        # self.mask_size = (int(resize_scale*img.shape[0]), int(resize_scale*img.shape[1]))
        # self.resize_scale = (self.mask_size[0]/float(img.shape[0]), self.mask_size[1]/float(img.shape[1]))

        ### Step 2: opt global_orient and transl for every frame
        batch_size = 1000
        video_for_debug = None
        for i in tqdm(range(0, frames_num, batch_size), desc = 'opt global_orient and transl'):
            start_id, end_id = i, min((i+batch_size), ldmks_2d_all_view.shape[1])
            cur_batch_size = end_id - start_id
            cur_ids = np.arange(start_id, end_id)
            left_pad_num = 0
            if start_id == 0: ### fix a strange bug
                left_pad_num = 2
                cur_ids = np.concatenate((np.zeros(left_pad_num, dtype=np.int64), cur_ids), axis=0)
                cur_batch_size += left_pad_num
            self.sel_ids = cur_ids
            dict_for_subinds = {}
            for key in self.paras_keys:
                if key in self.shared_keys:
                    dict_for_subinds[key] = self.frames_params[key].clone()
                else:
                    dict_for_subinds[key] = self.frames_params[key][cur_ids].clone()
            # masks, _, _ = self.load_masks(imgs_folder.replace('ori_imgs', 'seg_masks'), ldmks_names[cur_ids], self.mask_size)
            # self.masks = torch.as_tensor(masks, device = ldmks_2d_all_view.device).float() / 255.
            is_opt_dict = deepcopy(self.is_opt_dict_init)
            is_opt_dict.update({'global_orient': True, 'transl': True})
            with_losses_dict = deepcopy(self.with_losses_dict_init)
            dict_for_subinds = self.opt_smplx_paras(dict_for_subinds, cam_para, ldmks_2d_all_view[:,cur_ids], ldmks_3d_gt[cur_ids], is_opt_dict, with_losses_dict, confidence = ldmks_confidence_all_view[:,cur_ids], adam_lr = 1e-1, iter_nums=600, ldmk_ratio=1., current_step='global_orient_transl')
            cur_ids = np.arange(start_id, end_id)
            cur_batch_size = end_id - start_id
            for key in self.paras_keys:
                if key in self.shared_keys:
                    self.frames_params[key] = dict_for_subinds[key].clone()
                else:
                    self.frames_params[key][cur_ids] = dict_for_subinds[key].clone()[left_pad_num:]
            if debug_dir == 'none':
                continue
            paras_dict = {}
            for key in self.paras_keys:
                if key in self.shared_keys:
                    paras_dict[key] = self.frames_params[key].clone()
                    paras_dict[key] = paras_dict[key].expand(cur_batch_size, *(paras_dict[key]).shape[1:])
                else:
                    paras_dict[key] = self.frames_params[key][cur_ids].clone()

        ### Step 3: opt shared shape and fixed later
        batch_size = 100
        cur_ids = np.arange(0, ldmks_2d_all_view.shape[1], max(1, ldmks_2d_all_view.shape[1]//batch_size))
        self.sel_ids = cur_ids
        cur_batch_size = cur_ids.shape[0]
        dict_for_subinds = {}
        for key in self.paras_keys:
            if key in self.shared_keys:
                dict_for_subinds[key] = self.frames_params[key].clone()
            else:
                dict_for_subinds[key] = self.frames_params[key][cur_ids].clone()
        # masks, _, _ = self.load_masks(imgs_folder.replace('ori_imgs', 'seg_masks'), ldmks_names[cur_ids], self.mask_size)
        # self.masks = torch.as_tensor(masks, device = ldmks_2d_all_view.device).float() / 255.

        is_opt_dict = deepcopy(self.is_opt_dict_init)
        for key in is_opt_dict.keys():
            is_opt_dict[key] = True
        with_losses_dict = deepcopy(self.with_losses_dict_init)
        with_losses_dict.update({'mask': True, 'feature': True})
        dict_for_subinds = self.opt_smplx_paras(dict_for_subinds, cam_para, ldmks_2d_all_view[:,cur_ids], ldmks_3d_gt[cur_ids], is_opt_dict, with_losses_dict, confidence = ldmks_confidence_all_view[:,cur_ids], adam_lr = 3e-3, iter_nums=900, ldmk_ratio=1.5, ldmk_feature_ratio=0., current_step='opt_shared_shape', init_body_pose = body_pose_init[cur_ids])
        for key in self.paras_keys:
            if key in self.shared_keys:
                self.frames_params[key] = dict_for_subinds[key].clone()
            # else:
            #     self.frames_params[key][cur_ids] = dict_for_subinds[key].clone()

        ### Step 4: fixed shared and opt frame dependent params
        batch_size = 1000
        video_for_debug = None
        for i in tqdm(range(0, frames_num, batch_size), desc = 'opt frames dependent params'):
            start_id, end_id = i, min((i+batch_size), ldmks_2d_all_view.shape[1])
            cur_batch_size = end_id - start_id
            cur_ids = np.arange(start_id, end_id)
            left_pad_num = 0
            if start_id == 0: ### fix a strange bug
                left_pad_num = 2
                cur_ids = np.concatenate((np.zeros(left_pad_num, dtype=np.int64), cur_ids), axis=0)
                cur_batch_size += left_pad_num
            self.sel_ids = cur_ids
            dict_for_subinds = {}
            for key in self.paras_keys:
                if key in self.shared_keys:
                    dict_for_subinds[key] = self.frames_params[key].clone()
                else:
                    dict_for_subinds[key] = self.frames_params[key][cur_ids].clone()
            # masks, _, _ = self.load_masks(imgs_folder.replace('ori_imgs', 'seg_masks'), ldmks_names[cur_ids], self.mask_size)
            # self.masks = torch.as_tensor(masks, device = ldmks_2d.device).float() / 255.

            is_opt_dict = deepcopy(self.is_opt_dict_init)
            for key in self.paras_keys:
                if key not in self.shared_keys:
                    is_opt_dict[key] = True
            
            with_losses_dict = deepcopy(self.with_losses_dict_init)
            for key in with_losses_dict.keys():
                if key != 'normal':
                    with_losses_dict[key] = True
            dict_for_subinds = self.opt_smplx_paras(dict_for_subinds, cam_para, ldmks_2d_all_view[:,cur_ids], ldmks_3d_gt[cur_ids], is_opt_dict, with_losses_dict, confidence = ldmks_confidence_all_view[:,cur_ids], adam_lr = 3e-3, iter_nums=600, with_fixed_bary=True, ldmk_ratio=2., ldmk_mask_ratio = .1, ldmk_feature_ratio=.3, current_step='frames batch', init_body_pose = body_pose_init[cur_ids])
            cur_ids = np.arange(start_id, end_id)
            cur_batch_size = end_id - start_id
            for key in self.paras_keys:
                if key in self.shared_keys:
                    self.frames_params[key] = dict_for_subinds[key].clone()
                else:
                    self.frames_params[key][cur_ids] = dict_for_subinds[key].clone()[left_pad_num:]

            if len(self.smoother_buffers) > 0:
                for smoother_key in self.smoother_wts.keys():
                    self.smoother_pre_values[smoother_key] = self.smoother_buffers[smoother_key].detach().clone()    

            if debug_dir == 'none':
                continue
            paras_dict = {}
            for key in self.paras_keys:
                if key in self.shared_keys:
                    paras_dict[key] = self.frames_params[key].clone()
                    paras_dict[key] = paras_dict[key].expand(cur_batch_size, *(paras_dict[key]).shape[1:])
                else:
                    paras_dict[key] = self.frames_params[key][cur_ids].clone()

        paras_dict = {}
        for key in self.paras_keys:
            if key in self.shared_keys:
                paras_dict[key] = self.frames_params[key].clone()
                paras_dict[key] = paras_dict[key].expand(frames_num, *(paras_dict[key]).shape[1:])
            else:
                paras_dict[key] = self.frames_params[key].clone()

        paras_dict['body_pose'] = torch.cat((paras_dict['upper_body_pose'], paras_dict['lower_body_pose']), dim=1)[:, self.get_body_inds, :].reshape(frames_num, -1)

        # save for angs input
        save_keys = ['body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'betas', 'expression', 'global_orient', 'transl']
        save_dict = {}
        save_dict['body_pose'] = paras_dict['body_pose'].detach().cpu().numpy()
        save_dict['left_hand_pose'] = paras_dict['lhand_pose'].detach().cpu().numpy()
        save_dict['right_hand_pose'] = paras_dict['rhand_pose'].detach().cpu().numpy()
        save_dict['jaw_pose'] = paras_dict['jaw_pose'].detach().cpu().numpy()
        save_dict['leye_pose'] = paras_dict['leye_pose'].detach().cpu().numpy()
        save_dict['reye_pose'] = paras_dict['reye_pose'].detach().cpu().numpy()
        save_dict['betas'] = paras_dict['shape'].detach().cpu().numpy()[0:1,:]
        save_dict['expression'] = paras_dict['expr'].detach().cpu().numpy()
        save_dict['global_orient'] = paras_dict['global_orient'].detach().cpu().numpy()
        save_dict['transl'] = paras_dict['transl'].detach().cpu().numpy()
        for save_key in save_keys:
            print(f'{save_key} : {save_dict[save_key].shape}')
        np.savez(os.path.join(save_folder, 'smpl_params.npz'), **save_dict)

        # visualization
        for nf in tqdm(range(1,frames_num), desc='render'):
            images = load_image(imgs_folder, sub_vis, nf)
            smplx_out = self.smplx_model.forward(betas = paras_dict['shape'][nf:nf+1], body_pose = paras_dict['body_pose'][nf:nf+1], left_hand_pose = paras_dict['lhand_pose'][nf:nf+1], right_hand_pose = paras_dict['rhand_pose'][nf:nf+1], jaw_pose = paras_dict['jaw_pose'][nf:nf+1], expression = paras_dict['expr'][nf:nf+1], with_iris_return=False, leye_pose=paras_dict['leye_pose'][nf:nf+1], reye_pose=paras_dict['reye_pose'][nf:nf+1], global_orient=paras_dict['global_orient'][nf:nf+1], transl=paras_dict['transl'][nf:nf+1])
            vertice = smplx_out.vertices.squeeze().detach().cpu().numpy()
            faces = self.tris.squeeze().detach().cpu().numpy()
            self.vis_smpl(vertice, faces, images, save_folder, nf, sub_vis, add_back=True)
        images_folder = os.path.join(save_folder, 'smplx')
        images_to_video(images_folder, os.path.join(save_folder, 'smplx.mp4'))



            