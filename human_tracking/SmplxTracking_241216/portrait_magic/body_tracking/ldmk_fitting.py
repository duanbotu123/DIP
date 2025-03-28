import os
import cv2
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
from ..render_utils.geometry_utils import solve_pnp_cv, proj_pts
from .opt_utils.bundle_adjustment_bfgs import cal_best_focal
from .loss_utils import cal_euclidean_distance, cal_l2_loss, cal_smooth_loss
from ..dmm_models import FLAME, flame_config, SMPLX, smplx_model_path
from ..dmm_models.utils.keypoints_mapping import convert_kps
from ..utils_funcs import VideoWriter
from ..feature_tracking_all import TrackingLosser

class LDMK_Fitting():
    def __init__(self):
        self.device = 'cuda:0'
        self.mica_model = MICAModel().cuda().eval()
        self.flame_model = FLAME(flame_config).cuda().eval()
        self.shape_dim, self.expr_dim = 300, 100
        self.smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=self.expr_dim, num_betas = self.shape_dim, use_face_contour= True, use_pca = False).cuda().eval()

        for param in self.flame_model.parameters():
            param.requires_grad = False
        for param in self.mica_model.parameters():
            param.requires_grad = False
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
        self.confidence_scale[:, [22+62, 22+63, 22+64, 22+66, 22+67, 22+68], :] = 3. ### make mouth accurate
        self.confidence_scale[:, [22+38, 22+39, 22+41, 22+42, 22+44, 22+45, 22+47, 22+48], :] = 3. ### make eye accurate
        
        self.smoother_buffers, self.smoother_wts, self.smoother_pre_values = {}, {}, {}
        self.smoother_wts['verts'] = 1e4 * 1e-1 * 0.
        self.smoother_wts['upper_body_poses'] = 1e2 * 0.
        self.smoother_wts['lower_body_poses'] = 1e3 * 0.
        self.smoother_wts['lhand_pose'] = 1e2 * 0.
        self.smoother_wts['rhand_pose'] = 1e2 * 0.
        self.smoother_wts['global_orient'] = 1e3 * 1. * 0.
        self.smoother_wts['transl'] = 1e3 * 1. * 0.
        self.smoother_wts['expr'] = 1. * 0.
        self.smoother_wts['jaw'] = 2. * 0.

        for smoother_key in self.smoother_wts.keys():
            self.smoother_pre_values[smoother_key] = None

        self.tracking_losser = TrackingLosser()
        self.smplx_sample_ids = None


    def cal_loss(self, print_info = None, init_body_pose = None):
        paras_dict = {}
        confidence = None
        if self.tmp_fixed_dict['confidence'] is not None:
            confidence = self.tmp_fixed_dict['confidence'].clone()
        cam_para, pts2d_gt = self.tmp_fixed_dict['cam_para'], self.tmp_fixed_dict['pts2d_gt']
        for para_key in self.paras_keys:
            if para_key in self.tmp_fixed_dict:
                paras_dict[para_key] = self.tmp_fixed_dict[para_key]
            else:
                assert(para_key in self.opted_paras_dict.keys())
                paras_dict[para_key] = self.opted_paras_dict[para_key]
        body_joint_num = 21
        batch_size = paras_dict['expr'].shape[0]

        for para_key in self.paras_keys:
            if para_key in self.shared_keys:
                paras_dict[para_key] = paras_dict[para_key].expand(batch_size, *(paras_dict[para_key]).shape[1:])

        paras_dict['body_pose'] = torch.cat((paras_dict['upper_body_pose'], paras_dict['lower_body_pose']), dim=1)[:, self.get_body_inds, :].reshape(-1, body_joint_num * 3)
        

        smplx_out, iris_verts = self.smplx_model.forward(betas = paras_dict['shape'][:1], body_pose = paras_dict['body_pose'], left_hand_pose = paras_dict['lhand_pose'], right_hand_pose = paras_dict['rhand_pose'], jaw_pose = paras_dict['jaw_pose'], expression = paras_dict['expr'], with_iris_return=True, leye_pose=paras_dict['leye_pose'], reye_pose=paras_dict['reye_pose'], global_orient=paras_dict['global_orient'], transl=paras_dict['transl'])

        pts3d, confidence_ = convert_kps(smplx_out.joints)

        # print(torch.mean(pts3d, dim=(0,1)).reshape(1,-1))
        if confidence is not None:
            confidence *= confidence_.unsqueeze(-1)
        else:
            confidence = confidence_.unsqueeze(-1)
        confidence *= self.confidence_scale
        
        proj_pts_ = proj_pts(pts3d, cam_para)
        loss_ldmks = torch.mean(cal_euclidean_distance(proj_pts_, pts2d_gt, confidence=confidence))

        iris_gt = self.tmp_fixed_dict['iris_gt']
        proj_iris_ = proj_pts(iris_verts, cam_para)
        loss_iris = torch.mean(cal_euclidean_distance(proj_iris_, iris_gt[..., :2], confidence=iris_gt[..., 2:]))

        loss_shape, loss_exp = cal_l2_loss(paras_dict['shape'][0]), cal_l2_loss(paras_dict['expr'])

        loss_mask = torch.tensor(0., device = loss_ldmks.device)
        loss_smooth = torch.tensor(0., device = loss_ldmks.device)
        loss_feature = torch.tensor(0., device = loss_ldmks.device)
        loss_normal = torch.tensor(0., device = loss_ldmks.device)

        loss_pose = torch.tensor(0., device = loss_ldmks.device)
        if init_body_pose is not None:
            loss_pose = torch.mean(torch.abs(paras_dict['body_pose'] - init_body_pose))

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
        
        if (not self.with_losses_dict['mask']) and (not self.with_losses_dict['feature']):
            if print_info is not None:
                print(print_info, 'ldmk: ', loss_ldmks.item(), 'shape: ', loss_shape.item(), 'exp: ', loss_exp.item(), 'iris: ', loss_iris.item())
            return loss_ldmks * 1.  + loss_shape *1e-1 * 0.1 + loss_exp * .5 + loss_iris * .3

        vertices_cam = smplx_out.vertices.clone()

        # 'label_names':  {'0: background', '1: neck', '2: face', '3: cloth', '4: rr', '5: lr', '6: rb', '7: lb', '8: re', '9: le', '10: nose', 
        #                  '11: imouth', '12: llip', '13: ulip', '14: hair', '15: eyeg', '16: hat', '17: earr', '18: neck_l'}       
        if self.with_losses_dict['mask']:
            render_cam_para = cam_para.clone()
            render_cam_para[:, [0, 2]] *= self.resize_scale[1]
            render_cam_para[:, [1, 3]] *= self.resize_scale[0]
            rendered_masks, rendered_normals = self.mesh_renderer.forward_differentiable_mask_normal(vertices_cam, self.tris, render_cam_para, self.mask_size)
            valid_mask = enlarge_human_masks(self.masks)
            valid_mask[(self.parsing_maps == 14) | (self.parsing_maps == 16)] = 0. #### exclude hair & hat
            loss_mask = cal_l2_loss(rendered_masks.squeeze(-1) - self.masks, valid_mask)

            if self.with_losses_dict['normal']:
                normal_valid_mask = self.masks.clone()
                normal_valid_mask[(self.parsing_maps == 14) | (self.parsing_maps == 16) | (self.parsing_maps == 15) | (self.parsing_maps == 11)] = 0.
                loss_normal = cal_l2_loss(rendered_normals - self.normal_maps.detach(), normal_valid_mask.detach().unsqueeze(-1))

        if self.with_losses_dict['feature']:
            if self.with_fixed_bary:
                loss_feature, _ = self.tracking_losser.cal_tracking_loss(vertices_cam, cam_para, self.sel_ids, self.tris, self.bary_info)
            else:
                loss_feature, self.bary_info = self.tracking_losser.cal_tracking_loss(vertices_cam, cam_para, self.sel_ids, self.tris)
        if print_info is not None:   
            print(print_info, 'mask:', loss_mask.item(), 'ldmk: ', loss_ldmks.item(), 'normal: ', loss_normal.item(), 'feature: ', loss_feature.item(), 'smooth: ', loss_smooth.item(), 'shape: ', loss_shape.item(), 'exp: ', loss_exp.item(), 'iris: ', loss_iris.item())

        ldmk_mask_ratio = loss_ldmks.item() / (loss_mask.item() + 1e-5) * self.ldmk_mask_ratio
        ldmk_normal_ratio = loss_ldmks.item() / (loss_mask.item() + 1e-5) * .0
        ldmk_feature_ratio = loss_ldmks.item() / (loss_feature.item() + 1e-5) * self.ldmk_feature_ratio

        return (loss_ldmks*self.ldmk_ratio + loss_mask*ldmk_mask_ratio  + loss_feature*ldmk_feature_ratio + loss_normal*ldmk_normal_ratio) * 1. + loss_shape * 4e-3 * 0.1 + loss_exp *.25 + loss_smooth + loss_iris * .3 + loss_pose * 1e-1
        

    def opt_smplx_paras(self, paras_dict_init, cam_para, pts2d_gt, iris_ldmks, is_opt_dict, with_losses_dict, confidence = None, with_fixed_bary = False, adam_lr = 1e-2, iter_nums = 300, ldmk_ratio = 1., ldmk_mask_ratio = 1., ldmk_feature_ratio = .2, current_step = None, init_body_pose = None):
        self.is_opt_dict = is_opt_dict
        self.with_losses_dict = with_losses_dict
        self.with_fixed_bary = with_fixed_bary
        self.ldmk_ratio = ldmk_ratio
        self.ldmk_mask_ratio = ldmk_mask_ratio
        self.ldmk_feature_ratio = ldmk_feature_ratio
        self.tmp_fixed_dict = {}
        self.tmp_fixed_dict['cam_para'] = cam_para.detach()
        self.tmp_fixed_dict['pts2d_gt'] = pts2d_gt.detach()
        self.tmp_fixed_dict['iris_gt'] = iris_ldmks.detach()
        self.tmp_fixed_dict['confidence'] = confidence
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
            print_info = None
            if iter % 50 == 0:
                print_info = current_step + ' iter: ' + str(iter)
            loss = self.cal_loss(print_info, init_body_pose = init_body_pose)
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

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        self.tracking_losser.set_folder(imgs_folder)

        ldmks_folder = imgs_folder.replace('ori_imgs', 'ldmks')
        recon_folder = imgs_folder.replace('ori_imgs', 'smplx_recon')
        ldmks_names = np.array(natsorted([f for f in os.listdir(ldmks_folder) if f.endswith('.wb')]))

        ldmks_all = []
        recon_smplx_params = {'shape': [], 'body_pose': [], 'lhand_pose': [], 'rhand_pose': []}
        for ldmk_name in ldmks_names:
            ldmks = np.loadtxt(os.path.join(ldmks_folder, ldmk_name), dtype=np.float32)
            ldmks_all.append(ldmks)

            aios_recon_file = os.path.join(recon_folder, ldmk_name.replace('.wb', '_personId_0.pkl'))
            with open(aios_recon_file, 'rb') as f:
                recon_params = pickle.load(f)['params']
            recon_smplx_params['shape'].append(np.array(recon_params['betas']).reshape(-1))
            recon_smplx_params['body_pose'].append(np.array(recon_params['body_pose']).reshape(-1))
            recon_smplx_params['lhand_pose'].append(np.array(recon_params['left_hand_pose']).reshape(-1))
            recon_smplx_params['rhand_pose'].append(np.array(recon_params['right_hand_pose']).reshape(-1))

        # recon_load_params = torch.load(os.path.join(recon_folder, 'smplx_recon.pth'), map_location = 'cpu')
        # recon_smplx_params['shape'] = recon_load_params['shape']
        # recon_smplx_params['body_pose'] = recon_load_params['body_pose'].reshape(-1, 63)
        # recon_smplx_params['lhand_pose'] = recon_load_params['lhand_pose'].reshape(-1, 45)
        # recon_smplx_params['rhand_pose'] = recon_load_params['rhand_pose'].reshape(-1, 45)
        # body_pose_init = torch.as_tensor(recon_load_params['body_pose'].reshape(-1, 63)).to(self.device)

        body_pose_init = torch.as_tensor(recon_smplx_params['body_pose']).to(self.device)

        ldmks_all = np.array(ldmks_all)
        recon_smplx_params = {k: np.array(v) for k, v in recon_smplx_params.items()}

        iris_folder = imgs_folder.replace('ori_imgs', 'iris_ldmks')
        iris_ldmks_all = []
        for ldmk_name in ldmks_names:
            iris_ldmks = np.loadtxt(os.path.join(iris_folder, ldmk_name.replace('.wb', '.iris')), dtype=np.float32)
            iris_ldmks_all.append(iris_ldmks)
        iris_ldmks_all = np.array(iris_ldmks_all)
        iris_ldmks_all = torch.as_tensor(iris_ldmks_all).cuda()

        ### Step 1: find best focal length by reprojecting rigid head landmarks (4 eye corners + 1 nosetip)
        head_ldmk_score = np.min(ldmks_all[:, [59, 62, 65, 68, 50, 51, 52, 53], 2], axis=1)
        valid_ids = np.where(head_ldmk_score > 0.5)[0]
        mica_sel_ids = valid_ids[np.arange(0, valid_ids.shape[0], max(valid_ids.shape[0]//10, 1))]
        shape_paras = []
        with torch.inference_mode():
            for mica_sel_id in mica_sel_ids:
                img = cv2.imread(os.path.join(imgs_folder, ldmks_names[mica_sel_id][:-3] + '.jpg'))
                lms_68 = ldmks_all[mica_sel_id, 23:91, :2]
                shape_paras.append(self.mica_model.infer_img(img, lms_68))
            shape_code = torch.median(torch.cat(shape_paras, dim=0), dim=0, keepdim=True)[0]
        neutral_geo = self.flame_model.forward_geo(shape_code, shape_code.new_zeros(1, flame_config.n_exp))
        rigid_ids_in_68 = np.array((59, 62, 65, 68, 53), dtype=np.int64) - 23
        rigid_pts = self.flame_model.get_3dlandmarks(neutral_geo, self.flame_model.pts68_fid[rigid_ids_in_68], self.flame_model.pts68_bary_coords[rigid_ids_in_68])
        rigid_sel_ids = valid_ids[np.arange(0, valid_ids.shape[0], max(valid_ids.shape[0]//200, 1))]
        rigid_lmss = ldmks_all[rigid_sel_ids, 23:, :2][:, rigid_ids_in_68, :]
        rigid_lmss = torch.as_tensor(rigid_lmss).cuda()
        B, L = rigid_lmss.shape[:2]
        rigid_lmss = sample_farthest_points(rigid_lmss.reshape(B, -1).unsqueeze(0), K = min(B, 20))[0]
        rigid_lmss = rigid_lmss.reshape(-1, L, 2)
        best_focal, arg_dis, arg_z = cal_best_focal(rigid_lmss, (img.shape[0], img.shape[1]), rigid_pts)
        print('find best focel', best_focal, 'with proj error', round(arg_dis, 4), 'with distance to camera', round(arg_z, 4))
        cam_para = torch.tensor((best_focal, best_focal, img.shape[1]/2., img.shape[0]/2.), dtype=torch.float32).cuda().reshape(1, 4)


        ### initialize all para_keys: ['upper_body_pose', 'lower_body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'shape', 'expr', 'global_orient', 'transl']
        ldmks_2d = torch.as_tensor(ldmks_all[..., :2]).to(self.device)
        ldmks_confidence = torch.as_tensor(ldmks_all[..., 2:]).to(self.device)
        ldmks_confidence[ldmks_confidence <= self.confidence_min] = 0.
        frames_num = ldmks_2d.shape[0]
        self.frames_params = {}
        self.frames_params['shape'] = torch.zeros((1, self.shape_dim), dtype=torch.float32)
        self.frames_params['expr'] = torch.zeros((frames_num, self.expr_dim), dtype=torch.float32)
        self.frames_params['global_orient'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        self.frames_params['transl'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        self.frames_params['transl'][:, 2] += float(arg_z)
        self.frames_params['upper_body_pose'] = torch.zeros((frames_num, self.upper_body_inds.shape[0],3), dtype=torch.float32)
        self.frames_params['lower_body_pose'] = torch.zeros((frames_num, self.lower_body_inds.shape[0], 3), dtype=torch.float32)
        self.frames_params['lhand_pose'] = torch.zeros((frames_num, 45), dtype=torch.float32)
        self.frames_params['rhand_pose'] = torch.zeros((frames_num, 45), dtype=torch.float32)
        self.frames_params['jaw_pose'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        self.frames_params['leye_pose'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        self.frames_params['reye_pose'] = torch.zeros((frames_num, 3), dtype=torch.float32)
        mean_betas = np.mean(recon_smplx_params['shape'], axis=0, keepdims=True)
        self.frames_params['shape'][:, :mean_betas.shape[1]] = torch.as_tensor(mean_betas)
        self.frames_params['upper_body_pose'] = torch.as_tensor(recon_smplx_params['body_pose']).reshape(-1, 21, 3)[:, self.upper_body_inds, :]
        self.frames_params['lower_body_pose'] = torch.as_tensor(recon_smplx_params['body_pose']).reshape(-1, 21, 3)[:, self.lower_body_inds, :]
        self.frames_params['lhand_pose'] = torch.as_tensor(recon_smplx_params['lhand_pose'])
        self.frames_params['rhand_pose'] = torch.as_tensor(recon_smplx_params['rhand_pose'])
        for para_key in self.paras_keys:
            if para_key in self.shared_keys:
                self.frames_params[para_key] = self.frames_params[para_key][0:1]  ### make the shared param only have single value
            self.frames_params[para_key] = self.frames_params[para_key].to(self.device)
        
        
        resize_scale = min(1., 400./max(img.shape[:2]))  ### resolution for differentiable rendering loss computation
        self.mask_size = (int(resize_scale*img.shape[0]), int(resize_scale*img.shape[1]))
        self.resize_scale = (self.mask_size[0]/float(img.shape[0]), self.mask_size[1]/float(img.shape[1]))

        ### Step 2: opt global_orient and transl for every frame
        batch_size = 500
        video_for_debug = None
        for i in tqdm(range(0, frames_num, batch_size), desc = 'opt global_orient and transl'):
            start_id, end_id = i, min((i+batch_size), ldmks_2d.shape[0])
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
            masks, parsing_maps, normal_maps = self.load_masks(imgs_folder.replace('ori_imgs', 'seg_masks'), ldmks_names[cur_ids], self.mask_size)
            self.masks = torch.as_tensor(masks, device = ldmks_2d.device).float() / 255.
            self.parsing_maps = torch.as_tensor(parsing_maps, device = ldmks_2d.device)
            self.normal_maps = torch.as_tensor(normal_maps, device = ldmks_2d.device)
            is_opt_dict = deepcopy(self.is_opt_dict_init)
            is_opt_dict.update({'global_orient': True, 'transl': True})
            with_losses_dict = deepcopy(self.with_losses_dict_init)
            dict_for_subinds = self.opt_smplx_paras(dict_for_subinds, cam_para.expand(cur_batch_size, -1), ldmks_2d[cur_ids], iris_ldmks_all[cur_ids], is_opt_dict, with_losses_dict, confidence = ldmks_confidence[cur_ids], adam_lr = 1e-1, iter_nums=600, ldmk_ratio=1., current_step='global_orient&transl')
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

            body_pose = torch.cat((paras_dict['upper_body_pose'], paras_dict['lower_body_pose']), dim=1)[:, self.get_body_inds, :].reshape(cur_batch_size, -1)
            smplx_out, iris_verts = self.smplx_model.forward(betas = paras_dict['shape'], body_pose = body_pose, left_hand_pose = paras_dict['lhand_pose'], right_hand_pose = paras_dict['rhand_pose'], jaw_pose = paras_dict['jaw_pose'], expression = paras_dict['expr'], with_iris_return=True, leye_pose=paras_dict['leye_pose'], reye_pose=paras_dict['reye_pose'], global_orient=paras_dict['global_orient'], transl=paras_dict['transl'])
            if video_for_debug is None:
                video_for_debug = VideoWriter(os.path.join(debug_dir, 'smplx_fitting_rigid.mp4'))
            for j in range(0, end_id-start_id):
                img = cv2.imread(os.path.join(imgs_folder, ldmks_names[j+start_id].replace('ldmks', 'ori_imgs').replace('.wb', '.jpg')))
                render_vis = self.mesh_renderer.forward_visualization_geo(smplx_out.vertices[j:j+1], self.tris, cam_para, img.shape[:2], torch.ones_like(torch.as_tensor(img).to(self.device)[None, :, :, [2,1,0]].float()/255.))
                vis_img = render_vis[0] 
                vis_img = np.concatenate((img[:,:,::-1], vis_img), axis=1)
                video_for_debug.write_frame(vis_img)
        if video_for_debug is not None:
            video_for_debug.close()
            video_for_debug = None

        
        ### Step 3: opt shared shape and fixed later
        batch_size = 100
        cur_ids = np.arange(0, ldmks_2d.shape[0], max(1, ldmks_2d.shape[0]//batch_size))
        self.sel_ids = cur_ids
        cur_batch_size = cur_ids.shape[0]
        dict_for_subinds = {}
        for key in self.paras_keys:
            if key in self.shared_keys:
                dict_for_subinds[key] = self.frames_params[key].clone()
            else:
                dict_for_subinds[key] = self.frames_params[key][cur_ids].clone()
        masks, parsing_maps, normal_maps = self.load_masks(imgs_folder.replace('ori_imgs', 'seg_masks'), ldmks_names[cur_ids], self.mask_size)
        self.masks = torch.as_tensor(masks, device = ldmks_2d.device).float() / 255.
        self.parsing_maps = torch.as_tensor(parsing_maps, device = ldmks_2d.device)
        self.normal_maps = torch.as_tensor(normal_maps, device = ldmks_2d.device)
        is_opt_dict = deepcopy(self.is_opt_dict_init)
        for key in is_opt_dict.keys():
            is_opt_dict[key] = True
        with_losses_dict = deepcopy(self.with_losses_dict_init)
        with_losses_dict.update({'mask': True, 'feature': True})
        dict_for_subinds = self.opt_smplx_paras(dict_for_subinds, cam_para.expand(cur_batch_size, -1), ldmks_2d[cur_ids], iris_ldmks_all[cur_ids], is_opt_dict, with_losses_dict, confidence = ldmks_confidence[cur_ids], adam_lr = 3e-3, iter_nums=900, ldmk_ratio=1.5, ldmk_feature_ratio=0., current_step='opt shared shape', init_body_pose = body_pose_init[cur_ids])
        for key in self.paras_keys:
            if key in self.shared_keys:
                self.frames_params[key] = dict_for_subinds[key].clone()
            # else:
            #     self.frames_params[key][cur_ids] = dict_for_subinds[key].clone()

        ### Step 4: fixed shared and opt frame dependent params
        batch_size = 500
        video_for_debug = None
        for i in tqdm(range(0, frames_num, batch_size), desc = 'opt frames dependent params'):
            start_id, end_id = i, min((i+batch_size), ldmks_2d.shape[0])
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
            masks, parsing_maps, normal_maps = self.load_masks(imgs_folder.replace('ori_imgs', 'seg_masks'), ldmks_names[cur_ids], self.mask_size)
            self.masks = torch.as_tensor(masks, device = ldmks_2d.device).float() / 255.
            self.parsing_maps = torch.as_tensor(parsing_maps, device = ldmks_2d.device)
            self.normal_maps = torch.as_tensor(normal_maps, device = ldmks_2d.device)
            is_opt_dict = deepcopy(self.is_opt_dict_init)
            for key in self.paras_keys:
                if key not in self.shared_keys:
                    is_opt_dict[key] = True
            
            with_losses_dict = deepcopy(self.with_losses_dict_init)
            for key in with_losses_dict.keys():
                if key != 'normal':
                    with_losses_dict[key] = True
            dict_for_subinds = self.opt_smplx_paras(dict_for_subinds, cam_para.expand(cur_batch_size, -1), ldmks_2d[cur_ids], iris_ldmks_all[cur_ids], is_opt_dict, with_losses_dict, confidence = ldmks_confidence[cur_ids], adam_lr = 3e-3, iter_nums=600, with_fixed_bary=True, ldmk_ratio=2., ldmk_mask_ratio = .1, ldmk_feature_ratio=.3, current_step='frames batch' + str(i), init_body_pose = body_pose_init[cur_ids])
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

            body_pose = torch.cat((paras_dict['upper_body_pose'], paras_dict['lower_body_pose']), dim=1)[:, self.get_body_inds, :].reshape(cur_batch_size, -1)
            smplx_out, iris_verts = self.smplx_model.forward(betas = paras_dict['shape'], body_pose = body_pose, left_hand_pose = paras_dict['lhand_pose'], right_hand_pose = paras_dict['rhand_pose'], jaw_pose = paras_dict['jaw_pose'], expression = paras_dict['expr'], with_iris_return=True, leye_pose=paras_dict['leye_pose'], reye_pose=paras_dict['reye_pose'], global_orient=paras_dict['global_orient'], transl=paras_dict['transl'])
            if video_for_debug is None:
                video_for_debug = VideoWriter(os.path.join(debug_dir, 'smplx_fitting_final.mp4'))
            for j in range(0, end_id-start_id):
                img = cv2.imread(os.path.join(imgs_folder, ldmks_names[j+start_id].replace('ldmks', 'ori_imgs').replace('.wb', '.jpg')))
                render_vis = self.mesh_renderer.forward_visualization_geo(smplx_out.vertices[j:j+1], self.tris, cam_para, img.shape[:2], torch.ones_like(torch.as_tensor(img).to(self.device)[None, :, :, [2,1,0]].float()/255.))
                vis_img = render_vis[0] 
                vis_img = np.concatenate((img[:,:,::-1], vis_img), axis=1)
                video_for_debug.write_frame(vis_img)
        if video_for_debug is not None:
            video_for_debug.close()
            video_for_debug = None
        

        paras_dict = {}
        for key in self.paras_keys:
            if key in self.shared_keys:
                paras_dict[key] = self.frames_params[key].clone()
                paras_dict[key] = paras_dict[key].expand(frames_num, *(paras_dict[key]).shape[1:])
            else:
                paras_dict[key] = self.frames_params[key].clone()

        paras_dict['body_pose'] = torch.cat((paras_dict['upper_body_pose'], paras_dict['lower_body_pose']), dim=1)[:, self.get_body_inds, :].reshape(frames_num, -1)

        save_keys = ['body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'shape', 'expr', 'global_orient', 'transl']
        save_dict = {}
        save_dict['cam_para'] = cam_para.detach().cpu().numpy()
        for save_key in save_keys:
            save_dict[save_key] = paras_dict[save_key].detach().cpu().numpy()
        torch.save(save_dict, os.path.join(save_folder, 'smplx_track.pth'))
