import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.io import load_obj
from pytorch3d.transforms import so3_exp_map
from pytorch3d.ops import sample_farthest_points
from .folder_loader import Folder_Dataset
from ..utils_funcs.video_writer import VideoWriter
from ..render_utils.render_nvdiff import MeshRenderer_Cuda
from ..render_utils.geometry_utils import unproj_pts, pts2obj
from ..render_utils import enlarge_upper_masks
from ..dmm_models import SMPLX, smplx_model_path
from ..dmm_models.adnerf_rendering import adnerf_rendering_part_ids_path, adnerf_teeth_mesh_file, adnerf_gaussian_info_file, template_mesh_file


class TrainImgsWriter():
    def __init__(self):
        self.device = 'cuda:0'
        self.mesh_renderer = MeshRenderer_Cuda()
        shape_dim, expr_dim = 300, 100
        self.smplx_model = SMPLX(smplx_model_path, num_expression_coeffs=expr_dim, num_betas=shape_dim, use_pca = False).to(self.device).eval()

        self.adnerf_rendering_part_ids = np.loadtxt(adnerf_rendering_part_ids_path, dtype=np.int64)
        if os.path.isfile(template_mesh_file):
            _, faces, _ = load_obj(template_mesh_file)
        else:
            _, faces, _ = load_obj(adnerf_teeth_mesh_file)
        self.tris = faces.verts_idx[None, ...].to(self.device).int()

        self.dict_keys = ['body_pose', 'lhand_pose', 'rhand_pose', 'jaw_pose', 'expr', 'img_ori', 'seg_img', 'shape', 'cam_para', 'parsing_img', 'global_orient', 'transl', 'leye_pose', 'reye_pose']

        adnerf_gaussian_info = torch.load(adnerf_gaussian_info_file)
        self.hair_vid = adnerf_gaussian_info['hair_vid']

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        folder_dataset = Folder_Dataset(imgs_folder.replace('ori_imgs', 'ldmks'))
        folder_loader = torch.utils.data.DataLoader(
            folder_dataset, batch_size=5, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
        hair_frame_id = 0
        video_out = None
        for idx, ret_dict in enumerate(tqdm(folder_loader, desc='training imgs saver')):            
            for key in self.dict_keys:
                ret_dict[key] = ret_dict[key].to(self.device)
            smplx_out, verts_rott, v_shaped = self.smplx_model.forward(betas = ret_dict['shape'], body_pose = ret_dict['body_pose'], left_hand_pose = ret_dict['lhand_pose'], right_hand_pose = ret_dict['rhand_pose'], jaw_pose = ret_dict['jaw_pose'], expression = ret_dict['expr'], with_rott_return = True, leye_pose=ret_dict['leye_pose'], reye_pose=ret_dict['reye_pose'], global_orient=ret_dict['global_orient'], transl=ret_dict['transl'])
            adnerf_verts_cam = smplx_out.vertices
            rast_out = self.mesh_renderer.forward_rasterization(adnerf_verts_cam, ret_dict['cam_para'], self.tris, ret_dict['img_ori'].shape[1:3])
            rendering_mask = rast_out[..., 3]>0
            ret_dict['seg_img'][ret_dict['parsing_img'] == 11] = 255

            if False: ### full head:
                rendering_mask = (enlarge_upper_masks(rendering_mask.float()) > .5)
                rendering_mask[ret_dict['parsing_img'] == 14] = True
                rendering_mask[ret_dict['parsing_img'] == 3] = False #### remove clothes
                rendering_mask[ret_dict['parsing_img'] == 0] = False #### bg
            else:
                rendering_mask = (enlarge_upper_masks(rendering_mask.float()) > .5)
                rendering_mask[ret_dict['parsing_img'] == 14] = True
                # rendering_mask[ret_dict['parsing_img'] == 0] = False #### bg


            # ret_dict['seg_img'][~rendering_mask] = 0
            # if idx == hair_frame_id:
            #     hair_mask = (ret_dict['seg_img'][0]>1) & (rast_out[0][..., 3]<0.5)
            #     xys = torch.nonzero(hair_mask)[:, [1,0]].float().unsqueeze(0)
            #     Zs = torch.ones_like(xys[..., 0]) * adnerf_verts_cam[0, self.hair_vid, 2]
            #     hair_verts_cam = unproj_pts(ret_dict['cam_para'][:1], xys, Zs)
            #     hair_verts = torch.bmm(hair_verts_cam - ret_dict['cam_trans'][:1].unsqueeze(1), rots[:1])
            #     hair_verts = torch.bmm(hair_verts - verts_rott[:, self.adnerf_rendering_part_ids][:1, self.hair_vid, :, 3].unsqueeze(1), verts_rott[:1, self.hair_vid, :, :3])
            #     hair_verts_canonical = sample_farthest_points(hair_verts, K = hair_verts.shape[1]//10)[0].squeeze(0)
            #     print('hair point size', hair_verts_canonical.shape[0])
            #     torch.save(hair_verts_canonical.cpu(), os.path.join(save_folder, 'hair_canonical_pts.pt'))

            ret_dict['seg_img'] = ret_dict['seg_img'].unsqueeze(-1)
            ret_dict['img_ori'] = (ret_dict['img_ori'] * ret_dict['seg_img'].float() / 255. + (1.-ret_dict['seg_img'].float() / 255.)*torch.tensor((255, 255, 255), dtype=torch.float32, device=self.device).view(1, 1, 1, 3)).byte()

            imgs_saved = torch.cat((ret_dict['img_ori'], ret_dict['seg_img']), dim=-1).cpu().numpy()
            for i in range(ret_dict['img_ori'].shape[0]):
                cv2.imwrite(os.path.join(save_folder, ret_dict['img_name'][i][:-4] + '.png'), imgs_saved[i])

            if debug_dir == 'none':
                continue
            if video_out is None:
                video_out = VideoWriter(os.path.join(debug_dir, 'for_training.mp4'))
            for i in range(ret_dict['img_ori'].shape[0]):
                video_out.write_frame(ret_dict['img_ori'][i, :, :, [2,1,0]].cpu().numpy())
        if video_out is not None:
            video_out.close()        
                