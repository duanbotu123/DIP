import torch
from .folder_loader import Folder_Dataset
from .uv_gaussian import GaussianScene
from.config import pipeline_config, optim_config, model_config
from tqdm import tqdm
import cv2
from .gaussian_renderer import render_batch
from ..render_utils.loss_utils import l1_loss, ssim, l1_loss_with_mask
from ..utils_funcs.video_writer import VideoWriter
import numpy as np
import lpips
import time
import os

class GaussianTrainer():
    def __init__(self, data_dir, head_data_dir, device = 'cuda:0'):
        self.device = device
        self.folder_dataset = Folder_Dataset(data_dir, head_data_dir, 0., 1.)
        self.eval_dataset = Folder_Dataset(data_dir, head_data_dir, .0, 1.)
        print('datset size', len(self.folder_dataset), len(self.eval_dataset))
        self.save_dir = os.path.join(data_dir, 'gaussian_scene')
        self.log_dir = os.path.join(data_dir, 'gaussian_scene', 'logs')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.img_size = self.folder_dataset.img_size
        self.gaussian_scene = GaussianScene(data_dir, self.img_size, head_data_dir, self.folder_dataset.head_img_size, sh_degree=model_config.sh_degree, device=self.device)
        self.gaussian_scene.training_setup(optim_config)

        if os.path.isfile(os.path.join(self.save_dir, 'recon.pt')):
            self.gaussian_scene.restore(torch.load(os.path.join(self.save_dir, 'recon.pt')))

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.percep_module = lpips.LPIPS(net='vgg').to(self.device)

        self.test_folder_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=5, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
    
    def eval(self, epoch, filename = None):
        if filename is None:
            vid_out = VideoWriter(os.path.join(self.log_dir, str(epoch).zfill(4) + '_eval.mp4'), fps=30)
        else:
            vid_out = VideoWriter(filename, fps=30)
        for img_gt, img_parsing, _, head_img_gt, head_img_parsing, _, frame_ids in tqdm(self.test_folder_loader, desc='eval'):
            img_gt = img_gt.to(self.device)
            batch_size = img_gt.shape[0]
            with torch.no_grad():
                viewpoint_cams = [self.gaussian_scene.cameras[int(frame_id)] for frame_id in frame_ids]
                mesh_vis = self.gaussian_scene.update_xyz_rot(frame_ids, in_train=False, mode='mixture')
                render_pkg = render_batch(viewpoint_cams, self.gaussian_scene, pipeline_config, self.background)
                render_img, render_mask = render_pkg["render"], render_pkg['mask']
                
                render_img = torch.clamp(render_img.permute(0,2,3,1).detach()*255, 0, 255)
                img_gt = img_gt[..., :3].float()

# 'label_names':  {'0: background', '1: neck', '2: face', '3: cloth', '4: rr', '5: lr', '6: rb', '7: lb', '8: re',
#                     '9: le', '10: nose', '11: imouth', '12: llip', '13: ulip', '14: hair',
#                     '15: eyeg', '16: hat', '17: earr', '18: neck_l'}

                gt_parsing = img_parsing.to(self.device)
                gt_img_head_mask = ((gt_parsing==1) | (gt_parsing==2) | (gt_parsing==4) | (gt_parsing==5) | (gt_parsing==6) | (gt_parsing==7) | (gt_parsing==8) | (gt_parsing==9) | (gt_parsing==10) | (gt_parsing==11) | (gt_parsing==12) | (gt_parsing==13) | (gt_parsing==14) | (gt_parsing==15) | (gt_parsing==17) | (gt_parsing==18)).float()

                render_img = render_img * gt_img_head_mask[..., None] + img_gt * (1 - gt_img_head_mask[..., None])

                vis_img = torch.cat((render_img.byte(), img_gt.byte()), dim=2)

                for i in range(batch_size):
                    write_img = vis_img[i].cpu().numpy()
                    write_img = np.concatenate((mesh_vis[i], write_img), axis=1)
                    vid_out.write_frame(write_img)
        vid_out.close()

    def cal_G_loss(self, render_img, gt_img, rects):
        batch_size = render_img.shape[0]
        loss_G = 0
        for i in range(batch_size):
            rect = rects[i]
            loss_G += torch.mean(self.percep_module(render_img[i:i+1, :, rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]*2.-1., gt_img[i:i+1, :, rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]*2.-1.))
        return loss_G / batch_size

    def train(self):
        epoch_num = min(int((40000000*40-1)/len(self.folder_dataset)) + 1, 21)
        train_batch_size = 2
        folder_loader = torch.utils.data.DataLoader(
            self.folder_dataset, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=False)
        percep_epoch_start = epoch_num // 2
        save_epoch_freq = 50
        upsh_per_epoch = 10

        loss_keys = ['l1', 'mask', 'mesh_mask', 'ssim', 'G', 'lap', 'dis', 'scale', 'mesh_lap', 'normal_consistent', 'gs_mesh_concistent', 'head_l1', 'head_G']
        losses_wts = {'l1': 1.0 - optim_config.lambda_dssim, 'ssim': optim_config.lambda_dssim, 'mask': 1., 'mesh_mask': 1., 'G': .1, 'lap': 1e1, 'dis': 1e-1, 'scale': 1e1, 'mesh_lap': 5e1, 'normal_consistent': 1e-2, 'gs_mesh_concistent': 1e1, 'head_l1': 1.0, 'head_G': 1e-1}

        for epoch in tqdm(range(epoch_num)):
            if epoch % save_epoch_freq == 0:
                self.eval(epoch)
            losses_accum = {}
            for loss_key in loss_keys:
                losses_accum[loss_key] = []

            for img_gt, img_parsing, precep_rects, head_img_gt, head_img_parsing, head_precep_rects, frame_ids in tqdm(folder_loader, total=len(self.folder_dataset)//train_batch_size, desc='training'):
                #### prepare batched data
                img_gt = img_gt.to(self.device)
                gt_img = img_gt[..., :3].permute(0,3,1,2).float()/255.
                gt_parsing = img_parsing.to(self.device)
                gt_img_head_mask = (gt_parsing==11) | (gt_parsing==12) | (gt_parsing==13)
                color_valid_mask = 1. - gt_img_head_mask.unsqueeze(1).float()
                gt_mask = img_gt[..., 3].float()/255.
                gt_mesh_mask = gt_mask.clone()
                gt_mesh_mask[gt_mask>.8] = 1.
                mesh_mask, loss_normal_consistent = self.gaussian_scene.update_xyz_rot(frame_ids, in_train=True)
                viewpoint_cams = [self.gaussian_scene.cameras[int(frame_id)] for frame_id in frame_ids]

                #### GS splatting
                render_pkg = render_batch(viewpoint_cams, self.gaussian_scene, pipeline_config, self.background)
                render_img, render_mask = render_pkg["render"], render_pkg['mask']

                #### compute losses
                losses = {}
                losses['l1'] = l1_loss(render_img*color_valid_mask, gt_img*color_valid_mask)
                losses['mask'] = l1_loss(render_mask.squeeze(1), gt_mask)
                losses['mesh_mask'] = l1_loss(mesh_mask.squeeze(-1), gt_mesh_mask)
                losses['normal_consistent'] = loss_normal_consistent
                losses['ssim'] = 1.0 - ssim(render_img*color_valid_mask, gt_img*color_valid_mask)
                losses['scale'] = self.gaussian_scene.cal_scale_loss()
                losses['lap'] = self.gaussian_scene.cal_lap_loss()
                losses['dis'] = self.gaussian_scene.cal_dis_loss()
                losses['mesh_lap'] = self.gaussian_scene.cal_mesh_lap_loss()
                losses['gs_mesh_concistent'] = self.gaussian_scene.cal_gs_mesh_consistent_loss()
                if epoch > percep_epoch_start:
                    losses['G'] = self.cal_G_loss(render_img*color_valid_mask, gt_img*color_valid_mask, precep_rects) 
                else:
                    losses['G'] = torch.zeros_like(losses['l1'])

                head_img_gt = head_img_gt.to(self.device)
                head_gt_img = head_img_gt[..., :3].permute(0,3,1,2).float()/255.
                self.gaussian_scene.update_xyz_rot(frame_ids, in_train=True, with_mesh_rendering=False, mode='head')
                head_viewpoint_cams = [self.gaussian_scene.head_cameras[int(frame_id)] for frame_id in frame_ids]
                head_render_pkg = render_batch(head_viewpoint_cams, self.gaussian_scene, pipeline_config, self.background)
                head_render_img = head_render_pkg["render"]
                head_gt_parsing = head_img_parsing.to(self.device)



                # head_gt_img_head_mask = (head_gt_parsing==8) | (head_gt_parsing==9) |  (head_gt_parsing==11) | (head_gt_parsing==12) | (head_gt_parsing==13) | (head_gt_parsing==15)
                head_gt_img_head_mask = (head_gt_parsing==2) | (head_gt_parsing==4) | (head_gt_parsing==5) | (head_gt_parsing==6) | (head_gt_parsing==7) | (head_gt_parsing==8) | (head_gt_parsing==9) |  (head_gt_parsing==10) | (head_gt_parsing==11) | (head_gt_parsing==12) | (head_gt_parsing==13) | (head_gt_parsing==14) | (head_gt_parsing==15) | (head_gt_parsing==17)

                head_color_valid_mask = head_gt_img_head_mask.unsqueeze(1).float()
                losses['head_l1'] = l1_loss(head_render_img*head_color_valid_mask, head_gt_img*head_color_valid_mask)
                if epoch > percep_epoch_start:
                    losses['head_G'] = self.cal_G_loss(head_render_img*head_color_valid_mask, head_gt_img*head_color_valid_mask, head_precep_rects)
                else:
                    losses['head_G'] = torch.zeros_like(losses['l1'])
                
                
                #### accum losses and backward
                loss = 0.
                for loss_key in loss_keys:
                    losses_accum[loss_key].append(losses[loss_key].item())
                    loss += losses[loss_key] * losses_wts[loss_key]
                loss.backward()
                with torch.no_grad():
                    self.gaussian_scene.optimizer.step()
                    self.gaussian_scene.optimizer.zero_grad(set_to_none=True)

            if (epoch+1) % upsh_per_epoch == 0:
                self.gaussian_scene.oneupSHdegree()

            print_info = ''
            for loss_key in loss_keys:
                print_info += loss_key + ': ' + format(np.mean(np.array(losses_accum[loss_key])), '.5f') + ', '
            print(print_info)
            if (epoch+1) % save_epoch_freq == 0:
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'recon.pt'))
        self.eval(-1, os.path.join(self.save_dir, 'recon.mp4'))
        torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'recon.pt'))
