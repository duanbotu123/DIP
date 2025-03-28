import torch
from .folder_loader import Folder_Dataset
from .scene_adteeth import GaussianScene
from.config import pipeline_config, optim_config, model_config
from tqdm import tqdm
import cv2
from .gaussian_renderer import render
from ..render_utils.loss_utils import l1_loss, ssim
from ..utils_funcs.video_writer import VideoWriter
import numpy as np
import lpips
import time
import os

class GaussianTrainer():
    def __init__(self, data_dir, device = 'cuda:0'):
        self.device = device
        self.folder_dataset = Folder_Dataset(data_dir, 0., 1.)
        self.eval_dataset = Folder_Dataset(data_dir, .8, 1.)
        self.save_dir = os.path.join(data_dir, 'gaussian_scene')
        self.log_dir = os.path.join(data_dir, 'gaussian_scene', 'logs')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.img_size = self.folder_dataset.img_size
        self.gaussian_scene = GaussianScene(data_dir, self.img_size, sh_degree=model_config.sh_degree, device=self.device)
        self.gaussian_scene.training_setup(optim_config)

        if os.path.isfile(os.path.join(self.save_dir, 'recon.pt')):
            self.gaussian_scene.restore(torch.load(os.path.join(self.save_dir, 'recon.pt')))

        bg_color = [0, 1, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.percep_module = lpips.LPIPS(net='vgg').to(self.device)
    
    def eval(self, epoch, filename = None):
        test_folder_loader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=5, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
        if filename is None:
            vid_out = VideoWriter(os.path.join(self.log_dir, str(epoch).zfill(4) + '_eval.mp4'))
        else:
            vid_out = VideoWriter(filename)
        eval_time = 0
        for img_gt, frame_ids, mouth_rects in test_folder_loader:
            img_gt = img_gt.to(self.device)
            batch_size = img_gt.shape[0]
            with torch.no_grad():
                for i in range(batch_size):
                    frame_id = int(frame_ids[i])
                    time_start = time.time()
                    self.gaussian_scene.update_xyz_rot(frame_id, False, inverse_frame_expr = False, with_pose_delta=True)
                    viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                    render_pkg = render(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background)
                    torch.cuda.synchronize()
                    eval_time += time.time() - time_start
                    render_img, render_mask = render_pkg["render"], render_pkg['mask']
                    render_img = render_img*render_mask + (torch.tensor((1, 0, 0)).to(render_img.device).view(3, 1, 1))*(1.-render_mask)
                    img_gt_view = img_gt[i, :, :, :3]
                    vis_img = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view), dim=1)
                    vid_out.write_frame(vis_img.cpu().numpy())
        vid_out.close()
        print('rendering fps', 1./(eval_time/len(self.folder_dataset)))

    def train(self):
        epoch_num = min(int((optim_config.iterations-1)/len(self.folder_dataset)) + 1, 301)
        upsh_per_epoch = int(2000/len(self.folder_dataset)) + 1
        train_batch_size = 10
        folder_loader = torch.utils.data.DataLoader(
            self.folder_dataset, batch_size=train_batch_size, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
        percep_epoch_start = 10
        save_epoch_freq = 20
        for epoch in tqdm(range(epoch_num)):
            if epoch % save_epoch_freq == 0:
                self.eval(epoch)
            loss_l1s = []
            loss_masks = []
            loss_ssims = []
            loss_Gs = []
            loss_laps = []
            loss_poses = []
            loss_diss = []
            loss_rotts = []
            loss_scales = []
            for img_gt, frame_ids, mouth_rects in tqdm(folder_loader, total=len(self.folder_dataset)//train_batch_size, desc='training'):
                img_gt = img_gt.to(self.device)
                batch_size = img_gt.shape[0]
                loss_l1, loss_ssim, loss_G, loss_mask, loss_rott = 0., 0., 0., 0., 0.
                for i in range(batch_size):
                    frame_id = int(frame_ids[i])
                    mouth_rect = mouth_rects[i]
                    self.gaussian_scene.update_xyz_rot(frame_id, False, with_pose_delta=True, with_pose_disturb=True)
                    viewpoint_cam = self.gaussian_scene.cameras[frame_id]
                    render_pkg = render(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background)
                    render_img, render_mask = render_pkg["render"], render_pkg['mask']
                    gt_img = img_gt[i, :, :, :3].permute(2,0,1).float()/255.
                    gt_mask = img_gt[i, :, :, 3].float()/255.
                    l1_dis = l1_loss(render_img, gt_img)
                    l1_mask = l1_loss(render_mask.squeeze(0), gt_mask)
                    loss_l1 += l1_dis
                    loss_mask += l1_mask
                    ssim_dis = 1.0 - ssim(render_img, gt_img)
                    loss_ssim += ssim_dis
                    rott_delta_dis = self.gaussian_scene.cal_rott_loss()
                    loss_rott += rott_delta_dis

                    percep_dis = torch.zeros_like(loss_l1)
                    if epoch > percep_epoch_start:
                        # percep_dis = torch.mean(self.percep_module(render_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)*2.-1., gt_img[:, mouth_rect[1]:mouth_rect[1]+mouth_rect[3], mouth_rect[0]:mouth_rect[0]+mouth_rect[2]].unsqueeze(0)*2.-1.))
                        percep_dis = torch.mean(self.percep_module(render_img.unsqueeze(0)*2.-1., gt_img.unsqueeze(0)*2.-1.))
                    loss_G += percep_dis
                    
                    loss_l1s.append(l1_dis.item())
                    loss_ssims.append(ssim_dis.item())
                    loss_Gs.append(percep_dis.item())
                    loss_masks.append(l1_mask.item())
                    loss_rotts.append(rott_delta_dis.item())
                    
                loss_lap = self.gaussian_scene.cal_lap_loss()
                loss_scale = self.gaussian_scene.cal_scale_loss()
                loss_pose = self.gaussian_scene.cal_pose_loss(frame_ids)
                loss_dis = self.gaussian_scene.cal_dis_loss()
                loss_diss.append(loss_dis.item())
                loss_l1 /= batch_size
                loss_ssim /= batch_size
                loss_G /= batch_size
                loss_mask /= batch_size
                loss_rott /= batch_size
                loss = (1.0 - optim_config.lambda_dssim) * loss_l1 + optim_config.lambda_dssim * loss_ssim + .5 * loss_mask + loss_lap * 1e2 + loss_dis*1e2 + loss_rott*1e1 + loss_scale * 1e-0 # + loss_pose*1e-1
                loss_laps.append(loss_lap.item())
                loss_poses.append(loss_pose.item())
                loss_scales.append(loss_scale.item())
                if epoch > percep_epoch_start:
                    loss += loss_G * 1e-1
                loss.backward()
                with torch.no_grad():
                    self.gaussian_scene.optimizer.step()
                    self.gaussian_scene.optimizer.zero_grad(set_to_none=True)
                # print(loss_l1.item(), loss_mask.item(), loss_lap.item())
            if (epoch+1) % upsh_per_epoch == 0:
                self.gaussian_scene.oneupSHdegree()

            print(np.mean(np.array(loss_l1s)), np.mean(np.array(loss_ssims)), np.mean(np.array(loss_Gs)), np.mean(np.array(loss_masks)), np.mean(np.array(loss_laps)), np.mean(np.array(loss_diss)), np.mean(np.array(loss_poses)), np.mean(np.array(loss_rotts)), np.mean(np.array(loss_scales)), '\n')
            img_gt_view = img_gt[-1, :, :, :3]
            img_debug = torch.cat((torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte(), img_gt_view), dim=1)[..., [2,1,0]]
            cv2.imwrite(os.path.join(self.log_dir, 'train_' + str(epoch).zfill(4) + '.jpg'), img_debug.cpu().numpy())
            if (epoch+1) % save_epoch_freq == 0:
                torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'recon.pt'))
            # self.gaussian_scene.save_updated_canonical('debug_canoncial.obj')
        self.eval(-1, os.path.join(self.save_dir, 'recon.mp4'))
        torch.save(self.gaussian_scene.capture(), os.path.join(self.save_dir, 'recon.pt'))
