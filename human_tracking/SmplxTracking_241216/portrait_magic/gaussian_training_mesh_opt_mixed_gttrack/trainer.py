import torch
from .folder_loader import Folder_Dataset
from .uv_gaussian import GaussianScene
from.config import pipeline_config, optim_config, model_config
from tqdm import tqdm
import cv2
from .gaussian_renderer import render_batch
from ..render_utils.loss_utils import l1_loss, ssim
from ..utils_funcs.video_writer import VideoWriter
import numpy as np
import lpips
import time
import os

class GaussianEvaler():
    def __init__(self, data_dir, head_smplx_path, device = 'cuda:0'):
        self.device = device
        self.eval_dataset = Folder_Dataset(data_dir, .0, 1.)
        self.save_dir = os.path.join(data_dir, 'gaussian_scene')
        os.makedirs(self.save_dir, exist_ok=True)
        self.img_size = self.eval_dataset.img_size
        self.gaussian_scene = GaussianScene(data_dir, self.img_size, head_smplx_path, sh_degree=model_config.sh_degree, device=self.device)
        self.gaussian_scene.training_setup(optim_config)

        if os.path.isfile(os.path.join(self.save_dir, 'recon.pt')):
            self.gaussian_scene.restore(torch.load(os.path.join(self.save_dir, 'recon.pt')))

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.test_folder_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=5, shuffle=False, num_workers=5, pin_memory=True, drop_last=False)
    
    def eval(self):
        filename = os.path.join(self.save_dir, 'recon_no_track.mp4')
        vid_out = VideoWriter(filename, fps=30)
        save_folder = os.path.join(self.save_dir, 'recon_rgba')
        os.makedirs(save_folder, exist_ok=True)
        save_id = 0
        for img_gt, frame_ids in tqdm(self.test_folder_loader, desc='eval'):
            img_gt = img_gt.to(self.device)
            batch_size = img_gt.shape[0]
            with torch.no_grad():
                viewpoint_cams = [self.gaussian_scene.cameras[int(frame_id)] for frame_id in frame_ids]
                mesh_vis = self.gaussian_scene.update_xyz_rot(frame_ids, in_train=False)
                render_pkg = render_batch(viewpoint_cams, self.gaussian_scene, pipeline_config, self.background)
                render_img, render_mask = render_pkg["render"], render_pkg['mask']
                rgba_img = torch.clamp(torch.cat((render_img.permute(0,2,3,1).detach()*255, render_mask.permute(0,2,3,1).detach()*255), dim=-1), 0, 255).byte().cpu().numpy()
                bgra_img = rgba_img[..., [2, 1, 0, 3]]
                vis_img = torch.cat((torch.clamp(render_img.permute(0,2,3,1).detach()*255, 0, 255).byte(), img_gt[..., :3]), dim=2)
                for i in range(batch_size):
                    cv2.imwrite(os.path.join(save_folder, f'{save_id:06d}.png'), bgra_img[i])
                    save_id += 1
                for i in range(batch_size):
                    write_img = vis_img[i].cpu().numpy()
                    write_img = np.concatenate((mesh_vis[i], write_img), axis=1)
                    vid_out.write_frame(write_img)
        vid_out.close()
