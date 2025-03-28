import torch
from .folder_loader import Folder_Dataset
from .scene_adteeth import GaussianScene
from.config import pipeline_config, optim_config
from tqdm import tqdm
import cv2
from .gaussian_renderer import render
from ..render_utils.loss_utils import l1_loss, ssim
from ..utils_funcs.video_writer import VideoWriter
import numpy as np
import lpips
import time
import os

class GaussianDriver():
    def __init__(self, data_dir, device = 'cuda:0'):
        self.device = device
        self.folder_dataset = Folder_Dataset(data_dir)
        self.save_dir = os.path.join(data_dir, 'gaussian_scene')
        os.makedirs(self.save_dir, exist_ok=True)
        self.img_size = self.folder_dataset.img_size
        self.gaussian_scene = GaussianScene(data_dir, self.img_size, device=self.device)
        self.gaussian_scene.training_setup(optim_config)

        if os.path.isfile(os.path.join(self.save_dir, 'recon.pt')):
            self.gaussian_scene.restore(torch.load(os.path.join(self.save_dir, 'recon.pt')))

        bg_color = [0, 1, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
    
    def driving(self, driven_track_path, filename):
        vid_out = VideoWriter(filename)
        frames_num = self.gaussian_scene.load_driven_track(driven_track_path)
        for frame_id in tqdm(range(frames_num)):
            with torch.no_grad():
                self.gaussian_scene.update_xyz_rot_driven(frame_id)
                viewpoint_cam = self.gaussian_scene.cameras[0]
                render_pkg = render(viewpoint_cam, self.gaussian_scene, pipeline_config, self.background)
                render_img = render_pkg["render"]
                vis_img = torch.clamp(render_img.permute(1,2,0).detach()*255, 0, 255).byte()
                vid_out.write_frame(vis_img.cpu().numpy())
        vid_out.close()
        