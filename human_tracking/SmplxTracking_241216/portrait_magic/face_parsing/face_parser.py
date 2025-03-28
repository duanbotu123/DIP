import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from .folder_loader import Folder_Dataset
from ..utils_funcs.video_writer import VideoWriter
import seaborn as sns

sns.color_palette

from ..third_parties.facer import face_parser


class FaceParser():
    def __init__(self):
        torch._C._jit_set_profiling_mode(False)
        self.device = 'cuda:0'
        self.face_parser = face_parser('farl/celebm/448', device=self.device)
        n_colors = 19
        self.colors = torch.as_tensor(np.array(sns.color_palette(None, n_colors)), dtype=torch.float, device = self.device)

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        folder_dataset = Folder_Dataset(imgs_folder.replace('ori_imgs', 'ldmks'))
        folder_loader = torch.utils.data.DataLoader(
            folder_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, drop_last=False)
        video_out = None
        for img_rgb, pts5d, img_names in tqdm(folder_loader, desc='face parsing'):
            # print('batch start')
            imgs = img_rgb.to(self.device)[..., [2,1,0]].permute(0, 3, 1, 2).float() / 255.
            pts5d = pts5d.to(self.device)
            with torch.inference_mode():
                parsing_results = self.face_parser.forward_with_imgs_points(imgs, pts5d)
                parsing_imgs = parsing_results.squeeze(1).byte().cpu().numpy()
                for i in range(imgs.shape[0]):
                    cv2.imwrite(os.path.join(save_folder, img_names[i].replace('.wb', '.png')), (parsing_imgs[i]))
                
            if debug_dir != 'none':
                if video_out is None:
                    video_out = VideoWriter(os.path.join(debug_dir, 'face_parsing.mp4'))
                vis_col = self.colors[parsing_results.squeeze(1).long()]
                imgs_blend = ((imgs.permute(0, 2, 3, 1)[..., [2,1,0]] * .2 + vis_col * .8)*255.).byte().cpu().numpy()
                for i in range(imgs.shape[0]):
                    video_out.write_frame(imgs_blend[i])
        if video_out is not None:
            video_out.close()

            
                
                