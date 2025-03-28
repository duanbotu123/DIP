from .RobustVideoMatting.rvm_wrapper import RVMWrapper
from .folder_loader import Folder_Dataset
from ..utils_funcs.video_writer import VideoWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import os
import numpy as np
import torch

def largest_connected(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2 > 155

class RVMatter():
    def __init__(self, device = 'cuda:0'):
        self.device = device
        self.rvm_wrapper = RVMWrapper(self.device)

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        self.rvm_wrapper.init_rec()
        folder_dataset = Folder_Dataset(imgs_folder)
        folder_dataloader = DataLoader(
            folder_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False)
        video_out = None
        for img_ori, img_name in tqdm(folder_dataloader, desc = 'matting portrait'):
            img_cuda = img_ori.to(self.device).permute(0,3,1,2).float()/255.
            pha = self.rvm_wrapper.run(img_cuda)

            # for i in range(len(img_name)):
            #     is_valid = largest_connected((pha[i,0].cpu().numpy() > .5).astype(np.uint8))
            #     pha[i,0] = pha[i,0] * torch.from_numpy(is_valid).to(self.device).float()

            for i in range(len(img_name)):
                cv2.imwrite(os.path.join(save_folder, img_name[i][:-4]+'.png'), (pha[i,0,:,:]*255.).byte().cpu().numpy().astype(np.uint8))

            if debug_dir != 'none':
                imgs_ori = img_ori.to(self.device)
                bg_col = torch.tensor((255, 0, 0)).cuda().float().view(1, 1, 1, -1)
                imgs_debug = imgs_ori * pha.squeeze(0).unsqueeze(-1) + bg_col * (1.-pha.squeeze(0).unsqueeze(-1))
                imgs_debug = imgs_debug.detach().byte().cpu().numpy()
                if video_out is None:
                    video_out = VideoWriter(os.path.join(debug_dir, 'segs.mp4'))
                for i in range(imgs_debug.shape[0]):
                    video_out.write_frame(imgs_debug[i])
        if video_out is not None:
            video_out.close()