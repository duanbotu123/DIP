from .folder_loader import Folder_Dataset
from ..utils_funcs.video_writer import VideoWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import os
import numpy as np
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image

class ImagePreprocessor():
    def __init__(self, resolution = (1024, 1024)) -> None:
        self.transform_image = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image

def refine_foreground(image, mask, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(image, mask, r=r)
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    return image_masked

def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(
        image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * \
        (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B


class BiRefNet():
    def __init__(self, device = 'cuda:0'):
        self.device = device
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(os.path.join(file_dir_path, 'vid_mat_models'), local_files_only=True, trust_remote_code=True)
        self.birefnet.to(device)
        self.birefnet.eval()
        self.resolution = (1024, 1024)
        self.resizer = transforms.Resize(self.resolution)
        self.normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def run_folder(self, imgs_folder, save_folder, debug_dir='none'):
        folder_dataset = Folder_Dataset(imgs_folder)
        folder_dataloader = DataLoader(
            folder_dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=False)
        video_out = None
        for img_ori, img_name in tqdm(folder_dataloader, desc = 'matting portrait'):
            img_proc = img_ori.to(self.device).permute(0,3,1,2)
            img_proc = self.resizer(img_proc)
            img_proc = self.normalizer(img_proc.float()/255.)

            with torch.no_grad():
                preds = self.birefnet(img_proc)[-1].sigmoid().cpu()
            for i in range(preds.shape[0]):
                alpha = preds[0].squeeze().numpy()
                alpha = cv2.resize(alpha, (img_ori[i].shape[1], img_ori[i].shape[0]))
                pred_mask = (alpha*255.).astype(np.uint8)
                cv2.imwrite(os.path.join(save_folder, img_name[i][:-4]+'.png'), pred_mask)
                if debug_dir != 'none':
                    img = img_ori[i].numpy()
                    bg_col = np.array((255, 0, 0)).reshape(1, 1, 3)
                    alpha = alpha[:,:,None]
                    imgs_debug = img*alpha + bg_col*(1.-alpha) 
                    imgs_debug = imgs_debug.astype(np.uint8)
                    if video_out is None:
                        video_out = VideoWriter(os.path.join(debug_dir, 'segs.mp4'))
                    video_out.write_frame(imgs_debug)

        if video_out is not None:
            video_out.close()