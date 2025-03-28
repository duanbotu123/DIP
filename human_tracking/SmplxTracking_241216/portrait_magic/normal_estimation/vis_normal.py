# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import multiprocessing as mp
import os
import time
# from multiprocessing import cpu_count, Pool

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from .adhoc_image_dataset import AdhocImageDataset
from tqdm import tqdm

# from .worker_pool import WorkerPool

from ..utils_funcs.video_writer import VideoWriter

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 8


def warmup_model(model, batch_size):
    # Warm up the model with a dummy input.
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.bfloat16).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s


def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        results = model(imgs.to(dtype).cuda())
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)

def normal_vis(image, result):
    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)
    normal_map = seg_logits.float().data.numpy().transpose(1, 2, 0)  ## H x W. seg ids.
    normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map_normalized = normal_map / (normal_map_norm + 1e-5)  # Add a small e

    normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
    normal_map = normal_map[:, :, ::-1]

    vis_image = np.concatenate([image, normal_map], axis=1)
    return vis_image[:,:,::-1]


def img_save_and_viz(image, result, output_path, seg_dir):
    output_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", ".npy")
    )

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)
    normal_map = seg_logits.float().data.numpy().transpose(1, 2, 0)  ## H x W. seg ids.
    if seg_dir is not None:
        mask_path = os.path.join(
            seg_dir,
            os.path.basename(output_path)
            .replace(".png", ".npy")
            .replace(".jpg", ".npy")
            .replace(".jpeg", ".npy"),
        )
        mask = np.load(mask_path)
    else:
        mask = np.ones_like(normal_map)
    normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map_normalized = normal_map / (normal_map_norm + 1e-5)  # Add a small e
    np.save(output_file, normal_map_normalized)

    normal_map_normalized[mask == 0] = -1  ## visualize background (nan) as black
    normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
    normal_map = normal_map[:, :, ::-1]

    vis_image = np.concatenate([image, normal_map], axis=1)
    cv2.imwrite(output_path, vis_image)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

class NormalEstimator():
    def __init__(self):
        self.device = 'cuda:0'
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        checkpoint = os.path.join(file_dir_path, 'sapiens_2b_normal_render_people_epoch_70_torchscript.pt2')
        self.model = load_model(checkpoint, True).to(self.device)
        self.batch_size = 2
        self.input_shape = [1024, 768]
        self.fp16 = False

        if len(self.input_shape) == 1:
            self.input_shape = (3, self.input_shape[0], self.input_shape[0])
        elif len(self.input_shape) == 2:
            self.input_shape = (3,) + tuple(self.input_shape)
        else:
            raise ValueError("invalid input shape")

    def run_folder(self, imgs_folder, normal_folder, debug_dir='none'):
        # mp.log_to_stderr()
        # torch._inductor.config.force_fuse_int_mm_with_mul = True
        # torch._inductor.config.use_mixed_mm = True

        start = time.time()
        dtype = torch.float16 if self.fp16 else torch.float32
        

        image_names = []
        input_dir = imgs_folder  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg")
            or image_name.endswith(".png")
            or image_name.endswith(".jpeg")
        ]
        if not os.path.exists(normal_folder):
            os.makedirs(normal_folder)
        global BATCH_SIZE
        BATCH_SIZE = self.batch_size

        inference_dataset = AdhocImageDataset(
            [os.path.join(input_dir, img_name) for img_name in image_names],
            (self.input_shape[1], self.input_shape[2]),
            mean=[123.5, 116.5, 103.5],
            std=[58.5, 57.0, 57.5],
        )
        inference_dataloader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # img_save_pool = WorkerPool(
        #     img_save_and_viz, processes=max(min(self.batch_size, cpu_count()), 1)
        # )

        video_out = None

        for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
            enumerate(inference_dataloader), total=len(inference_dataloader)
        ):
            valid_images_len = len(batch_imgs)
            batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
            result = inference_model(self.model, batch_imgs, dtype=dtype)

            # args_list = [
            #     (
            #         i,
            #         r,
            #         os.path.join(normal_folder, os.path.basename(img_name)),
            #         None,
            #     )
            #     for i, r, img_name in zip(
            #         batch_orig_imgs[:valid_images_len],
            #         result[:valid_images_len],
            #         batch_image_name,
            #     )
            # ]
            # img_save_pool.run_async(args_list)

            for i, r, img_name in zip(
                batch_orig_imgs[:valid_images_len],
                result[:valid_images_len],
                batch_image_name,
            ):
                img_save_and_viz(
                    i,
                    r,
                    os.path.join(normal_folder, os.path.basename(img_name)),
                    None,
                )
                if debug_dir != 'none':
                    if video_out is None:
                        video_out = VideoWriter(os.path.join(debug_dir, 'normal_maps.mp4'))
                    vis_img = normal_vis(i, r)
                    video_out.write_frame(vis_img)

        if video_out is not None:
            video_out.close()
            
        total_time = time.time() - start
        fps = 1 / ((time.time() - start) / len(image_names))
        print(
            f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
        )

