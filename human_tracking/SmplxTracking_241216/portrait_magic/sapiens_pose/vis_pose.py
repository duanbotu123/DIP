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
from .adhoc_image_dataset import AdhocImageDataset
from tqdm import tqdm
import torch.nn as nn
from typing import List, Union


from ..utils_funcs.video_writer import VideoWriter

from .pose_utils import top_down_affine_transform, udp_decode

from .classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    COCO_SKELETON_INFO,
    COCO_WHOLEBODY_SKELETON_INFO
)

timings = {}
BATCH_SIZE = 8

def preprocess_pose(orig_img, bboxes_list, input_shape, mean, std):
    """Preprocess pose images and bboxes."""
    preprocessed_images = []
    centers = []
    scales = []
    for bbox in bboxes_list:
        img, center, scale = top_down_affine_transform(orig_img.copy(), bbox)
        img = cv2.resize(
            img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        img = (img - mean) / std
        preprocessed_images.append(img)
        centers.extend(center)
        scales.extend(scale)
    return preprocessed_images, centers, scales

def batch_inference_topdown(
    model: nn.Module,
    imgs: List[Union[np.ndarray, str]],
    dtype=torch.bfloat16,
    flip=False,
):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        heatmaps = model(imgs.cuda())
        if flip:
            heatmaps_ = model(imgs.to(dtype).cuda().flip(-1))
            heatmaps = (heatmaps + heatmaps_) * 0.5
        imgs.cpu()
    return heatmaps.cpu()

def img_save_and_vis(
    img, results, img_name, input_shape, heatmap_scale, kpt_colors, kpt_thr, radius, skeleton_info, thickness
):
    # pred_instances_list = split_instances(result)
    heatmap = results["heatmaps"]
    centres = results["centres"]
    scales = results["scales"]
    img_shape = img.shape
    instance_keypoints = []
    instance_scores = []
    # print(scales[0], centres[0])
    for i in range(len(heatmap)):
        result = udp_decode(
            heatmap[i].cpu().unsqueeze(0).float().data[0].numpy(),
            input_shape,
            (int(input_shape[0] / heatmap_scale), int(input_shape[1] / heatmap_scale)),
        )

        keypoints, keypoint_scores = result
        keypoints = (keypoints / input_shape) * scales[i] + centres[i] - 0.5 * scales[i]
        instance_keypoints.append(keypoints[0])
        instance_scores.append(keypoint_scores[0])

    instance_keypoints = np.array(instance_keypoints).astype(np.float32)
    instance_scores = np.array(instance_scores).astype(np.float32)
    instance_scores[instance_scores<.5] = 0.

    pts_with_scores = np.concatenate((instance_keypoints.reshape(-1,2), instance_scores.reshape(-1,1)), axis=-1)

    keypoints_visible = np.ones(instance_keypoints.shape[:-1])
    for kpts, score, visible in zip(
        instance_keypoints, instance_scores, keypoints_visible
    ):
        kpts = np.array(kpts, copy=False)

        if (
            kpt_colors is None
            or isinstance(kpt_colors, str)
            or len(kpt_colors) != len(kpts)
        ):
            raise ValueError(
                f"the length of kpt_color "
                f"({len(kpt_colors)}) does not matches "
                f"that of keypoints ({len(kpts)})"
            )

        # draw each point on image
        for kid, kpt in enumerate(kpts):
            if score[kid] < kpt_thr or not visible[kid] or kpt_colors[kid] is None:
                # skip the point that should not be drawn
                continue

            color = kpt_colors[kid]
            if not isinstance(color, str):
                color = tuple(int(c) for c in color[::-1])
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius), color, -1)
    
        # draw skeleton
        for skid, link_info in skeleton_info.items():
            pt1_idx, pt2_idx = link_info['link']
            color = link_info['color'][::-1] # BGR

            pt1 = kpts[pt1_idx]; pt1_score = score[pt1_idx]
            pt2 = kpts[pt2_idx]; pt2_score = score[pt2_idx]

            if pt1_score > kpt_thr and pt2_score > kpt_thr:
                x1_coord = int(pt1[0]); y1_coord = int(pt1[1])
                x2_coord = int(pt2[0]); y2_coord = int(pt2[1])
                cv2.line(img, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)
        cv2.putText(img, os.path.basename(img_name), (100, 100), 1, 1, (0,0,255))

    return pts_with_scores, img[:,:,::-1]

def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

class PoseEstimator():
    def __init__(self):
        self.device = 'cuda:0'
        file_dir_path = os.path.dirname(os.path.realpath(__file__))
        checkpoint = os.path.join(file_dir_path, 'sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2')
        self.model = load_model(checkpoint, True).to(self.device)
        self.batch_size = 16
        self.input_shape = [1024, 768]
        self.fp16 = False

        if len(self.input_shape) == 1:
            self.input_shape = (3, self.input_shape[0], self.input_shape[0])
        elif len(self.input_shape) == 2:
            self.input_shape = (3,) + tuple(self.input_shape)
        else:
            raise ValueError("invalid input shape")
        torch._inductor.config.force_fuse_int_mm_with_mul = True
        torch._inductor.config.use_mixed_mm = True

    def run_folder(self, imgs_folder, pose_folder, debug_dir='none'):
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
        if not os.path.exists(pose_folder):
            os.makedirs(pose_folder)
        global BATCH_SIZE
        BATCH_SIZE = self.batch_size

        inference_dataset = AdhocImageDataset([os.path.join(input_dir, img_name) for img_name in image_names])
        inference_dataloader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )

        video_out = None

        for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs, bboxes) in tqdm(
            enumerate(inference_dataloader), total=len(inference_dataloader)
        ):
            # print(batch_idx)
            valid_images_len = len(batch_orig_imgs)
            batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
            bboxes = bboxes.numpy()

            bboxes_batch = []
            for i in range(bboxes.shape[0]):
                bboxes_batch.append(bboxes[i].reshape(1,4))

            img_bbox_map = {}
            for i, bboxes in enumerate(bboxes_batch):
                img_bbox_map[i] = len(bboxes)

            pose_ops = []
            for i, bbox_list in zip(batch_orig_imgs.numpy(), bboxes_batch):
                pose_op = preprocess_pose(
                    i,
                    bbox_list,
                    (self.input_shape[1], self.input_shape[2]),
                    [123.5, 116.5, 103.5],
                    [58.5, 57.0, 57.5],
                )
                pose_ops.append(pose_op)

            pose_imgs, pose_img_centers, pose_img_scales = [], [], []
            for op in pose_ops:
                pose_imgs.extend(op[0])
                pose_img_centers.extend(op[1])
                pose_img_scales.extend(op[2])

            n_pose_batches = (len(pose_imgs) + self.batch_size - 1) // self.batch_size

            # use this to tell torch compiler the start of model invocation as in 'flip' mode the tensor output is overwritten
            torch.compiler.cudagraph_mark_step_begin()  
            pose_results = []
            for i in range(n_pose_batches):
                imgs = torch.stack(
                    pose_imgs[i * self.batch_size : (i + 1) * self.batch_size], dim=0
                )
                valid_len = len(imgs)
                imgs = fake_pad_images_to_batchsize(imgs)
                pose_results.extend(
                    batch_inference_topdown(self.model, imgs, dtype=dtype)[:valid_len]
                )

            batched_results = []
            for _, bbox_len in img_bbox_map.items():
                result = {
                    "heatmaps": pose_results[:bbox_len].copy(),
                    "centres": pose_img_centers[:bbox_len].copy(),
                    "scales": pose_img_scales[:bbox_len].copy(),
                }
                batched_results.append(result)
                del (
                    pose_results[:bbox_len],
                    pose_img_centers[:bbox_len],
                    pose_img_scales[:bbox_len],
                )

            assert len(batched_results) == len(batch_orig_imgs)

            for i, r, img_name in zip(
                batch_orig_imgs[:valid_images_len],
                batched_results[:valid_images_len],
                batch_image_name,
            ):
                pts_with_scores, vis_img = img_save_and_vis(
                    i.numpy(),
                    r,
                    img_name,
                    (self.input_shape[2], self.input_shape[1]),
                    4,
                    COCO_WHOLEBODY_KPTS_COLORS,
                    0.3,
                    6,
                    COCO_WHOLEBODY_SKELETON_INFO,
                    6,
                )
                np.savetxt(os.path.join(pose_folder, os.path.basename(img_name)[:-4] + '.wb'), pts_with_scores)
                if debug_dir != 'none':
                    if video_out is None:
                        video_out = VideoWriter(os.path.join(debug_dir, 'sapiens_pose.mp4'))
                    video_out.write_frame(vis_img)
        
        if video_out is not None:
            video_out.close()
        total_time = time.time() - start
        fps = 1 / ((time.time() - start) / len(image_names))
        print(
            f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
        )

