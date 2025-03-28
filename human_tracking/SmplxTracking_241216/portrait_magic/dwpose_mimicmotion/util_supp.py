import os, sys
import numpy as np

from .util import draw_pose

coco_kps_idx = (55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 60, 
    61, 62, 63, 64, 65, # body 23
    127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 
    139, 140, 141, 142, 143, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 
    124, 125, 126, # face 68
    20, # left hand root
    37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 
    31, 32, 33, 70, 
    21, # right hand root
    52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 
    46, 47, 48, 75
    )

def pts_to_image_scale(keypoints_info, height, width, hip_scale=1):
    # convert coco keypoints 2 dwpose keypoints following /home/juyonggroup/xiangjun/github/MimicMotion/mimicmotion/dwpose/wholebody.py
    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:4] = np.logical_and(
        keypoints_info[:, 5, 2:4] > 0.3,
        keypoints_info[:, 6, 2:4] > 0.3).astype(int)
    new_keypoints_info = np.insert(
        keypoints_info, 17, neck, axis=1)
    mmpose_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    openpose_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    new_keypoints_info[:, openpose_idx] = \
        new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    keypoints, scores = keypoints_info[
        ..., :2], keypoints_info[..., 2]
    
    # /home/juyonggroup/xiangjun/github/MimicMotion/mimicmotion/dwpose/dwpose_detector.py
    candidate, score = keypoints, scores
    nums, _, locs = candidate.shape

    body = candidate[:, :18].copy()
    body = body.reshape(nums * 18, locs)

    # rescale hips
    hip_left = body[8]
    hip_right = body[11]
    hip_middle = (hip_left + hip_right) / 2
    # hip_scale = 1
    hip_left_scale = hip_middle + (hip_left - hip_middle)*hip_scale
    hip_right_scale = hip_middle + (hip_right - hip_middle)*hip_scale
    body[8] = hip_left_scale
    body[11] = hip_right_scale
    # ipdb.set_trace()

    subset = score[:, :18].copy()
    for i in range(len(subset)):
        for j in range(len(subset[i])):
            if subset[i][j] > 0.3:
                subset[i][j] = int(18 * i + j)
            else:
                subset[i][j] = -1

    # un_visible = subset < 0.3
    # candidate[un_visible] = -1

    # foot = candidate[:, 18:24]

    faces = candidate[:, 24:92]

    hands = candidate[:, 92:113]
    hands = np.vstack([hands, candidate[:, 113:]])

    faces_score = score[:, 24:92]
    hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

    bodies = dict(candidate=body, subset=subset, score=score[:, :18])
    pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

    pose_img = draw_pose(pose, height, width)

    return pose_img  