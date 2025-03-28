import cv2
import numpy as np
from portrait_magic.face_landmarks import DSLPT_utils
import argparse
from natsort import natsorted
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default='./data/test.mp4')
    parser.add_argument('--target_path', type=str, default='./data/test_dir')

    parser.add_argument('--save_path', type=str, default='./data/test_dir')
    args = parser.parse_args()

    source_path = args.source_path
    target_path = args.target_path

    source_img_paths = natsorted([os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.jpg')])
    target_img_paths = natsorted([os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith('.jpg')])

    print(source_img_paths[0], target_img_paths[0])
    source_img = cv2.imread(source_img_paths[0])
    target_img = cv2.imread(target_img_paths[0])

    face_landmark_util = DSLPT_utils()
    source_ldmks = face_landmark_util.run_image(source_img[:,:,::-1])
    target_ldmks = face_landmark_util.run_image(target_img[:,:,::-1])

    p_num = source_ldmks.shape[0]
    mat_A = np.zeros((p_num*2, 3), dtype=np.float32)
    vec_b = np.zeros(p_num*2).astype(np.float32)
    mat_A[:p_num, 1] += 1.
    mat_A[p_num:, 2] += 1.
    mat_A[:p_num, 0] = source_ldmks[:, 0]
    mat_A[p_num:, 0] = source_ldmks[:, 1]
    vec_b[:p_num] = target_ldmks[:, 0]
    vec_b[p_num:] = target_ldmks[:, 1]
    vec_x = np.linalg.pinv(mat_A.T @ mat_A) @ mat_A.T @ vec_b
    trans_mat = np.array((vec_x[0], 0, vec_x[1], 0, vec_x[0], vec_x[2])).reshape(2, 3)
    
    for source_img_path in source_img_paths:
        source_img = cv2.imread(source_img_path)
        transed_source_img = cv2.warpAffine(source_img, trans_mat, (target_img.shape[1], target_img.shape[0]))
        cv2.imwrite(os.path.join(args.save_path, os.path.basename(source_img_path)), transed_source_img)
