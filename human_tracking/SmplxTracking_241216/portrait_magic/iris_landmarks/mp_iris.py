import cv2
from tqdm import tqdm
import os
import torch
import numpy as np
from ..utils_funcs.video_writer import VideoWriter
import mediapipe as mp
from natsort import natsorted

def check_ldmks_intersect(mp_iris_ldmks, wb_eyes_ldmks):
    """
    check if two landmarks sets intersect
    """
    wb_ldmks = wb_eyes_ldmks
    mp_ldmks = mp_iris_ldmks
    a_l, a_r, a_t, a_b = np.min(wb_ldmks[:, 0]), np.max(wb_ldmks[:, 0]), np.min(wb_ldmks[:, 1]), np.max(wb_ldmks[:, 1])
    out_box_num = (np.sum(mp_ldmks[:, 0]<a_l) + np.sum(mp_ldmks[:, 0]>a_r) + np.sum(mp_ldmks[:, 1]<a_t) + np.sum(mp_ldmks[:, 1]>a_b))/4.
    # print(out_box_num, mp_ldmks.shape[0])
    return out_box_num<(mp_ldmks.shape[0]*.3)

class MPIris():
    def __init__(self):
       mp_face_mesh = mp.solutions.face_mesh
       self.mpface = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def run_folder(self, imgs_folder, save_folder, debug_dir = 'none'):
        ldmks_folder = imgs_folder.replace('ori_imgs', 'ldmks')
        ldmk_names = natsorted([f for f in os.listdir(ldmks_folder) if f.endswith('.wb')])
        iris_num = 10

        video_out = None

        for ldmk_name in tqdm(ldmk_names, total = len(ldmk_names)):
            img_path = os.path.join(imgs_folder, ldmk_name.replace('.wb', '.jpg'))
            wb_ldmks = np.loadtxt(os.path.join(ldmks_folder, ldmk_name), dtype=np.float32)
            
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            face_results = self.mpface.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not face_results.multi_face_landmarks:
                iris_tmp = np.zeros((iris_num, 3), dtype=np.float32)
                np.savetxt(os.path.join(save_folder, ldmk_name.replace('.wb', '.iris')), iris_tmp, '%f')
                continue

            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in face_results.multi_face_landmarks[0].landmark])
            iris_lands_np = mesh_points[468:, :2]

            show_col = (0, 255, 0)
            confi = np.ones(iris_num)
            mp_iris_lands = np.concatenate((iris_lands_np, confi.reshape(-1, 1)), axis=-1)
            is_intersect = check_ldmks_intersect(mp_iris_lands, wb_ldmks[[42, 47, 59, 68, 51]])
            if not is_intersect:
                mp_iris_lands[:, 2] = 0.
            np.savetxt(os.path.join(save_folder, ldmk_name.replace('.wb', '.iris')), mp_iris_lands, '%f')

            for ldmk_idx in range(iris_num):
                cv2.circle(img, (int(iris_lands_np[ldmk_idx, 0]), int(iris_lands_np[ldmk_idx, 1])), 3, show_col, -1)

            if debug_dir != 'none':
                if video_out is None:
                    video_out = VideoWriter(os.path.join(debug_dir, 'iris_landmarks.mp4')) 
                video_out.write_frame(img[:, :, ::-1])
        if video_out is not None:
            video_out.close()

