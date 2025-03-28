import cv2
from tqdm import tqdm
import os
import torch
import numpy as np
from ..utils_funcs.video_writer import VideoWriter
import mediapipe as mp
from natsort import natsorted

def check_ldmks_intersect(mp_hands_ldmks, wb_hands_ldmks):
    """
    check if two landmarks sets intersect
    """
    is_vis = (wb_hands_ldmks[..., 2] > .7)
    if np.sum(is_vis.astype(np.float32)) < 5:
        return False
    wb_ldmks = wb_hands_ldmks[is_vis, :2]
    mp_ldmks = mp_hands_ldmks[is_vis, :2]
    a_l, a_r, a_t, a_b = np.min(wb_ldmks[:, 0]), np.max(wb_ldmks[:, 0]), np.min(wb_ldmks[:, 1]), np.max(wb_ldmks[:, 1])
    box_size = max(a_r-a_l, a_b-a_t)
    ldmks_mean_dis = np.mean(np.abs(mp_ldmks - wb_ldmks))
    return (ldmks_mean_dis<(.5*box_size))

class MPHands():
    def __init__(self):
       mp_hands = mp.solutions.hands
       self.mphands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


    def run_folder_refine(self, imgs_folder, save_folder, debug_dir = 'none'):
        ldmks_folder = imgs_folder.replace('ori_imgs', 'ldmks')
        ldmk_names = natsorted([f for f in os.listdir(ldmks_folder) if f.endswith('.wb')])

        video_out = None
        hand_ldmks_num = 21

        for ldmk_name in tqdm(ldmk_names, total = len(ldmk_names)):
            img_path = os.path.join(imgs_folder, ldmk_name.replace('.wb', '.jpg'))
            wb_ldmks = np.loadtxt(os.path.join(ldmks_folder, ldmk_name), dtype=np.float32)
            lhand_ldmks = wb_ldmks[91:112]
            rhand_ldmks = wb_ldmks[112:133]
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            hand_results = self.mphands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if not hand_results.multi_hand_landmarks:
                continue
            
            is_refined = False
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_lands_np = np.zeros((hand_ldmks_num, 2), dtype=np.float32)
                for ldmk_idx, ldmk in enumerate(hand_landmarks.landmark):
                    hand_lands_np[ldmk_idx] = [ldmk.x * img_w, ldmk.y * img_h]

                show_col = (0, 255, 0)
                confi = np.ones(hand_ldmks_num)
                lhand_dist = np.mean(np.abs(hand_lands_np - lhand_ldmks[..., :2]))
                rhand_dist = np.mean(np.abs(hand_lands_np - rhand_ldmks[..., :2]))

                if lhand_dist < rhand_dist:
                    show_col = (255, 0, 0)
                    wb_ldmks[91:112, :2] = np.array(hand_lands_np)
                    is_refined = True
                    wb_ldmks[91:112, 2] = 1.
                else:
                    show_col = (0, 0, 255)
                    wb_ldmks[112:133, :2] = np.array(hand_lands_np)
                    is_refined = True
                    wb_ldmks[112:133, 2] = 1.

                # if check_ldmks_intersect(hand_lands_np, lhand_ldmks):
                #     show_col = (255, 0, 0)
                #     wb_ldmks[91:112, :2] = np.array(hand_lands_np)
                #     is_refined = True
                #     wb_ldmks[91:112, 2] = 1.
                # elif check_ldmks_intersect(hand_lands_np, rhand_ldmks):
                #     show_col = (0, 0, 255)
                #     wb_ldmks[112:133, :2] = np.array(hand_lands_np)
                #     is_refined = True
                #     wb_ldmks[112:133, 2] = 1.
                # else:
                #     print('no existing hand')

                for ldmk_idx in range(hand_ldmks_num):
                    if confi[ldmk_idx] > .7:
                        cv2.circle(img, (int(hand_lands_np[ldmk_idx, 0]), int(hand_lands_np[ldmk_idx, 1])), 3, show_col, -1)
            if is_refined:
                np.savetxt(os.path.join(ldmks_folder, ldmk_name), wb_ldmks, '%f') 

            if debug_dir != 'none':
                if video_out is None:
                    video_out = VideoWriter(os.path.join(debug_dir, 'hand_ldmks_refine.mp4')) 
                video_out.write_frame(img[:, :, ::-1])

        if video_out is not None:
            video_out.close()

        

        
    def run_image(self, oriImg):
        ### oriImg: numpy.array RGB image -> (133, 3) xs,ys,scores
        input, trans = self.folder_dataset_tmp.process_img(oriImg)
        with torch.inference_mode():
            outputs = self.dslpt_model(input.cuda().unsqueeze(0))[0]
        ldmks = outputs[2][:, -1].cpu().numpy() * 256.0
        trans = trans.numpy()
        ldmks[0] = transform_pixel_v2(ldmks[0], trans, inverse=True)
        return ldmks[0]

